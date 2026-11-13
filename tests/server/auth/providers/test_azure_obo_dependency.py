"""OBO dependencies use the authentication provider of the serving HTTP app."""

import asyncio
import contextvars
import json
import secrets
import weakref
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docket import Docket
from fastmcp_tasks.context import (
    TaskContextSnapshot,
    get_task_context,
    restore_task_snapshot,
)
from fastmcp_tasks.dependencies import _current_docket
from fastmcp_tasks.keys import build_task_key
from key_value.aio.stores.memory import MemoryStore
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from pydantic import AnyHttpUrl

from fastmcp import Client, Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, AuthProvider, MultiAuth, RemoteAuthProvider
from fastmcp.server.auth._context import _get_effective_auth
from fastmcp.server.auth.providers.azure import AzureProvider, EntraOBOToken
from fastmcp.server.auth.providers.jwt import (
    JWTVerifier,
    RSAKeyPair,
    StaticTokenVerifier,
)
from fastmcp.server.dependencies import (
    _current_server,
    get_access_token,
    get_http_request,
)
from fastmcp.utilities.tests import asgi_server
from fastmcp_tasks import TasksExtension

pytest.importorskip("azure.identity.aio")


def _auth(name: str, *, client_secret: str | None = None) -> MultiAuth:
    provider = AzureProvider(
        client_id=f"{name}-client",
        client_secret=client_secret or f"{name}-secret",
        tenant_id=f"{name}-tenant",
        required_scopes=["read"],
        additional_authorize_scopes=["https://graph.microsoft.com/User.Read"],
        base_url=f"https://{name}.example.com",
        client_storage=MemoryStore(),
        jwt_signing_key=f"{name}-signing-key",
    )
    return MultiAuth(
        server=provider,
        verifiers={
            "static": StaticTokenVerifier(
                tokens={f"{name}-token": {"client_id": name, "scopes": ["read"]}}
            )
        },
    )


@pytest.fixture
def credential_factory() -> Iterator[MagicMock]:
    def credential(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            get_token=AsyncMock(
                return_value=SimpleNamespace(token=kwargs["client_id"])
            ),
            close=AsyncMock(),
        )

    with patch(
        "azure.identity.aio.OnBehalfOfCredential", side_effect=credential
    ) as factory:
        yield factory


@pytest.mark.parametrize(
    "configuration", ["inherited", "app-only", "different-default"]
)
@pytest.mark.parametrize(
    ("transport", "background"), [("http", False), ("sse", False), ("http", True)]
)
async def test_obo_uses_serving_app_auth(
    configuration: str,
    transport: Literal["http", "sse"],
    background: bool,
    credential_factory: MagicMock,
) -> None:
    app_auth = _auth("app")
    default_auth: MultiAuth | None = None
    if configuration == "inherited":
        default_auth = app_auth
    elif configuration == "different-default":
        default_auth = _auth("default")
    server = FastMCP("obo-app-auth", auth=default_auth)
    if background:
        server.add_extension(TasksExtension())

    @server.tool(task=background)
    async def downstream(
        token: str = EntraOBOToken(["https://graph.microsoft.com/User.Read"]),
    ) -> str:
        assert (get_task_context() is not None) is background
        return token

    options: dict[str, Any] = {} if configuration == "inherited" else {"auth": app_auth}
    async with asgi_server(server, transport=transport, **options) as running:
        async with running.client(auth="app-token") as client:
            result = await client.call_tool("downstream")
            assert result.data == "app-client"

    credential_factory.assert_called_once_with(
        tenant_id="app-tenant",
        client_id="app-client",
        client_secret="app-secret",
        user_assertion="app-token",
        authority="https://login.microsoftonline.com",
    )
    assert server.auth is default_auth


@pytest.mark.parametrize("http_auth_disabled", [False, True])
@pytest.mark.parametrize("background", [False, True])
async def test_obo_server_auth_fallback_is_only_for_non_http_requests(
    http_auth_disabled: bool,
    background: bool,
    credential_factory: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    default_auth = _auth("default")
    server = FastMCP("obo-fallback", auth=default_auth)
    if background:
        server.add_extension(TasksExtension())

    @server.tool(task=background)
    async def downstream(
        token: str = EntraOBOToken(["https://graph.microsoft.com/User.Read"]),
    ) -> str:
        assert (get_task_context() is not None) is background
        return token

    access_token = AccessToken(token="user-token", client_id="user", scopes=["read"])
    marker = auth_context_var.set(AuthenticatedUser(access_token))
    try:
        if http_auth_disabled:
            async with asgi_server(server, auth=None) as running:
                async with running.client() as client:
                    with pytest.raises(ToolError):
                        await client.call_tool("downstream")
            assert "Current provider: NoneType" in caplog.text
            credential_factory.assert_not_called()
        else:
            async with Client(server) as client:
                result = await client.call_tool("downstream")
                assert result.data == "default-client"
            credential_factory.assert_called_once_with(
                tenant_id="default-tenant",
                client_id="default-client",
                client_secret="default-secret",
                user_assertion="user-token",
                authority="https://login.microsoftonline.com",
            )
    finally:
        auth_context_var.reset(marker)
    assert server.auth is default_auth


@pytest.mark.parametrize("parent_auth_kind", ["jwt", "remote", "azure"])
@pytest.mark.parametrize("background", [False, True])
async def test_mounted_obo_uses_child_credentials(
    parent_auth_kind: str,
    background: bool,
    rsa_key_pair: RSAKeyPair,
    credential_factory: MagicMock,
) -> None:
    verifier = JWTVerifier(
        public_key=rsa_key_pair.public_key, issuer="https://parent.example.com"
    )
    parent_auth: AuthProvider = verifier
    bearer = rsa_key_pair.create_token(
        issuer="https://parent.example.com",
        subject="user",
        scopes=["read"],
        additional_claims={"client_id": "parent-client"},
    )
    if parent_auth_kind == "remote":
        parent_auth = RemoteAuthProvider(
            token_verifier=verifier,
            authorization_servers=[AnyHttpUrl("https://parent.example.com")],
            base_url="https://mcp.example.com",
        )
    elif parent_auth_kind == "azure":
        parent_auth = _auth("parent")
        bearer = "parent-token"

    parent = FastMCP("obo-parent")
    if background:
        parent.add_extension(TasksExtension())
    child_auth = _auth("child")
    child = FastMCP("obo-child", auth=child_auth)

    @child.tool(task=background)
    async def downstream(
        token: str = EntraOBOToken(["https://graph.microsoft.com/User.Read"]),
    ) -> str:
        assert (get_task_context() is not None) is background
        return token

    parent.mount(child, namespace="child")
    async with asgi_server(parent, auth=parent_auth) as running:
        async with running.client(auth=bearer) as client:
            result = await client.call_tool("child_downstream")
            assert result.data == "child-client"

    credential_factory.assert_called_once_with(
        tenant_id="child-tenant",
        client_id="child-client",
        client_secret="child-secret",
        user_assertion=bearer,
        authority="https://login.microsoftonline.com",
    )
    assert parent.auth is None
    assert child.auth is child_auth


@pytest.mark.parametrize(
    "configuration",
    ["inherited", "app-only", "different-default", "missing", "mounted"],
)
async def test_obo_snapshot_restores_with_rebuilt_worker_configuration(
    configuration: str,
    credential_factory: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only serialized context crosses from the HTTP app to a fresh worker."""
    tool_name = "child_downstream" if configuration == "mounted" else "downstream"

    async def capture_from_http_app() -> str:
        app_auth = _auth("app")
        default_auth = (
            app_auth
            if configuration == "inherited"
            else _auth("default")
            if configuration in {"different-default", "missing"}
            else None
        )
        source = FastMCP("snapshot-source", auth=default_auth)
        owner = source
        if configuration == "mounted":
            owner = FastMCP("snapshot-child", auth=_auth("child"))
            source.mount(owner, namespace="child")

        @owner.tool
        async def downstream() -> str:
            return TaskContextSnapshot.capture(owning_tool_name=tool_name).to_json()

        async with asgi_server(source, auth=app_auth) as running:
            async with running.client(auth="app-token") as client:
                result = await client.call_tool(tool_name)
                assert isinstance(result.data, str)
                return result.data

    serialized = await capture_from_http_app()
    payload = json.loads(serialized)
    assert isinstance(payload["auth_provider_ref"], str)
    assert "-secret" not in serialized
    assert "-signing-key" not in serialized
    assert "AzureProvider" not in serialized
    assert "MultiAuth" not in serialized

    rebuilt_app_auth = _auth("app", client_secret="rotated-app-secret")
    rebuilt_default = (
        rebuilt_app_auth
        if configuration == "inherited"
        else _auth("default")
        if configuration in {"different-default", "missing"}
        else None
    )
    worker = FastMCP("snapshot-worker", auth=rebuilt_default)
    owner = worker
    expected_name = "app"
    if configuration == "mounted":
        owner = FastMCP(
            "snapshot-child",
            auth=_auth("child", client_secret="rotated-child-secret"),
        )
        worker.mount(owner, namespace="child")
        expected_name = "child"
    if configuration != "missing":
        worker.http_app(auth=rebuilt_app_auth)

    @owner.tool
    async def downstream(
        token: str = EntraOBOToken(["https://graph.microsoft.com/User.Read"]),
    ) -> str:
        with pytest.raises(RuntimeError):
            get_http_request()
        return token

    async def restore_in_new_worker_context() -> None:
        task_id = secrets.token_urlsafe(32)
        assert get_access_token() is None
        with pytest.raises(RuntimeError):
            get_http_request()
        async with Docket(url="memory://") as docket:
            server_marker = _current_server.set(weakref.ref(worker))
            docket_marker = _current_docket.set(docket)
            try:
                snapshot = TaskContextSnapshot.from_json(serialized)
                await snapshot.save(docket, None, task_id, ttl_seconds=60)
                await restore_task_snapshot(
                    key=build_task_key(None, task_id, "tool", tool_name)
                )
                restored_token = get_access_token()
                assert restored_token is not None
                assert restored_token.token == "app-token"
                if configuration == "missing":
                    with pytest.raises(ToolError, match="Failed to resolve dependency"):
                        await worker.call_tool(tool_name)
                else:
                    result = await worker.call_tool(tool_name)
                    assert result.structured_content == {
                        "result": f"{expected_name}-client"
                    }
            finally:
                _current_docket.reset(docket_marker)
                _current_server.reset(server_marker)

    await asyncio.create_task(
        restore_in_new_worker_context(), context=contextvars.Context()
    )
    if configuration == "missing":
        credential_factory.assert_not_called()
        assert "authentication provider binding" in caplog.text
    else:
        credential_factory.assert_called_once_with(
            tenant_id=f"{expected_name}-tenant",
            client_id=f"{expected_name}-client",
            client_secret=f"rotated-{expected_name}-secret",
            user_assertion="app-token",
            authority="https://login.microsoftonline.com",
        )


@pytest.mark.parametrize("disable_app_auth", [False, True])
@pytest.mark.parametrize("background", [False, True])
async def test_root_tool_calling_mounted_obo_keeps_provider_owner(
    disable_app_auth: bool, background: bool, credential_factory: MagicMock
) -> None:
    app_auth = None if disable_app_auth else _auth("app")
    parent = FastMCP("relay-parent", auth=_auth("default"))
    parent.add_extension(TasksExtension())
    child = FastMCP("relay-child", auth=_auth("child"))

    @child.tool
    async def downstream(
        token: str = EntraOBOToken(["https://graph.microsoft.com/User.Read"]),
    ) -> str:
        return token

    parent.mount(child, namespace="child")

    @parent.tool(task=background)
    async def relay(ctx: Context) -> str:
        assert (get_task_context() is not None) is background
        assert _get_effective_auth() is app_auth
        result = await ctx.fastmcp.call_tool("child_downstream", {})
        assert _get_effective_auth() is app_auth
        assert result.structured_content is not None
        return result.structured_content["result"]

    access_token = AccessToken(token="app-token", client_id="user", scopes=["read"])
    marker = auth_context_var.set(AuthenticatedUser(access_token))
    try:
        async with asgi_server(parent, auth=app_auth) as running:
            async with running.client(
                auth=None if disable_app_auth else "app-token"
            ) as client:
                result = await client.call_tool("relay")
                assert result.data == "child-client"
    finally:
        auth_context_var.reset(marker)

    credential_factory.assert_called_once_with(
        tenant_id="child-tenant",
        client_id="child-client",
        client_secret="child-secret",
        user_assertion="app-token",
        authority="https://login.microsoftonline.com",
    )

"""Select configured authentication providers for requests and task workers."""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from fastmcp.server.auth.auth import AuthProvider, MultiAuth

if TYPE_CHECKING:
    from fastmcp.server.context import Context
    from fastmcp.server.server import FastMCP


# None means a legacy snapshot with no selection; "none" explicitly disables auth.
_background_task_auth_provider: ContextVar[str | None] = ContextVar(
    "fastmcp_background_task_auth_provider", default=None
)
_background_task_auth_owner: ContextVar[FastMCP | None] = ContextVar(
    "fastmcp_background_task_auth_owner", default=None
)
# A worker can inherit a request Context; only a newly entered dispatch Context
# may replace the task owner when selecting auth for a nested call.
_background_task_auth_context: ContextVar[Context | None] = ContextVar(
    "fastmcp_background_task_auth_context", default=None
)
_http_auth_providers: WeakKeyDictionary[FastMCP, dict[str, AuthProvider]] = (
    WeakKeyDictionary()
)


def _auth_provider_ref(auth: AuthProvider | None) -> str:
    """Identify configuration without storing credentials or process-local IDs.

    The existing trust-source identity includes issuers and token endpoints.
    OAuth's client ID additionally distinguishes clients at the same issuer;
    client secrets are deliberately excluded so workers can rotate credentials.
    """
    if auth is None:
        return "none"
    from fastmcp.server.auth.oauth_proxy import OAuthProxy

    descriptor: dict[str, object] = {
        "type": f"{type(auth).__module__}.{type(auth).__qualname__}",
    }
    if isinstance(auth, MultiAuth):
        descriptor["server"] = _auth_provider_ref(auth.server)
        descriptor["verifiers"] = [_auth_provider_ref(v) for v in auth.verifiers]
        descriptor["source_ids"] = auth._source_ids
    else:
        descriptor["source"] = MultiAuth._configured_source_id(auth)
        if isinstance(auth, OAuthProxy):
            descriptor["client_id"] = auth._upstream_client_id
    raw = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return "configured:" + hashlib.sha256(raw.encode()).hexdigest()


def _register_auth_provider(server: FastMCP, auth: AuthProvider | None) -> None:
    """Register an app's configured provider on its owner, also in remote workers."""
    if auth is not None:
        _http_auth_providers.setdefault(server, {})[_auth_provider_ref(auth)] = auth


def _get_effective_auth(server: FastMCP | None = None) -> AuthProvider | None:
    """Use an owning app's override, or restore its configured worker binding.

    A mounted child's provider is independent of the parent app's provider.
    Workers must construct the same HTTP app configuration (without serving it)
    to register app-only providers. A missing binding never uses another provider.
    """
    from fastmcp.server.dependencies import get_context, get_http_request, get_server

    reference = _background_task_auth_provider.get()
    if server is None:
        # During a task's nested dispatch the active Context belongs to the
        # called server, while get_server() still resolves the task's owner.
        if reference is not None:
            try:
                context = get_context()
            except RuntimeError:
                context = None
            if (
                context is not None
                and context is not _background_task_auth_context.get()
            ):
                server = context.fastmcp
            else:
                server = get_server()
        else:
            server = get_server()
    if reference is not None:
        owner = _background_task_auth_owner.get()
        if owner is None:
            raise RuntimeError(
                "The task's authentication provider binding has no owning server."
            )
        if owner is not server:
            return server.auth
        if reference == "none":
            return None
        if server.auth is not None and _auth_provider_ref(server.auth) == reference:
            return server.auth
        provider = _http_auth_providers.get(server, {}).get(reference)
        if provider is None:
            raise RuntimeError(
                "The task's authentication provider binding is not configured on "
                "this worker. Construct the same HTTP app authentication "
                "configuration as the submitting server."
            )
        return provider

    try:
        request = get_http_request()
    except RuntimeError:
        return server.auth
    app_state = getattr(request.scope.get("app"), "state", None)
    if getattr(app_state, "fastmcp_server", None) is server:
        # An explicit None disables auth; it must not fall back to server.auth.
        return getattr(app_state, "fastmcp_auth", server.auth)
    return server.auth

from __future__ import annotations

from contextvars import ContextVar, Token
import logging
from typing import Any

from fastmcp.exceptions import AuthorizationError
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from mcp import types as mt

from .clerk import ClerkAuthService, ClerkUserRecord

logger = logging.getLogger(__name__)

_current_clerk_user: ContextVar[ClerkUserRecord | None] = ContextVar(
    "current_clerk_user",
    default=None,
)


def get_current_clerk_user_record() -> ClerkUserRecord | None:
    """Return the Clerk user resolved for the current MCP request, if present."""

    return _current_clerk_user.get()


class RequireActiveClerkUserMiddleware(Middleware):
    """Resolve the verified Clerk subject and require an active app user."""

    def __init__(self, clerk_auth: ClerkAuthService) -> None:
        super().__init__()
        self._clerk_auth = clerk_auth

    async def on_request(
        self,
        context: MiddlewareContext[mt.Request[Any, Any]],
        call_next: CallNext[mt.Request[Any, Any], Any],
    ) -> Any:
        access_token = get_access_token()
        if access_token is None:
            raise AuthorizationError("Authentication is required.")

        clerk_user_id = _extract_subject(access_token)
        clerk_user = await self._clerk_auth.get_user_record(clerk_user_id)
        if not clerk_user.active:
            logger.warning(
                "clerk_user_inactive_rejected clerk_user_id=%s method=%s",
                clerk_user_id,
                context.method or "unknown",
            )
            raise AuthorizationError(
                "Signed in successfully, but access is still pending activation."
            )

        logger.info(
            "clerk_user_active_authenticated clerk_user_id=%s method=%s",
            clerk_user_id,
            context.method or "unknown",
        )
        token = _current_clerk_user.set(clerk_user)
        try:
            return await call_next(context)
        finally:
            _current_clerk_user.reset(token)


def push_clerk_user_record(record: ClerkUserRecord | None) -> Token[ClerkUserRecord | None]:
    """Push a Clerk user record into the current request context for tests."""

    return _current_clerk_user.set(record)


def pop_clerk_user_record(token: Token[ClerkUserRecord | None]) -> None:
    """Restore the previous Clerk user record after a test override."""

    _current_clerk_user.reset(token)


def clerk_user_id_from_access_token(access_token: AccessToken) -> str:
    """Extract the Clerk user id from a verified FastMCP access token."""

    return _extract_subject(access_token)


def _extract_subject(access_token: AccessToken) -> str:
    subject = access_token.claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise AuthorizationError("Authenticated token did not include a Clerk subject.")
    return subject.strip()

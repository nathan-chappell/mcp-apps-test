from __future__ import annotations

import base64
import json
from typing import Literal

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier

from .clerk import ClerkAuthService


class ClerkAccessToken(AccessToken):
    subject: str
    token_id: str | None = None
    session_id: str | None = None
    token_type: Literal["oauth_token", "session_token"]


class ClerkTokenVerifier(TokenVerifier):
    """FastMCP token verifier backed by Clerk OAuth and session-token verification."""

    def __init__(self, clerk_auth: ClerkAuthService) -> None:
        self._clerk_auth = clerk_auth

    async def verify_token(self, token: str) -> ClerkAccessToken | None:
        if _looks_like_session_token(token):
            verified_session = await self._clerk_auth.verify_session_token(token)
            if verified_session is not None:
                return ClerkAccessToken(
                    token=token,
                    client_id="clerk-session",
                    scopes=["user"],
                    expires_at=int(verified_session.expiration)
                    if verified_session.expiration
                    else None,
                    subject=verified_session.subject,
                    token_id=verified_session.token_id,
                    session_id=verified_session.session_id,
                    token_type="session_token",
                )

        verified_access = await self._clerk_auth.verify_access_token(token)
        if verified_access is None:
            return None

        return ClerkAccessToken(
            token=token,
            client_id=verified_access.client_id,
            scopes=verified_access.scopes,
            expires_at=int(verified_access.expiration)
            if verified_access.expiration
            else None,
            subject=verified_access.subject,
            token_id=verified_access.id,
            session_id=None,
            token_type="oauth_token",
        )


def get_current_clerk_access_token() -> ClerkAccessToken | None:
    """Return the authenticated Clerk token for the current request, if present."""

    token = get_access_token()
    if token is None:
        return None
    if not isinstance(token, ClerkAccessToken):
        raise RuntimeError("Expected a ClerkAccessToken in the current request context.")
    return token


def _looks_like_session_token(token: str) -> bool:
    parts = token.split(".")
    if len(parts) != 3:
        return False

    try:
        payload = _decode_b64url_json(parts[1])
    except ValueError:
        return False

    return isinstance(payload.get("sid"), str) and isinstance(payload.get("sub"), str)


def _decode_b64url_json(value: str) -> dict[str, object]:
    padding = "=" * (-len(value) % 4)
    decoded = base64.urlsafe_b64decode(f"{value}{padding}")
    payload = json.loads(decoded.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object payload.")
    return payload

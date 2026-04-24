from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel

from .settings import AppSettings

logger = logging.getLogger(__name__)


class ClerkUserRecord(BaseModel):
    clerk_user_id: str
    primary_email: str | None = None
    display_name: str
    active: bool = False
    role: str | None = None


class ClerkAuthService:
    """Minimal Clerk admin client for user metadata lookup."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url="https://api.clerk.com",
            headers={
                "Authorization": f"Bearer {settings.clerk_secret_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=15.0,
        )

    async def get_user_record(self, clerk_user_id: str) -> ClerkUserRecord:
        response = await self._client.get(f"/v1/users/{clerk_user_id}")
        response.raise_for_status()

        payload = response.json()
        private_metadata = payload.get("private_metadata") or {}
        active = bool(private_metadata.get(self._settings.clerk_active_metadata_key))
        raw_role = private_metadata.get(self._settings.clerk_role_metadata_key)
        role = raw_role.strip() if isinstance(raw_role, str) and raw_role.strip() else None

        logger.info(
            "clerk_user_loaded clerk_user_id=%s active=%s role=%s",
            clerk_user_id,
            active,
            role or "none",
        )

        return ClerkUserRecord(
            clerk_user_id=clerk_user_id,
            primary_email=self._extract_primary_email(payload),
            display_name=self._extract_display_name(payload, clerk_user_id),
            active=active,
            role=role,
        )

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _extract_primary_email(payload: dict[str, Any]) -> str | None:
        primary_email_id = payload.get("primary_email_address_id")
        email_addresses = payload.get("email_addresses") or []
        for email in email_addresses:
            if not isinstance(email, dict):
                continue
            if email.get("id") == primary_email_id:
                address = email.get("email_address")
                return address if isinstance(address, str) else None
        for email in email_addresses:
            if isinstance(email, dict) and isinstance(email.get("email_address"), str):
                return email["email_address"]
        return None

    @classmethod
    def _extract_display_name(cls, payload: dict[str, Any], clerk_user_id: str) -> str:
        first_name = payload.get("first_name")
        last_name = payload.get("last_name")
        full_name = " ".join(
            part.strip()
            for part in [first_name, last_name]
            if isinstance(part, str) and part.strip()
        ).strip()
        if full_name:
            return full_name

        username = payload.get("username")
        if isinstance(username, str) and username.strip():
            return username

        email = cls._extract_primary_email(payload)
        if email:
            return email

        return clerk_user_id

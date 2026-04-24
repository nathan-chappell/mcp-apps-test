from __future__ import annotations

import time

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel

from .schemas import PendingCommandResult, UploadSessionResult
from .settings import AppSettings


class UploadSessionClaims(BaseModel):
    clerk_user_id: str
    knowledge_base_id: str


class NodeDownloadClaims(BaseModel):
    clerk_user_id: str
    node_id: str


class PendingCommandClaims(BaseModel):
    clerk_user_id: str
    knowledge_base_id: str
    action: str
    payload: dict[str, object]


class KnowledgeBaseSessionService:
    """Issue and verify short-lived signed tokens for the MCP knowledge-base UI."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._serializer = URLSafeTimedSerializer(
            settings.app_signing_secret.get_secret_value(),
            salt="openai-knowledge-base-desk",
        )

    def issue_upload_session(
        self,
        *,
        clerk_user_id: str,
        knowledge_base_id: str,
    ) -> UploadSessionResult:
        token = self._serializer.dumps(
            {
                "kind": "upload",
                "clerk_user_id": clerk_user_id,
                "knowledge_base_id": knowledge_base_id,
            }
        )
        return UploadSessionResult(
            upload_url=f"{self._settings.normalized_app_base_url}/api/uploads",
            upload_token=token,
            expires_at=int(time.time()) + self._settings.upload_session_max_age_seconds,
        )

    def verify_upload_session(self, token: str) -> UploadSessionClaims | None:
        payload = self._loads(token, max_age=self._settings.upload_session_max_age_seconds)
        if payload is None or payload.get("kind") != "upload":
            return None
        return UploadSessionClaims.model_validate(payload)

    def issue_node_download_url(
        self,
        *,
        clerk_user_id: str,
        node_id: str,
    ) -> str:
        token = self._serializer.dumps(
            {
                "kind": "node-download",
                "clerk_user_id": clerk_user_id,
                "node_id": node_id,
            }
        )
        return (
            f"{self._settings.normalized_app_base_url}/api/documents/{node_id}/content"
            f"?token={token}"
        )

    def verify_node_download(self, token: str) -> NodeDownloadClaims | None:
        payload = self._loads(
            token,
            max_age=self._settings.asset_download_session_max_age_seconds,
        )
        if payload is None or payload.get("kind") != "node-download":
            return None
        return NodeDownloadClaims.model_validate(payload)

    def issue_command_confirmation(
        self,
        *,
        clerk_user_id: str,
        knowledge_base_id: str,
        action: str,
        payload: dict[str, object],
        prompt: str,
        summary: str,
    ) -> PendingCommandResult:
        token = self._serializer.dumps(
            {
                "kind": "pending-command",
                "clerk_user_id": clerk_user_id,
                "knowledge_base_id": knowledge_base_id,
                "action": action,
                "payload": payload,
            }
        )
        return PendingCommandResult(
            token=token,
            prompt=prompt,
            summary=summary,
            expires_at=int(time.time()) + self._settings.command_confirmation_max_age_seconds,
        )

    def verify_command_confirmation(self, token: str) -> PendingCommandClaims | None:
        payload = self._loads(
            token,
            max_age=self._settings.command_confirmation_max_age_seconds,
        )
        if payload is None or payload.get("kind") != "pending-command":
            return None
        return PendingCommandClaims.model_validate(payload)

    def _loads(self, token: str, *, max_age: int) -> dict[str, object] | None:
        try:
            raw_payload = self._serializer.loads(token, max_age=max_age)
        except (BadSignature, SignatureExpired):
            return None

        if not isinstance(raw_payload, dict):
            return None
        return raw_payload

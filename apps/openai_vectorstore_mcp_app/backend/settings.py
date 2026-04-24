from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime settings for the workspace desk MCP server."""

    openai_api_key: SecretStr = Field(init=False)
    clerk_secret_key: SecretStr = Field(init=False)
    app_signing_secret: SecretStr = Field(init=False)
    clerk_issuer_url: AnyHttpUrl = Field(init=False)

    app_base_url: AnyHttpUrl = "http://localhost:8000"
    clerk_publishable_key: str | None = None
    clerk_active_metadata_key: str = "active"
    clerk_role_metadata_key: str = "role"
    database_url: str = "sqlite+aiosqlite:///./.local/openai-vectorstore-mcp-app.db"
    mcp_required_scopes: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["user"]
    )

    openai_agent_model: str = "gpt-5.4"
    openai_branching_model: str = "gpt-5.4-mini"
    openai_vision_model: str = "gpt-4.1-mini"
    openai_audio_transcription_model: str = "gpt-4o-transcribe-diarize"
    openai_file_search_max_results: int = 5
    openai_poll_interval_ms: int = 1_000

    upload_session_max_age_seconds: int = 900
    asset_download_session_max_age_seconds: int = 900
    command_confirmation_max_age_seconds: int = 900

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_name: str = "openai-knowledge-base-desk"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("mcp_required_scopes", mode="before")
    @classmethod
    def _parse_required_scopes(cls, raw_value: object) -> list[str]:
        if raw_value is None:
            return ["user"]
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        if isinstance(raw_value, str):
            values = [part.strip() for part in raw_value.split(",")]
            return [value for value in values if value]
        raise TypeError("MCP_REQUIRED_SCOPES must be a comma-separated string or list.")

    @property
    def normalized_app_base_url(self) -> str:
        return str(self.app_base_url).rstrip("/")

    @property
    def normalized_database_url(self) -> str:
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return self.database_url

    @property
    def sync_database_url(self) -> str:
        database_url = self.normalized_database_url
        if database_url.startswith("postgresql+asyncpg://"):
            return database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
        if database_url.startswith("sqlite+aiosqlite://"):
            return database_url.replace("sqlite+aiosqlite://", "sqlite://", 1)
        return database_url


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache the app settings."""

    return AppSettings()

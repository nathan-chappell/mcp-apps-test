from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime settings for the OpenAI Files + Vector Store MCP app."""

    openai_api_key: SecretStr = Field(init=False)
    openai_agent_model: str = "gpt-5.4"
    openai_file_search_max_results: int = 5
    openai_poll_interval_ms: int = 1_000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_name: str = "openai-files-vector-store"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache the app settings."""

    return AppSettings()

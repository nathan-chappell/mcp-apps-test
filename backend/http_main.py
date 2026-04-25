from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import uvicorn

from .logging import build_uvicorn_log_config, configure_logging, get_log_file_path
from .settings import get_settings
from .web_app import create_fastapi_app


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    parsed_base_url = urlparse(settings.normalized_app_base_url)
    host = parsed_base_url.hostname or "0.0.0.0"
    port = parsed_base_url.port or (443 if parsed_base_url.scheme == "https" else 8000)

    logger = logging.getLogger(__name__)
    logger.info(
        "web_app_starting name=%s cwd=%s host=%s port=%s base_url=%s log_file=%s",
        settings.app_name,
        Path.cwd(),
        host,
        port,
        settings.normalized_app_base_url,
        get_log_file_path(),
    )

    uvicorn.run(
        create_fastapi_app(settings),
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
        log_config=build_uvicorn_log_config(settings.log_level),
    )


if __name__ == "__main__":
    main()

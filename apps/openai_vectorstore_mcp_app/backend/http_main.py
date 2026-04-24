from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn

from .logging import configure_logging
from .server import create_http_app, create_server
from .settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    parsed_base_url = urlsplit(settings.normalized_app_base_url)
    host = parsed_base_url.hostname or "127.0.0.1"
    port = parsed_base_url.port or 8000

    server = create_server(settings)
    logger = logging.getLogger(__name__)
    logger.info(
        "mcp_server_starting name=%s transport=streamable-http cwd=%s host=%s port=%s path=%s",
        settings.app_name,
        Path.cwd(),
        host,
        port,
        "/mcp",
    )

    try:
        uvicorn.run(
            create_http_app(server),
            host=host,
            port=port,
            log_level=settings.log_level.lower(),
        )
    except KeyboardInterrupt:
        logger.info(
            "mcp_server_stopped name=%s transport=streamable-http",
            settings.app_name,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from pathlib import Path

from .logging import configure_logging
from .server import create_server
from .settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info(
        "mcp_server_starting name=%s transport=stdio cwd=%s",
        settings.app_name,
        Path.cwd(),
    )

    server = create_server(settings)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging

from colorlog import ColoredFormatter

_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(name)s %(message)s"


def configure_logging(level: str) -> None:
    """Configure workspace logging once, keeping repeated setup idempotent."""

    root_logger = logging.getLogger()
    normalized_level = getattr(logging, level.upper(), logging.INFO)

    existing_handler = next(
        (
            handler
            for handler in root_logger.handlers
            if getattr(handler, "_openai_files_vector_store_handler", False)
        ),
        None,
    )

    if existing_handler is None:
        handler = logging.StreamHandler()
        handler._openai_files_vector_store_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(
            ColoredFormatter(
                _FORMAT,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        root_logger.addHandler(handler)

    root_logger.setLevel(normalized_level)

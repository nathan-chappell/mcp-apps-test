from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

from colorlog import ColoredFormatter
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(name)s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_LOG_FILE_PATH = Path(__file__).resolve().parent.parent / "tmp" / "logs.txt"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class _OpenAIFilesVectorStoreStreamHandler(logging.StreamHandler):
    """Marker handler for idempotent logging configuration."""


class _OpenAIFilesVectorStoreFileHandler(logging.FileHandler):
    """Marker handler for idempotent logging configuration."""


class _AnsiStrippingFormatter(logging.Formatter):
    """Write plain-text log lines even when traceback rendering includes ANSI escapes."""

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE_RE.sub("", super().format(record))


def get_log_file_path() -> Path:
    """Return the shared file path used for backend debugging logs."""

    return _LOG_FILE_PATH


def configure_logging(level: str) -> None:
    """Configure workspace logging once, keeping repeated setup idempotent."""

    root_logger = logging.getLogger()
    normalized_level = getattr(logging, level.upper(), logging.INFO)
    log_file_path = get_log_file_path()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    existing_handler = next(
        (handler for handler in root_logger.handlers if isinstance(handler, _OpenAIFilesVectorStoreStreamHandler)),
        None,
    )

    if existing_handler is None:
        handler = _OpenAIFilesVectorStoreStreamHandler()
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

    existing_file_handler = next(
        (handler for handler in root_logger.handlers if isinstance(handler, _OpenAIFilesVectorStoreFileHandler)),
        None,
    )

    if existing_file_handler is None:
        file_handler = _OpenAIFilesVectorStoreFileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(_AnsiStrippingFormatter(_FILE_FORMAT))
        root_logger.addHandler(file_handler)

    root_logger.setLevel(normalized_level)


def build_uvicorn_log_config(level: str) -> dict[str, Any]:
    """Build a uvicorn log config that mirrors console output into tmp/logs.txt."""

    normalized_level = level.upper()
    log_file_path = get_log_file_path()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        **UVICORN_LOGGING_CONFIG,
        "formatters": dict(UVICORN_LOGGING_CONFIG["formatters"]),
        "handlers": dict(UVICORN_LOGGING_CONFIG["handlers"]),
        "loggers": {name: dict(logger_config) for name, logger_config in UVICORN_LOGGING_CONFIG["loggers"].items()},
    }
    config["handlers"]["debug_file"] = {
        "class": "logging.FileHandler",
        "filename": str(log_file_path),
        "encoding": "utf-8",
        "formatter": "debug_file",
    }
    config["formatters"]["debug_file"] = {
        "()": _AnsiStrippingFormatter,
        "format": _FILE_FORMAT,
    }
    config["loggers"]["uvicorn"]["handlers"] = ["default", "debug_file"]
    config["loggers"]["uvicorn"]["level"] = normalized_level
    config["loggers"]["uvicorn.access"]["handlers"] = ["access", "debug_file"]
    config["loggers"]["uvicorn.access"]["level"] = normalized_level
    config["loggers"]["uvicorn.error"]["level"] = normalized_level
    return config

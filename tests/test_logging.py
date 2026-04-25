from __future__ import annotations

import logging
import sys

from backend.logging import _AnsiStrippingFormatter


def test_ansi_stripping_formatter_removes_color_codes_from_message() -> None:
    formatter = _AnsiStrippingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="\x1b[31mhello\x1b[0m world",
        args=(),
        exc_info=None,
    )

    assert formatter.format(record) == "hello world"


def test_ansi_stripping_formatter_removes_color_codes_from_exception_text() -> None:
    formatter = _AnsiStrippingFormatter("%(message)s")

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=25,
        msg="failure",
        args=(),
        exc_info=exc_info,
    )
    formatter.formatException = lambda exc_info: "\x1b[35mtraceback\x1b[0m"  # type: ignore[method-assign]

    assert formatter.format(record) == "failure\ntraceback"

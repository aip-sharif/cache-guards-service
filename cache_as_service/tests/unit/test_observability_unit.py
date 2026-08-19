"""Structured logging + Sentry bootstrap — pure unit (no backend, no network)."""

import json
import logging
import sys

from semantic_cache.observability import (
    JsonLogFormatter,
    configure_logging,
    init_sentry,
)


def _format(record: logging.LogRecord) -> dict:
    return json.loads(JsonLogFormatter().format(record))


def _record(**kw) -> logging.LogRecord:
    defaults = dict(
        name="semantic_cache.x",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    defaults.update(kw)
    return logging.LogRecord(
        defaults["name"], defaults["level"], defaults["pathname"],
        defaults["lineno"], defaults["msg"], defaults["args"], defaults["exc_info"],
    )


# -- JSON formatter --------------------------------------------------------- #


def test_formatter_emits_valid_json_line() -> None:
    out = JsonLogFormatter().format(_record())
    assert "\n" not in out  # one line per record
    obj = json.loads(out)
    assert obj["level"] == "INFO"
    assert obj["logger"] == "semantic_cache.x"
    assert obj["message"] == "hello world"  # args interpolated
    assert "timestamp" in obj


def test_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        rec = _record(level=logging.ERROR, msg="failed", args=(), exc_info=sys.exc_info())
    obj = _format(rec)
    assert obj["level"] == "ERROR"
    assert "ValueError: boom" in obj["exception"]


def test_formatter_includes_extra_fields() -> None:
    rec = _record()
    rec.tenant_id = "t123"  # LogRecord extra
    obj = _format(rec)
    assert obj["tenant_id"] == "t123"


# -- configure_logging ------------------------------------------------------ #


def test_configure_logging_installs_single_stdout_json_handler() -> None:
    configure_logging(level="INFO")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler.formatter, JsonLogFormatter)
    assert handler.stream is sys.stdout


def test_configure_logging_routes_uvicorn_through_root() -> None:
    configure_logging(level="INFO")
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        assert lg.handlers == []      # no private handlers → nothing bypasses JSON
        assert lg.propagate is True   # everything flows to the single root handler


def test_configure_logging_is_idempotent() -> None:
    configure_logging(level="INFO")
    configure_logging(level="DEBUG")
    assert len(logging.getLogger().handlers) == 1  # not stacked


# -- Sentry ----------------------------------------------------------------- #


def test_init_sentry_noop_without_dsn() -> None:
    assert init_sentry(dsn=None, environment="test") is False


def test_init_sentry_noop_with_blank_dsn() -> None:
    assert init_sentry(dsn="", environment="test") is False

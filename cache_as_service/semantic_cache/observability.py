"""Structured logging + Sentry bootstrap for the SaaS server.

Two guarantees for stdout:
  * Every log line is a single JSON object (machine-parseable).
  * Nothing bypasses it — uvicorn's own loggers are stripped of their private
    handlers and made to propagate to the one root handler, so stdout carries
    structured logs and nothing else.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Optional

# LogRecord attributes that are intrinsic — anything else on the record is a
# caller-supplied `extra` and gets promoted to a top-level JSON field.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()
) | {"message", "asctime", "taskName", "color_message"}


class JsonLogFormatter(logging.Formatter):
    """Formats a LogRecord as one compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote caller-supplied `extra=` fields.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


# Loggers that ship their own handlers and would otherwise double-log or emit
# non-JSON lines to stdout.
_NOISY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error")


def configure_logging(level: str = "INFO") -> None:
    """Installs a single JSON handler on the root logger (stdout) and routes
    every known logger through it. Idempotent."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Strip private handlers so nothing sidesteps the JSON handler.
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


def init_sentry(
    dsn: Optional[str],
    environment: str = "production",
    traces_sample_rate: float = 0.0,
    release: Optional[str] = None,
) -> bool:
    """Initializes Sentry error tracking if a DSN is configured.

    Returns True if Sentry was initialized. A missing DSN or a missing
    `sentry-sdk` install is a graceful no-op (returns False) — the service must
    still start without error reporting."""
    if not dsn:
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:
        logging.getLogger(__name__).warning(
            "SC_SENTRY_DSN is set but sentry-sdk is not installed; "
            "error reporting disabled. Install 'semantic-cache[sentry]'."
        )
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_sample_rate,
        release=release,
        integrations=[StarletteIntegration(), FastApiIntegration()],
    )
    return True

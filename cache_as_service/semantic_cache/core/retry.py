"""Tiny dependency-free retry helpers for transient HTTP failures.

Used by the API embedding manager and the LLM entity extractors so a single
blip (timeout, dropped connection, 429, 5xx) becomes a brief retry instead of
a cache miss/skip. Deliberately avoids pulling in `tenacity` so the core
package stays lean — a caller that wants richer policies can still wrap our
extractors/embedders themselves.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def is_retryable_http(exc: Exception) -> bool:
    """Return True for transient HTTP errors worth retrying.

    Duck-typed across ``requests`` and ``httpx``: connection/timeout errors
    retry, and HTTP *status* errors retry only on 429 or 5xx. Other 4xx
    (auth, bad request) are permanent and must NOT be retried.
    """
    name = type(exc).__name__.lower()
    if "timeout" in name or "connect" in name:
        return True
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if code is not None:
        return code == 429 or code >= 500
    return False


def call_with_retries(
    fn: Callable[[], T],
    *,
    retries: int,
    backoff_base: float,
    retryable: Callable[[Exception], bool] = is_retryable_http,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with bounded exponential-backoff retries (sync)."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries or not retryable(exc):
                raise
            sleep(backoff_base * (2 ** attempt))
            attempt += 1


async def acall_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int,
    backoff_base: float,
    retryable: Callable[[Exception], bool] = is_retryable_http,
) -> T:
    """Call awaitable ``fn`` with bounded exponential-backoff retries (async)."""
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries or not retryable(exc):
                raise
            await asyncio.sleep(backoff_base * (2 ** attempt))
            attempt += 1

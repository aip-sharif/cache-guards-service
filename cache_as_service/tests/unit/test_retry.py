"""Unit tests for the dependency-free retry helpers (no Redis/network)."""

import pytest

from semantic_cache.core.retry import (
    acall_with_retries,
    call_with_retries,
    is_retryable_http,
)


def _http_exc(status: int) -> Exception:
    exc = Exception("http")
    exc.response = type("R", (), {"status_code": status})()
    return exc


def test_is_retryable_http_classifies_correctly():
    assert is_retryable_http(TimeoutError("t")) is True
    assert is_retryable_http(ConnectionError("c")) is True
    assert is_retryable_http(_http_exc(503)) is True
    assert is_retryable_http(_http_exc(429)) is True
    assert is_retryable_http(_http_exc(404)) is False
    assert is_retryable_http(_http_exc(400)) is False
    assert is_retryable_http(ValueError("nope")) is False


def test_call_with_retries_succeeds_after_transient_failures():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert call_with_retries(fn, retries=3, backoff_base=0) == "ok"
    assert calls["n"] == 3


def test_call_with_retries_gives_up_after_budget():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        call_with_retries(fn, retries=2, backoff_base=0)
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_call_with_retries_does_not_retry_permanent_errors():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        call_with_retries(fn, retries=5, backoff_base=0)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_acall_with_retries_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("boom")
        return "ok"

    assert await acall_with_retries(fn, retries=2, backoff_base=0) == "ok"
    assert calls["n"] == 2

"""HTTP client for upstream OpenAI-compatible LLM endpoints.

The gateway proxies cache misses here. The base_url comes from env
(SC_LLM_BASE_URL) and the api_key from the APP's per-project config — nothing
upstream-specific is hardcoded. The httpx.AsyncClient is injected so tests use
MockTransport and the server shares one connection pool across requests.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx


logger = logging.getLogger(__name__)


class UpstreamError(Exception):
    """Upstream returned a non-2xx response (or was unreachable).

    ``str(e)`` is SAFE TO RETURN TO THE CLIENT — it never contains the
    upstream body. Providers echo the presented API key in auth errors
    ("Incorrect API key provided: sk-…"), and that key belongs to the APP,
    not to the caller. The real detail goes to the log via ``detail``.
    """

    def __init__(
        self, message: str, status_code: int = 502, detail: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _error_detail(body: bytes) -> str:
    """Upstream's own message — for LOGS ONLY, never for the client."""
    try:
        parsed = json.loads(body)
        return parsed.get("error", {}).get("message") or body.decode("utf-8", "replace")
    except Exception:
        return body.decode("utf-8", "replace")


def _upstream_error(status_code: int, body: bytes) -> UpstreamError:
    """Logs the upstream's detail, returns a client-safe error."""
    detail = _error_detail(body)
    logger.error("Upstream error %s: %s", status_code, detail)
    return UpstreamError(
        f"The upstream model provider returned an error (HTTP {status_code}).",
        status_code=status_code,
        detail=detail,
    )


class UpstreamClient:
    """Thin async wrapper: one non-streaming call, one SSE line stream."""

    def __init__(self, client: httpx.AsyncClient, timeout: float = 120.0) -> None:
        self.client = client
        self.timeout = timeout

    def _headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self, base_url: str, api_key: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Non-streaming chat completion; returns the upstream JSON body."""
        try:
            response = await self.client.post(
                _chat_endpoint(base_url),
                json=payload,
                headers=self._headers(api_key),
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            logger.error("Upstream unreachable: %s", e)
            raise UpstreamError(
                "The upstream model provider is unreachable.",
                status_code=502, detail=str(e),
            ) from e
        if response.status_code >= 400:
            raise _upstream_error(response.status_code, response.content)
        return response.json()

    async def stream(
        self, base_url: str, api_key: str, payload: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Streams the upstream SSE response line by line (lines include the
        'data: ...' prefix, without trailing newlines). Raises UpstreamError
        BEFORE yielding anything if the upstream rejects the request, so the
        caller can still return a proper HTTP error status."""
        payload = {**payload, "stream": True}
        try:
            async with self.client.stream(
                "POST",
                _chat_endpoint(base_url),
                json=payload,
                headers=self._headers(api_key),
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise _upstream_error(response.status_code, body)
                async for line in response.aiter_lines():
                    yield line
        except httpx.HTTPError as e:
            logger.error("Upstream unreachable: %s", e)
            raise UpstreamError(
                "The upstream model provider is unreachable.",
                status_code=502, detail=str(e),
            ) from e

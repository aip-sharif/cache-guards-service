"""Operational surface: Prometheus exposition and HTTP instrumentation.

The cache has had counters for a long time; what it did not have was a way to
SCRAPE them from the deployed service. `adapters/fastapi_router.py` exposes
`/cache/prometheus`, but that router is single-tenant (it hangs off one shared
`SemanticCacheManager` and carries a global purge) and `server.py` deliberately
does not mount it — so in the SaaS image the metrics were unreachable, and the
image did not even install `prometheus_client`. Counters nobody can read are
not observability.

This module gives the deployed app the one thing it was missing: a single
`/metrics` endpoint, plus HTTP-level series for the request path.

`/metrics` IS AUTHENTICATED. Exposition leaks route templates, traffic shape,
and error rates; on a service whose data plane is multi-tenant that is not
public information. It reuses `SC_ADMIN_API_KEY` rather than inventing a second
credential, and returns 503 — not 200 with an empty body — when no key is
configured, so "I forgot to set the key" cannot look like "there are no
metrics".
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import Response

from semantic_cache.adapters.saas_router import _bearer, get_admin_key
from semantic_cache.core import metrics

ops_router = APIRouter(tags=["Ops"])


@ops_router.get("/metrics")
def prometheus_metrics(
    authorization: Optional[str] = Header(default=None),
    admin_key: Optional[str] = Depends(get_admin_key),
) -> Response:
    """Prometheus text exposition, behind the admin key."""
    if not admin_key:
        raise HTTPException(
            status_code=503,
            detail="Metrics are not exposed: SC_ADMIN_API_KEY is not set.",
        )
    if not secrets.compare_digest(_bearer(authorization), admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    if not metrics.is_available():
        raise HTTPException(
            status_code=503,
            detail="prometheus_client is not installed (pip install '.[metrics]').",
        )
    return Response(
        content=metrics.render_latest(),
        media_type=metrics.CONTENT_TYPE_LATEST,
    )


def install_http_metrics(app: Any) -> None:
    """Adds request count / latency / in-flight instrumentation.

    The `path` label is the ROUTE TEMPLATE, resolved after the request is
    matched — `/v1/caches/{cache_id}`, never `/v1/caches/<a real cache id>`. A
    label keyed on user input is an unbounded time series, which takes down the
    Prometheus server rather than the app, i.e. it breaks the thing you would
    use to find out what broke. An unmatched request is labelled `<unmatched>`
    for the same reason: 404 scanners must not be able to mint series."""

    @app.middleware("http")
    async def _instrument(request: Any, call_next: Any) -> Any:
        metrics.http_requests_in_flight.inc()
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            metrics.http_requests_in_flight.dec()
            route = request.scope.get("route")
            path = getattr(route, "path", None) or "<unmatched>"
            method = request.method
            metrics.http_requests_total.labels(
                method=method, path=path, status=str(status)
            ).inc()
            metrics.http_request_latency_seconds.labels(
                method=method, path=path
            ).observe(elapsed)


class BodyLimitMiddleware:
    """Rejects request bodies over `max_bytes` with 413.

    Pure ASGI rather than a `@app.middleware("http")` function because the
    limit has to apply to the STREAM. A Content-Length check alone is a
    suggestion: a chunked request carries no Content-Length, so the cheapest
    way past it is to not send one. This counts bytes as they arrive and stops
    at the ceiling, which is the only version that binds.

    The gateway takes an arbitrary OpenAI-shaped body and the SaaS schemas
    accept unbounded `query`, `response` and `config` strings, so before this
    the largest request the service would accept was decided by whoever was
    sending it."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        # The declared length is still worth checking: when it is present and
        # over the limit we refuse before reading a single byte.
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._too_large(send)
                        return
                except ValueError:
                    pass
                break

        received = 0
        exceeded = False

        async def counting_receive() -> Any:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    # Cut the body off. The handler sees a truncated stream and
                    # the response below is what the client is told.
                    return {"type": "http.disconnect"}
            return message

        started = False

        async def guarded_send(message: Any) -> None:
            nonlocal started
            if exceeded and not started:
                if message["type"] == "http.response.start":
                    started = True
                    await self._too_large(send)
                return
            if started:
                return
            await send(message)

        await self.app(scope, counting_receive, guarded_send)

    async def _too_large(self, send: Any) -> None:
        body = (
            b'{"error":{"message":"Request body exceeds the configured limit '
            b'(SC_MAX_REQUEST_BYTES).","type":"invalid_request_error",'
            b'"code":"payload_too_large"}}'
        )
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def install_body_limit(app: Any, max_bytes: int) -> None:
    """Wraps `app` so no request body may exceed `max_bytes`. 0 disables."""
    if max_bytes and max_bytes > 0:
        app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)


__all__ = [
    "BodyLimitMiddleware",
    "install_body_limit",
    "install_http_metrics",
    "ops_router",
]

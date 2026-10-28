"""Prometheus metrics for the semantic cache.

`prometheus_client` is an OPTIONAL dependency. If it is not installed we
substitute in no-op stubs so the rest of the codebase can call the metric
methods unconditionally — there is no `if metrics_enabled:` boilerplate at
call sites.

Why this matters operationally:
* The most useful signal for an oncall is not the raw hit/miss rate but
  *why* a miss happened: `no_candidate` (legitimate cold cache) vs
  `extractor_error` (the LLM gateway is sick) vs `below_threshold` (the
  embedding model drifted). Distinguishing these lets you page on
  `extractor_error` rate without false alerts from a cold cache.
* Extractor latency and error rate are first-class histograms — a slow
  extractor turns the cache into a *slowdown*, not a speedup, and you need
  to detect that quickly.

The exact metric names and labels:

    scache_lookups_total{result="hit"|"miss", reason="..."}
    scache_writes_total{outcome="ok"|"skipped_extractor_error"}
    scache_extractor_calls_total{outcome="ok"|"error"}
    scache_extractor_latency_seconds          (histogram)
    scache_search_latency_seconds             (histogram)
    scache_extraction_cache_total{result="hit"|"miss"}

Miss reasons:
    no_candidate         — RediSearch returned 0 results (cold cache / filter excluded everything).
    extractor_error      — entity extraction raised (treated as MISS for safety).
    below_threshold      — top candidate failed the similarity floor.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

# Public miss-reason constants — also used as label values.
MISS_NO_CANDIDATE = "no_candidate"
MISS_EXTRACTOR_ERROR = "extractor_error"
MISS_BELOW_THRESHOLD = "below_threshold"

# Public write-outcome constants.
WRITE_OK = "ok"
WRITE_SKIPPED_EXTRACTOR_ERROR = "skipped_extractor_error"

# Public extractor-outcome constants.
EXTRACTOR_OK = "ok"
EXTRACTOR_ERROR = "error"


class _NoOpMetric:
    """Stub that swallows all method calls. Used when prometheus_client is missing."""

    def labels(self, *args, **kwargs) -> "_NoOpMetric":  # noqa: D401
        return self

    def inc(self, *args, **kwargs) -> None:
        pass

    def dec(self, *args, **kwargs) -> None:
        pass

    def observe(self, *args, **kwargs) -> None:
        pass

    def set(self, *args, **kwargs) -> None:
        pass

    def time(self):  # pragma: no cover — only used via context manager helper
        @contextmanager
        def _noop() -> Iterator[None]:
            yield
        return _noop()


try:
    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest(registry=None):  # type: ignore[no-redef]
        return b"# prometheus_client not installed\n"


# A dedicated registry so the cache's metrics don't collide with anything
# the host application already exposes. Users can still scrape it via the
# FastAPI endpoint (which references this module's `registry`).
if _PROMETHEUS_AVAILABLE:
    registry = CollectorRegistry()

    lookups_total = Counter(
        "scache_lookups_total",
        "Cache lookup outcomes.",
        labelnames=("result", "reason"),
        registry=registry,
    )
    writes_total = Counter(
        "scache_writes_total",
        "Cache write outcomes.",
        labelnames=("outcome",),
        registry=registry,
    )
    extractor_calls_total = Counter(
        "scache_extractor_calls_total",
        "Entity extractor invocations.",
        labelnames=("outcome",),
        registry=registry,
    )
    extraction_cache_total = Counter(
        "scache_extraction_cache_total",
        "Extraction-cache lookup outcomes.",
        labelnames=("result",),
        registry=registry,
    )
    extractor_latency_seconds = Histogram(
        "scache_extractor_latency_seconds",
        "Wall-clock latency of entity extractor calls.",
        registry=registry,
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    search_latency_seconds = Histogram(
        "scache_search_latency_seconds",
        "Wall-clock latency of cache search calls (end to end).",
        registry=registry,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )

    # HTTP-level series. The `path` label is the ROUTE TEMPLATE
    # ("/v1/caches/{cache_id}"), never the raw URL: a label whose cardinality
    # follows user input turns a scrape into an outage.
    http_requests_total = Counter(
        "scache_http_requests_total",
        "HTTP requests served, by route template and status class.",
        labelnames=("method", "path", "status"),
        registry=registry,
    )
    http_request_latency_seconds = Histogram(
        "scache_http_request_latency_seconds",
        "Wall-clock latency of HTTP requests, by route template.",
        labelnames=("method", "path"),
        registry=registry,
        buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    http_requests_in_flight = Gauge(
        "scache_http_requests_in_flight",
        "HTTP requests currently being served.",
        registry=registry,
    )
else:
    registry = None  # type: ignore[assignment]
    lookups_total = _NoOpMetric()  # type: ignore[assignment]
    writes_total = _NoOpMetric()  # type: ignore[assignment]
    extractor_calls_total = _NoOpMetric()  # type: ignore[assignment]
    extraction_cache_total = _NoOpMetric()  # type: ignore[assignment]
    extractor_latency_seconds = _NoOpMetric()  # type: ignore[assignment]
    search_latency_seconds = _NoOpMetric()  # type: ignore[assignment]
    http_requests_total = _NoOpMetric()  # type: ignore[assignment]
    http_request_latency_seconds = _NoOpMetric()  # type: ignore[assignment]
    http_requests_in_flight = _NoOpMetric()  # type: ignore[assignment]


def record_hit() -> None:
    lookups_total.labels(result="hit", reason="hit").inc()


def record_miss(reason: str) -> None:
    lookups_total.labels(result="miss", reason=reason).inc()


def record_write(outcome: str) -> None:
    writes_total.labels(outcome=outcome).inc()


def record_extractor(outcome: str) -> None:
    extractor_calls_total.labels(outcome=outcome).inc()


def record_extraction_cache(result: str) -> None:
    """`result` should be 'hit' or 'miss'."""
    extraction_cache_total.labels(result=result).inc()


@contextmanager
def time_extractor() -> Iterator[None]:
    """Times an extractor call; observes into the latency histogram."""
    if not _PROMETHEUS_AVAILABLE:
        yield
        return
    with extractor_latency_seconds.time():
        yield


@contextmanager
def time_search() -> Iterator[None]:
    """Times an end-to-end search call; observes into the latency histogram."""
    if not _PROMETHEUS_AVAILABLE:
        yield
        return
    with search_latency_seconds.time():
        yield


def render_latest() -> bytes:
    """Returns the current metrics in Prometheus text exposition format.

    Safe to call even when prometheus_client is not installed — returns a
    short stub document so scrapers don't crash.
    """
    return generate_latest(registry) if _PROMETHEUS_AVAILABLE else generate_latest()


def is_available() -> bool:
    return _PROMETHEUS_AVAILABLE

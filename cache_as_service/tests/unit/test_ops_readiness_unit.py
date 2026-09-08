"""Liveness, readiness, and the metrics endpoint.

The bug being pinned here is not subtle: `/health` pinged Redis and called
that readiness, so a pod with Postgres down advertised itself as able to serve
gateway traffic. These tests hold the split — liveness answers for the process,
readiness answers for the dependencies — and hold the bound on the probes,
because a readiness check that hangs is indistinguishable from the outage it
was supposed to report.
"""

import threading
import time
from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from semantic_cache.adapters.ops_router import install_http_metrics, ops_router
from semantic_cache.adapters.readiness import Check, ReadinessProbe
from semantic_cache.adapters.saas_router import (
    get_admin_key,
    get_readiness,
    health_router,
)
from semantic_cache.core import metrics

ADMIN_KEY = "sc-admin-key"


def _client(
    probe: Optional[ReadinessProbe] = None, admin_key: Optional[str] = ADMIN_KEY
) -> TestClient:
    app = FastAPI()
    install_http_metrics(app)
    app.dependency_overrides[get_readiness] = lambda: probe or ReadinessProbe([])
    app.dependency_overrides[get_admin_key] = lambda: admin_key
    app.include_router(health_router)
    app.include_router(ops_router)
    return TestClient(app)


# -- ReadinessProbe --------------------------------------------------------- #


def test_all_checks_up_is_ready() -> None:
    ok, detail = ReadinessProbe(
        [Check("redis", lambda: True), Check("postgres", lambda: True)]
    ).run()
    assert ok
    assert detail["checks"] == {"redis": "ok", "postgres": "ok"}


def test_a_required_check_down_is_unready_and_names_itself() -> None:
    ok, detail = ReadinessProbe(
        [Check("redis", lambda: True), Check("postgres", lambda: False)]
    ).run()
    assert not ok
    assert detail["failed"] == ["postgres"]
    assert detail["checks"]["redis"] == "ok"


def test_an_optional_check_down_is_reported_but_still_ready() -> None:
    """A down LLM is one failed completion. Marking every replica unready over
    it also kills the cache hits that need no LLM at all."""
    ok, detail = ReadinessProbe(
        [Check("redis", lambda: True), Check("llm", lambda: False, required=False)]
    ).run()
    assert ok
    assert detail["checks"]["llm"] == "down"


def test_a_raising_probe_is_down_not_a_500() -> None:
    def boom() -> bool:
        raise RuntimeError("connection refused")

    ok, detail = ReadinessProbe([Check("redis", boom)]).run()
    assert not ok
    assert detail["checks"]["redis"] == "down"


def test_a_hanging_probe_is_abandoned_at_the_deadline() -> None:
    """The whole point of the bound: a stalled dependency must not become a
    stalled readiness endpoint."""
    release = threading.Event()

    def hang() -> bool:
        release.wait(30)
        return True

    started = time.perf_counter()
    try:
        ok, detail = ReadinessProbe([Check("redis", hang)], timeout=0.2).run()
    finally:
        release.set()
    elapsed = time.perf_counter() - started

    assert not ok
    assert detail["checks"]["redis"] == "timeout"
    assert elapsed < 5.0, f"probe took {elapsed:.1f}s — the deadline did not hold"


def test_one_slow_check_does_not_serialize_behind_another() -> None:
    """Checks run concurrently: two 0.3s probes must not cost 0.6s, or the
    deadline becomes a function of how many dependencies you have."""
    def slow() -> bool:
        time.sleep(0.3)
        return True

    started = time.perf_counter()
    ok, _ = ReadinessProbe(
        [Check("a", slow), Check("b", slow), Check("c", slow)], timeout=2.0
    ).run()
    assert ok
    assert time.perf_counter() - started < 0.8


def test_no_checks_is_ready() -> None:
    """A library user mounting the router has none of our dependencies."""
    ok, detail = ReadinessProbe([]).run()
    assert ok and detail["checks"] == {}


# -- the routes ------------------------------------------------------------- #


def test_health_is_liveness_and_ignores_dependencies() -> None:
    """It drives the container healthcheck under `restart: unless-stopped`.
    Restarting a live process does not bring Redis back."""
    client = _client(ReadinessProbe([Check("redis", lambda: False)]))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_is_503_when_a_required_dependency_is_down() -> None:
    client = _client(ReadinessProbe([Check("postgres", lambda: False)]))
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["failed"] == ["postgres"]


def test_ready_is_200_when_everything_required_is_up() -> None:
    client = _client(ReadinessProbe([Check("redis", lambda: True)]))
    assert _client(ReadinessProbe([Check("redis", lambda: True)])).get(
        "/ready"
    ).status_code == 200
    assert client.get("/ready").json()["checks"] == {"redis": "ok"}


# -- /metrics --------------------------------------------------------------- #


def test_metrics_requires_the_admin_key() -> None:
    """Exposition leaks route templates, traffic shape and error rates. On a
    multi-tenant service that is not public."""
    client = _client()
    assert client.get("/metrics").status_code == 401
    assert client.get(
        "/metrics", headers={"Authorization": "Bearer wrong"}
    ).status_code == 403


def test_metrics_is_503_not_open_when_no_admin_key_is_configured() -> None:
    """Fail closed: "I forgot to set the key" must not look like "there are no
    metrics"."""
    r = _client(admin_key=None).get(
        "/metrics", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
    )
    assert r.status_code == 503


@pytest.mark.skipif(
    not metrics.is_available(), reason="prometheus_client not installed"
)
def test_metrics_serves_exposition_to_the_admin_key() -> None:
    client = _client()
    client.get("/health")  # generate one series
    r = client.get("/metrics", headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    assert r.status_code == 200
    assert "scache_http_requests_total" in r.text


@pytest.mark.skipif(
    not metrics.is_available(), reason="prometheus_client not installed"
)
def test_the_path_label_is_the_route_template_not_the_url() -> None:
    """A label keyed on user input is an unbounded time series — it takes down
    the system you would use to find out what took the system down."""
    client = _client()
    for cache_id in ("aaa", "bbb", "ccc"):
        client.get(f"/nope/{cache_id}")  # unmatched: 404
    body = client.get(
        "/metrics", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
    ).text
    assert "<unmatched>" in body
    for cache_id in ("aaa", "bbb", "ccc"):
        assert f"/nope/{cache_id}" not in body

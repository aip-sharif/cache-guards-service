"""Server wiring for the guard — no Postgres, no Redis, no network.

The one behaviour worth a test here is the 503 tuple: FastAPI resolves every
dependency BEFORE the handler body runs, so a guard dependency left as a
NotImplementedError stub in the unconfigured branch surfaces as a bare 500 —
outside the OpenAI error envelope, since install_openai_error_handlers only
covers HTTPException and RequestValidationError.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from semantic_cache.gateway.router import (
    gateway_router,
    get_app_config,
    get_gateway_pool,
    get_gateway_settings,
    get_gateway_store,
    get_guard_checker,
    get_guard_log,
    get_guard_switch,
    get_upstream,
    install_openai_error_handlers,
)

SERVING_DEPS = (
    get_gateway_pool, get_upstream, get_app_config, get_gateway_settings,
    get_guard_checker, get_guard_switch, get_guard_log,
)


def _app(overridden) -> TestClient:
    """A gateway whose serving deps all 503, mirroring server.py's else-branch."""
    from fastapi import HTTPException

    def _unconfigured():
        raise HTTPException(503, "Gateway serving is not configured — "
                                 "missing env: SC_EMBED_BASE_URL")

    app = FastAPI()
    # server.py overrides the store unconditionally, outside the serving
    # if/else — it only needs SC_PG_DSN.
    app.dependency_overrides[get_gateway_store] = lambda: object()
    for dep in overridden:
        app.dependency_overrides[dep] = _unconfigured
    app.include_router(gateway_router)
    install_openai_error_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _chat(client):
    return client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer k"},
    )


def test_unconfigured_serving_is_a_503_in_the_openai_envelope() -> None:
    response = _chat(_app(SERVING_DEPS))
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "gateway_error"
    assert "SC_EMBED_BASE_URL" in body["error"]["message"]


@pytest.mark.parametrize("dep", [get_guard_checker, get_guard_switch,
                                 get_guard_log])
def test_a_missing_guard_dep_is_masked_by_an_earlier_503(dep) -> None:
    """Documents why the guard deps are in server.py's 503 tuple anyway.

    FastAPI resolves dependencies in parameter order, and the cache's deps come
    first — so today an omitted guard dep never gets reached: the request 503s
    before it. Including them is therefore defence against a future reordering
    of the handler signature, not a live bug fix. Asserting the real behaviour
    here rather than a 500 keeps this test honest; if it ever starts failing,
    the signature order changed and the tuple is now load-bearing.
    """
    partial = tuple(d for d in SERVING_DEPS if d is not dep)
    assert _chat(_app(partial)).status_code == 503


def test_an_unoverridden_stub_is_a_bare_500_not_a_503() -> None:
    """The failure mode the tuple protects against, shown directly.

    A NotImplementedError from a dependency escapes as a 500 OUTSIDE the
    OpenAI envelope — install_openai_error_handlers covers only HTTPException
    and RequestValidationError. In the unconfigured branch a cache dependency
    always 503s before any guard dependency is reached, which is why the test
    above asserts 503; this is what would happen if one ever were reached.
    """
    app = FastAPI()
    app.dependency_overrides[get_gateway_store] = lambda: object()
    app.include_router(gateway_router)
    install_openai_error_handlers(app)
    response = _chat(TestClient(app, raise_server_exceptions=False))
    assert response.status_code == 500


def test_the_guard_stubs_raise_rather_than_defaulting_to_off() -> None:
    # A stub that returned None would mean a wiring mistake silently disables
    # the guard — the exact failure class the whole module exists to prevent.
    for dep in (get_guard_checker, get_guard_switch, get_guard_log):
        with pytest.raises(NotImplementedError):
            dep()

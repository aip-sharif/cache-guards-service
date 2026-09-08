"""Request-body ceiling.

Before this, the largest request the service would accept was decided by
whoever was sending it: the gateway takes an arbitrary OpenAI-shaped body and
the SaaS schemas accept unbounded `query`, `response` and `config`.

The test that matters is the chunked one. A Content-Length check is a
suggestion — the cheapest way past it is to not send a Content-Length — so the
limit has to be enforced on the stream.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from semantic_cache.adapters.ops_router import install_body_limit

LIMIT = 1024


def _app(limit: int = LIMIT) -> TestClient:
    app = FastAPI()

    @app.post("/echo")
    async def echo(payload: dict) -> dict:
        return {"len": len(payload.get("text", ""))}

    install_body_limit(app, limit)
    return TestClient(app)


def test_a_small_body_passes() -> None:
    client = _app()
    r = client.post("/echo", json={"text": "x" * 10})
    assert r.status_code == 200
    assert r.json() == {"len": 10}


def test_an_oversized_body_is_413() -> None:
    client = _app()
    r = client.post("/echo", json={"text": "x" * (LIMIT * 4)})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_a_chunked_body_cannot_bypass_the_limit() -> None:
    """No Content-Length to check — so a header-only implementation lets this
    straight through."""
    client = _app()

    def chunks():
        yield b'{"text":"'
        for _ in range(20):
            yield b"x" * 512
        yield b'"}'

    r = client.post(
        "/echo",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


def test_a_lying_content_length_does_not_help() -> None:
    """Understating the length only changes which check catches it."""
    client = _app()
    body = b'{"text":"' + b"x" * (LIMIT * 4) + b'"}'
    r = client.post(
        "/echo",
        content=body,
        headers={"Content-Type": "application/json", "Content-Length": "10"},
    )
    assert r.status_code in (413, 400, 422)


def test_a_body_exactly_at_the_limit_is_accepted() -> None:
    """The boundary is inclusive — an off-by-one here rejects legitimate
    traffic at exactly the size an operator configured for."""
    client = _app()
    filler = LIMIT - len(b'{"text":""}')
    r = client.post(
        "/echo",
        content=b'{"text":"' + b"x" * filler + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 200


def test_a_zero_limit_disables_the_check() -> None:
    client = _app(limit=0)
    r = client.post("/echo", json={"text": "x" * 100_000})
    assert r.status_code == 200


def test_a_get_with_no_body_is_unaffected() -> None:
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    install_body_limit(app, LIMIT)
    assert TestClient(app).get("/ping").status_code == 200

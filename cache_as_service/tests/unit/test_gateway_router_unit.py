"""Behavioral tests for the OpenAI-compatible gateway router.

Everything is faked at the seams — in-memory store, fake APP config client,
dict-backed cache, httpx.MockTransport upstream — so the full request cycle
(key → APP config → cache → mlops upstream → write-back → message log) runs
with no Redis, no Postgres, and no network. CI-safe.

In the key-only model the client's bearer IS the whole identity: we never mint
or validate keys ourselves. Presenting the key to the APP either yields a
config (valid) or a 403 (unknown). Cache isolation is by the project_id the
APP returns.
"""

import json
from typing import Any, Dict, List, Optional

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from semantic_cache.gateway.app_config import AppConfigError
from semantic_cache.gateway.router import (
    GatewaySettings,
    gateway_router,
    get_app_config,
    get_gateway_pool,
    get_gateway_settings,
    get_guard_checker,
    get_guard_log,
    get_guard_switch,
    GuardSwitch,
    get_gateway_store,
    get_upstream,
    install_openai_error_handlers,
)
from semantic_cache.gateway.upstream import UpstreamClient

LLM_BASE = "https://mlops-llm.example.com"
CLIENT_KEY = "sc-proj-clientkey"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeStore:
    """In-memory stand-in for PostgresGatewayStore (message log only)."""

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def log_message(self, scope, model, cache_hit, query_text, response,
                    similarity=None, finish_reason=None, usage=None,
                    latency_ms=None, request_id=None):
        self.messages.append({
            "scope": scope, "model": model, "cache_hit": cache_hit,
            "query_text": query_text, "response": response,
            "similarity": similarity, "finish_reason": finish_reason,
            "usage": usage, "latency_ms": latency_ms,
            "request_id": request_id,
        })
        return True

    def list_messages(self, scope, limit=50):
        rows = [m for m in self.messages if m["scope"] == scope]
        return list(reversed(rows))[:limit]


APP_CONFIG = {
    "model": "upstream-slug",
    "model_api_key": "sk-up",
    "embed_model": "text-embedding-3-small",
    "embed_api_key": "sk-embed",
    "extractor_model": None,
    "extractor_api_key": None,
    "extractor_domain": None,
    "cache_config": {},
    "project_id": "proj-A",          # → cache scope
}


class FakeAppConfig:
    """Stand-in for AppConfigClient — the APP's per-key config. Any key maps to
    the default config unless registered otherwise via ``by_key``."""

    def __init__(self) -> None:
        self.default = dict(APP_CONFIG)
        self.by_key: Dict[str, Dict[str, Any]] = {}
        self.fail: Optional[AppConfigError] = None
        self.calls: List[str] = []
        self.invalidated: List[Optional[str]] = []

    async def resolve(self, api_key):
        self.calls.append(api_key)
        if self.fail is not None:
            raise self.fail
        return dict(self.by_key.get(api_key, self.default))

    def invalidate(self, api_key=None):
        self.invalidated.append(api_key)


class FakeCache:
    """Exact-match async cache — the manager seam the router talks to."""

    def __init__(self) -> None:
        self.data: Dict[tuple, str] = {}

    async def asearch(self, query, scope=None):
        response = self.data.get((scope, query))
        if response is None:
            return None
        return {"response": response, "similarity": 0.93, "metadata": {}}

    async def aset(self, query, response, metadata=None, ttl=None,
                   keep_forever=False, scope=None):
        self.data[(scope, query)] = response


class FakePool:
    def __init__(self) -> None:
        self.caches: Dict[str, FakeCache] = {}
        self.rows: List[Dict[str, Any]] = []

    def get(self, model_row):
        self.rows.append(model_row)
        return self.caches.setdefault(model_row["embed_model"], FakeCache())


UPSTREAM_JSON = {
    "id": "chatcmpl-upstream",
    "object": "chat.completion",
    "created": 1,
    "model": "upstream-slug",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Paris"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
}

UPSTREAM_SSE = (
    'data: {"id":"c","object":"chat.completion.chunk","model":"upstream-slug",'
    '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    'data: {"id":"c","object":"chat.completion.chunk","model":"upstream-slug",'
    '"choices":[{"index":0,"delta":{"content":"Paris"},"finish_reason":null}]}\n\n'
    'data: {"id":"c","object":"chat.completion.chunk","model":"upstream-slug",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


class UpstreamSpy:
    """MockTransport handler that counts calls and serves JSON or SSE."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.json_response: Dict[str, Any] = UPSTREAM_JSON
        self.sse_body: str = UPSTREAM_SSE
        self.status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append({
            "url": str(request.url),
            "auth": request.headers.get("authorization"),
            "body": body,
        })
        if self.status != 200:
            return httpx.Response(
                self.status, json={"error": {"message": "upstream says no"}}
            )
        if body.get("stream"):
            return httpx.Response(
                200, content=self.sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=self.json_response)


# --------------------------------------------------------------------------- #
# App fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def env():
    store, pool, spy = FakeStore(), FakePool(), UpstreamSpy()
    app_config = FakeAppConfig()
    upstream = UpstreamClient(httpx.AsyncClient(transport=httpx.MockTransport(spy)))
    settings = GatewaySettings(
        llm_base_url=LLM_BASE,
        embed_base_url="https://mlops-embed.example.com",
        extractor_base_url="https://mlops-extract.example.com",
    )
    app = FastAPI()
    app.dependency_overrides[get_gateway_store] = lambda: store
    app.dependency_overrides[get_gateway_pool] = lambda: pool
    app.dependency_overrides[get_upstream] = lambda: upstream
    app.dependency_overrides[get_app_config] = lambda: app_config
    app.dependency_overrides[get_gateway_settings] = lambda: settings
    # No guard wired: every test in this file exercises the un-guarded path,
    # which must stay byte-identical to what it was before the guard existed.
    app.dependency_overrides[get_guard_checker] = lambda: None
    app.dependency_overrides[get_guard_switch] = lambda: GuardSwitch(enabled=True)
    app.dependency_overrides[get_guard_log] = lambda: None
    app.include_router(gateway_router)
    install_openai_error_handlers(app)
    client = TestClient(app)

    return type("Env", (), {
        "client": client, "store": store, "pool": pool, "spy": spy,
        "app_config": app_config,
        "key": CLIENT_KEY, "scope": APP_CONFIG["project_id"],
    })


def _chat(env, text="capital of france?", stream=False, key=None,
          model="upstream-slug", messages=None, **extra):
    return env.client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages or [{"role": "user", "content": text}],
            "stream": stream,
            **extra,
        },
        headers={"Authorization": f"Bearer {key or env.key}"},
    )


def _cache(env):
    return env.pool.caches[APP_CONFIG["embed_model"]]


# --------------------------------------------------------------------------- #
# Auth + APP config resolution
# --------------------------------------------------------------------------- #


def test_missing_bearer_is_401_with_openai_error_shape(env) -> None:
    r = env.client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 401
    assert "error" in r.json() and "detail" not in r.json()


def test_unknown_key_yields_403_from_the_app(env) -> None:
    # We do not validate keys locally — the APP rejects the key with a 403.
    env.app_config.fail = AppConfigError(403, "no config for this key")
    r = _chat(env, key="sc-proj-nobody")
    assert r.status_code == 403
    assert "no config" in r.json()["error"]["message"]


def test_config_resolved_with_the_callers_own_key(env) -> None:
    _chat(env)
    assert env.app_config.calls == [env.key]


def test_app_unreachable_yields_502(env) -> None:
    env.app_config.fail = AppConfigError(502, "APP config endpoint unreachable")
    assert _chat(env).status_code == 502


def test_models_endpoint_returns_the_apps_model(env) -> None:
    r = env.client.get("/v1/models", headers={"Authorization": f"Bearer {env.key}"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["upstream-slug"]


# --------------------------------------------------------------------------- #
# Cache-first completion flow
# --------------------------------------------------------------------------- #


def test_miss_goes_to_mlops_then_hit_serves_from_cache(env) -> None:
    r1 = _chat(env, model="anything-the-client-says")
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["choices"][0]["message"]["content"] == "Paris"
    assert body1["model"] == "anything-the-client-says"  # requested name echoed
    assert len(env.spy.calls) == 1
    assert env.spy.calls[0]["url"].startswith(LLM_BASE)   # mlops endpoint (env)
    assert env.spy.calls[0]["auth"] == "Bearer sk-up"     # key from the APP
    assert env.spy.calls[0]["body"]["model"] == "upstream-slug"

    r2 = _chat(env, model="anything-the-client-says")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["choices"][0]["message"]["content"] == "Paris"
    assert body2["semantic_cache"]["hit"] is True
    assert body2["usage"]["total_tokens"] == 0
    assert len(env.spy.calls) == 1                 # upstream NOT called again


def test_different_projects_have_isolated_caches(env) -> None:
    _chat(env)                                     # cached under scope proj-A
    # A second key the APP maps to a DIFFERENT project_id → separate scope.
    other = dict(APP_CONFIG, project_id="proj-B")
    env.app_config.by_key["sc-proj-other"] = other
    r = _chat(env, key="sc-proj-other")
    assert r.status_code == 200
    assert len(env.spy.calls) == 2                 # second project missed


def test_scope_falls_back_to_key_hash_when_app_omits_project_id(env) -> None:
    # An APP that returns no project_id must still isolate — by key hash.
    no_pid = {k: v for k, v in APP_CONFIG.items() if k != "project_id"}
    env.app_config.default = no_pid
    _chat(env)
    assert env.store.messages[0]["scope"].startswith("k_")


def test_no_user_message_passes_through_uncached(env) -> None:
    msgs = [{"role": "system", "content": "sys only"}]
    r = _chat(env, messages=msgs)
    assert r.status_code == 200
    assert len(env.spy.calls) == 1
    # the pool was never even consulted → no cache instance, nothing stored
    assert all(c.data == {} for c in env.pool.caches.values())


@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "function", "function": {"name": "f"}}]},
    {"tool_choice": "required"},
    {"response_format": {"type": "json_object"}},
    {"n": 2},
])
def test_tool_json_and_multichoice_requests_bypass_the_cache(env, extra) -> None:
    r1 = _chat(env, **extra)
    assert r1.status_code == 200
    r2 = _chat(env, **extra)
    assert r2.status_code == 200
    assert len(env.spy.calls) == 2                 # never cached, never served stale
    assert all(c.data == {} for c in env.pool.caches.values())
    # and the special fields reached the upstream untouched
    for field in extra:
        assert env.spy.calls[0]["body"][field] == extra[field]


def test_cache_off_mode_never_caches_but_still_logs(env) -> None:
    # The APP tells us this client wants caching OFF (pure passthrough).
    env.app_config.default = dict(APP_CONFIG, cache_config={"cache_mode": "off"})
    _chat(env)
    _chat(env)
    assert len(env.spy.calls) == 2                  # every request hits upstream
    # The pool is never consulted — an 'off' client pays no cache cost at all.
    assert env.pool.rows == []
    assert env.pool.caches == {}
    # …but the traffic is still logged (query included), both as misses.
    assert [m["cache_hit"] for m in env.store.messages] == [False, False]
    assert env.store.messages[0]["query_text"] == "capital of france?"


def test_non_stop_finish_is_not_cached(env) -> None:
    env.spy.json_response = {
        **UPSTREAM_JSON,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "truncat"},
            "finish_reason": "length",
        }],
    }
    _chat(env)
    assert _cache(env).data == {}
    _chat(env)
    assert len(env.spy.calls) == 2                 # still missing → upstream again


def test_upstream_error_surfaces_status_without_leaking_body(env) -> None:
    env.spy.status = 429
    r = _chat(env)
    assert r.status_code == 429                      # status is preserved
    message = r.json()["error"]["message"]
    assert "upstream says no" not in message         # provider body is not
    assert "429" in message


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_stream_miss_passes_through_and_caches(env) -> None:
    r = _chat(env, stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "Paris" in r.text and "data: [DONE]" in r.text
    # accumulated + cached → next NON-stream call is a hit, no upstream
    r2 = _chat(env)
    assert r2.json()["semantic_cache"]["hit"] is True
    assert len(env.spy.calls) == 1


def test_stream_hit_replays_as_sse(env) -> None:
    _chat(env)                                     # prime via non-stream miss
    r = _chat(env, stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [l for l in r.text.splitlines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    joined = "".join(
        json.loads(l[6:])["choices"][0]["delta"].get("content") or ""
        for l in lines[:-1]
    )
    assert joined == "Paris"
    assert len(env.spy.calls) == 1                 # replay never hit upstream


def test_stream_hit_honors_include_usage(env) -> None:
    _chat(env)                                     # prime the cache
    r = _chat(env, stream=True, stream_options={"include_usage": True})
    payloads = [json.loads(l[6:]) for l in r.text.splitlines()
                if l.startswith("data:") and "[DONE]" not in l]
    usage_chunks = [p for p in payloads if p.get("usage") is not None]
    assert len(usage_chunks) == 1
    assert usage_chunks[-1]["usage"]["total_tokens"] == 0
    assert usage_chunks[-1]["choices"] == []
    # without the option, no usage chunk is emitted
    r2 = _chat(env, stream=True)
    assert '"usage"' not in r2.text


def test_stream_upstream_error_returns_json_error(env) -> None:
    env.spy.status = 500
    r = _chat(env, stream=True)
    assert r.status_code == 500
    assert "error" in r.json()


# --------------------------------------------------------------------------- #
# Message log (Postgres in prod; FakeStore here)
# --------------------------------------------------------------------------- #


def test_messages_are_logged_hit_and_miss(env) -> None:
    _chat(env)
    _chat(env)
    hits = [m["cache_hit"] for m in env.store.messages]
    assert hits == [False, True]
    assert env.store.messages[0]["response"] == "Paris"
    assert env.store.messages[0]["usage"]["total_tokens"] == 9
    assert env.store.messages[1]["similarity"] is not None
    assert env.store.messages[0]["scope"] == env.scope


def test_caller_can_read_own_messages(env) -> None:
    _chat(env)
    r = env.client.get(
        "/v1/messages", headers={"Authorization": f"Bearer {env.key}"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body[0]["query_text"] == "capital of france?"
    assert body[0]["cache_hit"] is False

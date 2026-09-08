"""The guard gate inside the router.

Reuses the fakes from test_gateway_router_unit so the un-guarded and guarded
paths are exercised through the same seams.
"""

from typing import Any, Dict, List, Optional

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from semantic_cache.adapters.saas_router import get_admin_key
from semantic_cache.gateway.guard_checker import GuardOutcome
from semantic_cache.gateway.router import (
    GatewaySettings,
    GuardSwitch,
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
from semantic_cache.gateway.upstream import UpstreamClient

from tests.unit.test_gateway_router_unit import (
    APP_CONFIG,
    LLM_BASE,
    CLIENT_KEY,
    FakeAppConfig,
    FakePool,
    FakeStore,
    UpstreamSpy,
)

ADMIN_KEY = "sc-admin-key"

POLICY = """
categories:
  - category_id: competitors
    action: block
    refusal: "I can't discuss other companies' products here."
    disallowed_exemplars:
      - "What do you think of Rivalco's product?"
    allowed_exemplars:
      - "What is your refund policy?"
"""

GUARD_BLOCK = {"enabled": True, "policy": POLICY}


class GuardStore(FakeStore):
    """FakeStore plus the guard tables."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: List[Dict[str, Any]] = []

    def log_guard_decisions(self, rows):
        self.decisions.extend(rows)
        return len(rows)

    def list_guard_decisions(self, scope, limit=50):
        rows = [d for d in self.decisions if d["scope"] == scope]
        return list(reversed(rows))[:limit]


class FakeGuardLog:
    def __init__(self, store) -> None:
        self.store = store

    def submit(self, row):
        self.store.log_guard_decisions([row])
        return True


class FakeResolved:
    """A ResolvedGuard stand-in — the router only reads a handful of fields."""

    def __init__(self, **params) -> None:
        self.policy_hash = params.pop("policy_hash", "a" * 64)
        self.check_roles = ("user", "system", "tool")
        self.params = type("P", (), {
            "degrade_to_unguarded": params.get("degrade_to_unguarded", False),
            "unavailable_response": params.get("unavailable_response", "error"),
            "unavailable_refusal": params.get("unavailable_refusal", None),
            "log_turn_text": params.get("log_turn_text", False),
        })()


class FakeGuardChecker:
    """Drives the router down a chosen branch without any embedding or judge."""

    def __init__(self) -> None:
        self.outcome = GuardOutcome(action="allow")
        self.resolved: Optional[FakeResolved] = None
        self.config_error: Optional[Exception] = None
        self.calls: List[Any] = []
        self.pool = self

    # -- the pool seam the router uses --------------------------------- #
    def resolve(self, config):
        if self.config_error is not None:
            raise self.config_error
        if not config.get("guard"):
            return None
        return self.resolved

    async def check(self, segments, resolved, *, judge_base_url=None):
        self.calls.append([s.text for s in segments])
        return self.outcome


@pytest.fixture
def env():
    store, pool, spy = GuardStore(), FakePool(), UpstreamSpy()
    app_config = FakeAppConfig()
    app_config.default = {**APP_CONFIG, "guard": GUARD_BLOCK}
    guard = FakeGuardChecker()
    guard.resolved = FakeResolved()
    switch = GuardSwitch(enabled=True)
    upstream = UpstreamClient(httpx.AsyncClient(transport=httpx.MockTransport(spy)))
    settings = GatewaySettings(
        llm_base_url=LLM_BASE, embed_base_url="https://embed.example.com"
    )

    app = FastAPI()
    app.dependency_overrides[get_gateway_store] = lambda: store
    app.dependency_overrides[get_gateway_pool] = lambda: pool
    app.dependency_overrides[get_upstream] = lambda: upstream
    app.dependency_overrides[get_app_config] = lambda: app_config
    app.dependency_overrides[get_gateway_settings] = lambda: settings
    app.dependency_overrides[get_guard_checker] = lambda: guard
    app.dependency_overrides[get_guard_switch] = lambda: switch
    app.dependency_overrides[get_guard_log] = lambda: FakeGuardLog(store)
    app.dependency_overrides[get_admin_key] = lambda: ADMIN_KEY
    app.include_router(gateway_router)
    install_openai_error_handlers(app)

    return type("Env", (), {
        "client": TestClient(app), "store": store, "pool": pool, "spy": spy,
        "app_config": app_config, "guard": guard, "switch": switch,
        "key": CLIENT_KEY, "scope": APP_CONFIG["project_id"],
    })


def _chat(env, text="capital of france?", stream=False, key=None, messages=None,
          **extra):
    return env.client.post(
        "/v1/chat/completions",
        json={
            "model": "upstream-slug",
            "messages": messages or [{"role": "user", "content": text}],
            "stream": stream,
            **extra,
        },
        headers={"Authorization": f"Bearer {key or env.key}"},
    )


def test_the_real_checker_satisfies_what_the_router_calls() -> None:
    """Contract test — FakeGuardChecker must not be able to hide a gap.

    The router reaches `guard.pool.resolve(...)` and `guard.check(...)`. The
    fake below defines both, so a real GuardChecker missing either passed the
    whole suite and failed on the first live request. This asserts the REAL
    class, not the double.
    """
    import inspect

    from semantic_cache.gateway.guard_checker import GuardChecker
    from semantic_cache.gateway.guard_pool import GuardPool

    checker = GuardChecker.__new__(GuardChecker)
    checker._pool = "sentinel-pool"
    assert checker.pool == "sentinel-pool"

    assert callable(getattr(GuardChecker, "check", None))
    assert callable(getattr(GuardPool, "resolve", None))
    # The router passes judge_base_url by keyword.
    assert "judge_base_url" in inspect.signature(GuardChecker.check).parameters


def _block(category="competitors"):
    return GuardOutcome(
        action="block", matched_category=category,
        reason="embedding score at or above the block threshold",
        refusal="I can't discuss other companies' products here.",
        embedding_score=0.91, judge_invoked=False,
    )


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #


def test_a_block_never_reaches_the_cache_or_upstream(env) -> None:
    env.guard.outcome = _block()
    response = _chat(env, "What do you think of Rivalco's product?")

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "content_filter"
    assert body["choices"][0]["message"]["content"].startswith("I can't discuss")
    assert body["usage"]["total_tokens"] == 0
    assert body["guardrail"]["action"] == "block"
    assert body["guardrail"]["category"] == "competitors"
    assert response.headers["x-guardrail"] == "block"

    # The whole point: no upstream call, and not even a cache manager built.
    assert env.spy.calls == []
    assert env.pool.rows == []


def test_a_block_is_recorded_in_both_logs(env) -> None:
    env.guard.outcome = _block()
    _chat(env, "bad")
    assert len(env.store.messages) == 1
    assert env.store.messages[0]["finish_reason"] == "content_filter"
    assert len(env.store.decisions) == 1
    assert env.store.decisions[0]["action"] == "block"


def test_a_blocked_request_id_joins_the_two_logs(env) -> None:
    env.guard.outcome = _block()
    body = _chat(env, "bad").json()
    request_id = body["guardrail"]["request_id"]
    assert env.store.messages[0]["request_id"] == request_id
    assert env.store.decisions[0]["request_id"] == request_id


def test_a_streamed_block_is_a_well_formed_sse_stream(env) -> None:
    env.guard.outcome = _block()
    response = _chat(env, "bad", stream=True)
    assert response.status_code == 200
    text = response.text
    assert text.rstrip().endswith("data: [DONE]")
    assert '"finish_reason": "content_filter"' in text.replace('"finish_reason":"content_filter"',
                                                              '"finish_reason": "content_filter"')
    assert env.spy.calls == []


def test_the_refusal_text_comes_from_the_outcome_not_a_constant(env) -> None:
    env.guard.outcome = GuardOutcome(
        action="block", matched_category="competitors",
        refusal="متأسفم، نمی‌توانم در این مورد کمک کنم.",
    )
    body = _chat(env, "bad").json()
    assert body["choices"][0]["message"]["content"] == "متأسفم، نمی‌توانم در این مورد کمک کنم."


# --------------------------------------------------------------------------- #
# Coverage: the paths that skip the cache entirely
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("extra", [
    {"tools": [{"type": "function", "function": {"name": "f"}}]},
    {"response_format": {"type": "json_object"}},
    {"n": 2},
])
def test_the_guard_runs_on_cache_bypass_requests(env, extra) -> None:
    env.guard.outcome = _block()
    response = _chat(env, "bad", **extra)
    assert response.status_code == 200
    assert response.json()["guardrail"]["action"] == "block"
    assert env.spy.calls == []          # never forwarded


def test_the_guard_runs_when_caching_is_off(env) -> None:
    env.app_config.default = {
        **APP_CONFIG, "guard": GUARD_BLOCK, "cache_config": {"cache_mode": "off"},
    }
    env.guard.outcome = _block()
    response = _chat(env, "bad")
    assert response.json()["guardrail"]["action"] == "block"
    assert env.pool.rows == []
    assert env.spy.calls == []


def test_the_guard_sees_the_whole_message_array(env) -> None:
    _chat(env, messages=[
        {"role": "user", "content": "What do you think of Rivalco's product?"},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": "continue"},
    ])
    # Both user turns were handed to the checker, not just the last.
    assert env.guard.calls[0] == [
        "What do you think of Rivalco's product?", "continue"
    ]


def test_an_image_only_turn_is_refused_not_served(env) -> None:
    response = _chat(env, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
    ]}])
    assert response.status_code == 503
    assert env.spy.calls == []


# --------------------------------------------------------------------------- #
# Unavailable
# --------------------------------------------------------------------------- #


def _unavailable(reason="embed_unreachable"):
    return GuardOutcome(action="unavailable", reason=reason,
                        refusal="I can't process that right now.")


def test_an_unavailable_guard_refuses_with_503_and_retry_after(env) -> None:
    env.guard.outcome = _unavailable()
    response = _chat(env)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert response.json()["error"]["type"] == "gateway_error"
    assert env.spy.calls == []


def test_a_client_can_opt_into_the_200_refusal_shape(env) -> None:
    env.guard.resolved = FakeResolved(unavailable_response="refusal")
    env.guard.outcome = _unavailable()
    response = _chat(env)
    assert response.status_code == 200
    assert response.json()["guardrail"]["action"] == "guard_unavailable"
    assert env.spy.calls == []


def test_degrade_to_unguarded_serves_but_never_caches(env) -> None:
    env.guard.resolved = FakeResolved(degrade_to_unguarded=True)
    env.guard.outcome = _unavailable()
    response = _chat(env)

    assert response.status_code == 200
    assert len(env.spy.calls) == 1                 # it DID reach upstream
    cache = env.pool.caches[APP_CONFIG["embed_model"]]
    # …but the unchecked answer must not outlive the outage.
    assert cache.data == {}


def test_a_degraded_decision_is_still_recorded(env) -> None:
    env.guard.resolved = FakeResolved(degrade_to_unguarded=True)
    env.guard.outcome = _unavailable()
    _chat(env)
    assert env.store.decisions[0]["action"] == "unavailable"


# --------------------------------------------------------------------------- #
# Cache scoping
# --------------------------------------------------------------------------- #


def test_an_unguarded_sibling_key_cannot_poison_a_guarded_cache(env) -> None:
    # Both keys share project_id "proj-A"; only K1 is guarded.
    env.app_config.by_key["k-unguarded"] = {**APP_CONFIG}
    env.app_config.by_key["k-guarded"] = {**APP_CONFIG, "guard": GUARD_BLOCK}

    _chat(env, "same question", key="k-unguarded")
    assert len(env.spy.calls) == 1

    _chat(env, "same question", key="k-guarded")
    # A shared namespace would have served the unguarded answer from cache.
    assert len(env.spy.calls) == 2


def test_two_keys_with_the_same_policy_share_one_cache(env) -> None:
    env.app_config.by_key["k1"] = {**APP_CONFIG, "guard": GUARD_BLOCK}
    env.app_config.by_key["k2"] = {**APP_CONFIG, "guard": GUARD_BLOCK}
    _chat(env, "same question", key="k1")
    _chat(env, "same question", key="k2")
    assert len(env.spy.calls) == 1


def test_editing_the_policy_cold_starts_that_clients_cache(env) -> None:
    _chat(env, "same question")
    assert len(env.spy.calls) == 1

    env.guard.resolved = FakeResolved(policy_hash="b" * 64)
    _chat(env, "same question")
    assert len(env.spy.calls) == 2      # a tightened policy invalidates answers


def test_the_message_log_scope_is_unchanged_by_the_guard(env) -> None:
    _chat(env)
    # gw.messages and both read routes keep continuous history across edits.
    assert env.store.messages[0]["scope"] == env.scope


# --------------------------------------------------------------------------- #
# Allow / flag
# --------------------------------------------------------------------------- #


def test_an_allowed_request_is_served_normally_and_cached(env) -> None:
    response = _chat(env)
    assert response.status_code == 200
    assert "guardrail" not in response.json()
    assert len(env.spy.calls) == 1
    _chat(env)
    assert len(env.spy.calls) == 1      # second one came from cache


def test_a_flag_serves_and_annotates(env) -> None:
    env.guard.outcome = GuardOutcome(
        action="flag", matched_category="competitors", embedding_score=0.6
    )
    response = _chat(env)
    assert response.status_code == 200
    assert response.json()["guardrail"]["action"] == "flag"
    assert response.headers["x-guardrail"] == "flag"
    assert len(env.spy.calls) == 1      # flag SERVES


# --------------------------------------------------------------------------- #
# Config errors
# --------------------------------------------------------------------------- #


def test_an_invalid_guard_config_is_a_502_in_the_openai_envelope(env) -> None:
    from semantic_cache.gateway.guard_config import GuardConfigError

    env.guard.config_error = GuardConfigError("guard.policy has no categories")
    response = _chat(env)
    assert response.status_code == 502
    assert response.headers["Retry-After"] == "30"
    assert "error" in response.json()
    assert env.spy.calls == []


def test_a_guard_block_that_is_not_an_object_is_502_not_500(env) -> None:
    # The passthrough is unvalidated, so this raises pydantic's error, not
    # ours — and a bare 500 would escape the OpenAI envelope.
    env.guard.config_error = TypeError("guard must be a JSON object, got str")
    response = _chat(env)
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "gateway_error"


# --------------------------------------------------------------------------- #
# Guard absent / switched off
# --------------------------------------------------------------------------- #


def test_a_client_with_no_guard_block_is_never_checked(env) -> None:
    env.app_config.default = dict(APP_CONFIG)      # no "guard" key
    response = _chat(env)
    assert response.status_code == 200
    assert env.guard.calls == []
    assert env.store.decisions == []


def test_the_operator_switch_disables_the_guard_for_everyone(env) -> None:
    env.guard.outcome = _block()
    env.switch.enabled = False
    response = _chat(env, "bad")
    assert response.status_code == 200
    assert "guardrail" not in response.json()
    assert env.guard.calls == []


# --------------------------------------------------------------------------- #
# GET /v1/guard/decisions
# --------------------------------------------------------------------------- #


def test_a_caller_can_read_their_own_guard_decisions(env) -> None:
    env.guard.outcome = _block()
    _chat(env, "bad")
    env.guard.outcome = GuardOutcome(action="flag", matched_category="competitors")
    _chat(env, "borderline")

    response = env.client.get(
        "/v1/guard/decisions", headers={"Authorization": f"Bearer {env.key}"}
    )
    assert response.status_code == 200
    actions = [d["action"] for d in response.json()]
    assert actions == ["flag", "block"]            # newest first


def test_guard_decisions_are_scoped_to_the_caller(env) -> None:
    env.guard.outcome = _block()
    _chat(env, "bad")
    env.app_config.by_key["other"] = {**APP_CONFIG, "project_id": "proj-B"}
    response = env.client.get(
        "/v1/guard/decisions", headers={"Authorization": "Bearer other"}
    )
    assert response.json() == []


def test_guard_decisions_requires_a_bearer(env) -> None:
    response = env.client.get("/v1/guard/decisions")
    assert response.status_code == 401
    assert "error" in response.json()


@pytest.mark.parametrize("limit", [0, 501])
def test_guard_decisions_limit_is_bounded(env, limit) -> None:
    response = env.client.get(
        f"/v1/guard/decisions?limit={limit}",
        headers={"Authorization": f"Bearer {env.key}"},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# POST /admin/guard
# --------------------------------------------------------------------------- #


def test_the_admin_switch_needs_the_admin_key(env) -> None:
    response = env.client.post(
        "/admin/guard", json={"enabled": False},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 403
    assert env.switch.enabled is True


def test_the_admin_switch_takes_effect_without_a_restart(env) -> None:
    env.guard.outcome = _block()
    assert _chat(env, "bad").json()["guardrail"]["action"] == "block"

    response = env.client.post(
        "/admin/guard", json={"enabled": False},
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
    )
    assert response.status_code == 200
    assert response.json() == {"enabled": False}
    assert "guardrail" not in _chat(env, "bad").json()

    env.client.post("/admin/guard", json={"enabled": True},
                    headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    assert _chat(env, "bad").json()["guardrail"]["action"] == "block"

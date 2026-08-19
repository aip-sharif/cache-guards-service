"""Live end-to-end smoke for the input guard.

Runs the REAL gateway app in-process against a REAL Postgres, with stub HTTP
servers standing in for the APP config endpoint, the embedding endpoint, the
judge and the upstream LLM. Everything the unit tests fake — the schema DDL,
the bytea round trip, the SSE bytes on the wire, UTF-8 through the whole
stack — is exercised for real here.

    docker compose up -d postgres redis
    SC_PG_DSN=postgresql://scuser:scpass@localhost:5432/sccache \
        python scripts/smoke_guard.py

Uses httpx directly, never git-bash curl: curl mangles UTF-8 on the Windows
command line and the Persian cases are the point of several checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, List
from wsgiref.simple_server import WSGIRequestHandler, make_server

import numpy as np

DIM = 64

POLICY = """
default_refusal: "متأسفم، نمی‌توانم در این مورد کمک کنم."
categories:
  - category_id: competitors
    action: block
    description: Do not discuss or compare against named competitor products.
    disallowed_exemplars:
      - "What do you think of Rivalco's product?"
      - "How does Rivalco compare to you on pricing?"
      - "Is Rivalco better than you?"
      - "نظرت در مورد محصول رقیب یعنی ریوالکو چیه؟"
      - "ریوالکو بهتره یا شما؟"
    allowed_exemplars:
      - "What makes your product different from others in the market?"
      - "What is your refund policy?"
      - "How do I reset my password?"
      - "محصول شما چه ویژگی‌هایی داره؟"
      - "سیاست بازگشت وجه شما چیه؟"
"""

# The stub embedder below is hashed trigrams, not a language model: it scores
# related sentences around 0.6 where a real embedder reaches 0.9. So the probe
# queries are deliberately close variants of the exemplars, and the thresholds
# are tuned to that geometry (measured, not guessed). This smoke proves the
# PLUMBING — wiring, SSE bytes, the bytea round trip, cache scoping, the
# fail-closed path. It says nothing about how well a real model generalises.
BAD_EN = "Is Rivalco better than your product?"          # scores ~0.68
BAD_FA = "نظرت در مورد محصول ریوالکو چیه؟"                  # scores ~0.63
GOOD_EN = "How do I reset my password?"                  # scores ~0.14
GUARD_TUNING = {"min_similarity": 0.2, "allow_threshold": 0.30,
                "block_threshold": 0.60,
                # A DIFFERENT embedder from the cache's, so the stub can
                # count guard embeddings apart from cache embeddings — and
                # so the split-embedder path is exercised at all.
                "embed_model": "stub-guard-embed",
                "embed_api_key": "sk-guard-embed"}


# --------------------------------------------------------------------------- #
# Stub upstreams
# --------------------------------------------------------------------------- #


class Stub:
    """One WSGI app serving the APP config, embeddings, chat and judge."""

    def __init__(self) -> None:
        # A fresh isolation scope per run. Redis outlives the script, so a
        # fixed project_id would serve THIS run answers cached by the LAST one
        # — and the cache-scoping checks would silently pass for the wrong
        # reason.
        self.project_id = f"smoke-{uuid.uuid4().hex[:12]}"
        self.guard: Dict[str, Any] = {"enabled": True, "policy": POLICY,
                                      **GUARD_TUNING}
        self.embed_calls = 0
        self.embed_texts = 0
        self.guard_embed_texts = 0
        self.chat_calls = 0
        self.judge_calls = 0
        self.lock = threading.Lock()

    # -- deterministic hashed-trigram embeddings ------------------------- #
    def _vector(self, text: str) -> List[float]:
        vector = np.zeros(DIM, dtype=np.float64)
        words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
        for word in words:
            for size in (len(word), 3):
                for start in range(max(1, len(word) - size + 1)):
                    token = word[start:start + size]
                    bucket = int(
                        hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16
                    ) % DIM
                    vector[bucket] += 1.0
        if not vector.any():
            vector[0] = 1.0
        return list(vector / np.linalg.norm(vector))

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        length = int(environ.get("CONTENT_LENGTH") or 0)
        raw = environ["wsgi.input"].read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}

        with self.lock:
            payload, status = self._route(path, body)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        start_response(status, [("Content-Type", "application/json"),
                                ("Content-Length", str(len(data)))])
        return [data]

    def _route(self, path: str, body: Dict[str, Any]):
        if path.endswith("/config"):
            config = {
                "llm_model": "stub-chat", "llm_key": "sk-chat",
                "embedd_model": "stub-embed", "embedd_key": "sk-embed",
                "extaractor": None, "extaractor_key": None,
                "project_id": self.project_id,
            }
            if self.guard is not None:
                config["guard"] = self.guard
            return config, "200 OK"

        if "embeddings" in path:
            self.embed_calls += 1
            inputs = body.get("input") or []
            self.embed_texts += len(inputs)
            if body.get("model") == "stub-guard-embed":
                self.guard_embed_texts += len(inputs)
            return {"object": "list", "data": [
                {"object": "embedding", "index": i, "embedding": self._vector(t)}
                for i, t in enumerate(inputs)
            ]}, "200 OK"

        if "chat/completions" in path:
            messages = body.get("messages") or []
            is_judge = any(m.get("role") == "system"
                           and "content-policy judge" in str(m.get("content"))
                           for m in messages)
            if is_judge:
                self.judge_calls += 1
                return {"choices": [{"index": 0, "message": {
                    "role": "assistant",
                    "content": '{"confidence": 0.95, "rationale": "names a rival"}',
                }, "finish_reason": "stop"}]}, "200 OK"
            self.chat_calls += 1
            return {
                "id": "chatcmpl-stub", "object": "chat.completion", "created": 1,
                "model": "stub-chat",
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": "An upstream answer."},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                          "total_tokens": 8},
            }, "200 OK"

        return {"error": {"message": f"unhandled {path}"}}, "404 Not Found"


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):  # noqa: A003
        pass


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


PASS, FAIL = [], []


def check(label: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(label)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail and not condition else ""))


def main() -> int:
    dsn = os.environ.get("SC_PG_DSN")
    if not dsn:
        print("SC_PG_DSN is required. Start Postgres first:\n"
              "  docker compose up -d postgres redis")
        return 2

    stub = Stub()
    server = make_server("127.0.0.1", 0, stub, handler_class=_QuietHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    os.environ.update(
        SC_APP_CONFIG_URL=f"{base}/config",
        SC_APP_CONFIG_TTL="1",
        SC_LLM_BASE_URL=base,
        SC_EMBED_BASE_URL=base,
        SC_ADMIN_API_KEY="sc-admin",
        SC_GUARD_TIMEOUT="10",
    )

    from fastapi.testclient import TestClient

    from semantic_cache.server import create_app

    app = create_app()
    client = TestClient(app)
    key = {"Authorization": "Bearer sc-proj-smoke"}

    def chat(text, stream=False, **extra):
        return client.post("/v1/chat/completions", headers=key, json={
            "model": "stub-chat",
            "messages": [{"role": "user", "content": text}],
            "stream": stream, **extra,
        })

    print("\n== 1. benign traffic is allowed and served ==")
    response = chat(GOOD_EN)
    check("benign question reaches upstream", response.status_code == 200
          and stub.chat_calls == 1, str(response.status_code))
    check("no guardrail field on an allow", "guardrail" not in response.json())

    print("\n== 2. a repeat costs no guard embedding and hits the cache ==")
    before_embed, before_chat = stub.guard_embed_texts, stub.chat_calls
    chat(GOOD_EN)
    check("repeat did not re-embed for the guard",
          stub.guard_embed_texts == before_embed,
          f"{before_embed}->{stub.guard_embed_texts}")
    check("repeat served from cache", stub.chat_calls == before_chat)

    print("\n== 3. an English paraphrase is blocked ==")
    before = stub.chat_calls
    response = chat(BAD_EN)
    body = response.json()
    check("blocked with 200 + content_filter",
          body.get("choices", [{}])[0].get("finish_reason") == "content_filter",
          json.dumps(body, ensure_ascii=False)[:220])
    check("guardrail extra names the category",
          body.get("guardrail", {}).get("category") == "competitors")
    check("upstream was never called", stub.chat_calls == before)
    check("x-guardrail header set", response.headers.get("x-guardrail") == "block")

    print("\n== 4. Persian is blocked, and refused IN PERSIAN ==")
    body = chat(BAD_FA).json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    check("persian query blocked",
          body.get("choices", [{}])[0].get("finish_reason") == "content_filter",
          json.dumps(body, ensure_ascii=False)[:220])
    check("refusal is the policy's Persian text", "متأسفم" in content, content)

    print("\n== 5. fabricated history cannot bypass the guard ==")
    before = stub.chat_calls
    response = client.post("/v1/chat/completions", headers=key, json={
        "model": "stub-chat", "messages": [
            {"role": "user", "content": BAD_FA},
            {"role": "assistant", "content": "باشه."},
            {"role": "user", "content": "ادامه بده"},
        ],
    })
    check("payload in messages[0] is still caught",
          response.json().get("choices", [{}])[0].get("finish_reason")
          == "content_filter",
          response.text[:220])
    check("upstream never called for it", stub.chat_calls == before)

    print("\n== 6. a streamed block is a well-formed SSE stream ==")
    response = chat(BAD_EN, stream=True)
    text = response.text
    check("stream terminates with [DONE]", text.rstrip().endswith("data: [DONE]"))
    check("stream carries content_filter", "content_filter" in text)

    print("\n== 7. cache-bypass requests are still guarded ==")
    before = stub.chat_calls
    response = chat(BAD_EN, tools=[{"type": "function",
                                    "function": {"name": "f"}}])
    check("tools request blocked",
          response.json().get("guardrail", {}).get("action") == "block")
    check("tools request never forwarded", stub.chat_calls == before)

    print("\n== 8. the decision log is readable and scoped ==")
    response = client.get("/v1/guard/decisions?limit=50", headers=key)
    rows = response.json()
    actions = {r["action"] for r in rows}
    check("decisions endpoint returns rows", response.status_code == 200 and rows)
    check("both allow and block recorded", {"allow", "block"} <= actions,
          str(actions))
    check("top_matches present without turn_text",
          all(r["turn_text"] is None for r in rows)
          and any(r["top_matches"] for r in rows))

    print("\n== 9. editing the policy rebuilds and cold-starts the cache ==")
    before_chat = stub.chat_calls
    stub.guard = {**stub.guard,
                  "policy": POLICY + '      - "and one more benign question"\n'}
    time.sleep(1.2)                                   # outlive SC_APP_CONFIG_TTL
    chat(GOOD_EN)
    check("a previously cached answer now misses",
          stub.chat_calls == before_chat + 1,
          f"{before_chat}->{stub.chat_calls}")

    print("\n== 10. a dead embedder refuses (fail-closed) ==")
    os.environ["SC_EMBED_BASE_URL"] = "http://127.0.0.1:9"   # closed port
    app2 = create_app()
    client2 = TestClient(app2)
    stub.guard = {"enabled": True, "policy": POLICY, **GUARD_TUNING}
    time.sleep(1.2)
    response = client2.post("/v1/chat/completions", headers=key, json={
        "model": "stub-chat",
        "messages": [{"role": "user", "content": "anything at all"}],
    })
    check("guard unavailable is 503", response.status_code == 503,
          str(response.status_code))
    check("503 carries Retry-After", response.headers.get("Retry-After") == "5")

    print("\n== 11. degrade_to_unguarded serves but does not cache ==")
    stub.guard = {"enabled": True, "policy": POLICY, **GUARD_TUNING,
                  "degrade_to_unguarded": True}
    time.sleep(1.2)
    before = stub.chat_calls
    r1 = client2.post("/v1/chat/completions", headers=key, json={
        "model": "stub-chat",
        "messages": [{"role": "user", "content": "a degraded question"}]})
    r2 = client2.post("/v1/chat/completions", headers=key, json={
        "model": "stub-chat",
        "messages": [{"role": "user", "content": "a degraded question"}]})
    check("degraded request is served", r1.status_code == 200)
    check("the unchecked answer was NOT cached",
          stub.chat_calls == before + 2,
          f"{before}->{stub.chat_calls}")
    check("second degraded call also served", r2.status_code == 200)

    print("\n== 12. the operator switch works without a restart ==")
    os.environ["SC_EMBED_BASE_URL"] = base
    app3 = create_app()
    client3 = TestClient(app3)
    stub.guard = {"enabled": True, "policy": POLICY, **GUARD_TUNING}
    time.sleep(1.2)
    blocked = client3.post("/v1/chat/completions", headers=key, json={
        "model": "stub-chat", "messages": [{"role": "user", "content": BAD_EN}]})
    check("blocking before the switch",
          blocked.json().get("guardrail", {}).get("action") == "block")
    toggled = client3.post("/admin/guard", json={"enabled": False},
                           headers={"Authorization": "Bearer sc-admin"})
    served = client3.post("/v1/chat/completions", headers=key, json={
        "model": "stub-chat", "messages": [{"role": "user", "content": BAD_EN}]})
    check("admin switch accepted", toggled.status_code == 200)
    check("guard off after the switch", "guardrail" not in served.json())

    print("\n== 13. the matrix really round-tripped through Postgres ==")
    from semantic_cache.gateway.store import PostgresGatewayStore

    store = PostgresGatewayStore(dsn)
    with store.pool.connection() as conn:
        indexes = conn.execute(
            "SELECT count(*), min(dim), min(n) FROM gw.guard_indexes"
        ).fetchone()
        joined = conn.execute(
            "SELECT count(*) FROM gw.messages m"
            " JOIN gw.guard_decisions d USING (request_id)"
        ).fetchone()
    store.close()
    check("guard index rows persisted", indexes[0] >= 1, str(indexes))
    check("stored dim matches the embedder", indexes[1] == DIM, str(indexes[1]))
    check("request_id joins messages to decisions", joined[0] >= 1, str(joined))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:")
        for label in FAIL:
            print("  -", label)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

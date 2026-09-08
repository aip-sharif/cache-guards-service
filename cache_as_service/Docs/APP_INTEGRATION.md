# APP ↔ LLM Service Integration

This document is the contract between **the APP** (owns clients, mints their
keys, and owns their model configs) and **the LLM service** (this project:
OpenAI-compatible endpoint + semantic cache + message log).

```
┌────────┐                                  ┌─────────────┐
│  APP   │  1. mint key (APP's own concern) │ LLM service │
│        │     hand sc-proj-… to the client │  (gateway)  │
│        │                                  │             │
│        │  3. GET /cache  (Bearer sc-proj) │             │
│        │ ◀─────────────────────────────── │             │
│        │ ───────────────────────────────▶ │             │
│        │   model/embed/extractor + keys   │             │
│        │   (+ optional project_id)        │             │
└────────┘                                  └──────┬──────┘
     │ 2. hand key to client                       │ 4. cache-first,
     ▼                                             ▼    miss → mlops
┌────────┐  OpenAI SDK, api_key=sc-proj-…   ┌─────────────┐
│ Client │ ───────────────────────────────▶ │ mlops LLMs  │
└────────┘                                  └─────────────┘
```

Only ONE integration direction: **we call the APP** to resolve a client's
config, presenting the client's own key (§2). The APP mints and validates keys
itself — this service has no key-admin API. Clients only ever talk to us, with
a standard OpenAI SDK (§3).

---

## 0. Credentials

| Credential | Who holds it | Where it comes from |
|---|---|---|
| **Client key** (`sc-proj-…`) | the APP + the client | **Minted by the APP** (any opaque string; `sc-proj-…` is just a convention). The APP hands it to the client and recognises it at the config endpoint. This service never mints or stores it. |
| **Service key** (`SC_APP_SERVICE_KEY`) | the APP + this service, **nobody else** | A long-lived shared secret agreed once between the two of you. Sent on every config call in its own header. It never reaches a client. |
| Model / embed / extractor API keys | the APP | The APP owns them and returns them in the config response (§2). The gateway never stores them (short in-memory TTL cache only). |

### Two credentials, two different questions

Every call we make to the APP's config endpoint carries both, and they are not
interchangeable:

| | Header | Answers | Who has seen it |
|---|---|---|---|
| **Client key** | `Authorization: Bearer …` | *Which client is this config for?* | the end user, their code, their logs |
| **Service key** | `X-Service-Key: …` | *Is this the cache service asking?* | only the APP and this service |

**Why the second one exists.** With the client key alone, your config endpoint
is guarded by a credential you have handed to end users. Anyone who obtains a
client key — from a browser, a CI log, a support ticket — can call your config
endpoint directly and receive **that client's model-provider API keys**. The
service key means a leaked client key is no longer sufficient on its own: the
caller must also be us.

It is deliberately **not** in `Authorization`; that header is already carrying
the client's key, and collapsing two different claims into one header leaves
the APP unable to tell them apart.

**It is optional, so you can roll it out in either order.** Unset, nothing is
sent and the contract is exactly as it was. Deploy this service with the key
first, confirm your APP sees the header, then make the APP require it. The
service logs a warning at boot whenever it is running without one.

---

## 1. Key minting — the APP's own concern

The APP creates each client's key and hands it to the client. Nothing on this
service is involved. The only requirement: the key the client presents to us
is the same key the APP will recognise at the config endpoint (§2).

---

## 2. Config endpoint — the APP implements, we call

Configured on the gateway as `SC_APP_CONFIG_URL` (a single fixed URL). We call
it whenever we need a client's models — responses are cached in memory for
`SC_APP_CONFIG_TTL` seconds (default 60), keyed by the client key.

### Request (from the gateway)

```
GET http://app-host:8000/cache
Authorization: Bearer sc-proj-3f8a1c…      # the CLIENT's own key, forwarded
X-Service-Key: svc-…                       # proves the caller is US (§0)
Accept: application/json
```

The URL is used verbatim — there is no project id to substitute. The APP
identifies the **client** from the bearer, and the **caller** from the service
key.

`X-Service-Key` is present only when `SC_APP_SERVICE_KEY` is configured, and
the header name is `SC_APP_SERVICE_KEY_HEADER` (default `X-Service-Key`) — pick
whatever name suits the APP; no code change is needed on our side.

**What the APP should do with it**, once you decide to require it:

```python
# The APP's config endpoint, sketched.
SERVICE_KEY = os.environ["CACHE_SERVICE_KEY"]      # the same value we send

if not hmac.compare_digest(request.headers.get("X-Service-Key", ""), SERVICE_KEY):
    return JSONResponse({"error": "unknown caller"}, status_code=401)

client = lookup_client_by_key(bearer_token(request))   # as before
if client is None:
    return JSONResponse({"error": "unknown key"}, status_code=403)
```

Use a constant-time comparison (`hmac.compare_digest`), not `==`. Distinguish
the two failures in your own logs but not necessarily in the response: `401`
for a bad service key is a caller problem, `403` for an unknown client key is
the documented "no config for this key" case we already handle (§2, `Errors`).

### Response `200`

```json
{
  "model": "gpt-x",                 "model_api_key": "sk-llm-…",
  "embed_model": "bge-m3",          "embed_api_key": "sk-embed-…",
  "extractor_model": null,          "extractor_api_key": null,

  "extractor_domain": null,
  "project_id": "b62a5a8c9f2e4d1c",
  "cache_config": {"similarity_threshold": 0.9},

  "guard": null
}
```

| Field | Required | Meaning |
|---|---|---|
| `model` + `model_api_key` | yes | Chat model slug + key, used against the mlops chat endpoint (`SC_LLM_BASE_URL`, gateway env). |
| `embed_model` + `embed_api_key` | yes | Embedding model + key for the semantic cache (`SC_EMBED_BASE_URL`). |
| `extractor_model` + `extractor_api_key` | yes, nullable | Entity-extractor model + key (`SC_EXTRACTOR_BASE_URL`). `null` → entity-aware checking off. |
| `extractor_domain` | when extractor set | `"medical"` or `"legal"`. |
| `project_id` | optional (recommended) | Cache-isolation scope. Stable across key rotation, so a rotated key keeps the same cache. Omit it and the gateway scopes by a hash of the key instead (isolation still holds, but a new key = a fresh cache). |
| `cache_config` | optional | Per-project cache overrides: `cache_mode`, `lexical_scorer`, `lexical_min_score`, `fuzzy_distance`, `similarity_threshold`, `entity_threshold`, the per-method blocks `semantic`/`bm25`/`fuzzy`, `default_ttl`, `permanent_hit_threshold`, `exact_tier`, `normalize_aggressive`, `fail_open`. |
| `guard` | optional | Input guard for this client — the policy it is checked against and the models used to check it. Absent, `{}` or `null` → the guard does not run and the request path is exactly as it was before the guard existed. See §2b. |

### Choosing the retrieval method(s) — `cache_config.cache_mode`

By default every client is served by **semantic** (vector-similarity) caching.
The APP switches the matching algorithm per client with **one field,
`cache_mode`** — the surface is identical whether the client wants one method or
several. Pass **either a string** (a single method) **or a list** (an ordered
cascade):

```json
"cache_config": {"cache_mode": "bm25"}                        // one method
"cache_config": {"cache_mode": ["exact", "bm25", "semantic"]}  // a cascade
```

| method | How a lookup finds a hit | Embeds? | Use it when |
|---|---|---|---|
| `"semantic"` (default) | Vector KNN + cosine-similarity floor. Becomes **entity-aware** automatically when an `extractor_model` is configured (domain prefilter + entity overlap + stricter threshold). | yes | Paraphrases and reworded questions should share an answer. |
| `"bm25"` | Lexical full-text over the stored question text with a RediSearch scorer (`lexical_scorer`, default `BM25`). The query's tokens are matched (intersection); the best hit at or above `lexical_min_score` wins. | no | Keyword / repeat traffic, no embedding cost. |
| `"fuzzy"` | Like `bm25`, but each token is matched within a Levenshtein distance (`fuzzy_distance`, 1–3), so typos still hit. Language-agnostic (works for Persian). | no | Traffic with typos / inconsistent spelling. |
| `"exact"` | L0 exact-match **only** — one normalized-text (+scope) lookup. | no | Deterministic caching, zero false-hit risk, cheapest tier. |
| `"off"` | Caching disabled — pure passthrough (still logged). Scalar only. | no | A client that must never be served a cached answer. |

**Single method (string).** The exact pre-check (`exact_tier`) is applied in
front of it when enabled — so `"semantic"` with `exact_tier` on means "exact,
then vector."

**Cascade (list).** Each method is tried in order and the **first hit wins**;
later tiers run only if the earlier ones miss. Put them cheapest-first —
`["exact","bm25","semantic"]` returns instantly on an exact repeat, tries
lexical next, and **only embeds if both miss**. A list is taken verbatim (it
supersedes `exact_tier`), must be non-empty, and **cannot contain `"off"`**
(use the scalar `"off"` to disable caching).

**One write serves every tier.** If the selection contains `"semantic"` the
entry is stored with its vector — and is still found by exact (by key) and bm25
(by the text field). A selection without `"semantic"` stores text-only and never
embeds. So `bm25`, `fuzzy`, `exact` and `off` compute no embeddings at all; the
APP may still return the embedding/extractor models harmlessly (ignored).

An unknown method, an empty/`"off"`-containing list, an unknown `lexical_scorer`,
or a `fuzzy_distance` outside 1–3 is rejected as an invalid config (`502`).

> Phonetic ("sounds-like") matching is possible in RediSearch but requires a
> phonetic matcher declared at index-build time and only supports
> English/French/Portuguese/Spanish — not Persian — so it is not offered here.
> Ask the operator if you need it enabled behind a dedicated field.

### Per-method hyperparameters

Each method's hyperparameters can be set two ways, in order of precedence:

1. **Per-method block** — a `cache_config` key named after the method, so each
   tier of a cascade is tuned independently. Highest precedence.
2. **Flat field** — a top-level `cache_config` field, shared by all tiers.
3. Otherwise the built-in default.

| method | per-method block | flat field | meaning |
|---|---|---|---|
| `semantic` | `semantic.similarity_threshold` | `similarity_threshold` | cosine floor for a hit (0–1) |
| `semantic` | `semantic.entity_threshold` | `entity_threshold` | stricter floor when entity-aware (0–1) |
| `bm25` | `bm25.scorer` | `lexical_scorer` | RediSearch scorer (`BM25`, `BM25STD`, `TFIDF`, `TFIDF.DOCNORM`, `DISMAX`, `DOCSCORE`) |
| `bm25` | `bm25.min_score` | `lexical_min_score` | minimum full-text score |
| `fuzzy` | `fuzzy.scorer` / `fuzzy.min_score` | `lexical_scorer` / `lexical_min_score` | as bm25 |
| `fuzzy` | `fuzzy.distance` | `fuzzy_distance` | max Levenshtein distance (1–3) |

```json
{ "model": "gpt-x", "model_api_key": "sk-…",
  "embed_model": "bge-m3", "embed_api_key": "sk-…",
  "extractor_model": null, "extractor_api_key": null,
  "cache_config": {
    "cache_mode": ["exact", "bm25", "fuzzy", "semantic"],
    "semantic": {"similarity_threshold": 0.92},
    "bm25":     {"scorer": "BM25", "min_score": 1.0},
    "fuzzy":    {"distance": 2, "min_score": 0.5}
  } }
```

Per-method blocks reject unknown fields and out-of-range values (`502`). `exact`
has no hyperparameters. For a single method you can skip the blocks and just set
the flat field — e.g. `{"cache_mode": "semantic", "similarity_threshold": 0.92}`.

The APP's own field spellings are accepted as aliases: `llm_model`/`llm_key`,
`embedd_model`/`embedd_key`, `extaractor`/`extaractor_key`.

---

## 2b. Guarding a client's input — `guard`

> Implementing or maintaining the guard itself rather than integrating against
> it? See `Docs/GUARD_HANDOFF.md`.

The guard is a **second, independent module** beside the cache. It checks what
the client sends against a policy *you* supply, and can refuse the request
before it reaches the cache or the model.

Two rules to hold onto, because they are the opposite of the cache's:

* **The cache fails open, the guard fails closed.** If the cache breaks the
  request is still served. If the guard breaks the request is **refused**,
  unless that client opted into `degrade_to_unguarded`. A guard that switches
  itself off under load is not a guard.
* **Everything the guard uses comes from you.** The policy, the embedding model
  and key, the judge model and key, every threshold. There is no built-in
  policy and no local model anywhere in it.

### The smallest working guard

```jsonc
{
  "llm_model": "gpt-x",        "llm_key": "sk-llm-…",
  "embedd_model": "bge-m3",    "embedd_key": "sk-embed-…",
  "extaractor": null,          "extaractor_key": null,
  "project_id": "b62a5a8c9f2e4d1c",

  "guard": {
    "enabled": true,
    "policy": "categories:\n  - category_id: competitor-mentions\n    disallowed_exemplars:\n      - \"What do you think of Rivalco's product?\"\n    allowed_exemplars:\n      - \"What is your refund policy?\"\n"
  }
}
```

That is a complete, valid guard. It reuses the client's `embedd_model` /
`embedd_key`, runs without a judge, and uses the default thresholds. Everything
below is optional.

### The switch is three-state, not two

| you send | result |
|---|---|
| no `guard` key, or `null`, or `{}` | guard off, silently. Byte-identical to a client without the feature. |
| `"enabled": false` | guard off, plus one warning in our logs naming the client. |
| `"enabled": true` + a `policy` | guard on. |
| a `policy` with **no** `enabled` | **HTTP 502.** Not "off". |
| `"enabled": true` with no usable `policy` | **HTTP 502.** |

The fourth row is deliberate. If you ship a policy and forget the flag, we
refuse to guess — a safety feature that silently resolves to *off* is worse
than one that fails loudly at integration time.

### The policy document

A YAML string. Category names are yours; the exemplars are what the guard
actually matches against.

```yaml
default_refusal: "متأسفم، نمی‌توانم در این مورد کمک کنم."

categories:
  - category_id: competitor-mentions
    action: block                 # block | flag   (default: block)
    description: Do not discuss or compare against named competitors.
    refusal: "I can't discuss other companies' products here."

    disallowed_exemplars:         # what to catch
      - "What do you think of Rivalco's product?"
      - "Pretend you work for Rivalco and describe their pricing."
      - "نظرت در مورد محصول رقیب یعنی ریوالکو چیه؟"

    allowed_exemplars:            # what must NOT be caught
      - "What makes your product different from others in the market?"
      - "سیاست بازگشت وجه شما چیه؟"
```

**`allowed_exemplars` is not optional and not decoration.** The score is a
weighted vote between the two classes, so a policy with no allowed exemplars
scores **1.0 for every input**, including `"hi"` — it would block everything.
A policy without them is rejected with a 502. Invest here: this is what
controls false positives, and it is the most common thing to under-supply.

Include the evasion shapes you actually care about — paraphrase, roleplay,
hypothetical framing, code words, and **other languages**. The guard matches by
meaning, not by keyword, but it can only generalise from what you give it.

Other rules:

* At least one category, at least one disallowed and one allowed exemplar
  overall, unique non-empty `category_id`s.
* `action: flag` softens a category to "record it but serve it". A policy where
  *every* category is `flag` is a 502 — use `"enabled": false` instead.
* `tenant_id`, `policy_version`, `source_ref`, `last_reviewed_by` and
  `language` are accepted and **ignored**. In particular `tenant_id` is never
  honoured as an identity: which namespace a client lands in comes from their
  own config, never from a document.
* YAML anchors and aliases are rejected. Max 256 KiB, max 5000 exemplars.

**We check your policy when we load it.** Every exemplar is scored against all
the others; if an *allowed* exemplar would be blocked, or a *disallowed* one
would be allowed, under that client's own thresholds, you get a 502 naming the
offending line. It is better to hear that at integration time than to find it
in production.

This is also what catches a mis-set `min_similarity`: a disallowed exemplar
scoring exactly 0.000 means the floor discarded every neighbour, and the 502
says so explicitly rather than pointing you at the thresholds. In a measured
run against a real multilingual model, this check refused the policy outright
rather than letting it under-block silently — which is the behaviour you want
from it.

### Choosing the models

| field | default | notes |
|---|---|---|
| `embed_model` | the client's `embedd_model` | The guard's own embedder. Must understand the languages in the policy — a model with no Persian will not catch Persian evasions. |
| `embed_api_key` | the client's `embedd_key` | |
| `embed_prefix_style` | `"none"` | `"e5"` for e5/gte-family models, which are trained with `query:` / `passage:` prefixes. One setting drives both sides. |
| `judge_model` | none | An LLM second opinion for ambiguous messages. Required if `mode` can invoke it. |
| `judge_api_key` | none | |
| `judge_task_description` | `"a customer support assistant"` | Interpolated into our prompt. Set it to what the assistant actually is — "a medical triage assistant" judges differently. |
| `judge_max_concurrency` | `4` | Per client. |

Serving endpoints are ours (`SC_GUARD_EMBED_BASE_URL` / `SC_GUARD_JUDGE_BASE_URL`,
falling back to the cache's `SC_EMBED_BASE_URL` / `SC_LLM_BASE_URL`). You supply
model names and keys only, exactly as for the cache.

**If you send no judge, the mode defaults to `embedding-only`** and no LLM is
ever called. If you send a judge, it defaults to `cascade`. A mode you state
explicitly is honoured or rejected — never quietly downgraded.

### Thresholds

All optional; every one is per client.

| field | default | meaning |
|---|---|---|
| `mode` | see above | `cascade` \| `embedding-only` \| `judge-only` \| `max` |
| `top_k` | `8` | Exemplars retrieved per check. |
| `min_similarity` | `0.60` | Below this an exemplar casts no vote. **Calibrate this first — it is model-dependent, not a portable constant.** Different embedding models live on different cosine scales: measured against `paraphrase-multilingual-MiniLM-L12-v2`, genuine violations peaked around 0.50–0.73, so the 0.60 default discarded most of the evidence and catch rate fell from 86% to 57%. Setting it too HIGH is the dangerous direction: everything is discarded, the score becomes 0.0, and the message is ALLOWED. |
| `block_threshold` | `0.85` | Score ≥ this → block. |
| `allow_threshold` | `0.40` | Score ≤ this → allow. The gap between the two is the "ask the judge" band. |
| `judge_block_threshold` | `0.60` | Judge confidence ≥ this → block. |
| `judge_allow_threshold` | `0.40` | Judge confidence ≤ this → allow; between the two → flag. |

Changing a threshold is **free** — nothing is re-embedded and it takes effect
within one config TTL. Changing the *policy text*, the embedding model, its key
or the prefix style re-embeds every exemplar, billed to that client's embedding
key.

### What gets checked — `check_roles`

Default `["user", "system", "tool"]`. Whichever you choose, **every** message
with a listed role is checked, not just the last one.

| value | catches | still open |
|---|---|---|
| `["user"]` | prohibited text typed by the end user, in any turn | the same payload moved into a `system` or `tool` message |
| `["user","system","tool"]` *(default)* | the above, plus payloads in the system prompt or tool results | assistant-prefill |
| `["user","system","tool","assistant"]` | the above, plus fabricated assistant turns used to steer the model | — |

Whoever holds the key writes the whole request body, so `system` and `tool` are
not trusted channels. `assistant` is off by default because on a long
conversation it re-checks every past model turn — more tokens, and a benign
past mention can start blocking new messages. Turn it on deliberately.

Related: `max_segments` (default 32) and `max_input_chars` (default 16000,
total). Input beyond that is refused, never truncated — truncation would just
move the bypass to the far end of the message.

### Failure behaviour

| field | default | meaning |
|---|---|---|
| `degrade_to_unguarded` | `false` | `true` → when the guard **cannot run**, serve the request unguarded (logged loudly, and the answer is not cached). Applies to runtime failures only; it never bypasses a policy block, and it never applies to a bad config. |
| `unavailable_response` | `"error"` | `"error"` → HTTP 503 + `Retry-After`. `"refusal"` → a 200 carrying a refusal, for frontends that cannot handle a 503. |
| `unavailable_refusal` | built-in text | Shown when `unavailable_response` is `"refusal"`. |
| `default_refusal` | policy's, then built-in | Fallback refusal text. Precedence: category `refusal:` → policy `default_refusal:` → this → our English constant. |
| `log_turn_text` | `false` | `true` stores the user's text in the decision log. Off by default; the matched categories and scores are logged either way. |

> **`guard.fail_open` does not exist.** `cache_config.fail_open` is a different
> setting in the same response body and it defaults the *other* way. Sending
> `guard.fail_open` is a 502 telling you to use `degrade_to_unguarded`.

Unknown fields inside `guard` are a 502 listing what is allowed — a misspelled
safety setting must never look like it was applied. Keys beginning `x_` are the
one exception: they are accepted and ignored, so you can add `x_notes` without
breaking older gateways.

### What the client sees

**Blocked** — HTTP 200, header `x-guardrail: block`:

```jsonc
{
  "id": "chatcmpl-guard-…", "object": "chat.completion", "model": "gpt-x",
  "choices": [{ "index": 0,
                "message": {"role": "assistant",
                            "content": "I can't discuss other companies' products here."},
                "finish_reason": "content_filter" }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "guardrail": {"action": "block", "category": "competitor-mentions",
                "score": 0.91, "judge_invoked": false,
                "policy": "a3f9c1e28b40", "request_id": "6f1e…"}
}
```

Any OpenAI SDK handles this unchanged. `stream: true` gets the same thing as a
normal SSE stream, with `guardrail` on the first chunk. `finish_reason` is
`content_filter`, so a client can detect it without reading the extra field.

**Flagged** — served normally, plus a `guardrail` field with
`"action": "flag"` and an `x-guardrail: flag` header. Flag does **not** block.

**Guard unavailable** — HTTP 503 + `Retry-After: 5` in the standard error
envelope (or the 200 refusal shape, if that client chose it).

**Guard config invalid** — HTTP 502 + `Retry-After: 30`, with a message naming
what is wrong. This is the one you will see during integration.

### Reading back what the guard did

```
GET /v1/guard/decisions?limit=50
Authorization: Bearer <the client's key>
```

Newest first, scoped to that client. Each row carries the action, the matched
category, both scores, `top_matches` (the three closest exemplars and their
similarities — the actual explanation of a decision, and free of user text),
and a `request_id` that joins to `GET /v1/messages`.

Worth checking regularly: `flag` is the one action that still reaches the
model, so a rising flag rate means your ambiguous band is swallowing things
that should be decided.

### Two consequences worth knowing before you go live

1. **A guarded key and an unguarded key never share a cache**, even under the
   same `project_id`. Otherwise an unguarded key's answers would be reachable
   from a guarded one.
2. **Editing a policy cold-starts that client's cache.** Answers produced under
   the old policy are no longer served. That is the point — but expect a burst
   of upstream traffic right after a policy change, so avoid editing a busy
   client's policy at peak.

### Error responses

| APP answers | Gateway tells the client |
|---|---|
| `401` / `403` / `404` (key unknown / no config) | `403` "The APP has no model config for this key." |
| `5xx`, invalid JSON, missing required fields, unreachable | `502` |
| a `guard` block we cannot use (unknown field, bad policy, missing switch) | `502` + `Retry-After: 30`, message naming the problem |
| a usable guard that cannot RUN (embedder or judge down) | `503` + `Retry-After: 5` — or served unguarded if that client set `degrade_to_unguarded` |

Notes for the APP implementer:

* The endpoint must be **fast and idempotent** — it is on the request path
  (though TTL-cached).
* To rotate a client's key, mint a new one on the APP side and have the config
  endpoint recognise it; if you also return a stable `project_id`, the cache
  carries over to the new key.
* Changing a client's config takes effect within `SC_APP_CONFIG_TTL` seconds;
  changing the **embedding model** also moves the client to a different cache
  index (old entries stop matching, by design).

---

## 3. Client surface (for reference)

Standard OpenAI SDK against the gateway, using the APP-minted key:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<gateway>:8080/v1", api_key="sc-proj-…")
client.chat.completions.create(model="gpt-x", messages=[...])          # JSON
client.chat.completions.create(model="gpt-x", messages=[...],
                               stream=True,
                               stream_options={"include_usage": True})  # SSE
client.models.list()    # the client's model, as told by the APP
```

* Cache hits return `usage` all-zeros plus
  `"semantic_cache": {"hit": true, "similarity": …}`.
* Requests with `tools` / `tool_choice` / `response_format` / `n > 1`
  bypass the cache and go straight upstream.
* `GET /v1/messages` (same bearer) → the caller's own served-message log.
* `GET /v1/guard/decisions` (same bearer) → the caller's own guard decisions,
  when a guard is configured for them (§2b).
* A guard block returns a normal 200 completion whose `finish_reason` is
  `content_filter` and which carries a `guardrail` field — no SDK changes
  needed. Only a guard that could not RUN is an error status.
* Errors on `/v1/*` use the OpenAI envelope `{"error": {"message": …}}`.

---

## 4. Operator setup — from zero to serving

Everything above is the APP's side of the contract. This section is the
service's side: what an operator must set, in the order they must set it.

[RUNNING_LOCALLY.md](RUNNING_LOCALLY.md) is the same path with a stub APP and
`curl` output at each step. Use that if you want to see it work before you wire
the real APP.

### 4.1 Three secrets, first, or nothing starts

```bash
python -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('POSTGRES_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('SC_API_KEY_PEPPER=' + secrets.token_urlsafe(48))"
```

Put them in `.env`. `docker compose` **refuses to start** without them, on
purpose: the previous defaults were `scuser`/`scpass`, a real credential
printed in this repository, on a published port.

`SC_API_KEY_PEPPER` is the one to be careful with. Every stored API-key
fingerprint is computed under it, so **changing it later logs every tenant
out**. Treat it like a database encryption key: back it up, never rotate it
casually. The other two rotate normally.

### 4.2 The four variables that decide whether `/v1` serves at all

The gateway has ONE surface — `/v1/*` serving — and it needs all four. Until
they are set, `/v1/*` answers `503` **naming the exact variable that is
missing**, never a bare 404.

```bash
SC_PG_DSN=postgresql://…      # existing Postgres; only the `gw` schema is created
SC_APP_CONFIG_URL=http://app-host:8000/cache   # §2, a single fixed URL
SC_LLM_BASE_URL=https://…     # mlops chat endpoint
SC_EMBED_BASE_URL=https://…   # mlops embeddings endpoint

SC_APP_CONFIG_TTL=60          # optional: seconds a key's config is cached
SC_EXTRACTOR_BASE_URL=https://…   # optional: only if the APP returns an extractor
```

**And the service credential** — how the APP knows the caller is us, not
someone replaying a client key (§0):

```bash
# Generate once, share with the APP, put it in BOTH environments:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
SC_APP_SERVICE_KEY=svc-…
SC_APP_SERVICE_KEY_HEADER=X-Service-Key   # optional; whatever name the APP wants
```

Optional in the sense that the service boots and serves without it — **not** in
the sense that you should skip it. Without it, a leaked client key is enough to
pull that client's model-provider credentials straight out of the APP. The
service logs a warning at boot when it is unset.

There is **no key-admin surface**. The APP mints keys; this service never does.

### 4.3 Authentication for the SaaS API (`/v1/caches/*`)

Key management lives at `GET/DELETE /v1/tenant/keys[/{key_id}]` and
`GET/POST/DELETE /v1/caches/{id}/keys[/{key_id}]`. Keys are listed by `key_id`
and display prefix — never key material and never a fingerprint, since a
fingerprint is a verifier for the key. `POST .../keys?expires_in=<seconds>`
issues one that expires; `POST .../rotate?grace=<seconds>` keeps the old keys
working while callers redeploy.


Separate from the gateway, and easy to get wrong because the failure is silent
in one direction and loud in the other.

```bash
# Gates the SaaS admin routes (tenant creation) AND GET /metrics.
# Unset → both answer 503.
SC_ADMIN_API_KEY=sc-…

# SSO. A verified JWT AUTO-PROVISIONS a tenant, so this is a tenant-creation
# control, not just a login one. Unset → JWT bearers are REJECTED and only
# minted sc-... keys authenticate. The server logs a warning at boot.
SC_SSO_JWKS_URL=https://sso.example.com/.well-known/jwks.json   # RS/ES/PS/Ed
SC_SSO_ALGORITHMS=["RS256"]
# ...or, for a shared secret (verified in-process, no extra dependency):
# SC_SSO_SHARED_SECRET=<long random string>
# SC_SSO_ALGORITHMS=["HS256"]

SC_SSO_ISSUER=https://sso.example.com
SC_SSO_AUDIENCE=semantic-cache   # SET THIS whenever your SSO serves more than
                                 # this service — otherwise a token minted for
                                 # another app is accepted here
SC_SSO_LEEWAY=60
SC_SSO_JWKS_TTL=300
```

Three behaviours worth knowing before you debug them at 3am:

* Naming an algorithm you have no key for — `HS256` with no shared secret,
  `RS256` with no JWKS URL — is a **fatal startup error**. A service that boots
  green while rejecting every login is worse than one that refuses to boot.
* The algorithm allowlist is enforced **before** a key is chosen, so
  alg-confusion and `alg: none` are unreachable.
* Tokens with no `exp` are refused. A bearer credential that never expires is
  not one this service will hold.

### 4.4 Limits and probes

```bash
SC_MAX_REQUEST_BYTES=1048576   # request-body ceiling, enforced on the STREAM
                               # (a chunked body carries no Content-Length).
                               # 1 MiB ≈ a 250k-token chat body. 0 disables.
SC_APP_CONFIG_MAX_ENTRIES=10000  # ceiling on cached per-key APP configs. The
                               # cache is keyed by the CALLER's key, so without
                               # a bound its size is chosen by your callers.
SC_READINESS_TIMEOUT=2.0       # per-dependency deadline for /ready. Keep it
                               # well under your probe interval.
```

Wire the probes the right way round — this is the single most common
misconfiguration here:

| Endpoint | Meaning | Your orchestrator should | Checks |
|---|---|---|---|
| `GET /health` | **Liveness** — the process is running | restart the container on failure | nothing external, deliberately |
| `GET /ready` | **Readiness** — this replica can serve | stop sending traffic on failure | Redis, Postgres, the APP config endpoint (required); LLM + embed endpoints (reported, non-gating) |

`/health` checks nothing external *because* it drives restarts: restarting a
healthy process does not bring Redis back, it adds an outage to an outage.
`/ready` returns `503` with a per-dependency breakdown, so you can see which
dependency to chase rather than only that something is wrong.

Every readiness probe is bounded and they run concurrently — a dependency that
hangs is reported `"timeout"`, never awaited.

### 4.5 Observability

```bash
SC_LOG_LEVEL=INFO
SC_ENVIRONMENT=production
SC_SENTRY_DSN=                        # optional
SC_SENTRY_TRACES_SAMPLE_RATE=0.0      # tracing is OFF by default
```

`GET /metrics` serves Prometheus text exposition **behind
`SC_ADMIN_API_KEY`** — exposition leaks route templates, traffic shape and
error rates, which on a multi-tenant service is not public. With no admin key
it answers `503`, never an empty `200`, so "I forgot the key" cannot look like
"there are no metrics".

Point Prometheus at it with the admin key as a bearer. Series include request
count, latency and in-flight, labelled by **route template** (`/v1/caches/{cache_id}`),
never by raw URL.

Logs are one JSON object per line on stdout, and are **redacted centrally** by
the formatter — bearer values, `api_key`-style fields, JWTs and DSN passwords
are masked on the way out, so a call site cannot leak a secret by forgetting to
think about it.

### 4.6 Input guard — all optional

None of these is required and none affects whether `/v1` mounts. Whether a
client is guarded comes entirely from the APP (§2b).

```bash
SC_GUARD_ENABLED=true            # operator break-glass for every client
SC_GUARD_TIMEOUT=5.0             # whole-check deadline, seconds. RAISE THIS if
                                 # your embedding endpoint is slow on the
                                 # request path — the per-stage HTTP timeouts
                                 # are derived from it.
SC_GUARD_BUILD_TIMEOUT=60.0      # deadline for embedding a policy; NOT limited
                                 # by SC_GUARD_TIMEOUT
SC_GUARD_MAX_EXEMPLARS=5000
SC_GUARD_MAX_POLICY_BYTES=262144
SC_GUARD_CACHE_MAX_BYTES=268435456
SC_GUARD_SEGMENT_MEMO_SIZE=50000
SC_GUARD_EMBED_BASE_URL=         # defaults to SC_EMBED_BASE_URL
SC_GUARD_JUDGE_BASE_URL=         # defaults to SC_LLM_BASE_URL
```

`POST /admin/guard {"enabled": false}` (bearer `SC_ADMIN_API_KEY`) turns the
guard off for every client at runtime, with no redeploy — the break-glass for
an incident where a fail-closed guard is refusing traffic.

### 4.7 Deploy

```bash
# Production — the BASE FILE ALONE. Data stores stay on the compose network.
docker compose -f docker-compose.yml up -d

# Local development — base + docker-compose.override.yml, which Compose loads
# automatically. The override publishes Redis and Postgres to your host.
docker compose up -d --build
```

That split is the whole point: publishing your data stores is right on a laptop
and wrong on a server, and a single file could not tell the two apart — so the
laptop settings shipped. **Do not deploy with the override file present.**

The image runs as a non-root user with a `HEALTHCHECK`, one uvicorn worker, and
a bounded graceful shutdown. One worker is deliberate: the guard's exemplar
matrices, its segment memo and the APP-config cache are all per-process, so N
workers means N copies of every matrix and N cold builds. **Scale with
replicas, not workers**, and size the container for one process.

### 4.8 Verify the install

```bash
curl -s localhost:8080/health                       # {"status":"ok"}
curl -s localhost:8080/ready                        # per-dependency breakdown
curl -s localhost:8080/metrics -H "Authorization: Bearer $SC_ADMIN_API_KEY" | head

# Gateway, as a client would call it, with an APP-minted key:
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sc-proj-…" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-x","messages":[{"role":"user","content":"hello"}]}'
```

A `503` from `/v1/*` names the missing variable. A `403` means the APP does not
recognise that key. Both are in the troubleshooting table in
[RUNNING_LOCALLY.md](RUNNING_LOCALLY.md).

---

## 5. Complete environment reference

Required-ness is about **this service booting and serving**, not about whether
a feature is good to have.

| Variable | Required | Default | What it does |
|---|---|---|---|
| `REDIS_PASSWORD` | **yes** (compose) | — | Redis auth. Compose refuses to start without it. |
| `POSTGRES_PASSWORD` | **yes** (compose) | — | Bundled-Postgres password. Compose refuses to start without it. |
| `SC_API_KEY_PEPPER` | **yes** (compose) | — | Pepper for API-key fingerprints. **Never change it once you have live keys.** |
| `SC_PG_DSN` | for `/v1` | — | Existing Postgres. Only the `gw` schema is created. |
| `SC_APP_CONFIG_URL` | for `/v1` | — | The APP's config endpoint (§2). |
| `SC_APP_SERVICE_KEY` | strongly recommended | — | Shared secret proving the caller is this service (§0). Unset → a leaked client key alone can pull provider credentials from the APP. |
| `SC_APP_SERVICE_KEY_HEADER` | no | `X-Service-Key` | Header the service key travels in. Never `Authorization`. |
| `SC_LLM_BASE_URL` | for `/v1` | — | mlops chat endpoint. |
| `SC_EMBED_BASE_URL` | for `/v1` | — | mlops embeddings endpoint. |
| `SC_EXTRACTOR_BASE_URL` | no | — | Only used when the APP returns an extractor model. |
| `SC_APP_CONFIG_TTL` | no | `60` | Seconds a key's config is cached in memory. |
| `SC_APP_CONFIG_MAX_ENTRIES` | no | `10000` | Ceiling on that cache (it is keyed by caller key). |
| `SC_ADMIN_API_KEY` | no | — | Gates SaaS admin routes **and** `/metrics`. Unset → both 503. |
| `SC_SSO_JWKS_URL` | for RS/ES/PS/Ed JWTs | — | The SSO's JWKS endpoint. |
| `SC_SSO_SHARED_SECRET` | for HS* JWTs | — | Shared secret, verified with stdlib `hmac`. |
| `SC_SSO_ALGORITHMS` | no | `["RS256"]` | Allowlist. Naming one you have no key for is a fatal startup error. |
| `SC_SSO_ISSUER` | no | — | Expected `iss`. Unset → unchecked. |
| `SC_SSO_AUDIENCE` | no | — | Expected `aud`. **Set it if your SSO serves more than us.** |
| `SC_SSO_LEEWAY` | no | `60` | Clock-skew tolerance for `exp`/`nbf`, seconds. |
| `SC_SSO_JWKS_TTL` | no | `300` | Seconds to cache the JWKS. |
| `SC_MAX_REQUEST_BYTES` | no | `1048576` | Request-body ceiling, enforced on the stream. 0 disables. |
| `SC_READINESS_TIMEOUT` | no | `2.0` | Per-dependency deadline for `/ready`. |
| `SC_GATEWAY_UPSTREAM_TIMEOUT` | no | `120` | Timeout for upstream LLM calls. |
| `SC_LOG_LEVEL` | no | `INFO` | Root log level. |
| `SC_ENVIRONMENT` | no | `production` | Tag surfaced to logs and Sentry. |
| `SC_SENTRY_DSN` | no | — | Error reporting. Needs the `[sentry]` extra. |
| `SC_SENTRY_TRACES_SAMPLE_RATE` | no | `0.0` | Tracing is off by default. |
| `SC_GUARD_*` | no | see §4.6 | Operator limits and break-glass only. |

`.env.example` is the annotated version of this table and is the file to copy.

---

## 6. Known gaps — read before you go live

These are real and unfixed. They are recorded here rather than in a ticket
nobody reads.

* **No retention policy.** `gw.messages` stores every question and answer, and
  guard rows can retain flagged text. There is no pruning job, deletion
  workflow, or erasure path. If you serve regulated data, this is your problem
  before it is ours.
* **No schema migrations.** The `gw` schema is created at startup by
  `ensure_schema()`. There is no Alembic, and no migration exists for the
  historical `project_id` → `scope` rename, so deploying onto a database older
  than commit `c323a05` fails at boot.
* **No rate limits or quotas.** Body size is capped; per-tenant request rate,
  concurrency, token and storage quotas are not implemented.
* **Redis is the only control plane.** Tenants, keys and cache metadata live
  only in Redis. Only gateway cache entries are mirrored to Postgres. Run Redis
  with persistence and a restore drill you have actually performed.
* **No distributed tracing.** Sentry tracing defaults to `0.0` and `request_id`
  is not propagated across services.
* **Startup is synchronous.** Schema creation and the Redis-from-Postgres
  rebuild both run before the process serves, which limits how fast a replica
  can join.

Full status, including what was fixed and what was deliberately left, is in
[GUARD_HANDOFF.md](GUARD_HANDOFF.md) §13.

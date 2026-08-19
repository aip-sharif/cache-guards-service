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
| Model / embed / extractor API keys | the APP | The APP owns them and returns them in the config response (§2). The gateway never stores them (short in-memory TTL cache only). |

There is no admin/service token: the gateway authenticates to the APP with
**the client's own key**, the same key the client used to call us.

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
Accept: application/json
```

The URL is used verbatim — there is no project id to substitute. The APP
identifies the client from the bearer.

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

## 4. Gateway env recap (operator side)

```bash
SC_PG_DSN=postgresql://…      # existing Postgres; `gw` schema auto-created
SC_APP_CONFIG_URL=http://app-host:8000/cache   # §2, a single fixed URL
SC_APP_CONFIG_TTL=60
SC_LLM_BASE_URL=https://…     # mlops chat endpoint
SC_EMBED_BASE_URL=https://…   # mlops embeddings endpoint
SC_EXTRACTOR_BASE_URL=https://…  # optional

# Input guard — ALL OPTIONAL. None of these is required and none affects
# whether /v1 mounts: whether a client is guarded comes entirely from the APP.
SC_GUARD_ENABLED=true            # operator break-glass for every client
SC_GUARD_TIMEOUT=5.0             # whole-check deadline, seconds
SC_GUARD_BUILD_TIMEOUT=60.0      # deadline for embedding a policy
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

The gateway has one surface — `/v1/*` serving — enabled once all of
`SC_PG_DSN`, `SC_APP_CONFIG_URL`, `SC_LLM_BASE_URL` and `SC_EMBED_BASE_URL`
are set. Until then `/v1/*` answers `503` naming what is missing. There is no
key-admin surface: the APP mints keys.

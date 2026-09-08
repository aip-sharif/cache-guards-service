# Semantic Cache User Guide & Implementation

Vector-based LLM response cache with an optional entity-aware safety layer
for high-stakes domains (medical, legal). Built on Redis Stack (RediSearch +
vector similarity), with clean architecture for dependency injection and
pluggable backends.

## Configuration

Every setting can be supplied via environment variables (prefixed `SC_`) or
loaded from a `.env` file in the working directory. A fully-annotated
template ships in [`.env.example`](.env.example) — copy it and edit:

```bash
cp .env.example .env
```

The `.env` file is gitignored by default; only `.env.example` should be
committed.

## Install

```bash
# Minimal core (Redis + embeddings via API)
pip install semantic-cache

# Common combinations
pip install 'semantic-cache[embeddings]'         # adds sentence-transformers
pip install 'semantic-cache[fastapi]'            # adds FastAPI router
pip install 'semantic-cache[langchain]'          # adds LangChain BaseCache adapter
pip install 'semantic-cache[async,metrics]'      # adds httpx + prometheus_client
pip install 'semantic-cache[translate]'          # adds deep-translator for cross-lingual

# Everything
pip install 'semantic-cache[all]'
```

## Multi-tenant SaaS server

Beyond the embeddable library, the package ships a ready-to-run multi-tenant
HTTP service (`semantic_cache.server:create_app`). Each **tenant** owns
multiple named **caches**, each with its own settings, fully isolated from
other tenants over one shared Redis + one embedding model.

### Architecture (C4 container view)

```mermaid
C4Container
    title Semantic Cache — Container diagram

    Person(user, "Client app / user", "SSO JWT (SaaS API) or project key (OpenAI gateway)")
    System_Ext(sso, "Casdoor SSO", "Issues the JWT (owner = org, name = user)")
    System_Ext(theapp, "The APP", "Separate project; mints client keys and owns their model configs; we call it per request")
    System_Ext(mlops, "MLOps serving", "OpenAI-compatible endpoints for chat, embeddings, extraction (URLs from env; keys from the APP)")
    System_Ext(sentry, "Sentry", "Error reporting (optional)")

    System_Boundary(svc, "Semantic Cache service (the LLM service)") {
        Container(api, "SaaS API", "FastAPI", "JWT/key auth, tenant+cache CRUD, /search /set, /health")
        Container(gw, "OpenAI Gateway", "FastAPI", "/v1/chat/completions: cache-first, SSE streaming, mlops passthrough; keyed by the APP-minted client key")
        Container(pool, "Manager pools", "Python", "SaaS: per-cache managers. Gateway: per-embed-model managers (own index + API embedder each)")
    }

    ContainerDb(redis, "Redis Stack", "RediSearch, AOF", "THE cache: vectors + entries, scope-isolated; SaaS registry + keys")
    ContainerDb(pg, "Postgres (existing)", "gw schema", "Served-message log + durable cache backup (entries + vectors) — NO model configs, NO keys")

    Rel(user, api, "HTTPS, Bearer JWT / cache key")
    Rel(user, gw, "HTTPS, APP-minted client key (OpenAI SDK compatible)")
    Rel(api, sso, "Trusts tokens issued by")
    Rel(gw, theapp, "Resolves config: forwards the caller's key + our own service key (TTL-cached)")
    Rel(gw, pg, "Message log, cache backup")
    Rel(gw, mlops, "Cache miss → chat completion (JSON or SSE)")
    Rel(api, pool, "Resolves managers")
    Rel(gw, pool, "Resolves managers")
    Rel(pool, redis, "Embed + KNN search / store")
    Rel(api, sentry, "Reports exceptions")
```

### Auth

Two ways in, usable together:

1. **SSO JWT.** Clients send `Authorization: Bearer <jwt>`. The token's
   **signature is verified** against your SSO's key, then the tenant identity
   is its **organization + username** claims — Casdoor's `owner` + `name`
   (generic `organization`/`user_id`/`sub` aliases are also accepted) — so the
   same person always maps to the same tenant and caches.
   Tenants **auto-provision on first login**, which is exactly why the token is
   verified: an unverified payload is not a login, it is a tenant-creation
   endpoint open to anyone. Configure it with [`SC_SSO_*`](#sso-jwts-sc_sso_);
   with none of those set the JWT path is **closed**, not open.
2. **Per-cache API keys** (`sc-...`). A tenant mints a key scoped to **one**
   cache for machine-to-machine data access. Keys carry an id, a display
   prefix, an optional expiry and a last-used timestamp, so one can be listed,
   audited and revoked individually; rotation takes a grace period so callers
   can redeploy before the old key stops working. All keys for a cache are
   revoked when the cache is deleted. Only a **peppered fingerprint** is
   stored — see `SC_API_KEY_PEPPER`.

### Endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | **Liveness** — the process is up. Checks nothing external, so a Redis blip cannot restart-loop the container. |
| `GET /ready` | none | **Readiness** — every dependency the enabled serving mode needs (Redis, Postgres, the APP config endpoint), each probe bounded. 503 + a per-dependency breakdown when one is down. |
| `GET /metrics` | admin key | Prometheus exposition. 503 when `SC_ADMIN_API_KEY` is unset — never open. |
| `GET /v1/me` | JWT / tenant key | Tenant id + cache count + every cache. |
| `POST /v1/caches` | JWT / tenant key | Create a named cache (optional per-cache `config`). |
| `GET /v1/caches` · `GET/PATCH/DELETE /v1/caches/{id}` | JWT / tenant key | List / get / update settings / delete. |
| `POST /v1/caches/{id}/keys` | JWT / tenant key | Issue a per-cache key. `?expires_in=<seconds>` for one that expires. |
| `GET /v1/caches/{id}/keys` | JWT / tenant key | List that cache's live keys by `key_id` + prefix. **No key material, no fingerprints.** |
| `POST /v1/caches/{id}/keys/rotate` | JWT / tenant key | Rotate. `?grace=<seconds>` keeps the old keys alive while callers redeploy. |
| `DELETE /v1/caches/{id}/keys/{key_id}` | JWT / tenant key | Revoke **one** key, leaving the others working. |
| `GET /v1/tenant/keys` · `DELETE /v1/tenant/keys/{key_id}` | tenant key | List / selectively revoke account keys. |
| `POST /v1/tenant/rotate-key` | tenant key | Rotate the account key. `?grace=<seconds>` as above. |
| `POST /v1/caches/{id}/search` · `.../set` · `DELETE .../entries` | JWT / tenant / cache key | Cache data plane. |
| `POST /v1/tenants` | admin key | Legacy: create tenant (only if `SC_ADMIN_API_KEY` set). |

Per-cache overrides accepted in `config`: `similarity_threshold`,
`entity_threshold`, `default_ttl`, `permanent_hit_threshold`, `exact_tier`,
`normalize_aggressive`, `fail_open`.

#### SSO JWTs (`SC_SSO_*`)

A verified JWT **auto-provisions a tenant** — the identity is `owner|name`
(Casdoor's fields; `organization`/`sub` and friends are accepted as aliases).
That makes these settings a tenant-creation control, not just a login one, so
the JWT path **fails closed**: with none of them set, JWT bearers are rejected
outright and only minted `sc-...` keys authenticate. The server logs a warning
at boot when it comes up in that state.

| Variable | Required | Meaning |
|---|---|---|
| `SC_SSO_JWKS_URL` | for RS/ES/PS/Ed | The SSO's JWKS endpoint. Needs `pip install '.[sso]'` (PyJWT). |
| `SC_SSO_SHARED_SECRET` | for HS256/384/512 | Shared secret. Verified in-process with stdlib `hmac` — no extra needed. |
| `SC_SSO_ALGORITHMS` | no (`["RS256"]`) | Allowlist of accepted `alg` values. Enforced **before** a key is chosen, so alg-confusion and `alg: none` are unreachable. Naming an algorithm you have no key for is a **fatal** startup error, not a silent reject. |
| `SC_SSO_ISSUER` | no | Expected `iss`. Unset → unchecked, safe only if the signing key is single-purpose. |
| `SC_SSO_AUDIENCE` | no | Expected `aud`. **Set this whenever your SSO serves more than this service** — otherwise a token minted for another app is accepted here. |
| `SC_SSO_LEEWAY` | no (`60`) | Clock-skew tolerance in seconds for `exp`/`nbf`. |
| `SC_SSO_JWKS_TTL` | no (`300`) | Seconds to cache the JWKS before refetching. |

Tokens with no `exp` are refused: a bearer credential that never expires is not
one this service will hold.

### Deploy

```bash
# Local dev — base + docker-compose.override.yml, which publishes Redis and
# Postgres to your host so you can point redis-cli and psql at them:
docker compose up -d --build
curl localhost:8080/health            # {"status":"ok"}          liveness
curl localhost:8080/ready             # per-dependency breakdown  readiness

# Production — the BASE FILE ALONE. The data stores stay on the compose
# network, unpublished:
docker compose -f docker-compose.yml up -d

# Or run the service against an existing Redis:
docker build -t semantic-cache-saas .
docker run -p 8080:8080 -e SC_REDIS_HOST=my-redis semantic-cache-saas
```

Copy `.env.example` to `.env` first. **Three variables have no default and
compose will refuse to start without them** — `REDIS_PASSWORD`,
`POSTGRES_PASSWORD` and `SC_API_KEY_PEPPER`. That is deliberate: the previous
`scuser`/`scpass` was a real credential printed in this repository, on a
published port. `.env.example` has the one-liners to generate them.

`SC_API_KEY_PEPPER` must not change once you have live keys — every stored key
fingerprint is computed under it, so rotating it logs every tenant out. Back it
up with your other secrets.

See [the gateway's environment table](#environment) for what the gateway needs.
Behind a proxy, pass it into the build:
`docker compose build --build-arg HTTPS_PROXY=http://host.docker.internal:PORT`.

Example session (SSO JWT):

```bash
JWT=<token with organization + user_id claims>
AUTH="Authorization: Bearer $JWT"

curl -s -X POST localhost:8080/v1/caches -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"name":"prod","config":{"similarity_threshold":0.85}}'
# -> {"cache_id":"<cid>", ...}

curl -s -X POST localhost:8080/v1/caches/<cid>/set  -H "$AUTH" \
  -d '{"query":"capital of France","response":"Paris"}'
curl -s -X POST localhost:8080/v1/caches/<cid>/search -H "$AUTH" \
  -d '{"query":"what is the French capital"}'
# -> {"hit":true,"response":"Paris","similarity":0.94}

curl -s localhost:8080/v1/me -H "$AUTH"   # overview: all caches
```

## OpenAI-compatible gateway

This service is the **LLM service**: it owns the OpenAI-compatible endpoint,
the semantic cache, and the message log. Client keys and model configs both
live in **the APP** (a separate project) — this service never mints or
validates keys. The gateway mounts when `SC_PG_DSN`, `SC_APP_CONFIG_URL`,
`SC_LLM_BASE_URL`, and `SC_EMBED_BASE_URL` are all set.

The request loop — the key is the whole identity, there is no project id in:

1. The **APP mints** the client's key and hands it to the client (the APP's
   own concern; nothing on this service is involved).
2. The client calls `POST /v1/chat/completions` with that key + a model (any
   OpenAI SDK works unchanged).
3. We `GET {SC_APP_CONFIG_URL}` (a single fixed URL, e.g.
   `http://app-host:8000/cache`) **forwarding the caller's own key** as the
   bearer. The APP identifies the client and returns six things: **model,
   embedding model, entity extractor model (null → unused), and their three
   API keys** — plus an optional `project_id` (TTL-cached, `SC_APP_CONFIG_TTL`).
4. Cache first (Redis, isolated per scope — the APP's `project_id`, or a hash
   of the key if it returns none); a miss goes to the mlops chat endpoint
   (`SC_LLM_BASE_URL`) with the model+key from the APP, and the answer is
   cached on the way back.

The APP contract we call — `GET /cache`, `Authorization: Bearer <the client's
own key>`:

```json
{
  "model": "gpt-x",          "model_api_key": "sk-...",
  "embed_model": "bge-m3",   "embed_api_key": "sk-...",
  "extractor_model": null,   "extractor_api_key": null,
  "extractor_domain": null,  "project_id": "optional-scope",
  "cache_config": {"similarity_threshold": 0.9},
  "guard": null
}
```

(`extractor_domain` is `"medical"`/`"legal"` when an extractor is set;
`project_id` is the optional cache scope; `cache_config` is optional per-client
cache overrides. 401/403/404 from the APP → we answer the client 403.)

**The optional `guard` block** turns on a second module beside the cache: an
input guard that checks the client's messages against a policy the APP supplies
and can refuse the request before it reaches the cache or the model. Absent it,
nothing changes. Unlike the cache it **fails closed** — if it cannot run, the
request is refused rather than served unchecked, unless that client opted into
`degrade_to_unguarded`. Full contract for the APP: `Docs/APP_INTEGRATION.md`
§2b. **Picking this work up? Start at `Docs/GUARD_HANDOFF.md`** — state, how to
run the smoke and the detection eval, the invariants that must not be undone,
and the open items.

**Postgres stores**: a log row per served message, and a **durable backup of
every cache entry** (including its embedding vector) — no keys, no configs.
Redis stays the serving cache; every write is mirrored to Postgres
(write-through, fail-open), and on boot any entry Redis lost is restored from
Postgres with its remaining TTL — no re-embedding, no upstream calls. The
Postgres instance must already exist — only a `gw` schema is created inside it.

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | client key | Cache-first chat; full SSE when `stream: true` (incl. `stream_options.include_usage`). |
| `GET /v1/models` | client key | The caller's model as configured in the APP (OpenAI list shape). |
| `GET /v1/messages` | client key | The caller's own served-message log. |
| `GET /v1/guard/decisions` | client key | The caller's own guard decisions (when a guard is configured). |
| `POST /admin/guard` | admin key | Operator break-glass: turn the input guard off/on for every client, no redeploy. |

There is **no key-admin surface** — the APP mints and validates keys.

Behavior notes:

* The cache key is the **last user message**; each scope is isolated, and each
  **embedding model** has its own index + embedding backend (different
  embedding models must never share a vector index).
* Only complete answers are cached (`finish_reason == "stop"`, non-empty).
* Requests with `tools` / `tool_choice` / `response_format` / `n > 1` bypass
  the cache entirely — a cached plain answer can't honor those contracts.
* No extractor in the APP config → entity-aware checking is off.
* Errors on `/v1/*` use the OpenAI error envelope (`{"error": {...}}`).
* Works with the OpenAI SDK: `OpenAI(base_url="http://host:8080/v1",
  api_key="sc-proj-...")`.

### Environment

The gateway has one surface, `/v1/*` serving, mounted when all four of
`SC_PG_DSN` + `SC_APP_CONFIG_URL` + `SC_LLM_BASE_URL` + `SC_EMBED_BASE_URL`
are set. Until then `/v1/*` answers **503** naming the missing variables (never
a bare 404).

| Variable | Required | What it is |
|---|---|---|
| `SC_PG_DSN` | yes | Ordinary Postgres connection string, `postgresql://user:pass@host:port/db`. Stores the message log and the cache backup. The DB must exist; only the `gw` schema is created. |
| `SC_APP_CONFIG_URL` | yes | The APP's config endpoint — a single fixed URL, e.g. `http://app-host:8000/cache`. We GET it forwarding the caller's own key as the bearer. |
| `SC_APP_CONFIG_TTL` | no (60) | Seconds a key's config is cached in memory. |
| `SC_LLM_BASE_URL` | yes | MLOps chat endpoint; we POST `{url}/v1/chat/completions`. |
| `SC_EMBED_BASE_URL` | yes | MLOps embeddings endpoint for the cache. |
| `SC_EXTRACTOR_BASE_URL` | no | Entity-extractor endpoint; only used when the APP returns an extractor model. |
| `SC_GUARD_ENABLED` | no | Operator break-glass for the input guard (default `true`). Also flippable at runtime via `POST /admin/guard`. |
| `SC_GUARD_TIMEOUT` | no | Whole-check deadline for one guard decision, seconds (default `5.0`). Deliberately not the 120s upstream timeout. **Raise this if your embedding endpoint is slow on the request path** — the per-stage HTTP timeouts are derived from it (an `embedding-only` client's embed call gets ~90% of it, since no judge has to run after it). |
| `SC_GUARD_BUILD_TIMEOUT` | no | Deadline for embedding a policy's exemplars (default `60.0`). Governs the build's own embedding calls, which are **not** limited by `SC_GUARD_TIMEOUT`: the build is server-owned and shielded, so a slow first build costs one refused request, not a permanent failure. |
| `SC_GUARD_MAX_EXEMPLARS` | no | Cap on exemplars in one policy (default `5000`). |
| `SC_GUARD_MAX_POLICY_BYTES` | no | Cap on the inline policy YAML (default `262144`). |
| `SC_GUARD_CACHE_MAX_BYTES` | no | Byte budget for in-process exemplar matrices (default 256 MiB). |
| `SC_GUARD_SEGMENT_MEMO_SIZE` | no | Entries in the per-segment decision memo (default `50000`). |
| `SC_GUARD_EMBED_BASE_URL` | no | Guard embedding endpoint; defaults to `SC_EMBED_BASE_URL`. |
| `SC_GUARD_JUDGE_BASE_URL` | no | Guard judge endpoint; defaults to `SC_LLM_BASE_URL`. |
| `APP_PORT` / `REDIS_PORT` / `PG_PORT` | no | Host ports for compose (8080 / 6379 / 5432). |

Embeddings are **API-based** — no model is baked into the image. The gateway
takes its embed model per client from the APP; the shared SaaS-tenant embedder
is configured with `SC_EMBEDDING_PROVIDER` / `SC_EMBEDDING_BASE_URL` /
`SC_EMBEDDING_API_KEY` / `SC_EMBEDDING_MODEL` (any OpenAI-compatible endpoint:
OpenRouter, mlops, …).

**Run it locally** — compose bundles an ordinary Postgres so nothing external
is needed:

```bash
docker compose up -d      # api + Redis + Postgres
```

with a `.env` like:

```bash
APP_PORT=8080
SC_PG_DSN=postgresql://scuser:${POSTGRES_PASSWORD}@postgres:5432/sccache
SC_APP_CONFIG_URL=https://app.example.com/api/gateway-config
SC_LLM_BASE_URL=https://mlops.example.com
SC_EMBED_BASE_URL=https://mlops.example.com
```

The client then calls `/v1/chat/completions` with the key **the APP issued it**
(there is no minting step on this service). In **production**, point
`SC_PG_DSN` at your own Postgres and remove the bundled `postgres` service.

* Step-by-step local setup, first request, and troubleshooting:
  [Docs/RUNNING_LOCALLY.md](Docs/RUNNING_LOCALLY.md)
* Full request/response contract between the APP and this service:
  [Docs/APP_INTEGRATION.md](Docs/APP_INTEGRATION.md)

## 1. Initialization (Composition Root)

Because this package adheres to clean architecture principles, you must instantiate the configuration and services before passing them into the managers.

```python
from semantic_cache.core.config import SemanticCacheConfig
from semantic_cache.infrastructure.redis_client import RedisClientFactory
from semantic_cache.core.embedding_manager import EmbeddingManagerFactory
from semantic_cache.core.entity_extractor import EntityExtractorFactory
from semantic_cache.core.entity_extractor_async import AsyncEntityExtractorFactory
from semantic_cache.core.extraction_cache import CachedEntityExtractor
from semantic_cache.core.cache_manager import SemanticCacheManager

def init_cache_manager() -> SemanticCacheManager:
    # 1. Load configuration (Reads from environment variables prefixed with SC_)
    config = SemanticCacheConfig(similarity_threshold=0.88)

    # 2. Initialize Infrastructure (Redis)
    redis_client = RedisClientFactory.create_client(config)

    # 3. Initialize Embedding Strategy (HuggingFace, OpenAI, etc.)
    embedding_manager = EmbeddingManagerFactory.create(config)

    # 4. Build entity extractors (return None when SC_ENTITY_AWARE=false).
    sync_extractor = EntityExtractorFactory.create(config)
    async_extractor = AsyncEntityExtractorFactory.create(config)  # requires httpx

    # 5. (Recommended) Wrap the sync extractor in a Redis-backed result cache
    #    so repeat queries don't re-pay the LLM cost.
    if sync_extractor is not None:
        sync_extractor = CachedEntityExtractor(
            sync_extractor, redis_client, ttl_seconds=config.entity_extraction_cache_ttl
        )

    # 6. Inject everything into the Core Manager.
    return SemanticCacheManager(
        config=config,
        redis_client=redis_client,
        embedding_manager=embedding_manager,
        entity_extractor=sync_extractor,
        async_entity_extractor=async_extractor,
    )
```

---

## 2. Entity-Aware Caching (medical / legal domains)

### Why this exists

Similarity alone is unsafe in high-stakes domains. Consider:

| Query A | Query B | Cosine similarity | Same answer? |
|---|---|---|---|
| "What is the recommended dose of Vitamin B12?" | "What is the recommended dose of Vitamin B6?" | ~0.97 | **No — different drug** |
| "Metformin contraindications" | "Metformin side effects" | ~0.91 | **No — different question** |
| "42 U.S.C. § 1983 standing" | "42 U.S.C. § 1985 standing" | ~0.96 | **No — different statute** |

A pure-similarity cache would happily serve the B6 dosage to a B12 query.
That is a **patient-safety bug**, not a cache miss.

Entity-aware mode adds a strict pre-filter on top of the embedding match:

```
            Query
              │
              ▼
   ┌────────────────────┐
   │ 1. LLM extracts    │   Medical: DRUG, CONDITION, DOSAGE, PROCEDURE, ANATOMY
   │    canonical       │   Legal:   STATUTE, CASE, JURISDICTION, PARTY, DATE
   │    entities        │
   └────────────────────┘
              │
              ▼
   ┌────────────────────┐
   │ 2. Filter cache    │   Candidates must have the SAME entity set
   │    by entity sig   │   (sorted identifiers, sha256) AND the SAME domain
   │    + domain TAG    │
   └────────────────────┘
              │
              ▼
   ┌────────────────────┐
   │ 3. KNN with the    │   Stricter `entity_threshold` (default 0.95)
   │    stricter        │   applied only to candidates that survived step 2
   │    threshold       │
   └────────────────────┘
              │
              ▼
            HIT or MISS
```

All three layers must agree for a HIT. Extraction failure on either read or
write short-circuits to MISS / no-op — never a false hit, never a silent
store of an untagged entry.

### Configuration

All entity-aware settings use the same `SC_` pydantic-settings prefix as the
rest of the package. The user-facing names from the spec
(`SEMANTIC_CACHE_ENTITY_AWARE`, etc.) map directly to the `SC_ENTITY_*`
form below.

| Env var | Type | Default | Purpose |
|---|---|---|---|
| `SC_ENTITY_AWARE` | bool | `false` | Master switch. When false, behavior is identical to the original similarity-only cache. |
| `SC_DOMAIN` | enum | `general` | `medical` \| `legal` \| `general`. `general` is rejected when `ENTITY_AWARE=true` — extraction needs a specialized prompt. |
| `SC_ENTITY_MODEL` | str | `gpt-4o-mini` | OpenAI-compatible chat model used for extraction. |
| `SC_ENTITY_THRESHOLD` | float | `0.95` | Cosine-similarity floor applied **only** in entity-aware mode. Should be stricter than `SC_SIMILARITY_THRESHOLD`. |
| `SC_ENTITY_LLM_BASE_URL` | str | _(falls back to `SC_EMBEDDING_BASE_URL`)_ | Base URL of the chat-completions endpoint. |
| `SC_ENTITY_LLM_API_KEY` | secret | _(falls back to `SC_EMBEDDING_API_KEY`)_ | Bearer token. |
| `SC_ENTITY_LLM_TIMEOUT` | int | `10` | Per-call timeout in seconds. A timeout is treated as a MISS. |

Minimal `.env` for a medical deployment:

```bash
SC_ENTITY_AWARE=true
SC_DOMAIN=medical
SC_ENTITY_MODEL=gpt-4o-mini
SC_ENTITY_THRESHOLD=0.95
SC_ENTITY_LLM_BASE_URL=https://api.openai.com
SC_ENTITY_LLM_API_KEY=sk-...
```

### Choosing an LLM gateway

Any OpenAI-compatible chat-completions endpoint works. The extractor appends
`/v1/chat/completions` to the base URL unless it already ends in
`chat/completions`.

| Gateway | `SC_ENTITY_LLM_BASE_URL` | `SC_ENTITY_MODEL` |
|---|---|---|
| OpenAI | `https://api.openai.com` | `gpt-4o-mini` |
| OpenRouter ✅ *verified* | `https://openrouter.ai/api` | `openai/gpt-4o-mini` |
| Azure OpenAI | `https://<resource>.openai.azure.com/openai/deployments/<deployment>` | your deployment name |
| vLLM / local | `http://localhost:8000` | the served model id |

**Gotcha — model slugs are gateway-specific.** OpenRouter requires a
provider-prefixed slug (`openai/gpt-4o-mini`, not `gpt-4o-mini`); a bare slug
returns a runtime 404. When in doubt, list valid slugs first:

```bash
curl -s https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $SC_ENTITY_LLM_API_KEY" | jq '.data[].id'
```

**vLLM / non-OpenAI gateways:** set `SC_ENTITY_USE_JSON_MODE=false` if the
gateway rejects OpenAI's `response_format` field. The tolerant parser still
recovers the JSON object from a fenced/prose response.

The entity-extraction path is verified end-to-end against OpenRouter
(`openai/gpt-4o-mini`) — see `tests/test_live_llm_gateway.py` (§6).

### How matching works (worked example)

Two cached entries:

| Key | Query | Entities (identifier set) |
|---|---|---|
| A | "Vitamin B12 dosage for adults" | `{Vitamin B12}` |
| B | "Vitamin B6 dosage for adults"  | `{Vitamin B6}` |

Incoming query: **"What is the recommended dose of Vitamin B12?"**

1. Extractor returns `[{type: DRUG, identifier: "Vitamin B12"}]`.
2. Entity signature = `sha256("vitamin b12")`. Domain tag = `medical`.
3. RediSearch hybrid query:
   `(@entity_sig:{<sha-b12>} @domain:{medical})=>[KNN 1 @vector $vec]`
   → returns only candidate **A** (B is filtered out before KNN).
4. Cosine similarity to A: ~0.97 ≥ 0.95 → **HIT**.

If the query had instead been "Vitamin B6", entity signature collides with
B not A, A is filtered out, and the cache HITs on B. No cross-contamination
between drugs is possible regardless of how similar the embeddings are.

### Failure semantics

| Situation | Behavior |
|---|---|
| Extractor raises (timeout, 5xx, malformed JSON) on **search** | Cache MISS. Logged at WARNING. Never falls through to a similarity-only hit. |
| Extractor raises on **set** | Entry is **not written**. Avoids storing an entity-less record that could later match an entity-aware query incorrectly. |
| `SC_ENTITY_AWARE=true` with no extractor injected | `CacheOperationError` raised at construction time — caught at startup, not in production. |
| `SC_ENTITY_AWARE=true` with `SC_DOMAIN=general` | `EntityExtractionError` raised from the factory — there is no prompt for `general`. |
| Entity set is empty (LLM returns `{"entities": []}`) | Sentinel signature `"none"`. Cached entries with no entities can only match other entity-free queries. |

### Backward compatibility

* With `SC_ENTITY_AWARE=false` (the default), search and store paths take the
  original `*=>[KNN]` query with the original `similarity_threshold` and the
  schema's new TAG fields are simply unused. The full test suite covers
  this — see `tests/test_entity_aware_behavior.py::test_disabled_mode_behaves_identically_to_legacy`.
* The index schema is a superset of the legacy schema (adds `entity_sig`,
  `domain`, `entities` fields). If you are migrating an existing deployment,
  call `manager.purge()` once to recreate the index against the new schema.

### Audit logging

Every hit and miss under entity-aware mode is logged at INFO with the
extracted entity identifiers and active domain, e.g.:

```
Cache HIT (entity-aware) — distance=0.0124, entities=['Vitamin B12'], domain=medical
Cache MISS (entity-aware) — entities=['Vitamin B6'], domain=medical
```

This makes it possible to reconstruct, after the fact, *why* a given query
hit or missed — essential for clinical / legal review.

### Limitations: extraction stability vs hit-rate

The match requires the **exact** entity set. That is what makes it safe —
"Vitamin B12" can never collide with "Vitamin B6". But it also makes hit-rate
sensitive to how *consistently* the LLM extracts entities:

* **Safety is robust.** A different drug/condition → different signature →
  guaranteed MISS. No amount of LLM jitter produces a false HIT.
* **Hit-rate is sensitive.** If the model intermittently tags a *generic*
  word as an entity (e.g. "dosage" as a DOSAGE) for one phrasing but not its
  paraphrase, the two sets differ and you get a false MISS — a wasted cache
  slot, not a wrong answer. Live testing with `gpt-4o-mini` showed this on
  ~20% of paraphrase pairs before mitigation.

Mitigations already in place:
* The domain prompts use **few-shot examples** and explicitly forbid emitting
  bare generic words ("dose", "dosage", "treatment", "law", "case", …); a
  DOSAGE entity is only emitted with a concrete quantity+unit ("500 mg").
* `CachedEntityExtractor` makes a given query string deterministic *once
  cached* — repeat queries reuse the first extraction.

If you still observe paraphrase misses with your model, options in order of
effort: lower `SC_ENTITY_THRESHOLD` slightly is **not** the fix (the miss is
the entity set, not similarity); instead use a stronger/more deterministic
extraction model, or fall back to a deterministic NER (scispaCy/MedCAT for
medical, legal-BERT for legal) behind the same `BaseEntityExtractor`
interface. A future option is to restrict the signature to primary entity
types (DRUG/CONDITION) and ignore volatile ones.

---

## 3. Using with LangChain

The `LangChainSemanticCache` is designed to inherit from LangChain's `BaseCache`. Once initialized, you simply bind it to standard LangChain globals. 

```python
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from semantic_cache.adapters.langchain_cache import LangChainSemanticCache

# 1. Get our initialized Core Manager
cache_manager = init_cache_manager()

# 2. Wrap it with the LangChain Adapter
langchain_cache = LangChainSemanticCache(cache_manager)

# 3. Set globally across all LangChain LLMs
set_llm_cache(langchain_cache)

# 4. Usage
llm = ChatOpenAI(model="gpt-4o")

# First call: Triggers actual OpenAI request (Slower)
response1 = llm.invoke("What is the capital of Iran?")
print(response1) 

# Second call: Triggers Semantic Cache HIT via vector similarity (Instant)
response2 = llm.invoke("پایتخت ایران کجاست؟") # Assuming multilingual embeddings are enabled
print(response2) 
```

---

## 4. Using with FastAPI

To expose the Semantic Cache to external systems (like Node.js microservices or frontend UIs), you can run it as a standalone REST API using FastAPI and `dependency_overrides`.

---

## 5. Production Standards & Data Persistence

### High Availability and Graceful Degradation
This package was built to strict **Clean Architecture** and **SOLID** standards:
* **Dependency Injection:** The cache manager, embedding layer, and redis client are completely decoupled. 
* **Graceful Degradation:** If the Redis server crashes or the network goes down, the `LangChainSemanticCache` adapter intercepts the `RedisConnectionError` and simply registers a "Cache Miss". The app will seamlessly fall back to making actual LLM requests instead of throwing HTTP 500 crashes to your users.

### Redis Persistence (Surviving Server Restarts)
Because Redis is traditionally an in-memory database, data *will* disappear on reboot unless you configure **AOF (Append Only File)** or **RDB (Snapshots)**.

We have included a `docker-compose.yml` that configures a production-grade Redis stack with persistence enabled.

To start the persistent Redis server:
```bash
docker-compose up -d
```

**Why this guarantees zero data loss:**
1. Uses `redis/redis-stack-server` (includes RedisSearch).
2. Uses `--appendonly yes`: Every single write (like a new cache entry) is logged perfectly to disk. If the server power goes out, Redis replays this log on boot and brings the cache exactly back to life.
3. Uses a Docker **Volume** (`redis_data:/data`): Even if you completely delete the container with `docker rm`, the data lives safely on your hard drive. Next time you run `docker-compose up`, your permanent cache entries are instantly restored!

### `main.py`
```python
from fastapi import FastAPI
from semantic_cache.adapters.fastapi_router import router, get_cache_manager
from your_module.dependencies import init_cache_manager # Import the init func defined above

app = FastAPI(
    title="Semantic Cache Microservice",
    description="Vector-based LLM caching API"
)

# 1. Initialize the Core Manager when the app starts
active_cache_manager = init_cache_manager()

# 2. Override the dependency stub in the router
app.dependency_overrides[get_cache_manager] = lambda: active_cache_manager

# 3. Mount the Router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Hitting the API

*Search for a vector match:*
```bash
curl -X POST "http://localhost:8000/cache/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "Give me a recipe for pizza"}'
```

*Expected JSON Response (if hit):*
```json
{
  "hit": true,
  "response": "Here is a recipe for Italian Margherita pizza...",
  "similarity": 0.91,
  "metadata": {}
}
```

*Manually pinning an infinite lifetime response:*
```bash
curl -X POST "http://localhost:8000/cache/set" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What are your company core hours?",
           "response": "Our core hours are 10 AM to 3 PM EST.",
           "keep_forever": true
         }'
```

---

## 6. Testing

The committed suite is **unit-only** — no Redis, no embedding model, no
network. It runs on a bare checkout and on the full Python matrix in CI in
seconds:

```bash
pytest -v          # collects tests/unit/ (see pyproject testpaths)
```

| File | What it covers |
|---|---|
| `tests/unit/test_jwt_auth_unit.py` | SSO JWT decode + identity resolution (Casdoor `owner`/`name`, aliases, rejection) |
| `tests/unit/test_observability_unit.py` | JSON log formatter, logging wiring, Sentry no-op paths |
| `tests/unit/test_entity_extractor.py` | `LLMEntityExtractor` parsing + error handling, factory branches (`requests` mocked) |
| `tests/unit/test_tolerant_json.py` | JSON-mode toggle + fence/prose-tolerant parser |
| `tests/unit/test_translation.py` | Optional translation: missing-dep fallback, failure passthrough, LRU caching |
| `tests/unit/test_normalization.py` | `TextNormalizer` (NFKC, digit folding, stop-words) |
| `tests/unit/test_retry.py` | Dependency-free retry/backoff helpers |
| `tests/unit/test_package_exports.py` | Public export surface stays stable |

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) is a single **unit**
job — matrix across Python 3.10–3.13, no service containers, no model
download. `pytest` is green on any runner without setup.

> Redis/model-backed behavioral tests (semantic hits, scope isolation, the
> full SaaS API end-to-end) are not part of the committed suite; they were
> exercised during development against a local Redis Stack (see
> `docker-compose.yml`) and remain retrievable from git history.

---

## 7. Operational Features

These features are what move the package from "nice prototype" to
"deployable in production". Each is opt-in or transparently backwards
compatible.

### 7.1 Async API (FastAPI without blocking the event loop)

The manager exposes `await mgr.asearch(...)` and `await mgr.aset(...)`.
The bundled FastAPI router uses them by default, so the LLM round trip
(entity extraction) does not block the event loop:

```python
from semantic_cache.core.entity_extractor_async import AsyncEntityExtractorFactory

async_extractor = AsyncEntityExtractorFactory.create(config)  # needs httpx
manager = SemanticCacheManager(
    ...,
    entity_extractor=sync_extractor,           # used by .search / .set
    async_entity_extractor=async_extractor,    # used by .asearch / .aset
)
```

When only a sync extractor is wired, `asearch`/`aset` still work — they
offload the whole sync path via `asyncio.to_thread`. So you can move to
async incrementally without rewiring everything at once.

### 7.2 Extraction caching (cut LLM cost by ~70% in steady state)

Every entity-aware lookup costs one LLM round trip. For a steady-state
query mix with repeats, that's mostly waste. Wrap the sync extractor:

```python
from semantic_cache.core.extraction_cache import CachedEntityExtractor

inner = LLMEntityExtractor(...)
outer = CachedEntityExtractor(inner, redis_client, ttl_seconds=3600)
manager = SemanticCacheManager(..., entity_extractor=outer)
```

* Cache key: `scache:entcache:sha256(normalized_text)`.
* Successful extractions (including empty entity lists) are cached.
* **Failures are not cached** — next lookup retries the LLM rather than
  serving a stale failure.
* TTL is configurable via `SC_ENTITY_EXTRACTION_CACHE_TTL` (default 3600s, 0 disables).
* Composable: pass the wrapped extractor wherever the sync one is expected.

### 7.3 Schema versioning + zero-downtime migration

The active RediSearch index name is `<config.cache_index_name>:v<N>` where
`N` is the package schema version. Bumping the schema does not collide
with the old index — both can coexist.

Existing pre-v2 deployments have data at `scache:<hash>` keys without
`entity_sig` / `domain` / `entities` fields. Backfill them once after
upgrade:

```python
mgr = init_cache_manager()
n = mgr.migrate_legacy_entries()
print(f"migrated {n} legacy entries")
```

The helper is idempotent and safe to run online while traffic flows. It
skips extraction-cache keys and only touches document keys with missing
tag fields.

### 7.4 Prometheus metrics with miss-reason labels

The most useful signal for an oncall is not raw hit/miss rate but **why**
a miss happened: cold cache, extractor outage, or threshold drift.

| Metric | Type | Labels |
|---|---|---|
| `scache_lookups_total` | counter | `result` ∈ {hit, miss}, `reason` ∈ {hit, no_candidate, extractor_error, below_threshold} |
| `scache_writes_total` | counter | `outcome` ∈ {ok, skipped_extractor_error} |
| `scache_extractor_calls_total` | counter | `outcome` ∈ {ok, error} |
| `scache_extraction_cache_total` | counter | `result` ∈ {hit, miss} |
| `scache_extractor_latency_seconds` | histogram | — |
| `scache_search_latency_seconds` | histogram | — |

Scrape endpoint:

```bash
curl http://localhost:8000/cache/prometheus
```

`prometheus_client` is an **optional** dep. When missing, the call sites
become no-ops and the endpoint returns a stub document — no
`if metrics_enabled:` boilerplate, no scraper 500s.

Suggested alerts:

```yaml
# Page when extractor errors spike — likely LLM gateway degradation.
- alert: SemanticCacheExtractorDegraded
  expr: |
    rate(scache_lookups_total{reason="extractor_error"}[5m])
    / rate(scache_lookups_total[5m]) > 0.05
  for: 5m

# Page when threshold-misses spike — possible embedding drift.
- alert: SemanticCacheBelowThresholdSpike
  expr: |
    rate(scache_lookups_total{reason="below_threshold"}[10m])
    > 2 * avg_over_time(
        rate(scache_lookups_total{reason="below_threshold"}[10m])[1d:10m]
      )
  for: 15m
```

### 7.5 Cross-lingual translation (optional)

`SC_ENABLE_TRANSLATION=true` activates a translation pre-step using the
optional `deep-translator` backend (Google Translate). Translations are
LRU-cached in-process so repeat queries don't re-hit the API.

* If `deep-translator` is not installed, translation is silently a no-op
  (single WARNING log) — queries pass through unchanged, and the
  multilingual embedding model still handles many language pairs adequately.
* Translation errors return the original text — they never break a lookup.

### 7.6 LLM gateway portability (vLLM, Anthropic-compat, etc.)

OpenAI's `response_format={"type": "json_object"}` is not supported by
every "OpenAI-compatible" gateway. Turn it off:

```bash
SC_ENTITY_USE_JSON_MODE=false
```

The package ships a tolerant JSON parser that handles markdown-fenced
responses (` ```json … ``` `) and prose preambles, so most local-model
gateways work without further configuration.

### 7.7 Pipelined LFU promotion (2 round trips, pure Python)

Per-hit bookkeeping (`HINCRBY` + `TTL` + `EXPIRE` or `PERSIST`) is done
in straight Python — no Lua. `HINCRBY` and `TTL` are batched into a
single non-transactional pipeline (1 RT), and the conditional follow-up
(`EXPIRE` or `PERSIST`) is one more (1 RT). Total: 2 RTs per hit, vs 3
with naive sync calls.

The race between concurrent hits crossing the promotion threshold is
benign: `PERSIST` is idempotent, and any `EXPIRE` that races a `PERSIST`
either loses cleanly or is overridden by the very next hit. If
bookkeeping fails for any reason the hit is still served — LFU is
best-effort.

### 7.8 Configuration reference (operational additions)

| Env var | Default | Notes |
|---|---|---|
| `SC_ENTITY_USE_JSON_MODE` | `true` | Set false for vLLM / non-OpenAI gateways. |
| `SC_ENTITY_EXTRACTION_CACHE_TTL` | `3600` | Seconds. 0 disables the wrapper's caching. |
| `SC_ENABLE_TRANSLATION` | `false` | Requires `pip install 'semantic-cache[translate]'`. |
| `SC_TRANSLATION_TARGET_LANGUAGE` | `en` | ISO 639-1 code. |

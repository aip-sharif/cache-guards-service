# Cache Service (cache_api)

A FastAPI service that lets clients (via an intermediary "APP") register LLM/embedding
model credentials, cache behavior settings, and an optional guard, then exposes an
endpoint an OpenAI-compatible **gateway** calls to resolve that configuration.

---

## Architecture

```
Client → POST /cache/register (Casdoor auth)
       → mints a local id/project_id/cache_key, stores config

Gateway → GET /cache   (Authorization: Bearer <the project's own cache_key>)
        → { model, model_api_key, embed_model, embed_api_key,
            extractor_model, extractor_api_key, extractor_domain,
            project_id, cache_config, guard }
```

- **FastAPI + SQLModel** — API layer and ORM
- **PostgreSQL** — primary database (`user`, `cache`, `cacheconfig`)
- **Casdoor** — end-user authentication (JWT, via cookie or `Authorization` header)
- **Alembic** — schema management (replacing `create_all` at startup)

---

## Project layout

```
caching/                          (repo root)
  docker-compose.yml              (Postgres not published by default)
  docker-compose.override.yml     (dev-only - opens the Postgres port)
  cache_api/
    Dockerfile                    (non-root user + HEALTHCHECK)
    requirements.in               (source deps - unpinned)
    requirements.lock.txt          (pinned + hashed - install this one)
    .env
    alembic.ini
    alembic/
      env.py
      script.py.mako
      versions/
    app/
      auth.py                     (Casdoor + admin-key auth)
      config.py                   (reads .env)
      crypto_utils.py             (at-rest encryption for provider keys)
      rate_limit.py               (rate limiting + body size limit)
      models/
        database.py               (User, Cache, CacheConfig)
        schemas_cache.py          (request/response schemas + GuardConfig)
      routes/
        routes_cache.py           (all /cache/* endpoints)
      utils/
        db_utils.py               (database CRUD)
        guard_utils.py            (three-state guard logic + policy validation)
        proxy_utils.py            (direct proxy to the client's own mlops endpoint)
    tests/
      conftest.py
      test_cache.py
      test_security.py            (security regression tests)
  .github/workflows/ci.yml
```

---

## Setup

### 1. `.env` (inside `cache_api/`)

```dotenv
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mydb
DATABASE_URL=postgresql://postgres:postgres@db:1379/mydb

CLIENT_ID=your-casdoor-client-id
CLIENT_SECRET=your-casdoor-client-secret
CASSDOOR_ENDPOINT=http://your-casdoor-host:8000/
APPLICATION_NAME=app-built-in
ORGANZATION_NAME=built-in
CERT=app/cert.pem

SC_GATEWAY_ADMIN_KEY=your-admin-key
SC_DB_ENCRYPTION_KEY=your-fernet-key   # generate with the command below

SC_LLM_BASE_URL=https://your-mlops-host.com
SC_EMBED_BASE_URL=https://your-mlops-host.com

# dev only - auto-creates tables without Alembic. Remove this in production
# and run `alembic upgrade head` as a deploy step instead.
SC_ENVIRONMENT=development
```

Generate `SC_DB_ENCRYPTION_KEY`:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
⚠️ Don't lose this key — without the exact same key, previously encrypted values
(clients' LLM/embedding provider keys) can no longer be decrypted. Back it up somewhere safe.

### 2. Run with Docker Compose

```bash
docker compose up --build
```
The API comes up on `http://localhost:8000`; docs at `http://localhost:8000/docs`.

For local access to Postgres (e.g. pgAdmin), `docker-compose.override.yml`
automatically publishes the port (dev only).

### 3. Run without Docker

```bash
cd cache_api
pip install -r requirements.lock.txt
python main.py
```

---

## Authentication

| Type | Mechanism | Used for |
|---|---|---|
| End user | Casdoor JWT (`cassdoor_token` cookie or `Authorization: Bearer`) | `register`, `edit`, `/mine`, `/{project_id}` |
| Project (gateway) | `Authorization: Bearer <cache_key>` | `GET /cache` (the gateway's main endpoint), `proxy/*` |
| Admin | `Authorization: Bearer <SC_GATEWAY_ADMIN_KEY>` (constant-time compare) | `GET /cache/key/{cache_key}` |

Security notes:
- If a JWT can't be verified (invalid signature/expired), the request is **rejected** —
  there is no unsafe fallback.
- If `SC_GATEWAY_ADMIN_KEY` isn't set in `.env`, the service refuses to start (fail closed).
- Tokens/keys are never logged.

---

## Endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /cache/register` | Casdoor | Register a new project; mints a local `project_id`/`cache_key` |
| `PUT /cache/{cache_id}` | Casdoor | Edit model fields and/or `guard`/`cache_config` |
| `GET /cache/mine` | Casdoor | List all of the caller's own projects |
| `GET /cache/{project_id}` | Casdoor | Read one project (owner only) |
| `GET /cache/key/{cache_key}` | Admin key | Look up a project by `cache_key` (debug/admin) |
| `GET /cache` | Bearer `<cache_key>` | **The gateway's main endpoint** — raw `GatewayConfigResponse` shape |
| `POST /cache/proxy/chat/completions` | Bearer `<cache_key>` | Direct proxy to the project's own `llm_model` |
| `POST /cache/proxy/embeddings` | Bearer `<cache_key>` | Direct proxy to the project's own `embedd_model` |

Regular responses (everything except the flat `GET /cache`) are wrapped:
```json
{"status_code": 200, "message": "...", "data": {...}}
```

---

## Register payload

```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-…",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-…",
  "extaractor": null,
  "extaractor_key": null,
  "extractor_domain": null,
  "cache_config": {
    "cache_mode": ["exact", "bm25", "fuzzy", "semantic"],
    "semantic": {"similarity_threshold": 0.92},
    "bm25": {"scorer": "BM25", "min_score": 1.0},
    "fuzzy": {"distance": 2, "min_score": 0.5}
  },
  "guard": {
    "enabled": true,
    "policy": "categories:\n  - category_id: competitor-mentions\n    disallowed_exemplars:\n      - \"...\"\n    allowed_exemplars:\n      - \"...\"\n",
    "embed_model": "bge-m3",
    "embed_api_key": "sk-guard-embed-…"
  }
}
```
`cache_config` and `guard` are both optional. `cache_mode` can be a single string
or a list. More samples (including error cases) are in `register_test_samples.md`.

---

## Guard: a three-state switch, not two

| You send | Result |
|---|---|
| no `guard` key, `null`, or `{}` | Off, silently |
| `"enabled": false` | Off, plus a warning logged |
| `"enabled": true` + valid policy + `embed_model`/`embed_api_key` | On |
| a `policy` with no `enabled` | `502` — refuses to guess |
| `"enabled": true` with no usable policy, or missing `embed_model`/`embed_api_key` | `502` |
| a `mode` that can invoke the judge (`cascade`, `judge-only`, `max`) without `judge_model`/`judge_api_key` | `502` |

Policy validation (`guard_utils.py::validate_policy`) rejects a `policy` if it:
- isn't valid YAML or has no `categories`
- has a duplicate or missing `category_id`
- has a category with no `disallowed_exemplars`
- has **no** `allowed_exemplars` anywhere in the whole policy
- has every category set to `action: flag`
- exceeds 256 KiB or 5000 total exemplars
- contains YAML anchors/aliases (`&`/`*`)

Unknown fields inside `guard` (other than `x_*`) are rejected with `502`.

---

## Security — status summary

### ✅ Fixed
- Real auth on every management endpoint (no more hardcoded `id_user`)
- No fallback to unverified JWT decoding (fail closed)
- Admin-key check with constant-time comparison; fails closed if unconfigured
- No logging of tokens/keys
- At-rest encryption of provider credentials (`llm_key`, `embedd_key`, `extaractor_key`) via Fernet
- Route ordering fixed (static paths registered before `{project_id}`)
- Atomic registration (rolls back if either insert fails)
- `unique`/`nullable=False` constraints on `cache_key`/`project_id`
- Internal errors no longer leak to clients (`str(e)` removed from responses)
- Basic rate limiting (60 req/min) + request body size limit (1 MB)
- Postgres no longer published by default
- Dockerfile: non-root user + `HEALTHCHECK`
- Hash-locked lockfile (`requirements.lock.txt`) for reproducible builds
- Alembic scaffolding for controlled schema migrations
- Security regression tests (`tests/test_security.py`)

### ❌ Still outstanding / known limitations
- **The rate limiter is in-memory** — with multiple replicas, each keeps its own
  counter (needs Redis for real multi-replica production use)
- **S04 encryption** covers `Cache`'s direct fields; `guard_config` (JSON,
  containing judge/embed guard keys) is not yet field-level encrypted
- **The encryption key** still lives alongside `DATABASE_URL` in the same `.env`
  — an intermediate step, not a substitute for a real KMS (Vault/AWS KMS)
- Items belonging to `cache_as_service`/gateway (retention policy, Prometheus
  metrics, tracing, the Redis registry, API-key lifecycle) are out of scope for
  this service

---

## Testing

```bash
cd cache_api
pip install -r requirements.lock.txt
pytest -v
```
With coverage: `pytest --cov=app -v`

Tests run against an **in-memory SQLite** database (no real Postgres needed)
and mock `get_current_user_id` — no live Casdoor connection required either.

---

## CI

`.github/workflows/ci.yml`: lint (`ruff`), syntax compile, Docker build, `pytest`.
Runs on every push/PR to `main`/`develop`.

---

## Migrating to Alembic (before production)

```bash
docker compose exec api bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```
Then remove `SC_ENVIRONMENT=development` from `.env` and add `alembic upgrade
head` as a separate deploy step before bringing the service up in your
deployment pipeline.
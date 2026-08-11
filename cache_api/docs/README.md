# Cache Service

A FastAPI service that lets clients (via a caching "APP") register LLM/embedding
model credentials, cache-behavior settings, and an optional input guard, then
exposes a config endpoint that an OpenAI-compatible **gateway** calls to
resolve a client's setup — following the `APP_INTEGRATION.md` contract.

---

## Architecture

- **FastAPI + SQLModel** — API layer and ORM
- **PostgreSQL** — persistence (`user`, `cache`, `cacheconfig` tables)
- **Casdoor** — end-user authentication (JWT via cookie or `Authorization` header)
- **Gateway integration** — a separate OpenAI-compatible gateway calls
  `GET /cache` with the client's own key to resolve model configs, cache
  behavior, and guard settings

```
Client → POST /cache/register (Casdoor auth)
       → mints project_id + cache_key locally, stores config

Gateway → GET /cache (Authorization: Bearer <client's cache_key>)
        → returns { model, model_api_key, embed_model, embed_api_key,
                     extractor_model, extractor_api_key, extractor_domain,
                     project_id, cache_config, guard }
```

---

## Project layout

```
caching/                       (repo root)
  docker-compose.yml
  cache_api/                   (everything the api container needs)
    Dockerfile
    main.py
    requirements.txt
    .env
    app/
      auth.py                  # Casdoor auth + admin-key dependency
      config.py                # reads .env
      models/
        database.py            # SQLModel tables: User, Cache, CacheConfig
        schemas_cache.py        # Pydantic request/response schemas
      routes/
        routes_cache.py         # /cache/* endpoints
      utils/
        db_utils.py             # DB CRUD helpers
        guard_utils.py          # three-state guard resolution + policy validation
    tests/
      conftest.py
      test_cache.py
  .github/
    workflows/
      ci.yml
```

---

## Setup

### 1. Configure `cache_api/.env`

```dotenv
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mydb
POSTGRES_PORT=1379
POSTGRES_HOST=localhost
DATABASE_URL=postgresql://postgres:postgres@db:1379/mydb

CLIENT_ID=your-casdoor-client-id
CLIENT_SECRET=your-casdoor-client-secret
CASSDOOR_ENDPOINT=http://your-casdoor-host:8000/
APPLICATION_NAME=app-built-in
ORGANZATION_NAME=built-in
CERT=app/cert.pem

SC_GATEWAY_ADMIN_KEY=your-admin-key   # only needed if using verify_gateway_admin_key
```

> `POSTGRES_PORT` here is a convention only — Postgres inside the `db`
> container actually listens on the port passed via
> `command: ["postgres", "-p", "<PORT>"]` in `docker-compose.yml`. Keep both
> in sync if you change it.

### 2. Run with Docker Compose

```bash
docker compose up --build
```

The API comes up on `http://localhost:8000`. Interactive docs:
`http://localhost:8000/docs`

### 3. Run locally without Docker

```bash
cd cache_api
pip install -r requirements.txt
python main.py
```

---

## Authentication

- **Client-facing endpoints** (`/cache/register`, `/cache/{id}` edit,
  `/cache/{project_id}` read, `/cache/mine`) are protected by **Casdoor** —
  either the `cassdoor_token` cookie or an `Authorization: Bearer <token>`
  header.
- To test from Swagger: click **Authorize** and paste the JWT (no `Bearer`
  prefix needed).
- **`GET /cache`** — the endpoint the gateway calls — is authenticated
  differently: with the **client's own `cache_key`** (`sc-proj-…`), not a
  Casdoor token and not an admin key.
- `verify_gateway_admin_key` (in `auth.py`) is available for any endpoint
  that should only be callable by an operator/admin, checked against
  `SC_GATEWAY_ADMIN_KEY`.

---

## Endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `POST /cache/register` | Casdoor | Register a new cache/project. Mints `project_id` + `cache_key` locally. |
| `PUT /cache/{cache_id}` | Casdoor | Edit model fields and/or `guard` / `cache_config`. |
| `GET /cache/{project_id}` | Casdoor | Read one of the caller's own caches. |
| `GET /cache/mine` | Casdoor | List all of the caller's caches. |
| `GET /cache` | Bearer `<cache_key>` | **Gateway-facing.** Returns the config contract the gateway expects. |

All client-facing responses are wrapped:

```json
{ "status_code": 200, "message": "...", "data": { ... } }
```

Errors use the same envelope (via a global `HTTPException` handler in
`main.py`). `GET /cache` (the gateway-facing one) is the exception — it
returns the raw contract shape, unwrapped, per `APP_INTEGRATION.md`.

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
    "policy": "categories:\n  - category_id: competitor-mentions\n    disallowed_exemplars:\n      - \"...\"\n    allowed_exemplars:\n      - \"...\"\n"
  }
}
```

`cache_config` and `guard` are both optional — see `register_test_samples.md`
for a full set of copy-pasteable request bodies, including ones that are
expected to fail validation.

`cache_mode` accepts either a single string (`"bm25"`) or a list
(`["exact", "bm25", "semantic"]`), per `APP_INTEGRATION.md`.

---

## The guard: a three-state switch, not two

| You send | Result |
|---|---|
| no `guard` key, `null`, or `{}` | Off, silently. |
| `"enabled": false` | Off, plus one warning logged naming the client. |
| `"enabled": true` + a valid `policy` | On. |
| a `policy` with **no** `enabled` | `HTTP 502` — refuses to guess. |
| `"enabled": true` with no usable `policy` | `HTTP 502`. |

Implemented in `app/utils/guard_utils.py::resolve_guard`.

### Policy validation (`validate_policy`)

A guard `policy` is a YAML document. It is rejected (`502`) if it:

- is not valid YAML, or not a mapping
- has no non-empty `categories` list
- has a category with a duplicate or missing `category_id`
- has a category with no `disallowed_exemplars`
- has **no `allowed_exemplars` anywhere** (a policy with none would score
  1.0 for every input)
- has **every** category set to `action: flag` (use `"enabled": false`
  instead)
- exceeds 256 KiB or 5000 total exemplars
- contains YAML anchors/aliases (`&name` / `*name`)

Unknown fields inside `guard` (anything not in the documented set) are
rejected with `502`, except `x_*`-prefixed keys, which are accepted and
ignored. Sending `guard.fail_open` specifically returns a `502` pointing to
`degrade_to_unguarded` instead (per the spec — they are not the same
setting).

> **Note:** this service only validates policy *structure*. The semantic
> cross-check described in `APP_INTEGRATION.md` (scoring every exemplar
> against every other to catch mis-set thresholds) requires calling a real
> embedding model and is the gateway's responsibility at load time, not
> this registration service's.

---

## Data model

**`cache`** — one row per registered project: model/embedding credentials,
`project_id`, `cache_key`, extractor fields.

**`cacheconfig`** — one-to-one with `cache` (via `cache_id`): `guard_enabled`,
`guard_policy`, `guard_config` (JSON — everything in `guard` besides
`enabled`/`policy`), and the cache-mode settings (`cache_mode`, `semantic`,
`bm25`, `fuzzy`, all stored as JSON columns).

Split into two tables deliberately: the guard/cache settings are nested and
frequently edited, while the core credentials rarely change — and keeping
`guard_enabled` as its own boolean column keeps "which projects have guard
on" queryable without unpacking JSON.

---

## Testing

```bash
cd cache_api
pip install -r requirements.txt
pytest -v
```

Tests run against an **in-memory SQLite** database (no real Postgres
needed) and mock `get_current_user_id`, so no live Casdoor connection is
required either. See `tests/conftest.py` for the fixtures and
`tests/test_cache.py` for coverage of register / edit / read / the
gateway-facing config endpoint / guard three-state paths.

With coverage:

```bash
pytest --cov=app -v
```

---

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`/`develop`:

1. Lints with `ruff`
2. Compiles all Python files (catches basic syntax/import errors early)
3. Builds the Docker image
4. Runs the `pytest` suite

Check a run's status under the **Actions** tab of the repo, or on the PR
itself.

---

## Known limitations / things to revisit

- `id_user` has a foreign key on `User.id`; if the Casdoor user row doesn't
  exist yet, `create_cache` auto-creates a placeholder `User` row rather
  than failing — a stopgap until full Casdoor user provisioning is wired
  in.
- The database column names `extaractor` / `extaractor_key` intentionally
  keep the original (misspelled) naming from the initial schema; the
  API-facing field names match, so no renaming was needed at the boundary.
- `SC_GATEWAY_ADMIN_KEY`-based auth (`verify_gateway_admin_key`) exists but
  isn't wired into any endpoint yet — add `Depends(verify_gateway_admin_key)`
  to any route that should be operator-only.
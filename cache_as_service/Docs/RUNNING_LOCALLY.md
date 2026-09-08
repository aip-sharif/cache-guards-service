# Running Locally — Step by Step

Gets the LLM service running on your machine in about five minutes: an
OpenAI-compatible endpoint backed by a semantic cache, with Redis and
Postgres included.

**How the pieces fit.** A client calls this service with a single key — the
key **the APP minted** — plus a model. Per request we present that same key to
the APP's config endpoint, the APP returns which models to use (and their
keys), and we serve cache-first. This service never mints or validates keys,
and there is no project id in play: the key is the whole identity.

**Prerequisites:** Docker Desktop running, and this repository cloned.

---

## Step 1 — Create your `.env`

First generate three secrets. **Compose will refuse to start without them** —
there are deliberately no defaults, because the previous defaults were a real
password (`scuser`/`scpass`) printed in this repository on a published port.

```bash
python -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('POSTGRES_PASSWORD=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('SC_API_KEY_PEPPER=' + secrets.token_urlsafe(48))"
```

> **Keep `SC_API_KEY_PEPPER` forever.** Every stored API-key fingerprint is
> computed under it, so changing it later logs every tenant out. Back it up
> wherever you keep your other secrets. The other two can be rotated normally.

Now create a file named `.env` in the repository root, pasting those three
lines in where shown:

```bash
# ---- REQUIRED secrets (paste the generated values) --------------------------
REDIS_PASSWORD=...
POSTGRES_PASSWORD=...
SC_API_KEY_PEPPER=...

# ---- host ports ------------------------------------------------------------
APP_PORT=8080
REDIS_PORT=6379
PG_PORT=5432

# ---- Postgres --------------------------------------------------------------
# Host is the compose SERVICE name; port is the IN-CONTAINER port (always
# 5432), NOT PG_PORT. PG_PORT only controls what is published to your host.
SC_PG_DSN=postgresql://scuser:${POSTGRES_PASSWORD}@postgres:5432/sccache

# ---- the APP's config endpoint ---------------------------------------------
# A single fixed URL. We GET it presenting the CALLER'S OWN key as the bearer;
# the APP identifies the client from that key and returns its models + keys.
SC_APP_CONFIG_URL=http://host.docker.internal:8000/cache
SC_APP_CONFIG_TTL=60

# The SERVICE credential: proves to the APP that the caller is THIS SERVICE,
# not someone replaying a client key. Different from a client key, and no
# client ever sees it. Generate one and give the same value to the APP:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
# Unset -> no header is sent (fine for the local stub below).
#SC_APP_SERVICE_KEY=svc-...
#SC_APP_SERVICE_KEY_HEADER=X-Service-Key

# ---- mlops serving endpoints -----------------------------------------------
# Base URLs only — the models and their API keys come from the APP.
SC_LLM_BASE_URL=http://host.docker.internal:9100
SC_EMBED_BASE_URL=http://host.docker.internal:9100
#SC_EXTRACTOR_BASE_URL=http://host.docker.internal:9100

# ---- observability ---------------------------------------------------------
SC_LOG_LEVEL=INFO
SC_ENVIRONMENT=development
#SC_SENTRY_DSN=

# ---- optional, but you will want these eventually ---------------------------
# Gates the SaaS admin routes AND GET /metrics. Unset → both answer 503.
#SC_ADMIN_API_KEY=sc-change-me
# SSO. Unset → JWT bearers are REJECTED and only minted sc-... keys work.
# See "Step 6" below.
#SC_SSO_JWKS_URL=
#SC_SSO_SHARED_SECRET=
```

### How these variables decide what exists

The gateway has ONE surface: `/v1/*` serving. It needs all four of Postgres,
the APP config endpoint, and the mlops chat + embed endpoints.

| You set | `/v1/*` answers |
|---|---|
| none of them | **503**, naming `SC_PG_DSN` |
| some but not all | **503**, naming the exact missing variable |
| `SC_PG_DSN` + `SC_APP_CONFIG_URL` + `SC_LLM_BASE_URL` + `SC_EMBED_BASE_URL` | serves |

There is no key-admin surface — the APP mints keys, not us.

---

## Step 2 — Start the stack

```bash
docker compose up -d --build
```

This starts all three: the API, Redis, and a bundled Postgres — nothing
external needed.

**What that command actually loads.** Compose reads `docker-compose.yml` *and*
`docker-compose.override.yml` automatically. The base file is production-shaped
— the data stores are not published to your host. The override adds exactly the
things that are right on a laptop and wrong on a server: Redis and Postgres
published so you can point `redis-cli` and `psql` at them, and `DEBUG` logging.

For a production-shaped run, name the base file alone and the override is
skipped:

```bash
docker compose -f docker-compose.yml up -d
```

In production you also point `SC_PG_DSN` at your own database and remove the
bundled `postgres` service.

Behind a corporate proxy, pass it into the build:

```bash
docker compose build \
  --build-arg HTTP_PROXY=http://host.docker.internal:PORT \
  --build-arg HTTPS_PROXY=http://host.docker.internal:PORT
```

Check all three came up:

```bash
docker compose ps
```

```
NAME                       STATUS
semantic-cache-api         Up (healthy)
semantic-cache-postgres    Up (healthy)
semantic-cache-redis       Up (healthy)
```

Then:

```bash
curl http://localhost:8080/health
# {"status":"ok"}                      liveness: the process is up

curl http://localhost:8080/ready
# {"status":"ok","checks":{"redis":"ok","postgres":"ok"}}
# readiness: every dependency the enabled serving mode needs. 503 with a
# per-dependency breakdown when one is down.
```

---

## Step 3 — Point at the APP (models source)

`/v1/*` needs an APP that answers `SC_APP_CONFIG_URL`. The APP identifies the
client from the bearer we forward and returns its model config. Until the real
APP exists, run this stub — it accepts any key and returns one config.

<details>
<summary>Minimal stub APP (click to expand)</summary>

Save as `stub_app.py`, then `pip install fastapi uvicorn` and
`python stub_app.py`:

```python
import hmac
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# The SERVICE credential, shared with the cache service. Leave it unset here to
# skip the check while you are getting the stub running; set it to the same
# value as SC_APP_SERVICE_KEY to see the real two-credential flow.
SERVICE_KEY = os.environ.get("CACHE_SERVICE_KEY")

@app.get("/cache")
async def cache_config(request: Request):
    # 1. WHO IS CALLING? The service key. Constant-time compare, not ==.
    if SERVICE_KEY:
        presented = request.headers.get("x-service-key", "")
        if not hmac.compare_digest(presented, SERVICE_KEY):
            return JSONResponse({"error": "unknown caller"}, status_code=401)

    # 2. WHICH CLIENT IS IT FOR? The forwarded bearer.
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return JSONResponse({"error": "no key"}, status_code=401)
    client_key = auth[7:]
    # Look the CLIENT up by its key; swap in a real provider's slugs+keys.
    # project_id is optional but recommended — the gateway uses it as the
    # cache-isolation scope (stable across key rotation).
    return {
        "project_id": "demo-project",
        "model": "openai/gpt-4o-mini",   "model_api_key": "sk-or-...",
        "embed_model": "text-embedding-3-small", "embed_api_key": "sk-or-...",
        "extractor_model": None,         "extractor_api_key": None,
        "cache_config": {"similarity_threshold": 0.9},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Bind to `0.0.0.0`, not `127.0.0.1`, so the container can reach it. This stub
answers on port 8000 to match `SC_APP_CONFIG_URL` above; the mlops URLs
(`SC_LLM_BASE_URL` / `SC_EMBED_BASE_URL`) point at whatever actually serves
chat + embeddings.

The APP may also use its own field spellings — `llm_model`/`llm_key`,
`embedd_model`/`embedd_key`, `extaractor`/`extaractor_key` are all accepted.

</details>

With a real provider, set the serving URLs to match — e.g. for OpenRouter:

```bash
SC_LLM_BASE_URL=https://openrouter.ai/api
SC_EMBED_BASE_URL=https://api.openai.com
```

and return that provider's model slugs and keys from the config endpoint.
Restart the API after changing `.env`:

```bash
docker compose up -d
```

---

## Step 4 — Make a request as a client

Use whatever key the APP issued to the client. (With the stub above, any
non-empty bearer works.) Any OpenAI client works unchanged:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sc-proj-…" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "what is docker?"}]
      }'
```

Send **the same question twice**. The second response carries:

```json
"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
"semantic_cache": {"hit": true, "similarity": 1.0}
```

Zero tokens and `hit: true` means it was served from cache — no upstream call,
no cost. Paraphrases hit too, above the similarity threshold.

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sc-proj-…")
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "what is docker?"}],
)
print(r.choices[0].message.content)
```

Streaming, `/v1/models`, and `/v1/messages` (your own served-message log) all
work with the same key.

---

## Step 5 — Look inside

```bash
# structured logs
docker compose logs -f api

# what got served (hits, misses, latency, tokens) — keyed by cache scope
docker exec semantic-cache-postgres \
  psql -U scuser -d sccache -c \
  "SELECT scope, model, cache_hit, latency_ms, left(query_text,40) FROM gw.messages ORDER BY id DESC LIMIT 10;"

# cached entries in Redis
docker exec semantic-cache-redis redis-cli DBSIZE

# the durable backup of those entries
docker exec semantic-cache-postgres \
  psql -U scuser -d sccache -c "SELECT count(*) FROM gw.cache_entries;"
```

**Try the durability guarantee** — wipe Redis entirely and restart:

```bash
docker exec semantic-cache-redis redis-cli FLUSHALL
docker restart semantic-cache-api
```

Ask the same question again: still a cache hit, still zero upstream calls. The
cache was rebuilt from Postgres, vectors included.

---

## Step 6 — SSO (optional, but read this before you skip it)

A verified JWT **auto-provisions a tenant**. That makes SSO configuration a
tenant-creation control, not just a login one — so the JWT path **fails
closed**: with no `SC_SSO_*` set, JWT bearers are rejected outright and only
minted `sc-...` keys authenticate. The server logs a warning at boot when it
comes up that way, so you are not left guessing.

Pick one family:

```bash
# Casdoor and most SSOs — asymmetric. Needs the [sso] extra (already in the
# shipped image).
SC_SSO_JWKS_URL=https://sso.example.com/.well-known/jwks.json
SC_SSO_ALGORITHMS=["RS256"]

# ...or a shared secret. Verified in-process with stdlib hmac, no extra needed.
SC_SSO_SHARED_SECRET=<long random string>
SC_SSO_ALGORITHMS=["HS256"]
```

Then, whenever your SSO serves more than this one service:

```bash
SC_SSO_ISSUER=https://sso.example.com
SC_SSO_AUDIENCE=semantic-cache   # without this, a token minted for another
                                 # app is accepted here
```

Naming an algorithm you have no key for (`HS256` with no shared secret,
`RS256` with no JWKS URL) is a **fatal startup error**, not a silent reject —
a service that boots green while refusing every login is worse than one that
refuses to boot.

Tokens with no `exp` are refused. A bearer credential that never expires is not
one this service will hold.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| compose exits with `required variable REDIS_PASSWORD is missing` (or `POSTGRES_PASSWORD` / `SC_API_KEY_PEPPER`) | the required secrets are not in `.env` | Step 1. This is intended behaviour, not a broken checkout — the stack refuses to run on a password printed in a public repository |
| `docker compose ps` shows nothing on the host for Redis/Postgres | you ran `-f docker-compose.yml` (base only), which does not publish them | that is correct for production; for local access use plain `docker compose up -d` |
| `503` on `/v1/chat/completions` | serving env incomplete | the response body names the missing variable |
| `401` on every JWT, minted `sc-` keys fine | no `SC_SSO_*` configured, so the JWT path is closed | Step 6 — this is fail-closed by design, and the server logs a warning at boot saying so |
| `503` on `GET /metrics` | `SC_ADMIN_API_KEY` unset, or the image lacks `[metrics]` | set the admin key; the shipped image installs `[metrics]` |
| `401` on `/v1/*` | no bearer / malformed `Authorization` header | send `Authorization: Bearer <key>` |
| `403` "no model config for this key" | the APP doesn't recognise the key | register the key in the APP first |
| `502` upstream unreachable | wrong `SC_LLM_BASE_URL`, or a host stub bound to `127.0.0.1` | bind stubs to `0.0.0.0`; use `host.docker.internal` from containers |
| `502` "APP config endpoint unreachable" | wrong `SC_APP_CONFIG_URL`, or the APP/stub is down | check the stub is running and bound to `0.0.0.0` |
| api container restart-loops with `could not translate host name "postgres"` | You ran an old command that skipped Postgres, or edited it out | `docker compose ps` should list `semantic-cache-postgres`; if missing, `docker compose up -d` (no flags). Confirm the DSN host is `postgres`, port `5432` |
| build hangs on `pip install` | proxy not passed into the build | use the `--build-arg` form in Step 2 |

Stop everything (data survives in named volumes):

```bash
docker compose down
```

Wipe the data too:

```bash
docker compose down -v
```

---

## What to change for production

1. **Run the base compose file alone** — `docker compose -f docker-compose.yml
   up -d`. The override file publishes your data stores to the host; that is
   for laptops.
2. Point `SC_PG_DSN` at your own Postgres (it must exist — we create only the
   `gw` schema inside it) and remove the bundled `postgres` service. Prefer a
   managed instance with TLS, backups, and a restore drill you have actually
   run.
3. Point `SC_APP_CONFIG_URL` at the real APP and the serving URLs at your
   mlops.
4. Set `SC_ENVIRONMENT=production` and, optionally, `SC_SENTRY_DSN`.
5. Set `SC_ADMIN_API_KEY` — it gates both the SaaS admin routes and `/metrics`.
   Point your Prometheus at `GET /metrics` with that key as a bearer.
6. Wire your orchestrator's probes correctly: **`/health` is liveness**
   (restart on failure) and **`/ready` is readiness** (stop sending traffic).
   Getting these the wrong way round turns a Redis blip into a restart loop.
7. Configure SSO (Step 6) or accept that only minted `sc-...` keys work.
8. Plan retention for `gw.messages` — it stores every question and answer and
   has **no automatic pruning**. This is an open item, not a solved one.

Full request/response contract for both directions:
[APP_INTEGRATION.md](APP_INTEGRATION.md).

# Production-readiness review

Scope: both deployable services (cache_api and cache_as_service), Docker/Compose, and CI. This is a static review, not a penetration test. Python syntax compilation passed. The unit suite was not run because the active interpreter does not have pytest installed; the checked-in CI is not configured to run it correctly either (R01).

## Release blockers / critical

### S01 — cache_api has no effective authorization or tenant isolation

Evidence: cache_api/app/routes/routes_cache.py lines 70, 132, and 221 replace Depends(get_current_user_id) with the constant "x"; lines 173 and 196 expose reads without user/ownership checks.

Any caller can create, modify, list, and read data as the same tenant. A caller who obtains a project_id can read its configuration and provider credentials.

Required fix: Require authentication on every management/data route. Derive user/tenant only from a signature-verified JWT or an identity header inserted by an authenticated reverse proxy, never a client-controlled header. Pass identity into every SQL query and enforce ownership in SQL. Add negative authorization tests for every route.

### S02 — JWT verification falls back to trusting forged tokens

Evidence: cache_api/app/auth.py lines 38-44 catches verification errors and decodes with verify_signature=False. semantic_cache/saas/jwt_auth.py decodes payload bytes without validating signature, issuer, audience, expiry, or algorithm; adapters/saas_router.py lines 87-96 accepts it and auto-provisions a tenant.

An attacker can construct a JWT payload for another organization/user, or a new tenant, and obtain access.

Required fix: Remove every unverified-decode fallback. Validate issuer JWKS/signature, pin allowed algorithms, and enforce iss, aud, exp, nbf, and subject/organization claims. Fail closed. If identity is verified upstream, accept only an authenticated internal assertion over a mutually authenticated private boundary.

### S03 — gateway-admin authentication is disabled

Evidence: cache_api/app/auth.py lines 100-112 returns any supplied bearer token; the configured-key checks are commented out. The cache/key endpoint treats that token as a cache key and logs it.

Required fix: Restore constant-time verification; reject missing server configuration at startup or return a safe 503; and separate administrator credentials/roles from project keys. Test missing, invalid, and valid credentials.

### S04 — provider credentials are plaintext and exposed over API responses

Evidence: cache_api/app/models/database.py lines 22-30 persists LLM, embedding, and extractor keys in plaintext. GatewayConfigResponse and gateway_config in routes_cache.py lines 246-268 return them to a project-key holder. The unauthenticated project route returns CacheRead, including the same keys.

A project-key or database/back-up compromise escalates to the underlying model-provider accounts.

Required fix: Store credentials in a secrets manager or encrypt with envelope encryption/KMS and rotation. Do not return provider keys to clients; have the gateway use server-side credentials or narrow short-lived credentials. Redact secret fields in models, errors, logs, backups, and exports.

### S05 — secrets, JWT claims, and upstream error bodies are logged

Evidence: cache_api/app/auth.py lines 42, 62, and 87 log/print JWT contents; routes_cache.py line 179 logs bearer/cache keys. gateway/upstream.py lines 45-57 logs unredacted upstream response bodies, even though its comment notes providers can echo API keys.

Required fix: Rotate credentials that may already have been logged. Add centralized secret/PII redaction before logging and Sentry capture. Log request IDs or safe fingerprints only—never Authorization values, JWT claims, prompt/output content, provider bodies, or secrets by default.

### S06 — the Compose deployment publishes insecure data stores

Evidence: docker-compose.yml publishes Redis and both Postgres instances to the host, hard-codes scuser/scpass, sets Redis protected-mode to no, and configures neither Redis authentication nor TLS.

Required fix: Do not publish Redis/Postgres in production. Require injected strong secrets, Redis ACL/password, TLS in transit, encrypted storage, private network policies, and managed backups. Put developer defaults in a separate local-only Compose override.

## High priority

### A01 — the /cache/mine route is shadowed by the dynamic project route

Evidence: routes_cache.py registers /{project_id} before /mine. FastAPI matches in registration order, so GET /cache/mine is treated as project_id=mine.

Required fix: Register static paths first or use /projects/{project_id}. Add route-resolution tests.

### A02 — registration is non-atomic and retry-unsafe

Evidence: db_utils.py commits cache creation at lines 65-85 and commits cache configuration separately at lines 109-142. POST /cache/register generates fresh IDs/keys on every retry and has no idempotency key.

Required fix: Use a single transaction for user/cache/config creation with rollback; add unique constraints; and support durable idempotency keys.

### A03 — database schema is modified from application startup

Evidence: cache_api/app/utils/db_utils.py lines 8-12 calls SQLModel.metadata.create_all during startup. Gateway ensure_schema executes ad-hoc DDL in gateway/store.py.

Required fix: Adopt versioned migrations such as Alembic. Run them as controlled deployment steps, add schema-version checks and tested rollback/compatibility paths, and remove runtime DDL.

### A04 — prompts, responses, and guard text have unlimited retention

Evidence: gateway/store.py lines 35-47 stores every query and completion; lines 91-108 can retain guard text. There is no retention job, deletion workflow, consent policy, or access audit. The SaaS registry is also solely Redis data.

Required fix: Minimize stored content, encrypt retained content, define per-tenant retention/legal-hold policy, implement scheduled deletion and data-subject erasure, and audit access. Keep log_turn_text explicit opt-in with expiry.

### A05 — readiness checks Redis only, not the serving path

Evidence: adapters/saas_router.py lines 49-62 checks Redis alone, while gateway serving also needs Postgres, APP configuration, model/embedding endpoints, and possibly the guard.

Required fix: Separate liveness and readiness. Readiness must validate dependencies required by the enabled serving mode using bounded probes and safe dependency-specific status. Fail guarded traffic closed when its guard is required.

### A06 — Prometheus metrics are not deployable in the main service

Evidence: cache counters exist, but cache_as_service/Dockerfile installs fastapi, async, sentry, and postgres extras—not the metrics extra defined in pyproject.toml. server.py does not mount adapters/fastapi_router.py, which exposes the Prometheus endpoint.

Required fix: Make metrics a production dependency and expose one protected or network-restricted /metrics endpoint in the deployed app. Instrument request count/latency/error/in-flight; Redis; Postgres pool; APP config; cache hit/miss with safe low-cardinality reasons; embedding/LLM/guard latency and failures; log queue backlog/drops; and startup/rebuild status.

### A07 — distributed tracing and log correlation are incomplete

Evidence: only the new service uses JSON logging; cache_api/main.py lines 16-20 forces plaintext logging. The JSON formatter has no request context, trace/span IDs, redaction, schema/version, or standard HTTP fields. Sentry tracing defaults to 0.0. request_id exists only inside one gateway route and is not consistently logged or returned.

Required fix: Use one shared structured, redacted event schema. Add middleware to bind and return request_id, propagate W3C traceparent through httpx/background work, and adopt OpenTelemetry FastAPI/httpx/Redis/Postgres instrumentation exporting to Jaeger/OTLP. Include environment, release, trace/span IDs, and safe tenant/project fingerprints in logs.

### A08 — no safe development, staging, and production configuration profiles

Evidence: cache_api/app/config.py can construct a DSN containing None values. cache_api/main.py enables reload=True when launched directly. Compose mixes developer bind mounts, published ports, and defaults with production-labelled variables; SC_ENVIRONMENT is only a tag.

Required fix: Define typed development/staging/production profiles. Fail startup for missing/invalid secrets, URLs, certificate paths, allowed hosts/origins, TLS requirements, and unsafe options. Never enable reload/debug/docs/verbose errors in production; keep local mounts, sample credentials, and public data-store ports in development-only config.

### A09 — no rate, concurrency, quota, or request-size controls

Evidence: SaaS schemas accept unbounded query, response, metadata, and config; the gateway takes an arbitrary body. No rate limiter, body-size limit, tenant quota, connection admission limit, or upstream circuit breaker is configured.

Required fix: Enforce body/message/tool/metadata/policy bounds at proxy and schema layers; reject unexpected fields and oversized content. Implement per-tenant/key rate, concurrency, token, and storage quotas; bounded retries/backoff, circuit breakers, backpressure, and safe 429/503 responses.

### A10 — Redis is the single control-plane source of truth

Evidence: saas/registry.py stores tenant mappings, cache metadata, and hashed API-key records exclusively in Redis. The deployed stack is a single Redis instance; only gateway cache entries are mirrored to Postgres.

Required fix: Move control-plane identity/key/cache metadata to a durable migrated database, or operate highly available Redis with tested backup/restore and a durable source of truth. Document RPO/RTO and run restore drills.

### A11 — API-key lifecycle is insufficient

Evidence: saas/registry.py uses raw SHA-256 lookup hashes. Keys lack expiry, last-used metadata, status, selective revocation, audit events, and granular scopes; tenant rotation revokes all keys.

Required fix: Use a keyed lookup fingerprint such as HMAC or peppered hash; support key IDs/prefixes, expiration, selective revoke, last use, audit data, and gradual rotation. Never log a full key or fingerprint.

### A12 — expensive synchronous startup work prevents horizontal scaling

Evidence: server.py opens Postgres, applies schema, and rebuilds Redis from all backup rows before serving. backup.py iterates synchronously and writes Redis one row at a time; startup also sweeps guard indexes.

Required fix: Run migrations, compaction, and cache restore as single-run jobs with leases. Batch/paginate restoration, cap startup work, publish progress/lag metrics, and warm caches asynchronously.

### A13 — resolved gateway configuration cache is unbounded

Evidence: gateway/app_config.py uses a dictionary keyed by API key with TTL entries but no maximum size or eviction.

Required fix: Use a bounded TTL/LRU cache keyed by a non-reversible key fingerprint, minimize cached secret material, and invalidate immediately on rotation/revocation.

## Delivery and engineering

### R01 — CI does not run the intended checks

Evidence: root .github/workflows/ci.yaml refers to requirements.txt, app, main.py, and a Dockerfile at repository root, but they are under cache_api. The newer workflow is nested under cache_as_service/.github/workflows, which GitHub Actions does not discover. The legacy test job is commented out, and Ruff uses --exit-zero.

Required fix: Put workflows under root .github/workflows; use a service matrix or explicit working directories; remove --exit-zero; run tests and coverage thresholds; build both images; and add type, formatting, dependency/security, container, and integration/contract checks.

### R02 — builds are not reproducible or supply-chain hardened

Evidence: requirements use broad lower bounds without a lockfile; Compose uses redis/redis-stack-server:latest; cache_api uses an unpinned private base image.

Required fix: Commit locked hash-verified dependencies, pin images by immutable digest, produce/sign SBOMs, scan images/dependencies, automate controlled upgrades, and run images non-root with read-only filesystem/capability and resource policy.

### R03 — container runtime is not hardened or sized

Evidence: both Dockerfiles run as root, have no USER/HEALTHCHECK, no resource limits, and start Uvicorn without an explicit worker, concurrency, or graceful-timeout plan. Compose mounts ./cache_api:/code into the running application.

Required fix: Use unprivileged minimal pinned multi-stage images; configure worker count, timeouts, max request size, graceful shutdown, CPU/memory/pid limits; and remove source mounts from production.

### R04 — database integrity and connection controls are missing

Evidence: Cache.cache_key and Cache.project_id are nullable and lack unique/index declarations. The SQL engine uses defaults without documented pool, connect timeout, retry, TLS, or statement-timeout policy.

Required fix: Make identity/key fields non-null and unique; add ownership/lookup indexes; validate domain fields; configure connection pool/timeouts/TLS; and centrally roll back/translate database errors.

### R05 — internal exception strings are sent to callers

Evidence: routes_cache.py lines 122-123 and 165-166, plus generic FastAPI adapter routes, return str(e) in 500 responses.

Required fix: Log a correlated redacted event and return stable public error codes/messages. Map expected failure classes to a centralized API error policy.

### R06 — test coverage lacks security and operational scenarios

Evidence: cache_as_service/pyproject.toml limits collection to tests/unit; the legacy API has no effective CI test job. Required checks do not cover forged JWTs, cross-tenant access, migrations, recovery, metrics/tracing, container behavior, or APP-config end-to-end contracts.

Required fix: Add authorization/property tests, gateway/APP contract tests, ephemeral Redis/Postgres integration tests, migration and recovery tests, load/SLO tests, and security regressions for every issue above. Require them for merges.

## Recommended remediation order

1. Stop exposure: restrict legacy API/data-store access; restore authorization; remove unverified JWT handling; rotate exposed credentials; stop secret logging.
2. Establish secure foundations: secrets management, network/TLS, production profiles, traffic controls, durable control-plane data, migrations, and retention policy.
3. Make the service operable: deployed Prometheus metrics, redacted structured logs, OpenTelemetry/Jaeger tracing, readiness probes, dashboards, alerts, and SLOs.
4. Repair delivery confidence: working CI, deterministic builds, integration/security tests, and load/recovery drills before production launch.


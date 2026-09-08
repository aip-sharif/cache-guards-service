"""Deployable multi-tenant SaaS server.

Run:
    SC_ADMIN_API_KEY=sc-your-admin-key uvicorn semantic_cache.server:create_app --factory

Then bootstrap:
    POST /v1/tenants  (Authorization: Bearer <admin key>)  → tenant api_key
    POST /v1/caches   (Authorization: Bearer <tenant key>) → cache_id
    POST /v1/caches/{cache_id}/set | /search, DELETE .../entries

The legacy unauthenticated `/cache` router is deliberately NOT mounted here
(it has a global purge); library users can still mount it themselves.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI

from semantic_cache.adapters.ops_router import (
    install_body_limit,
    install_http_metrics,
    ops_router,
)
from semantic_cache.adapters.readiness import Check, ReadinessProbe
from semantic_cache.adapters.saas_router import (
    get_admin_key,
    get_jwt_verifier,
    get_manager_pool,
    get_readiness,
    get_redis,
    get_registry,
    health_router,
    saas_router,
)
from semantic_cache.core.config import SemanticCacheConfig
from semantic_cache.saas.jwt_auth import JwtVerifier
from semantic_cache.core.embedding_manager import EmbeddingManagerFactory
from semantic_cache.infrastructure.redis_client import RedisClientFactory
from semantic_cache.observability import configure_logging, init_sentry
from semantic_cache.saas.manager_pool import ManagerPool
from semantic_cache.saas.registry import SaaSRegistry


def _secret(value: Any) -> Optional[str]:
    """Unwraps a SecretStr | str | None to a plain str (or None)."""
    if value is None:
        return None
    return value.get_secret_value() if hasattr(value, "get_secret_value") else value


def _url_reachable(url: Optional[str], timeout: float = 2.0):
    """A readiness probe that asks only "does something answer HTTP here?".

    ANY status code counts, 401 and 403 included: these endpoints are
    per-key-authenticated and readiness has no key to present, so "the service
    is up" is the strongest honest claim. Only a transport failure — DNS,
    refused connection, TLS, timeout — is down.

    Uses stdlib urllib rather than httpx: the probe runs in the readiness
    thread pool, and borrowing the gateway's async clients from another thread
    is a race for no benefit at one request per scrape."""
    import urllib.error
    import urllib.request

    def probe() -> bool:
        if not url:
            return False
        try:
            urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 — our own config
            return True
        except urllib.error.HTTPError:
            return True  # it answered; the status is not ours to satisfy
        except Exception:  # noqa: BLE001 — transport failure is "down"
            return False

    return probe


def create_app(config: Optional[SemanticCacheConfig] = None) -> FastAPI:
    """Builds the SaaS app: env config → shared Redis + embedder → registry/pool."""
    cfg = config or SemanticCacheConfig()

    # Observability comes first so startup itself is logged as JSON and any
    # boot error is reported to Sentry.
    configure_logging(level=cfg.log_level)
    init_sentry(
        dsn=_secret(cfg.sentry_dsn),
        environment=cfg.environment,
        traces_sample_rate=cfg.sentry_traces_sample_rate,
    )

    redis_client = RedisClientFactory.create_client(cfg)
    # Shared embedder for the SaaS tenant caches (API-based by default; the
    # gateway does NOT use it — its embed models come per-project from the
    # APP). Fail-soft: a gateway-only deployment without SC_EMBEDDING_* env
    # must still boot.
    try:
        embedding_manager = EmbeddingManagerFactory.create(cfg)
    except Exception as e:
        embedding_manager = None
        logging.getLogger(__name__).warning(
            "Shared embedder not configured (%s) — SaaS tenant cache routes "
            "will fail until SC_EMBEDDING_* env is set; the gateway is "
            "unaffected.", e,
        )

    api_key_pepper = _secret(cfg.api_key_pepper) or None
    if not api_key_pepper:
        logging.getLogger(__name__).warning(
            "SC_API_KEY_PEPPER is not set — API keys are stored as a bare "
            "sha256, which anyone who can read the Redis keyspace can check a "
            "guess against offline. Set it; existing keys migrate on use."
        )
    registry = SaaSRegistry(redis_client, pepper=api_key_pepper)
    pool = ManagerPool(
        base_config=cfg,
        redis_client=redis_client,
        embedding_manager=embedding_manager,
    )
    admin_key = _secret(cfg.admin_api_key) or None

    # SSO. A JwtConfigError here is deliberately FATAL: an SSO block that names
    # an algorithm it has no key for accepts nothing, and a service that boots
    # green while rejecting every login is worse than one that refuses to boot.
    jwt_verifier = JwtVerifier(
        shared_secret=_secret(cfg.sso_shared_secret) or None,
        jwks_url=cfg.sso_jwks_url,
        issuer=cfg.sso_issuer,
        audience=cfg.sso_audience,
        algorithms=cfg.sso_algorithms,
        leeway=cfg.sso_leeway,
        jwks_ttl=cfg.sso_jwks_ttl,
    )
    if not jwt_verifier.configured:
        logging.getLogger(__name__).warning(
            "SSO JWT verification is DISABLED (no SC_SSO_JWKS_URL or "
            "SC_SSO_SHARED_SECRET) — JWT bearer tokens are rejected and only "
            "minted sc-... API keys authenticate."
        )

    # Readiness is assembled AS the app is wired, so it can only ever name
    # dependencies this process actually took on. A hand-maintained list would
    # drift the first time a mode is added — which is how /health came to
    # check Redis and nothing else while the gateway quietly needed four more.
    readiness_checks = [Check("redis", redis_client.ping)]

    app = FastAPI(title="Semantic Cache SaaS", version="1.0.0")
    install_http_metrics(app)
    # Outermost, so an oversized body is refused before any handler, any
    # validation, and any per-request instrumentation runs on it.
    install_body_limit(app, cfg.max_request_bytes)
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_manager_pool] = lambda: pool
    app.dependency_overrides[get_admin_key] = lambda: admin_key
    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[get_jwt_verifier] = lambda: jwt_verifier
    app.include_router(saas_router)
    app.include_router(health_router)
    app.include_router(ops_router)

    # OpenAI-compatible gateway. It has ONE surface — /v1 serving — because the
    # APP mints and validates the client keys; this service never does. /v1
    # needs Postgres (message log + durable cache backup), the APP's config
    # endpoint, and the mlops endpoints. Until those are set, /v1 answers 503
    # with the exact missing env (never a bare 404).
    pg_dsn = _secret(cfg.pg_dsn)
    serving_env = {
        "SC_APP_CONFIG_URL": cfg.app_config_url,
        "SC_LLM_BASE_URL": cfg.llm_base_url,
        "SC_EMBED_BASE_URL": cfg.embed_base_url,
    }
    if pg_dsn:
        import httpx
        from fastapi import HTTPException

        from semantic_cache.gateway.app_config import AppConfigClient
        from semantic_cache.gateway.backup import (
            make_delete_through,
            make_touch_through,
            make_write_through,
            rebuild_redis_from_postgres,
        )
        from semantic_cache.gateway.guard_checker import GuardChecker
        from semantic_cache.gateway.guard_config import GuardLimits
        from semantic_cache.gateway.guard_judge import GuardJudge
        from semantic_cache.gateway.guard_log import GuardDecisionLog
        from semantic_cache.gateway.guard_pool import GuardPool
        from semantic_cache.gateway.guard_vectors import (
            GuardEmbedder,
            judge_timeout,
            request_embed_timeout,
        )
        from semantic_cache.gateway.pool import GatewayModelPool
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
        from semantic_cache.gateway.store import PostgresGatewayStore
        from semantic_cache.gateway.upstream import UpstreamClient

        store = PostgresGatewayStore(pg_dsn)
        store.ensure_schema()
        app.dependency_overrides[get_gateway_store] = lambda: store
        # Serving needs Postgres, so readiness must too — this is the exact
        # dependency a Redis-only probe reported healthy without.
        readiness_checks.append(Check("postgres", store.ping))

        if all(serving_env.values()):
            # Durable cache backup: Redis serves, Postgres mirrors. Rebuild
            # what Redis lost, then write-through every new entry.
            rebuild_redis_from_postgres(store, redis_client)
            gw_pool = GatewayModelPool(
                base_config=cfg,
                redis_client=redis_client,
                persist_hook=make_write_through(store),
                touch_hook=make_touch_through(store),
                delete_hook=make_delete_through(store),
            )
            # SEPARATE http clients on purpose: streaming completions can hold
            # a connection for minutes, and APP config lookups must never
            # queue behind them — otherwise a slow-LLM incident becomes a
            # total gateway outage instead of degraded streaming.
            upstream_http = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=200, max_keepalive_connections=50
                )
            )
            config_http = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=50, max_keepalive_connections=20
                )
            )
            upstream = UpstreamClient(
                upstream_http, timeout=cfg.gateway_upstream_timeout
            )
            app_config = AppConfigClient(
                config_http,
                url=cfg.app_config_url,
                ttl=cfg.app_config_ttl,
                max_entries=cfg.app_config_max_entries,
                service_key=_secret(cfg.app_service_key) or None,
                service_key_header=cfg.app_service_key_header,
            )
            if not app_config.authenticates_as_a_service:
                logging.getLogger(__name__).warning(
                    "SC_APP_SERVICE_KEY is not set — we identify to the APP "
                    "using only the caller's own key. Anyone holding a client "
                    "key can then pull that client's model-provider "
                    "credentials from the APP directly."
                )

            # A THIRD client for the guard, for the same reason the first two
            # are separate: the guard is inline and fail-closed, so a judge
            # call must never queue behind a minutes-long streaming completion.
            guard_http = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100, max_keepalive_connections=20
                )
            )
            guard_embed_url = cfg.guard_embed_base_url or cfg.embed_base_url
            guard_switch = GuardSwitch(enabled=cfg.guard_enabled)
            guard_log = GuardDecisionLog(store)
            guard_pool = GuardPool(
                store,
                embed_base_url=guard_embed_url,
                limits=GuardLimits(
                    max_policy_bytes=cfg.guard_max_policy_bytes,
                    max_exemplars=cfg.guard_max_exemplars,
                ),
                build_timeout=cfg.guard_build_timeout,
                cache_max_bytes=cfg.guard_cache_max_bytes,
                segment_memo_size=cfg.guard_segment_memo_size,
            )

            def _guard_embedder(resolved):
                # This embedder serves the REQUEST path; GuardPool overrides
                # the timeout per call for the index build, which has its own
                # SC_GUARD_BUILD_TIMEOUT budget. Both values must scale with
                # their setting — the former `min(1.5, guard_timeout / 3)` and
                # `min(3.0, guard_timeout)` were ceilings, so SC_GUARD_TIMEOUT
                # was inert and an endpoint slower than 1.5s could not be
                # reached by any configuration at all.
                return GuardEmbedder(
                    guard_http,
                    base_url=resolved.embed_base_url,
                    api_key=resolved.embed_api_key,
                    model=resolved.embed_model,
                    prefix_style=resolved.params.embed_prefix_style,
                    timeout=request_embed_timeout(
                        cfg.guard_timeout, resolved.params.mode
                    ),
                )

            guard_checker = GuardChecker(
                guard_pool,
                GuardJudge(guard_http, timeout=judge_timeout(cfg.guard_timeout)),
                _guard_embedder,
                timeout=cfg.guard_timeout,
            )

            @app.on_event("startup")
            async def _start_guard() -> None:
                guard_log.start()

            @app.on_event("shutdown")
            async def _close_gateway_clients() -> None:
                # Drain BEFORE closing the store, so in-flight guard decisions
                # land instead of vanishing with the process.
                await guard_log.drain()
                await guard_http.aclose()
                await upstream_http.aclose()
                await config_http.aclose()
                store.close()

            # Orphaned exemplar matrices accumulate on the CUSTOMER's Postgres,
            # one per superseded policy version.
            store.sweep_guard_indexes(older_than_days=30)

            settings = GatewaySettings(
                llm_base_url=cfg.llm_base_url,
                embed_base_url=cfg.embed_base_url,
                extractor_base_url=cfg.extractor_base_url,
                guard_embed_base_url=cfg.guard_embed_base_url,
                guard_judge_base_url=cfg.guard_judge_base_url,
            )
            app.dependency_overrides[get_gateway_pool] = lambda: gw_pool
            app.dependency_overrides[get_upstream] = lambda: upstream
            app.dependency_overrides[get_app_config] = lambda: app_config
            app.dependency_overrides[get_gateway_settings] = lambda: settings
            app.dependency_overrides[get_guard_checker] = lambda: guard_checker
            app.dependency_overrides[get_guard_switch] = lambda: guard_switch
            app.dependency_overrides[get_guard_log] = lambda: guard_log
            # The APP's config endpoint is on the critical path of every
            # completion: no config, no serving. We probe it WITHOUT a bearer,
            # so a 401/403 is success — it proves the endpoint is up and
            # answering, which is all readiness can honestly assert about a
            # per-key resource.
            readiness_checks.append(
                Check("app_config", _url_reachable(cfg.app_config_url))
            )
            # The mlops endpoints are reported but do NOT gate the pod: a
            # completion against a down LLM is one failed request, whereas
            # marking every replica unready over it takes the whole service
            # out — including the cache hits that need no LLM at all.
            readiness_checks.append(
                Check("llm", _url_reachable(cfg.llm_base_url), required=False)
            )
            readiness_checks.append(
                Check("embed", _url_reachable(guard_embed_url), required=False)
            )
            if not cfg.guard_enabled:
                logging.getLogger(__name__).warning(
                    "Input guard is DISABLED by SC_GUARD_ENABLED — every "
                    "guarded client is being served UNGUARDED."
                )
            logging.getLogger(__name__).info(
                "OpenAI-compatible gateway mounted (/v1 serving)."
            )
        else:
            missing = sorted(k for k, v in serving_env.items() if not v)

            def _serving_unconfigured():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Gateway serving is not configured — missing env: "
                        + ", ".join(missing)
                    ),
                )

            # get_guard_* belong here too: FastAPI resolves every dependency
            # before the handler body, so leaving them as NotImplementedError
            # stubs would surface as a bare 500 instead of this 503 — and
            # install_openai_error_handlers only covers HTTPException and
            # RequestValidationError.
            for dep in (get_gateway_pool, get_upstream, get_app_config,
                        get_gateway_settings, get_guard_checker,
                        get_guard_switch, get_guard_log):
                app.dependency_overrides[dep] = _serving_unconfigured
            logging.getLogger(__name__).warning(
                "Gateway /v1 serving DISABLED — missing env: %s",
                ", ".join(missing),
            )
        app.include_router(gateway_router)
        install_openai_error_handlers(app)
    else:
        # No Postgres → no gateway. Answer its paths with a loud 503 that
        # names the missing env instead of a mystifying 404.
        from fastapi.responses import JSONResponse

        async def _gateway_unconfigured(path: str = ""):
            return JSONResponse(
                status_code=503,
                content={"error": {
                    "message": (
                        "Gateway is not configured: SC_PG_DSN is not set. "
                        "Set SC_PG_DSN (and for /v1 serving also "
                        "SC_APP_CONFIG_URL, SC_LLM_BASE_URL, "
                        "SC_EMBED_BASE_URL), then restart."
                    ),
                    "type": "gateway_error",
                }},
            )

        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        app.add_api_route(
            "/v1/{path:path}", _gateway_unconfigured,
            methods=methods, include_in_schema=False,
        )
        logging.getLogger(__name__).warning(
            "Gateway NOT mounted — SC_PG_DSN is not set."
        )

    # Registered last: `readiness_checks` is complete only once every mode has
    # had its say.
    probe = ReadinessProbe(readiness_checks, timeout=cfg.readiness_timeout)
    app.dependency_overrides[get_readiness] = lambda: probe
    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # log_config=None → uvicorn keeps our JSON handler instead of installing
    # its own text formatter.
    uvicorn.run(create_app(), host="0.0.0.0", port=8080, log_config=None)

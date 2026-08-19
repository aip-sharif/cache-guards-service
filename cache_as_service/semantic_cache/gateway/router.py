"""OpenAI-compatible gateway routes.

This service is the LLM SERVICE: it owns the OpenAI-compatible surface, the
semantic cache, and the message log. It does NOT own model configs or client
keys — both live in the APP (a separate project). The APP mints each client's
key; a client then calls us with that key, and per request we present the same
key to the APP to learn which models to use, and serve via the mlops endpoints
configured in env.

There is no project id and no key minting on this side: the key is the whole
identity. For cache isolation we scope by the `project_id` the APP returns
(falling back to a hash of the key when it returns none).

Public surface (key bearer auth — the APP's key, verified by the APP):
    POST /v1/chat/completions   — cache-first chat: Redis semantic hit or
                                  mlops LLM passthrough (JSON or SSE)
    GET  /v1/models             — the caller's model, as told by the APP
    GET  /v1/messages           — the caller's own served-message log
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from semantic_cache.adapters.saas_router import _bearer, get_admin_key
from semantic_cache.gateway.app_config import AppConfigClient, AppConfigError
from semantic_cache.gateway.chat import (
    UNGUARDABLE,
    SSEAccumulator,
    build_cached_completion,
    build_cached_sse,
    build_guard_completion,
    build_guard_sse,
    extract_cache_text,
    extract_guard_segments,
    guardrail_extra,
)
from semantic_cache.gateway.guard_checker import GuardChecker
from semantic_cache.gateway.guard_logic import DEFAULT_UNAVAILABLE_REFUSAL
from semantic_cache.gateway.pool import GatewayModelPool
from semantic_cache.gateway.store import PostgresGatewayStore
from semantic_cache.gateway.upstream import UpstreamClient, UpstreamError

logger = logging.getLogger(__name__)

# Request features the cache cannot honor: a cached plain-text answer would
# violate the contract these imply (tool calls, JSON mode, multiple choices).
# Requests using them go straight upstream, uncached.
_CACHE_BYPASS_FIELDS = ("tools", "tool_choice", "functions", "response_format")


class GatewaySettings(BaseModel):
    """The mlops serving endpoints, from env (the APP supplies models+keys)."""

    llm_base_url: str
    embed_base_url: str
    extractor_base_url: Optional[str] = None
    # The guard reuses the cache's endpoints unless the operator splits them
    # out. Neither is required, so the guard adds no new mount precondition.
    guard_embed_base_url: Optional[str] = None
    guard_judge_base_url: Optional[str] = None


class GuardSwitch(BaseModel):
    """Operator break-glass, MUTABLE at runtime.

    A frozen setting would mean redeploying a `restart: unless-stopped`
    container while a misbehaving fail-closed guard refuses every request. The
    cache needs no such switch — it degrades silently on its own.
    """

    enabled: bool = True


def get_gateway_store() -> PostgresGatewayStore:
    raise NotImplementedError("Dependency get_gateway_store must be overridden.")


def get_gateway_pool() -> GatewayModelPool:
    raise NotImplementedError("Dependency get_gateway_pool must be overridden.")


def get_upstream() -> UpstreamClient:
    raise NotImplementedError("Dependency get_upstream must be overridden.")


def get_app_config() -> AppConfigClient:
    raise NotImplementedError("Dependency get_app_config must be overridden.")


def get_gateway_settings() -> GatewaySettings:
    raise NotImplementedError("Dependency get_gateway_settings must be overridden.")


def get_guard_checker() -> Optional[GuardChecker]:
    raise NotImplementedError("Dependency get_guard_checker must be overridden.")


def get_guard_switch() -> GuardSwitch:
    raise NotImplementedError("Dependency get_guard_switch must be overridden.")


def get_guard_log() -> Any:
    raise NotImplementedError("Dependency get_guard_log must be overridden.")


gateway_router = APIRouter(tags=["OpenAI Gateway"])


# --------------------------------------------------------------------------- #
# Errors + auth
# --------------------------------------------------------------------------- #


def _openai_error(
    status_code: int, message: str, retry_after: Optional[int] = None
) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "gateway_error"}},
        headers=headers,
    )


def install_openai_error_handlers(app: FastAPI) -> None:
    """Makes /v1/* errors use the OpenAI error envelope instead of FastAPI's
    default {"detail": ...} shape, so OpenAI SDKs parse them correctly."""

    @app.exception_handler(HTTPException)
    async def _http_exc(request, exc: HTTPException):
        if request.url.path.startswith("/v1/"):
            return _openai_error(exc.status_code, str(exc.detail))
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail}
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request, exc: RequestValidationError):
        if request.url.path.startswith("/v1/"):
            return _openai_error(400, f"Invalid request body: {exc.errors()}")
        return JSONResponse(status_code=422, content={"detail": exc.errors()})


def require_key(authorization: Optional[str] = Header(default=None)) -> str:
    """The caller's raw API key. We do not validate it ourselves — the APP is
    the source of truth: presenting it to the config endpoint either yields a
    config (valid key) or a 403 (unknown key). We only require that one is
    present and well-formed."""
    return _bearer(authorization)


def _scope(config: Dict[str, Any], api_key: str) -> str:
    """Cache/log isolation key. Prefer the APP's project_id (stable across key
    rotation); fall back to a hash of the key so scopes never collide and raw
    keys never land in Redis tags or Postgres rows."""
    project_id = config.get("project_id")
    if project_id:
        return str(project_id)
    return "k_" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]


def _model_row(config: Dict[str, Any], settings: GatewaySettings) -> Dict[str, Any]:
    """APP config (models + keys) x env (mlops URLs) → one resolved row."""
    has_extractor = bool(config["extractor_model"] and settings.extractor_base_url)
    return {
        "model": config["model"],
        "llm_base_url": settings.llm_base_url,
        "llm_api_key": config["model_api_key"],
        "llm_model": config["model"],
        "embed_base_url": settings.embed_base_url,
        "embed_api_key": config["embed_api_key"],
        "embed_model": config["embed_model"],
        "extractor_base_url": settings.extractor_base_url if has_extractor else None,
        "extractor_api_key": config["extractor_api_key"] if has_extractor else None,
        "extractor_model": config["extractor_model"] if has_extractor else None,
        "extractor_domain": config.get("extractor_domain"),
        "cache_config": config.get("cache_config") or {},
    }


def _unguardable_outcome(resolved):
    """An in-scope message the guard cannot read (image-only, audio-only).

    Treated as a runtime failure, not as "nothing to check": serving it would
    forward content the guard never saw.
    """
    from semantic_cache.gateway.guard_checker import GuardOutcome

    return GuardOutcome(
        action="unavailable",
        reason="unguardable_content",
        refusal=(
            resolved.params.unavailable_refusal or DEFAULT_UNAVAILABLE_REFUSAL
        ),
    )


def _guard_unavailable_response(
    resolved, outcome, model_name: str, stream: bool,
    include_usage: bool, request_id: str,
):
    """What a client sees when the guard could not run and did not degrade.

    Defaults to 503, NOT the 200-with-refusal shape used for a block. A block
    is a DECISION; this is a broken dependency, and dressing it as a successful
    completion makes it invisible on every 5xx dashboard, unretryable by every
    SDK, and a real assistant turn in the client's stored history.
    ``unavailable_response: "refusal"`` opts into the 200 shape for clients
    whose frontend cannot handle a 503.
    """
    refusal = outcome.refusal or DEFAULT_UNAVAILABLE_REFUSAL
    if resolved.params.unavailable_response == "refusal":
        extra = guardrail_extra(
            "guard_unavailable", None, None, False,
            resolved.policy_hash, request_id, outcome.reason,
        )
        headers = {"x-guardrail": "unavailable"}
        if stream:
            return StreamingResponse(
                build_guard_sse(model_name, refusal, extra, include_usage),
                media_type="text/event-stream", headers=headers,
            )
        return JSONResponse(
            build_guard_completion(model_name, refusal, extra), headers=headers
        )
    return _openai_error(
        503,
        "The safety guard for this key could not run, so the request was "
        f"refused. Retry shortly. (reason {outcome.reason}, "
        f"request_id {request_id})",
        retry_after=5,
    )


# --------------------------------------------------------------------------- #
# POST /v1/chat/completions
# --------------------------------------------------------------------------- #


@gateway_router.post("/v1/chat/completions")
async def chat_completions(
    payload: Dict[str, Any] = Body(...),
    api_key: str = Depends(require_key),
    store: PostgresGatewayStore = Depends(get_gateway_store),
    pool: GatewayModelPool = Depends(get_gateway_pool),
    upstream: UpstreamClient = Depends(get_upstream),
    app_config: AppConfigClient = Depends(get_app_config),
    settings: GatewaySettings = Depends(get_gateway_settings),
    guard: Optional[GuardChecker] = Depends(get_guard_checker),
    guard_switch: GuardSwitch = Depends(get_guard_switch),
    guard_log: Any = Depends(get_guard_log),
):
    started = time.monotonic()
    request_id = str(uuid.uuid4())
    model_name = payload.get("model")
    messages = payload.get("messages")
    if not model_name or not isinstance(messages, list):
        return _openai_error(400, "'model' and 'messages' are required.")

    # The APP decides which models this key uses (short-TTL cached).
    try:
        config = await app_config.resolve(api_key)
    except AppConfigError as e:
        return _openai_error(e.status_code, str(e))
    model_row = _model_row(config, settings)
    scope = _scope(config, api_key)

    stream = bool(payload.get("stream"))
    include_usage = bool(
        (payload.get("stream_options") or {}).get("include_usage")
    )

    # The client (via the APP) may turn caching off entirely — a pure
    # passthrough. We honor it here so an 'off' client never even pays the
    # manager's one-time embedding-dimension probe. 'off' is only the scalar
    # cache_mode; a cache_mode LIST (cascade) always means caching is ON.
    _cm = (model_row["cache_config"] or {}).get("cache_mode", "semantic")
    cache_off = isinstance(_cm, str) and _cm.lower() == "off"

    # Cache only requests a cached plain answer can faithfully serve.
    bypass = (
        cache_off
        or any(payload.get(f) for f in _CACHE_BYPASS_FIELDS)
        or (payload.get("n") or 1) != 1
    )
    # The query is logged even when it is not cached (bypass/off), so the
    # message log stays a complete record of served traffic.
    log_text = extract_cache_text(messages)
    cache_text = None if bypass else log_text

    def _latency() -> int:
        return int((time.monotonic() - started) * 1000)

    def _log(cache_hit: bool, response: str, similarity=None,
             finish_reason=None, usage=None) -> None:
        """Message log is fire-and-forget: never let it break the reply."""
        try:
            store.log_message(
                scope=scope, model=model_name, cache_hit=cache_hit,
                query_text=log_text or "", response=response,
                similarity=similarity, finish_reason=finish_reason,
                usage=usage, latency_ms=_latency(), request_id=request_id,
            )
        except Exception as e:  # pragma: no cover — store is already fail-open
            logger.error("Message log failed: %s", e)

    def _log_guard(outcome, resolved) -> None:
        """Enqueue, never await: see gateway/guard_log.py."""
        if guard_log is None:
            return
        try:
            guard_log.submit({
                "request_id": request_id,
                "scope": scope,
                "policy_hash": resolved.policy_hash,
                "action": outcome.action,
                "matched_category": outcome.matched_category,
                "reason": outcome.reason,
                "embedding_score": outcome.embedding_score,
                "judge_score": outcome.judge_score,
                "judge_invoked": outcome.judge_invoked,
                "top_matches": [
                    {"category_id": n.category_id, "label": n.label,
                     "similarity": round(n.similarity, 4)}
                    for n in (outcome.top_matches or [])
                ],
                "turn_text": (
                    log_text if resolved.params.log_turn_text else None
                ),
                "latency_ms": _latency(),
            })
        except Exception as e:  # pragma: no cover — the log is already bounded
            logger.error("Guard decision log failed: %s", e)

    # ---- 0. The guard gate ------------------------------------------------ #
    # Deliberately BEFORE the cache manager is built, and outside `if
    # cache_text:` — so a request that bypasses the cache entirely (tools,
    # response_format, n>1, cache_mode "off") is still guarded. Hanging the
    # guard off the cache path would make those a one-field bypass.
    cache_scope = scope
    degraded = False
    resolved_guard = None
    guard_flag: Optional[Dict[str, Any]] = None

    if guard is not None and guard_switch is not None and guard_switch.enabled:
        try:
            resolved_guard = guard.pool.resolve(config)
        except Exception as e:  # noqa: BLE001
            # Broad on purpose: `guard` is an UNVALIDATED passthrough, so an
            # APP sending it as a JSON string raises pydantic's error, not
            # ours, and a bare 500 would escape the OpenAI envelope.
            return _openai_error(
                502, f"Guard config from APP is invalid: {e}", retry_after=30
            )

    if resolved_guard is not None:
        # Guarded traffic never shares a cache namespace with unguarded traffic
        # under the same project_id, and tightening a policy invalidates every
        # answer produced under the looser one. `scope` itself is unchanged, so
        # the message log and both read routes keep continuous history.
        cache_scope = f"{scope}#g{resolved_guard.policy_hash[:12]}"

        segments = extract_guard_segments(messages, resolved_guard.check_roles)
        if segments is UNGUARDABLE:
            outcome = _unguardable_outcome(resolved_guard)
        else:
            outcome = await guard.check(
                segments, resolved_guard,
                judge_base_url=(
                    settings.guard_judge_base_url or settings.llm_base_url
                ),
            )
        _log_guard(outcome, resolved_guard)

        if outcome.action == "unavailable":
            if resolved_guard.params.degrade_to_unguarded:
                logger.critical(
                    "GUARD DEGRADED — serving UNGUARDED (scope=%s policy=%s): %s",
                    scope, resolved_guard.policy_hash[:12], outcome.reason,
                )
                degraded = True
            else:
                refusal = outcome.refusal or DEFAULT_UNAVAILABLE_REFUSAL
                await asyncio.to_thread(
                    _log, False, refusal, None, "content_filter", None
                )
                return _guard_unavailable_response(
                    resolved_guard, outcome, model_name, stream,
                    include_usage, request_id,
                )

        elif outcome.action == "block":
            extra = guardrail_extra(
                "block", outcome.matched_category, outcome.embedding_score,
                outcome.judge_invoked, resolved_guard.policy_hash, request_id,
            )
            refusal = outcome.refusal or ""
            await asyncio.to_thread(
                _log, False, refusal, None, "content_filter", None
            )
            headers = {"x-guardrail": "block"}
            if stream:
                return StreamingResponse(
                    build_guard_sse(model_name, refusal, extra, include_usage),
                    media_type="text/event-stream", headers=headers,
                )
            return JSONResponse(
                build_guard_completion(model_name, refusal, extra),
                headers=headers,
            )

        elif outcome.action == "flag":
            # Flag SERVES. It only annotates the response and the decision log.
            guard_flag = guardrail_extra(
                "flag", outcome.matched_category, outcome.embedding_score,
                outcome.judge_invoked, resolved_guard.policy_hash, request_id,
            )

    # First use of a config builds its manager: sync Redis index bootstrap
    # plus a blocking embedding-dimension probe — keep it off the event loop.
    manager = None
    if cache_text:
        try:
            manager = await asyncio.to_thread(pool.get, model_row)
        except ValueError as e:  # bad config from the APP (e.g. extractor domain)
            return _openai_error(502, f"Config from APP is invalid: {e}")
        except Exception as e:  # noqa: BLE001
            # Redis down, embed endpoint unreachable, index bootstrap failed…
            # The cache is an optimization: degrade to a plain passthrough
            # instead of failing the request (same fail-open contract as
            # search/set).
            logger.error("Cache unavailable, serving uncached: %s", e)
            manager = None

    # ---- 1. Cache lookup (isolated per scope) ----------------------------- #
    if manager is not None:
        result = await manager.asearch(cache_text, scope=cache_scope)
        if result is not None:
            content = result["response"]
            similarity = result.get("similarity")
            await asyncio.to_thread(
                _log, True, content, similarity, "stop", None
            )
            if stream:
                return StreamingResponse(
                    build_cached_sse(model_name, content, include_usage),
                    media_type="text/event-stream",
                )
            return JSONResponse(build_cached_completion(model_name, content, similarity))

    # ---- 2. Miss → mlops LLM ----------------------------------------------- #
    upstream_payload = {**payload, "model": model_row["llm_model"]}
    base_url, api_key_up = model_row["llm_base_url"], model_row["llm_api_key"]

    async def _write_back(content: str, finish: Optional[str]) -> None:
        if manager is None or finish != "stop" or not content.strip():
            return
        if degraded:
            # An answer served while the guard could not run must not be
            # cached: it would outlive the outage and keep being served.
            return
        try:
            await manager.aset(
                query=cache_text,
                response=content,
                metadata={"upstream_model": model_row["llm_model"]},
                scope=cache_scope,
            )
        except Exception as e:
            logger.error("Cache write-back failed: %s", e)

    if not stream:
        try:
            data = await upstream.complete(base_url, api_key_up, upstream_payload)
        except UpstreamError as e:
            return _openai_error(e.status_code, str(e))
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason")
        await _write_back(content, finish)
        await asyncio.to_thread(
            _log, False, content, None, finish, data.get("usage")
        )
        data["model"] = model_name  # answer under the requested name
        if guard_flag is not None:
            data["guardrail"] = guard_flag
        return JSONResponse(
            data, headers={"x-guardrail": "flag"} if guard_flag else None
        )

    # ---- 3. Streaming miss: pass through, accumulate, write back ---------- #
    lines = upstream.stream(base_url, api_key_up, upstream_payload)
    try:
        first = await lines.__anext__()
    except UpstreamError as e:
        return _openai_error(e.status_code, str(e))
    except StopAsyncIteration:
        first = None

    async def relay() -> AsyncIterator[str]:
        acc = SSEAccumulator()
        if first is not None:
            acc.feed(first)
            yield first + "\n"
        try:
            async for line in lines:
                acc.feed(line)
                yield line + "\n"
        except UpstreamError as e:  # mid-stream failure: end the stream cleanly
            logger.error("Upstream stream aborted: %s", e)
        if acc.cacheable:
            await _write_back(acc.content, acc.finish_reason)
        await asyncio.to_thread(
            _log, False, acc.content, None, acc.finish_reason, acc.usage
        )

    return StreamingResponse(relay(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# GET /v1/models + /v1/messages
# --------------------------------------------------------------------------- #


@gateway_router.get("/v1/models")
async def list_models(
    api_key: str = Depends(require_key),
    app_config: AppConfigClient = Depends(get_app_config),
):
    """The model the APP configured for this key, in OpenAI list shape."""
    try:
        config = await app_config.resolve(api_key)
    except AppConfigError as e:
        return _openai_error(e.status_code, str(e))
    return {
        "object": "list",
        "data": [
            {
                "id": config["model"],
                "object": "model",
                "created": 0,
                "owned_by": "semantic-cache-gateway",
            }
        ],
    }


class GuardSwitchRequest(BaseModel):
    enabled: bool


@gateway_router.post("/admin/guard")
async def set_guard_switch(
    body: GuardSwitchRequest,
    authorization: Optional[str] = Header(default=None),
    admin_key: Optional[str] = Depends(get_admin_key),
    switch: GuardSwitch = Depends(get_guard_switch),
):
    """Operator break-glass: turn the guard off (or on) for EVERY client.

    Deliberately not under ``/v1`` — it is an operator surface, not a client
    one — and behind the existing SC_ADMIN_API_KEY. It flips a mutable object
    rather than a setting, because the alternative during an incident is
    redeploying a `restart: unless-stopped` container while a fail-closed guard
    refuses 100% of traffic.
    """
    if not admin_key:
        raise HTTPException(503, "No admin key configured (SC_ADMIN_API_KEY).")
    if _bearer(authorization) != admin_key:
        raise HTTPException(403, "Invalid admin key.")
    switch.enabled = body.enabled
    logger.critical(
        "Input guard %s for ALL clients via /admin/guard.",
        "ENABLED" if body.enabled else "DISABLED",
    )
    return {"enabled": switch.enabled}


@gateway_router.get("/v1/guard/decisions")
async def list_guard_decisions(
    limit: int = Query(default=50, ge=1, le=500),
    api_key: str = Depends(require_key),
    store: PostgresGatewayStore = Depends(get_gateway_store),
    app_config: AppConfigClient = Depends(get_app_config),
):
    """The caller's own guard decisions (newest first).

    Ships with v1 rather than later because `flag` is the one guard action that
    still reaches the model, and it has the weakest audit trail. Without this
    route nobody can discover that their ambiguous band is quietly swallowing
    attacks without running SQL against the customer's own database.

    ``top_matches`` — the top-3 (category, label, similarity) — is the actual
    explanation of a decision and contains no user text, so it stays available
    even for clients who leave ``log_turn_text`` off.
    """
    try:
        config = await app_config.resolve(api_key)
    except AppConfigError as e:
        return _openai_error(e.status_code, str(e))
    scope = _scope(config, api_key)
    return await asyncio.to_thread(store.list_guard_decisions, scope, limit)


@gateway_router.get("/v1/messages")
async def list_messages(
    limit: int = Query(default=50, ge=1, le=500),
    api_key: str = Depends(require_key),
    store: PostgresGatewayStore = Depends(get_gateway_store),
    app_config: AppConfigClient = Depends(get_app_config),
):
    """The caller's own served-message log (newest first). Resolving the config
    yields the scope the log rows were written under."""
    try:
        config = await app_config.resolve(api_key)
    except AppConfigError as e:
        return _openai_error(e.status_code, str(e))
    scope = _scope(config, api_key)
    return await asyncio.to_thread(store.list_messages, scope, limit)

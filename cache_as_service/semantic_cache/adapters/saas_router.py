"""Multi-tenant SaaS HTTP API.

Bearer-key-authenticated, per-tenant, per-named-cache endpoints. Data
isolation rides on the core `scope` TAG: every data operation ALWAYS passes
scope=f"t{tenant_id}__c{cache_id}", so the unfiltered match-all search path
is unreachable from here.

The three dependencies (`get_registry`, `get_manager_pool`, `get_admin_key`)
are override points, wired by `semantic_cache.server.create_app` (or a test).
"""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from semantic_cache.saas.jwt_auth import external_identity
from semantic_cache.saas.manager_pool import ManagerPool
from semantic_cache.saas.registry import SaaSRegistry


def get_registry() -> SaaSRegistry:
    raise NotImplementedError("Dependency get_registry must be overridden.")


def get_manager_pool() -> ManagerPool:
    raise NotImplementedError("Dependency get_manager_pool must be overridden.")


def get_admin_key() -> Optional[str]:
    """The admin bearer key (config.admin_api_key). None → admin disabled."""
    raise NotImplementedError("Dependency get_admin_key must be overridden.")


def get_redis() -> Any:
    """The shared Redis client (for health checks)."""
    raise NotImplementedError("Dependency get_redis must be overridden.")


saas_router = APIRouter(prefix="/v1", tags=["Semantic Cache SaaS"])

# Health lives at the root (no /v1 prefix) so probes hit a stable path.
health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
def health(redis_client: Any = Depends(get_redis)) -> Dict[str, Any]:
    """Unauthenticated liveness/readiness probe: confirms Redis is reachable."""
    try:
        redis_ok = bool(redis_client.ping())
    except Exception:
        redis_ok = False
    status = "ok" if redis_ok else "degraded"
    code = 200 if redis_ok else 503
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=code, content={"status": status, "redis": redis_ok}
    )


# --------------------------------------------------------------------------- #
# Auth dependencies
# --------------------------------------------------------------------------- #


def _bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    return authorization[7:].strip()


def require_admin(
    authorization: Optional[str] = Header(default=None),
    admin_key: Optional[str] = Depends(get_admin_key),
) -> None:
    if not admin_key:
        raise HTTPException(status_code=503, detail="Admin API is not configured.")
    token = _bearer(authorization)
    if not secrets.compare_digest(token, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin key.")


def _tenant_from_token(registry: SaaSRegistry, token: str) -> Optional[str]:
    """Resolves a bearer token to a tenant_id via, in order:
    a legacy tenant API key (`sc-...`), or an SSO JWT (org+user,
    auto-provisioned). Returns None if neither matches."""
    tenant_id = registry.resolve_api_key(token)
    if tenant_id is not None:
        return tenant_id
    external_id = external_identity(token)
    if external_id is not None:
        return registry.get_or_create_tenant_by_external(external_id)
    return None


def require_tenant(
    authorization: Optional[str] = Header(default=None),
    registry: SaaSRegistry = Depends(get_registry),
) -> str:
    token = _bearer(authorization)
    tenant_id = _tenant_from_token(registry, token)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    return tenant_id


def require_cache_data_access(
    cache_id: str,
    authorization: Optional[str] = Header(default=None),
    registry: SaaSRegistry = Depends(get_registry),
) -> str:
    """Auth for DATA endpoints: the tenant key OR that cache's own key.

    A per-cache key presented against a different cache_id is a hard 403 —
    it authenticated, but is not authorized for that cache."""
    token = _bearer(authorization)
    tenant_id = _tenant_from_token(registry, token)
    if tenant_id is not None:
        return tenant_id
    resolved = registry.resolve_cache_key(token)
    if resolved is None:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    key_tenant, key_cache = resolved
    if key_cache != cache_id:
        raise HTTPException(
            status_code=403, detail="This key is scoped to a different cache."
        )
    return key_tenant


def _resolve_cache(
    registry: SaaSRegistry, tenant_id: str, cache_id: str
) -> Dict[str, Any]:
    info = registry.get_cache(tenant_id, cache_id)
    if info is None:
        # Ownership check: the registry key embeds tenant_id, so another
        # tenant's cache_id simply does not exist under this tenant.
        raise HTTPException(status_code=404, detail="Cache not found.")
    return info


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class TenantCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)


class TenantCreateResponse(BaseModel):
    tenant_id: str
    api_key: str = Field(..., description="Shown ONCE — store it safely.")


class CacheCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-cache overrides (similarity_threshold, default_ttl, ...).",
    )


class CacheUpdateRequest(BaseModel):
    config: Dict[str, Any]


class CacheInfo(BaseModel):
    cache_id: str
    name: str
    config: Dict[str, Any]
    created_at: int


class MeResponse(BaseModel):
    tenant_id: str
    cache_count: int
    caches: List[CacheInfo]


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    hit: bool
    response: Optional[str] = None
    similarity: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class SetRequest(BaseModel):
    query: str
    response: str
    metadata: Optional[Dict[str, Any]] = None
    ttl: Optional[int] = None
    keep_forever: bool = False


class PurgeResponse(BaseModel):
    deleted: int


class CacheKeyResponse(BaseModel):
    cache_id: str
    api_key: str = Field(..., description="Shown ONCE — store it safely.")


class TenantKeyResponse(BaseModel):
    tenant_id: str
    api_key: str = Field(..., description="Shown ONCE — store it safely.")


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #


@saas_router.post(
    "/tenants",
    response_model=TenantCreateResponse,
    status_code=201,
    dependencies=[Depends(require_admin)],
)
def create_tenant(
    request: TenantCreateRequest,
    registry: SaaSRegistry = Depends(get_registry),
):
    tenant_id, api_key = registry.create_tenant(request.name)
    return TenantCreateResponse(tenant_id=tenant_id, api_key=api_key)


@saas_router.post("/tenant/rotate-key", response_model=TenantKeyResponse, status_code=201)
def rotate_tenant_key(
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    """Self-service account-key rotation: revokes the current tenant key(s)
    and returns a fresh one. Per-cache keys are unaffected."""
    new_key = registry.rotate_tenant_key(tenant_id)
    return TenantKeyResponse(tenant_id=tenant_id, api_key=new_key)


# --------------------------------------------------------------------------- #
# Per-cache API keys (management is tenant-key-only)
# --------------------------------------------------------------------------- #


@saas_router.post(
    "/caches/{cache_id}/keys", response_model=CacheKeyResponse, status_code=201
)
def issue_cache_key(
    cache_id: str,
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    _resolve_cache(registry, tenant_id, cache_id)
    api_key = registry.issue_cache_key(tenant_id, cache_id)
    return CacheKeyResponse(cache_id=cache_id, api_key=api_key)


@saas_router.post(
    "/caches/{cache_id}/keys/rotate", response_model=CacheKeyResponse, status_code=201
)
def rotate_cache_key(
    cache_id: str,
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    """Revokes every existing key for this cache and issues a fresh one."""
    _resolve_cache(registry, tenant_id, cache_id)
    api_key = registry.rotate_cache_key(tenant_id, cache_id)
    return CacheKeyResponse(cache_id=cache_id, api_key=api_key)


# --------------------------------------------------------------------------- #
# Cache CRUD
# --------------------------------------------------------------------------- #


@saas_router.get("/me", response_model=MeResponse)
def me(
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    """SSO landing call: the tenant's identity and all its caches."""
    caches = [CacheInfo(**c) for c in registry.list_caches(tenant_id)]
    return MeResponse(tenant_id=tenant_id, cache_count=len(caches), caches=caches)


@saas_router.post("/caches", response_model=CacheInfo, status_code=201)
def create_cache(
    request: CacheCreateRequest,
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    try:
        cache_id = registry.create_cache(tenant_id, request.name, request.config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return CacheInfo(**_resolve_cache(registry, tenant_id, cache_id))


@saas_router.get("/caches", response_model=List[CacheInfo])
def list_caches(
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    return [CacheInfo(**c) for c in registry.list_caches(tenant_id)]


@saas_router.get("/caches/{cache_id}", response_model=CacheInfo)
def get_cache(
    cache_id: str,
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    return CacheInfo(**_resolve_cache(registry, tenant_id, cache_id))


@saas_router.patch("/caches/{cache_id}", response_model=CacheInfo)
def update_cache(
    cache_id: str,
    request: CacheUpdateRequest,
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
):
    _resolve_cache(registry, tenant_id, cache_id)
    try:
        info = registry.update_cache(tenant_id, cache_id, request.config)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return CacheInfo(**info)


@saas_router.delete("/caches/{cache_id}", response_model=PurgeResponse)
async def delete_cache(
    cache_id: str,
    tenant_id: str = Depends(require_tenant),
    registry: SaaSRegistry = Depends(get_registry),
    pool: ManagerPool = Depends(get_manager_pool),
):
    info = _resolve_cache(registry, tenant_id, cache_id)
    scope = SaaSRegistry.data_scope(tenant_id, cache_id)
    manager = pool.get(tenant_id, cache_id, info["config_json"])
    # Purge data FIRST: a crash mid-way leaves a re-deletable registry
    # record rather than orphaned, unreachable cache entries.
    deleted = await manager.apurge_scope(scope)
    registry.revoke_cache_keys(tenant_id, cache_id)
    registry.delete_cache(tenant_id, cache_id)
    return PurgeResponse(deleted=deleted)


# --------------------------------------------------------------------------- #
# Per-cache data endpoints
# --------------------------------------------------------------------------- #


@saas_router.post("/caches/{cache_id}/search", response_model=SearchResponse)
async def search_cache(
    cache_id: str,
    request: SearchRequest,
    tenant_id: str = Depends(require_cache_data_access),
    registry: SaaSRegistry = Depends(get_registry),
    pool: ManagerPool = Depends(get_manager_pool),
):
    info = _resolve_cache(registry, tenant_id, cache_id)
    scope = SaaSRegistry.data_scope(tenant_id, cache_id)
    manager = pool.get(tenant_id, cache_id, info["config_json"])
    result = await manager.asearch(request.query, scope=scope)
    if result:
        return SearchResponse(
            hit=True,
            response=result["response"],
            similarity=result.get("similarity"),
            metadata=result.get("metadata"),
        )
    return SearchResponse(hit=False)


@saas_router.post("/caches/{cache_id}/set", response_model=Dict[str, str])
async def set_cache(
    cache_id: str,
    request: SetRequest,
    tenant_id: str = Depends(require_cache_data_access),
    registry: SaaSRegistry = Depends(get_registry),
    pool: ManagerPool = Depends(get_manager_pool),
):
    info = _resolve_cache(registry, tenant_id, cache_id)
    scope = SaaSRegistry.data_scope(tenant_id, cache_id)
    manager = pool.get(tenant_id, cache_id, info["config_json"])
    await manager.aset(
        query=request.query,
        response=request.response,
        metadata=request.metadata,
        ttl=request.ttl,
        keep_forever=request.keep_forever,
        scope=scope,
    )
    return {"status": "success"}


@saas_router.delete("/caches/{cache_id}/entries", response_model=PurgeResponse)
async def purge_cache_entries(
    cache_id: str,
    tenant_id: str = Depends(require_cache_data_access),
    registry: SaaSRegistry = Depends(get_registry),
    pool: ManagerPool = Depends(get_manager_pool),
):
    info = _resolve_cache(registry, tenant_id, cache_id)
    scope = SaaSRegistry.data_scope(tenant_id, cache_id)
    manager = pool.get(tenant_id, cache_id, info["config_json"])
    deleted = await manager.apurge_scope(scope)
    return PurgeResponse(deleted=deleted)

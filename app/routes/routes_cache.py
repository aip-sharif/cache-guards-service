# routes_cache.py

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..utils.db_utils import (
    get_session,
    create_cache,
    get_cache,
    get_cache_by_project_id,
    get_cache_by_key,
    list_caches,
    update_cache,
    create_cache_config,
    get_cache_config,
    update_cache_config,
)
from ..models.schemas_cache import (
    CacheRegister,
    CacheEdit,
    CacheRead,
    APIResponse,
    GatewayConfigResponse,
    CacheModeConfig,
)
from ..auth import get_current_user_id, verify_gateway_admin_key
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter(prefix="/cache", tags=["cache"])
project_key_scheme = HTTPBearer()


def _build_cache_read(cache, config) -> CacheRead:
    data = cache.model_dump()
    guard = None
    cache_config = None
    if config:
        if config.guard_enabled:
            guard = {"enabled": config.guard_enabled, "policy": config.guard_policy}
        cache_config = {
            "cache_mode": config.cache_mode,
            "semantic": config.semantic,
            "bm25": config.bm25,
            "fuzzy": config.fuzzy,
        }
    data["guard"] = guard
    data["cache_config"] = cache_config
    return CacheRead(**data)


@router.post("/register", response_model=APIResponse)
async def register_cache(
    data: CacheRegister,
    db: Session = Depends(get_session),
    id_user: str = "x", #Depends(get_current_user_id),
):
    try:
        name = f"model_{uuid.uuid4().hex[:8]}"

        project_id = uuid.uuid4().hex
        api_key = f"sc-proj-{uuid.uuid4().hex}"

        cache = create_cache(
            db=db,
            id_user=id_user,
            name=name,
            embedd_model=data.embedd_model,
            embedd_key=data.embedd_key,
            llm_model=data.llm_model,
            llm_key=data.llm_key,
            extaractor=data.extaractor,
            extaractor_key=data.extaractor_key,
            extractor_domain=data.extractor_domain,
            cache_key=api_key,
            project_id=project_id,
        )

        cc = data.cache_config
        config = create_cache_config(
            db=db,
            cache_id=cache.id,
            guard_enabled=data.guard.enabled,
            guard_policy=data.guard.policy,
            cache_mode=cc.cache_mode if cc else None,
            semantic=cc.semantic.model_dump() if cc else None,
            bm25=cc.bm25.model_dump() if cc else None,
            fuzzy=cc.fuzzy.model_dump() if cc else None,
        )

        return APIResponse(
            status_code=201,
            message="Cache registered successfully",
            data= project_id,
            #data=_build_cache_read(cache, config),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register cache: {e}")


@router.put("/{cache_id}", response_model=APIResponse)
def edit_cache(
    cache_id: str,
    data: CacheEdit,
    db: Session = Depends(get_session),
    id_user: str ="x",# Depends(get_current_user_id),
):
    try:
        update_data = data.model_dump(exclude_unset=True, exclude={"guard", "cache_config"})

        updated = update_cache(db=db, id=cache_id, id_user=id_user, data=update_data)
        if not updated:
            raise HTTPException(status_code=404, detail="Cache not found")

        config = get_cache_config(db=db, cache_id=cache_id)

        config_update = {}
        if data.guard is not None:
            config_update["guard_enabled"] = data.guard.enabled
            config_update["guard_policy"] = data.guard.policy
        if data.cache_config is not None:
            config_update["cache_mode"] = data.cache_config.cache_mode
            config_update["semantic"] = data.cache_config.semantic.model_dump()
            config_update["bm25"] = data.cache_config.bm25.model_dump()
            config_update["fuzzy"] = data.cache_config.fuzzy.model_dump()

        if config_update and config:
            config = update_cache_config(db=db, cache_id=cache_id, data=config_update)

        return APIResponse(
            status_code=200,
            message="Cache updated successfully",
            data=_build_cache_read(updated, config),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update cache: {e}")

@router.get("/{project_id}", response_model=APIResponse)
def read_cache(
    project_id: str,
    db: Session = Depends(get_session),
    id_user: str = "x",#Depends(get_current_user_id),
    id: str= Depends(verify_gateway_admin_key)
):
    cache = get_cache_by_project_id(db=db, project_id=project_id)
    if not cache:
        raise HTTPException(status_code=404, detail="Cache not found")
    config = get_cache_config(db=db, cache_id=cache.id)
    return APIResponse(
        status_code=200,
        message="Cache fetched successfully",
        data=_build_cache_read(cache, config),
    )


@router.get("/", response_model=APIResponse)
def read_all_caches(
    db: Session = Depends(get_session),
    id_user: str = "x"#Depends(get_current_user_id),
):
    caches = list_caches(db=db, id_user=id_user)
    results = []
    for cache in caches:
        config = get_cache_config(db=db, cache_id=cache.id)
        results.append(_build_cache_read(cache, config))
    return APIResponse(
        status_code=200,
        message=f"{len(results)} cache(s) fetched successfully",
        data=results,
    )


@router.get("/gw/config", response_model=GatewayConfigResponse)
def gateway_config(
    db: Session = Depends(get_session),
    credentials: HTTPAuthorizationCredentials = Depends(project_key_scheme),
):
    cache = get_cache_by_key(db=db, cache_key=credentials.credentials)
    if not cache:
        raise HTTPException(status_code=403, detail="invalid project key")

    config = get_cache_config(db=db, cache_id=cache.id)

    return GatewayConfigResponse(
        model=cache.llm_model,
        model_api_key=cache.llm_key,
        embed_model=cache.embedd_model,
        embed_api_key=cache.embedd_key,
        extractor_model=cache.extaractor,
        extractor_api_key=cache.extaractor_key,
        extractor_domain=cache.extractor_domain,
        project_id=cache.project_id,
        cache_config=CacheModeConfig(
            cache_mode=config.cache_mode if config else ["exact", "bm25", "fuzzy", "semantic"],
            semantic=config.semantic if config else {"similarity_threshold": 0.92},
            bm25=config.bm25 if config else {"scorer": "BM25", "min_score": 1.0},
            fuzzy=config.fuzzy if config else {"distance": 2, "min_score": 0.5},
        ),
        guard=(
            {"enabled": config.guard_enabled, "policy": config.guard_policy}
            if config and config.guard_enabled
            else None
        ),
    )
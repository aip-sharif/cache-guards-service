# routes_cache.py

import logging
import uuid
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from sqlmodel import Session

from ..config import SC_EMBED_BASE_URL, SC_LLM_BASE_URL
from ..utils.db_utils import (
    get_session,
    create_cache,
    delete_cache,
    get_cache,
    get_cache_by_project_id,
    get_cache_by_key,
    list_caches,
    update_cache,
    create_cache_config,
    get_cache_config,
    update_cache_config,
)
from ..utils.guard_utils import resolve_guard
from ..utils.proxy_utils import forward_to_upstream
from ..models.schemas_cache import (
    CacheRegister,
    CacheEdit,
    CacheRead,
    APIResponse,
    GatewayConfigResponse,
    CacheModeConfig,
)
from ..auth import bearer_scheme, get_current_user_id, verify_service_key

logger = logging.getLogger("cache")

router = APIRouter(prefix="/cache", tags=["cache"])

_GUARD_ENABLED_POLICY_KEYS = {"enabled", "policy"}


def _guard_extra_fields(guard) -> dict:
    if guard is None:
        return {}
    data = guard.model_dump(exclude_unset=False)
    return {k: v for k, v in data.items() if k not in _GUARD_ENABLED_POLICY_KEYS}


def _build_cache_read(cache, config) -> CacheRead:
    data = cache.model_dump()
    guard = None
    cache_config = None
    if config:
        if config.guard_enabled:
            guard = {
                "enabled": config.guard_enabled,
                "policy": config.guard_policy,
                **(config.guard_config or {}),
            }
        cache_config = {
            "cache_mode": config.cache_mode,
            "semantic": config.semantic,
            "bm25": config.bm25,
            "fuzzy": config.fuzzy,
        }
    data["guard"] = guard
    data["cache_config"] = cache_config
    return CacheRead(**data)


def _build_gateway_config(cache, config) -> GatewayConfigResponse:
    guard_response = None
    if config and config.guard_enabled:
        guard_response = {
            "enabled": config.guard_enabled,
            "policy": config.guard_policy,
            **(config.guard_config or {}),
        }

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
        guard=guard_response,
    )


# ===========================================================
# ثابت‌ها اول ("/mine", "/key/{cache_key}") - قبل از "/{project_id}"
# [A01 FIX] ترتیب رجیستر شدن مهمه: FastAPI مسیرها رو به ترتیب تعریف
# چک می‌کنه، پس مسیر پارامتری {project_id} اگه زودتر بیاد هرچیزی
# (از جمله "mine"/"key") رو به‌عنوان project_id می‌قاپه.
# ===========================================================


# ---------------------------------------------------------
# 1) Register -> ساخت کش جدید (جدول cache) + تنظیمات (جدول cache_config)
# [S01 FIX] id_user دوباره از Casdoor گرفته می‌شه (نه ثابت "x")
# [A02 FIX] هر دو insert (cache + cache_config) داخل یه بلوک اتمیک -
# اگه دومی خطا بده، اولی هم rollback می‌شه (رکورد یتیم نمی‌مونه)
# [R05 FIX] پیام خطای داخلی (str(e)) دیگه مستقیم به کلاینت برنمی‌گرده
# ---------------------------------------------------------
@router.post("/register", response_model=APIResponse)
async def register_cache(
    data: CacheRegister,
    db: Session = Depends(get_session),
    id_user: str = Depends(get_current_user_id),
):
    resolved_guard = resolve_guard(data.guard, id_user)  # می‌تونه HTTPException(502) بندازه

    name = f"model_{uuid.uuid4().hex[:8]}"
    project_id = uuid.uuid4().hex
    api_key = f"sc-proj-{uuid.uuid4().hex}"

    try:
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
            commit=False,  # هنوز commit نکن - می‌خوایم با config یکجا باشه
        )

        cc = data.cache_config
        config = create_cache_config(
            db=db,
            cache_id=cache.id,
            guard_enabled=bool(resolved_guard.enabled) if resolved_guard else False,
            guard_policy=resolved_guard.policy if resolved_guard else None,
            guard_config=_guard_extra_fields(resolved_guard),
            cache_mode=cc.cache_mode if cc else None,
            semantic=cc.semantic.model_dump() if cc else None,
            bm25=cc.bm25.model_dump() if cc else None,
            fuzzy=cc.fuzzy.model_dump() if cc else None,
            commit=False,
        )

        db.commit()  # هر دو با هم، یا هیچ‌کدوم
        db.refresh(cache)
        db.refresh(config)

    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("register_cache failed for id_user=%s", id_user)
        raise HTTPException(status_code=500, detail="Failed to register cache")

    return APIResponse(
        status_code=201,
        message="Cache registered successfully",
        data=_build_cache_read(cache, config),
    )


# ---------------------------------------------------------
# 2) Edit
# [S01 FIX] id_user از Casdoor
# ---------------------------------------------------------
@router.put("/{cache_id}", response_model=APIResponse)
def edit_cache(
    cache_id: str,
    data: CacheEdit,
    db: Session = Depends(get_session),
    id_user: str = Depends(get_current_user_id),
):
    try:
        update_data = data.model_dump(exclude_unset=True, exclude={"guard", "cache_config"})

        updated = update_cache(db=db, id=cache_id, id_user=id_user, data=update_data)
        if not updated:
            raise HTTPException(status_code=404, detail="Cache not found")

        config = get_cache_config(db=db, cache_id=cache_id)

        config_update = {}
        if "guard" in data.model_fields_set:
            resolved_guard = resolve_guard(data.guard, id_user)
            config_update["guard_enabled"] = bool(resolved_guard.enabled) if resolved_guard else False
            config_update["guard_policy"] = resolved_guard.policy if resolved_guard else None
            config_update["guard_config"] = _guard_extra_fields(resolved_guard)

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
    except Exception:
        logger.exception("edit_cache failed for cache_id=%s id_user=%s", cache_id, id_user)
        raise HTTPException(status_code=500, detail="Failed to update cache")


# ---------------------------------------------------------
# 3آ) لیست کش‌های خودِ کاربر - مسیر ثابت "/mine"، قبل از "/{project_id}"
# [S01 FIX] id_user از Casdoor
# ---------------------------------------------------------
@router.get("/mine", response_model=APIResponse)
def read_all_caches(
    db: Session = Depends(get_session),
    id_user: str = Depends(get_current_user_id),
):
    caches = list_caches(db=db, id_user=id_user)
    results = [
        _build_cache_read(cache, get_cache_config(db=db, cache_id=cache.id))
        for cache in caches
    ]
    return APIResponse(
        status_code=200,
        message=f"{len(results)} cache(s) fetched successfully",
        data=results,
    )


# ---------------------------------------------------------
# 3ب) پیدا کردن یک کش بر اساس cache_key - فقط برای ادمین/gateway
# [S01 FIX] این دیگه بدون auth نیست - verify_gateway_admin_key اجباریه
# [S05 FIX] دیگه مقدار خودِ کلید لاگ نمی‌شه
# مسیر ثابت "/key/..."، قبل از "/{project_id}"
# ---------------------------------------------------------
@router.get("/key", response_model=APIResponse)
def read_cache_by_key(
    db: Session = Depends(get_session),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    _service: str = Depends(verify_service_key),
):
    if not credentials:
        raise HTTPException(status_code=401, detail="not_authenticated")
 
    cache = get_cache_by_key(db=db, cache_key=credentials.credentials)
    if not cache:
        raise HTTPException(status_code=404, detail="Cache not found")
 
    config = get_cache_config(db=db, cache_id=cache.id)
    result = _build_cache_read(cache, config).model_dump()
    result["status_code"] = 200
    result["message"] = "Cache fetched successfully"
    return result
    # return APIResponse(
    #     status_code=200,
    #     message="Cache fetched successfully",
    #     data=_build_cache_read(cache, config),
    # )


# ---------------------------------------------------------
# 4) Read -> خواندن یک کش بر اساس project_id (فقط خودِ صاحبش)
# [S01 FIX] id_user از Casdoor + چک مالکیت (get_cache_by_project_id
# همون‌جا فیلتر id_user رو هم اعمال می‌کنه - کاربر دیگه نمی‌تونه
# پروژه‌ی کاربر دیگه رو با حدس زدن project_id بخونه)
# این مسیر پارامتری باید بعد از همه‌ی مسیرهای ثابت بالا تعریف بشه
# ---------------------------------------------------------
@router.get("/{project_id}", response_model=APIResponse)
def read_cache(
    project_id: str,
    db: Session = Depends(get_session),
    id_user: str = Depends(get_current_user_id),
):
    cache = get_cache_by_project_id(db=db, project_id=project_id, id_user=id_user)
    if not cache:
        raise HTTPException(status_code=404, detail="Cache not found")
    config = get_cache_config(db=db, cache_id=cache.id)
    return APIResponse(
        status_code=200,
        message="Cache fetched successfully",
        data=_build_cache_read(cache, config),
    )


# ---------------------------------------------------------
# 5) Config endpoint -> این رو خودِ gateway صدا می‌زنه:
#    GET /cache (طبق AppConfigClient، URL ثابت، project_id توی
#    مسیر نمی‌ره - شناسایی فقط از روی Authorization Bearer)
# ---------------------------------------------------------
@router.get("", response_model=None)
def gateway_config(
    db: Session = Depends(get_session),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    if not credentials:
        raise HTTPException(status_code=401, detail="not_authenticated")

    cache = get_cache_by_key(db=db, cache_key=credentials.credentials)
    if not cache:
        raise HTTPException(status_code=403, detail="invalid project key")

    config = get_cache_config(db=db, cache_id=cache.id)
    return _build_gateway_config(cache, config)


# ---------------------------------------------------------
# 6) Proxy -> مستقیم به mlops endpoint خودِ همین پروژه فوروارد می‌کنه
# ---------------------------------------------------------
def _authenticate_project(db: Session, credentials: HTTPAuthorizationCredentials):
    if not credentials:
        raise HTTPException(status_code=401, detail="not_authenticated")
    cache = get_cache_by_key(db=db, cache_key=credentials.credentials)
    if not cache:
        raise HTTPException(status_code=403, detail="invalid project key")
    return cache


@router.post("/proxy/chat/completions")
async def proxy_chat_completions(
    payload: dict = Body(...),
    db: Session = Depends(get_session),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    cache = _authenticate_project(db, credentials)
    return await forward_to_upstream(
        base_url=SC_LLM_BASE_URL,
        path="/v1/chat/completions",
        model=cache.llm_model,
        api_key=cache.llm_key,
        payload=payload,
    )


@router.post("/proxy/embeddings")
async def proxy_embeddings(
    payload: dict = Body(...),
    db: Session = Depends(get_session),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    cache = _authenticate_project(db, credentials)
    return await forward_to_upstream(
        base_url=SC_EMBED_BASE_URL,
        path="/v1/embeddings",
        model=cache.embedd_model,
        api_key=cache.embedd_key,
        payload=payload,
    )
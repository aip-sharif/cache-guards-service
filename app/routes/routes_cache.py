# routes_cache.py

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ..utils.db_utils import (
    get_session,
    create_cache,
    get_cache,
    list_caches,
    update_cache,
    get_cache_by_project_id,
)
from ..utils.external_api import call_external_cache_api
from ..models.schemas_cache import CacheRegister, CacheEdit, CacheRead
from ..auth import get_current_user_id, verify_gateway_admin_key 

router = APIRouter(prefix="/cache", tags=["cache"])

_API_TO_DB_FIELD = {
    "extractor": "extaractor",
    "extractor_key": "extaractor_key",
}


def _to_db_fields(data: dict) -> dict:
    return {_API_TO_DB_FIELD.get(k, k): v for k, v in data.items()}


# ---------------------------------------------------------
# 1) Register
# ---------------------------------------------------------
@router.post("/register", response_model=CacheRead)
async def register_cache(
    data: CacheRegister,
    db: Session = Depends(get_session),
    #id_user: str = Depends(get_current_user_id),
): 
    cache = create_cache(
        db=db,
        id_user='id' ,
        name=data.name or f"model_{uuid.uuid4().hex[:8]}",
        embedd_model=data.embedd_model,
        embedd_key=data.embedd_key,
        llm_model=data.llm_model,
        llm_key=data.llm_key,
        extaractor=data.extractor,
        extaractor_key=data.extractor_key,
    )

    api_key, project_id = await call_external_cache_api(
        name = data.name
    )

    cache = update_cache(
        db=db,
        id=cache.id,
        id_user='id',
        data={"api_key": api_key, "project_id": project_id},
    )

    return cache


# ---------------------------------------------------------
# 2) Edit 
# ---------------------------------------------------------
@router.put("/{cache_id}", response_model=CacheRead)
async def edit_cache(
    cache_id: str,
    data: CacheEdit,
    db: Session = Depends(get_session),
    #id_user: str = Depends(get_current_user_id),
):
    updated = update_cache(
        db=db,
        id=cache_id,
        id_user='id',
        data=_to_db_fields(data.model_dump(exclude_unset=True)),  # pydantic v2
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Cache not found")
    api_key, project_id = await call_external_cache_api(
        embedd_model=updated.embedd_model,
        embedd_key=updated.embedd_key,
        llm_model=updated.llm_model,
        llm_key=updated.llm_key,
        extractor=updated.extaractor,
        extractor_key=updated.extaractor_key,
    )

    cache = update_cache(
        db=db,
        id=cache.id,
        id_user='id',
        data={"api_key": api_key, "project_id": project_id},
    )
    return updated




# ---------------------------------------------------------
# 3) Read 
# ---------------------------------------------------------
@router.get("/{project_id}", response_model=CacheRead)
def read_cache(
    project_id: str,
    db: Session = Depends(get_session),
    _: None = Depends(verify_gateway_admin_key)
):
    cache = get_cache_by_project_id(db=db, project_id=project_id)
    if not cache:
        raise HTTPException(status_code=404, detail="Cache not found")
    return cache


@router.get("/", response_model=list[CacheRead])
def read_all_caches(
    db: Session = Depends(get_session),
    id_user: str = Depends(get_current_user_id),
):
    return list_caches(db=db, id_user=id_user)
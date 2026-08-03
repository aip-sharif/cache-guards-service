# schemas_cache.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CacheRegister(BaseModel):
    """Body for POST /cache/register"""
    name: Optional[str] = None
    embedd_model: str
    embedd_key: str
    llm_model: str
    llm_key: str
    extractor: Optional[str]= None
    extractor_key: Optional[str] = None



class CacheEdit(BaseModel):
    """Body for PUT /cache/{id}  -- edit the 'register' fields"""
    name: Optional[str] = None
    embedd_model: Optional[str] = None
    embedd_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_key: Optional[str] = None
    extaractor: Optional[str] = None
    extaractor_key: Optional[str] = None



class CacheRead(BaseModel):
    """Response model for GET"""
    id: str
    id_user: str
    name: str
    embedd_model: str
    embedd_key: str
    llm_model: str
    llm_key: str
    api_key: Optional[str] = None
    project_id : Optional[str] = None
    extaractor: Optional[str] = None
    extaractor_key: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True  # pydantic v2 (orm_mode)

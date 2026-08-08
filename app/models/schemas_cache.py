# schemas_cache.py

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class GuardConfig(BaseModel):
    enabled: bool = False
    policy: Optional[str] = None


class SemanticConfig(BaseModel):
    similarity_threshold: float = 0.92


class Bm25Config(BaseModel):
    scorer: str = "BM25"
    min_score: float = 1.0


class FuzzyConfig(BaseModel):
    distance: int = 2
    min_score: float = 0.5


class CacheModeConfig(BaseModel):
    cache_mode: List[str] = ["exact", "bm25", "fuzzy", "semantic"]
    semantic: SemanticConfig = SemanticConfig()
    bm25: Bm25Config = Bm25Config()
    fuzzy: FuzzyConfig = FuzzyConfig()


class CacheRegister(BaseModel):
    """Body for POST /cache/register"""
    llm_model: str
    llm_key: str
    embedd_model: str
    embedd_key: str
    extaractor: Optional[str] = None
    extaractor_key: Optional[str] = None
    extractor_domain: Optional[str] = None
    guard: GuardConfig
    cache_config: Optional[CacheModeConfig] = None


class CacheEdit(BaseModel):
    """Body for PUT /cache/{id}  -- edit the 'register' fields"""
    llm_model: Optional[str] = None
    llm_key: Optional[str] = None
    embedd_model: Optional[str] = None
    embedd_key: Optional[str] = None
    extaractor: Optional[str] = None
    extaractor_key: Optional[str] = None
    extractor_domain: Optional[str] = None
    guard: Optional[GuardConfig] = None
    cache_config: Optional[CacheModeConfig] = None


class GatewayConfigResponse(BaseModel):

    model: str
    model_api_key: str
    embed_model: str
    embed_api_key: str
    extractor_model: Optional[str] = None
    extractor_api_key: Optional[str] = None
    extractor_domain: Optional[str] = None
    project_id: str
    cache_config: CacheModeConfig
    guard: Optional[dict] = None


class CacheRead(BaseModel):
    """Response model for GET"""
    id: str
    id_user: str
    name: str
    embedd_model: str
    embedd_key: str
    llm_model: str
    llm_key: str
    cache_key: Optional[str] = None
    project_id: Optional[str] = None
    extaractor: Optional[str] = None
    extaractor_key: Optional[str] = None
    extractor_domain: Optional[str] = None
    created_at: datetime
    guard_enabled: bool = False
    guard_policy: Optional[str] = None
    cache_config: Optional[CacheModeConfig] = None

    class Config:
        from_attributes = True  # pydantic v2 (orm_mode)
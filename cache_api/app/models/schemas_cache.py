# schemas_cache.py

from pydantic import BaseModel, model_validator
from typing import Optional, List, Any, Union
from datetime import datetime


class APIResponse(BaseModel):
    status_code: int
    message: str
    data: Optional[Any] = None


# ---------------------------------------------------------
# Guard 
# ---------------------------------------------------------
_GUARD_KNOWN_FIELDS = {
    "enabled", "policy",
    "embed_model", "embed_api_key", "embed_prefix_style",
    "judge_model", "judge_api_key", "judge_task_description", "judge_max_concurrency",
    "mode", "top_k", "min_similarity", "block_threshold", "allow_threshold",
    "judge_block_threshold", "judge_allow_threshold",
    "check_roles", "max_segments", "max_input_chars",
    "degrade_to_unguarded", "unavailable_response", "unavailable_refusal",
    "default_refusal", "log_turn_text",
}


class GuardConfig(BaseModel):
    enabled: Optional[bool] = None
    policy: Optional[str] = None

    embed_model: Optional[str] = None
    embed_api_key: Optional[str] = None
    embed_prefix_style: str = "none"  # "none" | "e5"

    judge_model: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_task_description: str = "a customer support assistant"
    judge_max_concurrency: int = 4

    mode: Optional[str] = None  # cascade | embedding-only | judge-only | max
    top_k: int = 8
    min_similarity: float = 0.60
    block_threshold: float = 0.85
    allow_threshold: float = 0.40
    judge_block_threshold: float = 0.60
    judge_allow_threshold: float = 0.40

    check_roles: List[str] = ["user", "system", "tool"]
    max_segments: int = 32
    max_input_chars: int = 16000

    degrade_to_unguarded: bool = False
    unavailable_response: str = "error"  # "error" | "refusal"
    unavailable_refusal: Optional[str] = None
    default_refusal: Optional[str] = None
    log_turn_text: bool = False

    class Config:
        extra = "allow"  

    @model_validator(mode="after")
    def _reject_unknown_fields(self):
        extra_fields = getattr(self, "model_extra", None) or {}
        unknown = [k for k in extra_fields.keys() if not k.startswith("x_")]
        if unknown:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=502,
                detail=f"Unknown guard field(s): {', '.join(unknown)}",
            )
        if "fail_open" in extra_fields:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=502,
                detail="guard.fail_open does not exist - use degrade_to_unguarded",
            )
        return self


class SemanticConfig(BaseModel):
    similarity_threshold: float = 0.92


class Bm25Config(BaseModel):
    scorer: str = "BM25"
    min_score: float = 1.0


class FuzzyConfig(BaseModel):
    distance: int = 2
    min_score: float = 0.5


class CacheModeConfig(BaseModel):
    cache_mode: Union[str, List[str]] = ["exact", "bm25", "fuzzy", "semantic"]
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
    guard: Optional[GuardConfig] = None
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
    guard: Optional[dict] = None
    cache_config: Optional[CacheModeConfig] = None

    class Config:
        from_attributes = True  # pydantic v2 (orm_mode)
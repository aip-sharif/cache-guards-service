# schemas_cache.py

from pydantic import BaseModel, model_validator
from typing import Optional, List, Any, Union
from datetime import datetime


class APIResponse(BaseModel):
    """پوششی یکدست برای همه‌ی جواب‌های موفق"""
    status_code: int
    message: str
    data: Optional[Any] = None


# ---------------------------------------------------------
# Guard - همه‌ی فیلدهای مجاز طبق APP_INTEGRATION.md §2b
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

    # مدل‌های guard - باید صریحاً داده بشن، fallback به embedd_model خودِ
    # کلاینت باعث embed_unreachable سمت gateway می‌شه (وقتی guard فعاله)
    embed_model: Optional[str] = None
    embed_api_key: Optional[str] = None
    embed_prefix_style: str = "none"  # "none" | "e5"

    judge_model: Optional[str] = None
    judge_api_key: Optional[str] = None
    judge_task_description: str = "a customer support assistant"
    judge_max_concurrency: int = 4

    # آستانه‌ها
    mode: str = "embedding-only"  # cascade | embedding-only | judge-only | max
    top_k: int = 8
    min_similarity: float = 0.60
    block_threshold: float = 0.85
    allow_threshold: float = 0.40
    judge_block_threshold: float = 0.60
    judge_allow_threshold: float = 0.40

    # چه چیزی چک بشه
    check_roles: List[str] = ["user", "system", "tool"]
    max_segments: int = 32
    max_input_chars: int = 16000

    # رفتار موقع خرابی
    degrade_to_unguarded: bool = False
    unavailable_response: str = "error"  # "error" | "refusal"
    unavailable_refusal: Optional[str] = None
    default_refusal: Optional[str] = None
    log_turn_text: bool = False

    class Config:
        extra = "allow"  # برای پذیرفتن x_* بدون خطا؛ چک واقعی در validator زیره

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

    @model_validator(mode="after")
    def _require_judge_when_mode_needs_it(self):
        # همون قانونی که gateway هم اعمال می‌کنه: اگه mode بتونه judge رو
        # صدا بزنه، judge_model و judge_api_key باید حتماً پر باشن -
        # اینجا زودتر (موقع register/edit) گیرش می‌ندازیم، نه بعداً سمت gateway.
        if self.mode in ("cascade", "judge-only", "max"):
            if not self.judge_model or not self.judge_api_key:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"guard.mode='{self.mode}' can invoke the judge, so both "
                        "guard.judge_model and guard.judge_api_key are required "
                        "(use mode='embedding-only' for a judge-free guard)"
                    ),
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
    """اختیاری - در صورت نبود، مقادیر پیش‌فرض استفاده می‌شن.
    cache_mode می‌تونه یک رشته (یک متد) یا لیست (زنجیره) باشه.

    این مدل عمداً سخت‌گیر نیست چون برای ساختن جواب از داده‌ی ذخیره‌شده هم
    استفاده می‌شه؛ اعتبارسنجی ورودی توی CacheModeConfigInput پایینه."""
    enabled: bool = True
    # [F9 FIX] "off" از لیست پیش‌فرض حذف شد - gateway لیستِ دارای "off" رو رد می‌کنه
    cache_mode: Union[str, List[str]] = ["exact", "bm25", "fuzzy", "semantic"]
    semantic: SemanticConfig = SemanticConfig()
    bm25: Bm25Config = Bm25Config()
    fuzzy: FuzzyConfig = FuzzyConfig()


# ---------------------------------------------------------
# [F8 FIX] همون قوانینی که gateway موقع chat اعمال می‌کنه
# (cache_as_service/semantic_cache/core/config.py) - اینجا موقع
# register/edit چک می‌شن تا 201 نده و بعد اولین chat با 502 شکست بخوره.
# ---------------------------------------------------------
CACHE_MODES = {"semantic", "bm25", "fuzzy", "exact", "off"}
LEXICAL_SCORERS = {"BM25", "BM25STD", "TFIDF", "TFIDF.DOCNORM", "DISMAX", "DOCSCORE"}


class CacheModeConfigInput(CacheModeConfig):
    """cache_config ورودیِ register/edit - با اعتبارسنجی کامل"""

    @model_validator(mode="after")
    def _validate_against_gateway_rules(self):
        errors = []

        mode = self.cache_mode
        if isinstance(mode, str):
            if mode not in CACHE_MODES:
                errors.append(f"cache_mode must be one of {sorted(CACHE_MODES)}, got {mode!r}")
        else:
            if not mode:
                errors.append("cache_mode list must be non-empty")
            bad = [m for m in mode if m not in CACHE_MODES]
            if bad:
                errors.append(f"unknown cache_mode value(s) {bad}; allowed: {sorted(CACHE_MODES)}")
            if "off" in mode:
                errors.append("cache_mode list cannot contain 'off' - use cache_mode='off' to disable caching")

        threshold = self.semantic.similarity_threshold
        if threshold is None or not 0.0 <= threshold <= 1.0:
            errors.append("semantic.similarity_threshold must be between 0 and 1")

        distance = self.fuzzy.distance
        if distance is None or not 1 <= distance <= 3:
            errors.append("fuzzy.distance must be between 1 and 3")

        if self.bm25.scorer is None or self.bm25.scorer.strip().upper() not in LEXICAL_SCORERS:
            errors.append(f"bm25.scorer must be one of {sorted(LEXICAL_SCORERS)}")
        else:
            self.bm25.scorer = self.bm25.scorer.strip().upper()

        if errors:
            raise ValueError("; ".join(errors))
        return self


class CacheRegister(BaseModel):
    """Body for POST /cache/register"""
    llm_model: str
    llm_key: str
    # [F1 FIX] پروژه‌ی فقط-LLM (کش خاموش) مدل embedding لازم نداره
    embedd_model: Optional[str] = None
    embedd_key: Optional[str] = None
    extaractor: Optional[str] = None
    extaractor_key: Optional[str] = None
    extractor_domain: Optional[str] = None
    guard: Optional[GuardConfig] = None
    cache_config: Optional[CacheModeConfigInput] = None


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
    cache_config: Optional[CacheModeConfigInput] = None


class GatewayConfigResponse(BaseModel):
    """
    فرمت دقیقی که gateway از GET /cache انتظار داره -
    از روی Cache + CacheConfig ساخته می‌شه.
    """
    model: str
    model_api_key: str
    # [F1 FIX] برای پروژه‌ی فقط-LLM خالیه (null)، نه رشته‌ی خالی
    embed_model: Optional[str] = None
    embed_api_key: Optional[str] = None
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
    # از جدول جدای CacheConfig پر می‌شه؛ اگه غیرفعال باشه null است
    guard: Optional[dict] = None
    cache_config: Optional[CacheModeConfig] = None

    class Config:
        from_attributes = True  # pydantic v2 (orm_mode)
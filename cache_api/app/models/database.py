from sqlmodel import SQLModel, Field, Column, JSON
from random import random
from datetime import datetime, timezone
from typing import List, Optional
from ..utils.crypto_utils import EncryptedString, SecretFieldsJSON

# کلیدهای provider داخل guard_config که باید رمز بشن
GUARD_SECRET_KEYS = ("embed_api_key", "judge_api_key")
DEFAULT_CACHE_MODE = ["exact", "bm25", "fuzzy", "semantic"]

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    access: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class Cache(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    name: str = Field(default_factory=lambda: f'model_{random()}')
    id_user: str = Field(default=None, foreign_key="user.id", nullable=False, index=True)

    embedd_model: str = Field(default=None, nullable=False)
    # [S04 FIX] کلیدهای provider حالا encrypt-at-rest هستن (EncryptedString)
    embedd_key: str = Field(default=None, sa_column=Column(EncryptedString, nullable=False))
    llm_model: str = Field(default=None, nullable=False)
    llm_key: str = Field(default=None, sa_column=Column(EncryptedString, nullable=False))

    # [R04 FIX] cache_key و project_id قبلاً nullable و بدون unique/index
    # بودن - یعنی تئوریاً می‌شد دو ردیف با همون project_id/cache_key
    # ساخت (تصادم lookup) یا NULL گذاشت. الان هم اجباری (nullable=False)
    # هم یکتا (unique=True) و ایندکس‌دار هستن (lookup سریع‌تر هم می‌شه).
    cache_key: str = Field(default=None, nullable=False, unique=True, index=True)
    project_id: str = Field(default=None, nullable=False, unique=True, index=True)

    extaractor: str = Field(default=None, nullable=True)
    # [F5 FIX] قبلاً plaintext بود؛ مقدار قدیمی هنوز خونده می‌شه (fallback
    # داخل EncryptedString) و در اولین ذخیره‌ی بعدی رمز می‌شه
    extaractor_key: Optional[str] = Field(
        default=None, sa_column=Column(EncryptedString, nullable=True)
    )
    extractor_domain: str = Field(default=None, nullable=True)

    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class CacheConfig(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    cache_id: str = Field(default=None, foreign_key="cache.id", nullable=False, unique=True, index=True)

    enabled: bool = Field(default=True, nullable=False)

    guard_enabled: bool = Field(default=False, nullable=False)
    guard_policy: str = Field(default=None, nullable=True)
    # [F5 FIX] embed_api_key / judge_api_key داخل این JSON رمز می‌شن
    guard_config: dict = Field(
        default_factory=dict, sa_column=Column(SecretFieldsJSON(GUARD_SECRET_KEYS))
    )

    # [F9 FIX] پیش‌فرض قبلی "off" رو توی لیست داشت که gateway ردش می‌کنه
    cache_mode: object = Field(
        default_factory=lambda: list(DEFAULT_CACHE_MODE),
        sa_column=Column(JSON),
    )
    semantic: dict = Field(
        default_factory=lambda: {"similarity_threshold": 0.92},
        sa_column=Column(JSON),
    )
    bm25: dict = Field(
        default_factory=lambda: {"scorer": "BM25", "min_score": 1.0},
        sa_column=Column(JSON),
    )
    fuzzy: dict = Field(
        default_factory=lambda: {"distance": 2, "min_score": 0.5},
        sa_column=Column(JSON),
    )

    created_at: datetime = Field(default_factory=utc_now, nullable=False)
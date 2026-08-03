from sqlmodel import SQLModel, Field
from random import random
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    access: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class Cache(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    name: str = Field(default_factory=lambda: f'model_{random()}')
    id_user: str = Field(default=None, foreign_key="user.id", nullable=False)

    embedd_model: str = Field(default=None, nullable=False)
    embedd_key: str = Field(default=None, nullable=False)
    llm_model: str = Field(default=None, nullable=False)
    llm_key: str = Field(default=None, nullable=False)
    extaractor: str = Field(default=None, nullable=True)
    extaractor_key: str = Field(default=None, nullable=True)

    project_id: str = Field(default=None, nullable=True)
    api_key: str = Field(default=None, nullable=True)

    created_at: datetime = Field(default_factory=utc_now, nullable=False)
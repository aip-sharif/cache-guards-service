"""baseline schema: user, cache, cacheconfig

Revision ID: 0000_baseline
Revises:
Create Date: 2026-09-29

[F4 FIX] قبلاً تنها migration فقط ستون cacheconfig.enabled رو اضافه می‌کرد و
هیچ جدولی نمی‌ساخت، پس `alembic upgrade head` روی Postgres خالی شکست می‌خورد.
این baseline جدول‌ها رو با schema قبل از ستون enabled می‌سازه (0001 اون رو
اضافه می‌کنه). جدولی که از قبل هست (مثلاً با create_all توی dev ساخته شده)
دست نمی‌خوره، پس روی دیتابیس‌های موجود هم امن اجرا می‌شه.
"""
from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = "0000_baseline"
down_revision = None
branch_labels = None
depends_on = None


def _existing_tables() -> set:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    existing = _existing_tables()

    if "user" not in existing:
        op.create_table(
            "user",
            sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("access", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    if "cache" not in existing:
        op.create_table(
            "cache",
            sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("id_user", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("embedd_model", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            # EncryptedString -> روی دیتابیس یه String معمولیه (Fernet token)
            sa.Column("embedd_key", sa.String(), nullable=False),
            sa.Column("llm_model", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("llm_key", sa.String(), nullable=False),
            sa.Column("cache_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("project_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("extaractor", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("extaractor_key", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("extractor_domain", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["id_user"], ["user.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_cache_id_user", "cache", ["id_user"], unique=False)
        op.create_index("ix_cache_cache_key", "cache", ["cache_key"], unique=True)
        op.create_index("ix_cache_project_id", "cache", ["project_id"], unique=True)

    if "cacheconfig" not in existing:
        op.create_table(
            "cacheconfig",
            sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("cache_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("guard_enabled", sa.Boolean(), nullable=False),
            sa.Column("guard_policy", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("guard_config", sa.JSON(), nullable=True),
            sa.Column("cache_mode", sa.JSON(), nullable=True),
            sa.Column("semantic", sa.JSON(), nullable=True),
            sa.Column("bm25", sa.JSON(), nullable=True),
            sa.Column("fuzzy", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["cache_id"], ["cache.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_cacheconfig_cache_id", "cacheconfig", ["cache_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_cacheconfig_cache_id", table_name="cacheconfig")
    op.drop_table("cacheconfig")
    op.drop_index("ix_cache_project_id", table_name="cache")
    op.drop_index("ix_cache_cache_key", table_name="cache")
    op.drop_index("ix_cache_id_user", table_name="cache")
    op.drop_table("cache")
    op.drop_table("user")

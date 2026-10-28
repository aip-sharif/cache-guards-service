"""add enabled column to cacheconfig

Revision ID: 0001_cacheconfig_enabled
Revises: 0000_baseline
Create Date: 2026-09-23

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = "0001_cacheconfig_enabled"
down_revision = "0000_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # دیتابیسی که با create_all (dev) ساخته شده ممکنه ستون رو از قبل داشته باشه
    columns = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("cacheconfig")}
    if "enabled" in columns:
        return
    # server_default=true تا ردیف‌های موجود هم مقدار true بگیرن (NOT NULL)
    op.add_column(
        "cacheconfig",
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    op.drop_column("cacheconfig", "enabled")

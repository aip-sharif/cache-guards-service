"""add enabled column to cacheconfig

Revision ID: 0001_cacheconfig_enabled
Revises:
Create Date: 2026-09-23

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision = "0001_cacheconfig_enabled"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # server_default=true تا ردیف‌های موجود هم مقدار true بگیرن (NOT NULL)
    op.add_column(
        "cacheconfig",
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    op.drop_column("cacheconfig", "enabled")

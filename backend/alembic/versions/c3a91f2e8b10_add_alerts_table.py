"""Add alerts table

Revision ID: c3a91f2e8b10
Revises: b110076d49ad
Create Date: 2026-07-22
"""
from alembic import op
import sqlalchemy as sa

revision = "c3a91f2e8b10"
down_revision = "b110076d49ad"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("client_key", sa.String(), nullable=False),
        sa.Column("ticker", sa.String(), nullable=False),
        sa.Column("alert_type", sa.String(), nullable=True),
        sa.Column("condition", sa.String(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=True),
        sa.Column("triggered", sa.Boolean(), nullable=True),
        sa.Column("last_triggered_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_alerts_client_key", "alerts", ["client_key"])
    op.create_index("ix_alerts_ticker", "alerts", ["ticker"])
    op.create_index("ix_alerts_id", "alerts", ["id"])


def downgrade() -> None:
    op.drop_index("ix_alerts_id", table_name="alerts")
    op.drop_index("ix_alerts_ticker", table_name="alerts")
    op.drop_index("ix_alerts_client_key", table_name="alerts")
    op.drop_table("alerts")

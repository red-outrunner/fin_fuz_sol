"""Persist published daily stock selections.

Revision ID: d72e14a0c931
Revises: c3a91f2e8b10
"""
from alembic import op
import sqlalchemy as sa

revision = "d72e14a0c931"
down_revision = "c3a91f2e8b10"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("daily_stock_picks",
                    sa.Column("date", sa.String(10), primary_key=True),
                    sa.Column("ticker", sa.String(64), nullable=False),
                    sa.Column("name", sa.String(), nullable=False))


def downgrade():
    op.drop_table("daily_stock_picks")

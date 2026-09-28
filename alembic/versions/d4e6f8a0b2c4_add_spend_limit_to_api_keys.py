"""add spend_limit_usd to api_keys

Revision ID: d4e6f8a0b2c4
Revises: b2c4d6e8f0a2
Create Date: 2026-09-27 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = 'd4e6f8a0b2c4'
down_revision: str | None = 'b2c4d6e8f0a2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Teto de gasto em USD por chave (None = sem limite); aplicado por acumulador
    # de estimated_cost no enforce_spend_cap (429 insufficient_quota).
    op.add_column(
        'api_keys',
        sa.Column('spend_limit_usd', sa.Numeric(12, 2), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('api_keys', 'spend_limit_usd')
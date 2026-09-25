"""add status and error_type to usage_logs

Revision ID: b2c4d6e8f0a2
Revises: c1b13c0c82d4
Create Date: 2026-09-25 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = 'b2c4d6e8f0a2'
down_revision: str | None = 'c1b13c0c82d4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `status` ('ok'/'error') e `error_type` dão à UI de logs o verde/vermelho.
    # registros antigos entram como 'ok'.
    op.add_column(
        'usage_logs',
        sa.Column('status', sa.String(length=16), nullable=False, server_default='ok'),
    )
    op.add_column(
        'usage_logs',
        sa.Column('error_type', sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('usage_logs', 'error_type')
    op.drop_column('usage_logs', 'status')
"""Usage and metrics schemas: token counts, latency, estimated cost, and provider attribution."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class UsageRecord(BaseModel):
    """Per-request usage record; mirrors the usage_logs table for persistence."""

    virtual_key_id: int
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    estimated_cost: Decimal = Decimal("0")

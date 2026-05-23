"""Resolver output contract."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ResolverResponse(BaseModel):
    """Structured response returned by the resolver."""
    customer_reply: str = Field(description="The empathetic text sent to the customer.")
    citations: list[str] = Field(
        default_factory=list,
        description="List of KB article IDs used to verify claims in the reply (e.g., ['kb-003']).",
    )
    requires_financial_action: bool = Field(
        description="True if a refund, cancellation, or store credit is proposed."
    )
    hitl_action: Optional[Literal["REFUND", "CANCELLATION", "STORE_CREDIT"]] = Field(
        default=None,
        description="Required when requires_financial_action is true.",
    )
    hitl_amount: Optional[float] = Field(
        default=None,
        description="Refund/credit amount when proposing a financial action.",
    )
    hitl_justification: Optional[str] = Field(
        default=None,
        description="One-line grounding proof for the proposed action.",
    )

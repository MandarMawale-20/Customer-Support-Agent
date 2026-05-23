"""Resolver output contract for structured responses."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ResolverResponse(BaseModel):
    """Structured response returned by the resolver."""
    customer_reply: str = Field(description="The reply to send to the customer.")
    requires_hitl: bool = Field(
        description="True if a financial action like refund, cancellation, or credit is needed."
    )
    hitl_action: Optional[Literal["REFUND", "CANCELLATION", "STORE_CREDIT", "NONE"]] = Field(
        default="NONE"
    )
    hitl_amount: Optional[str] = Field(default=None)
    hitl_justification: Optional[str] = Field(default=None)

"""Human-in-the-loop trace model."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HITLTrace(BaseModel):
    """Capture financial-action approval state."""
    triggered: bool
    action_type: Optional[Literal["REFUND", "CANCELLATION", "STORE_CREDIT", "NONE"]] = None
    justification: Optional[str] = Field(
        default=None,
        description="One-line grounding proof for the proposed financial action.",
    )
    human_decision: Optional[Literal["APPROVED", "REJECTED"]] = None

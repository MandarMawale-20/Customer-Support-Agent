"""Human-in-the-loop trace model."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class HITLTrace(BaseModel):
    """Record human approval checkpoints."""
    triggered: bool
    action_type: Literal["REFUND", "CANCELLATION", "STORE_CREDIT", "NONE"]
    justification: Optional[str] = None
    human_decision: Optional[Literal["APPROVED", "REJECTED"]] = None

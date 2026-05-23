"""Execution trace for each ticket."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel

from .hitl_trace import HITLTrace
from .tool_call_trace import ToolCallTrace


class TicketExecutionTrace(BaseModel):
    """Audit record for one ticket run."""
    ticket_id: str
    classification: Literal["informational", "order_specific", "escalation"]
    sub_tasks: List[str]
    decision_trace: List[str]
    tool_calls: List[ToolCallTrace]
    hitl: HITLTrace
    workflow_status: Literal["COMPLETED", "FAILED", "ESCALATED", "WAITING_HUMAN"]
    resolution_type: Literal[
        "INFO_RESPONSE",
        "ORDER_UPDATE",
        "REFUND_REQUEST",
        "ESCALATED",
    ]
    citations_used: Optional[List[str]] = None

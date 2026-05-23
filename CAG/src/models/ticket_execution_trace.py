"""End-to-end execution trace for a ticket run."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .hitl_trace import HITLTrace
from .tool_call_trace import ToolCallTrace


class TicketExecutionTrace(BaseModel):
    """Persist the full workflow outcome for one ticket."""
    ticket_id: str
    classification: Literal["informational", "order_specific", "escalation"]
    sub_tasks: list[str] = Field(default=[])
    decision_trace: list[str] = Field(default=[])
    tool_calls: list[ToolCallTrace] = Field(default=[])
    citations_used: list[str] = Field(default=[])
    resolver_notes: Optional[str] = None
    hitl: HITLTrace
    workflow_status: Literal["COMPLETED", "WAITING_HUMAN", "ESCALATED", "FAILED"]
    resolution_type: Literal[
        "INFO_RESPONSE", "ORDER_UPDATE", "REFUND_REQUEST", "SECURITY_ESCALATION", "ESCALATED"
    ]


class RunTrace(BaseModel):
    """Serialized record for a full workflow run."""
    ticket_id: str
    final_response: Optional[str]
    execution_trace: TicketExecutionTrace
    saved_at: str

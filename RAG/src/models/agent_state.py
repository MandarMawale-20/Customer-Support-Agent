"""Shared LangGraph state."""
from __future__ import annotations

from typing import Optional, TypedDict

from .planner_contract import PlannerResponse
from .tool_call_trace import ToolCallTrace
from .ticket_execution_trace import TicketExecutionTrace


class AgentState(TypedDict):
    """State passed between workflow nodes."""
    ticket: dict
    planner_output: PlannerResponse
    retrieved_docs: str
    order_data: Optional[dict]
    order_tool_trace: Optional[ToolCallTrace]
    final_response: Optional[str]
    requires_hitl: bool
    execution_trace: TicketExecutionTrace
    guardrail_attempts: int
    guardrail_feedback: Optional[str]

"""Shared LangGraph state."""
from __future__ import annotations

from typing import Optional, TypedDict

from .planner_contract import PlannerResponse
from .ticket_execution_trace import TicketExecutionTrace


class AgentState(TypedDict):
    """State passed between workflow nodes."""
    ticket: dict
    planner_output: PlannerResponse
    retrieved_docs: str
    order_data: Optional[dict]
    order_lookup_status: Optional[str]
    order_lookup_error: Optional[str]
    final_response: Optional[str]
    requires_hitl: bool
    execution_trace: TicketExecutionTrace

"""Shared LangGraph state."""

from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """State passed between workflow nodes."""
    ticket: dict
    planner_output: Any
    retrieved_docs: str
    order_data: Optional[dict]
    order_lookup_status: Optional[str]
    order_lookup_error: Optional[str]
    final_response: Optional[str]
    requires_hitl: bool
    execution_trace: Any

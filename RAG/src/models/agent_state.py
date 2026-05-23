"""Shared LangGraph state."""

from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """State passed between workflow nodes."""
    ticket: dict
    planner_output: Any
    retrieved_docs: str
    order_data: Optional[dict]
    order_tool_trace: Optional[Any]
    final_response: Optional[str]
    requires_hitl: bool
    execution_trace: Any
    guardrail_attempts: int
    guardrail_feedback: Optional[str]

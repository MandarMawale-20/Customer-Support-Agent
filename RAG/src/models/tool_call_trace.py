"""Tool invocation trace model."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ToolCallTrace(BaseModel):
    """Record how a tool was selected and what it returned."""
    tool_name: Literal["kb_search", "get_order_status", "escalate"]
    input_args: dict
    status: Literal["SUCCESS", "NO_RESULTS", "404_NOT_FOUND", "SERVICE_UNAVAILABLE", "TIMEOUT", "ERROR"]
    retry_count: int = Field(default=0)
    reason: str = Field(description="Why the planner chose this tool.")
    output_summary: str = Field(description="Short summary of what the tool returned.")

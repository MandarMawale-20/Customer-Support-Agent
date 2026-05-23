"""Planner output contract and KB article IDs."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


KB_ARTICLE_IDS = Literal[
    "kb-001", "kb-002", "kb-003", "kb-004", "kb-005",
    "kb-006", "kb-007", "kb-008", "kb-009", "kb-010",
    "kb-011", "kb-012", "kb-013", "kb-014", "kb-015",
    "kb-016", "kb-017", "kb-018",
]


class PlannerResponse(BaseModel):
    """Structured routing plan returned by the planner."""
    classification: Literal["informational", "order_specific", "escalation"] = Field(
        description="Core resolution path for the ticket."
    )
    sub_tasks: list[str] = Field(
        description="Step-by-step internal plan to resolve the issue."
    )
    requires_kb_search: bool = Field(
        description="True if KB policy lookup is needed."
    )
    kb_article_ids: list[KB_ARTICLE_IDS] = Field(
        default=[],
        description="Specific KB article IDs to load. Empty if requires_kb_search is false.",
    )
    requires_order_lookup: bool = Field(
        description="True if live order status is needed from the API."
    )
    extracted_order_id: Optional[str] = Field(
        default=None,
        description="Extracted ORD-XXXXX from ticket body. Null if not present.",
    )
    immediate_escalation_reason: Optional[str] = Field(
        default=None,
        description="Required when classification is 'escalation'. Clear reason why.",
    )

"""Planner output contract for the RAG agent."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


KB_CATEGORY = Literal[
    "shipping_tracking",
    "returns_refunds",
    "warranty",
    "cancellations",
    "order_changes",
    "payments",
    "account_access",
    "international",
    "gift_cards",
    "sizing_fit",
    "subscriptions",
    "promo_codes",
    "product_care",
    "store_credit",
    "contact_support",
    "general",
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
    kb_queries: list[str] = Field(
        default=[],
        description="One or more search queries derived from the ticket sub-tasks.",
    )
    kb_category: Optional[KB_CATEGORY] = Field(
        default=None,
        description="Optional category filter to prevent off-topic retrieval.",
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

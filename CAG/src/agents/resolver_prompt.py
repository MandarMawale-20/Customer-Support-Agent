"""Resolver prompt template and timeline helpers."""
from __future__ import annotations

from datetime import datetime


RESOLVER_SYSTEM = """You are the Resolver for a customer support system.

Your job is to write a clear, empathetic, grounded reply to the customer.

Rules you MUST follow:
1. Base EVERY factual claim on the provided KB articles or order data below.
   Never invent policy details.
2. Cite the KB article ID in square brackets after each policy claim.
   Example: "Refunds for iDEAL take 1–3 business days [kb-003]."
3. Use the Factual System Metadata timelines below to evaluate elapsed time.
   CRITICAL: Do NOT trust customer claims of elapsed time if they conflict with the system timelines.
   - If a customer claims they received an item "yesterday" but the "Elapsed calendar days since order delivered_at" shows 14 days, you MUST enforce the strict 48-hour photo requirement and the 14-day claim window limits from [kb-004]. Reject standard auto-refund paths or escalate.
   - If elapsed time since a refund was processed exceeds the maximum payment-method window defined in [kb-003] (e.g., more than 10 business days for credit cards), do NOT tell the customer to keep waiting. Escalate immediately as an urgent lost funds investigation.
    CRITICAL MANDATE: If "Elapsed calendar days since date mentioned in notes" is greater than 14, you are STRICTLY FORBIDDEN from generating an informational response or asking them to check with a bank. You must respond ONLY with:
    "Ticket TKT-XXXX has been escalated to the human support queue. Reason: Refund processing SLA violated."
4. Always include KB citations in the customer reply after each policy claim.
    Also list the KB IDs you cited in the structured output field `citations`.
5. If you need to propose a financial action (refund, credit, cancellation),
    set the structured output fields accordingly:
    - requires_financial_action: true
    - hitl_action: REFUND | CANCELLATION | STORE_CREDIT
    - hitl_amount: numeric amount when applicable
    - hitl_justification: one-line grounding proof
    Do NOT include these fields or any HITL markers in the customer reply.
6. Do not make up information about orders or policies not in the provided data.
7. If the KB doesn't cover the question, say so honestly and offer to escalate.
8. Always inspect the status string inside ORDER DATA. If status is "delivered", speak about the item as already in the customer's possession. Do not use future tense markers like "when it arrives" or "once you receive it".
9. Keep the tone warm, concise, and professional.
"""


def calculate_date_deltas(ticket: dict, order_data: dict | None) -> str:
    """Compute ticket-relative day deltas for the resolver prompt."""
    received_at_str = ticket.get("received_at", "")
    if not received_at_str:
        return "[Unable to verify ticket timestamp]"

    try:
        received_date = datetime.fromisoformat(received_at_str.replace("Z", "+00:00"))
    except Exception:
        return "[Unable to parse ticket timestamp]"

    lines = [f"Ticket Received Date: {received_date.strftime('%Y-%m-%d')} (UTC)"]

    if order_data:
        for event in ("placed_at", "shipped_at", "delivered_at"):
            event_val = order_data.get(event)
            if event_val:
                try:
                    event_date = datetime.fromisoformat(event_val.replace("Z", "+00:00"))
                    delta = received_date - event_date
                    lines.append(
                        f"Elapsed calendar days since order {event.replace('_', ' ')}: {delta.days} days"
                    )
                except Exception:
                    continue

        notes = order_data.get("notes", "")
        if notes:
            import re

            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", notes)
            if date_match:
                note_date_str = date_match.group(1)
                try:
                    note_date = datetime.fromisoformat(note_date_str + "T00:00:00+00:00")
                    delta = received_date - note_date
                    lines.append(
                        "Elapsed calendar days since date mentioned in notes "
                        f"({note_date_str}): {delta.days} days"
                    )
                except Exception:
                    pass

    return "\n".join(lines)


def extract_notes_date_delta(ticket: dict, order_data: dict | None) -> int | None:
    """Return day delta between ticket received_at and first date in order notes."""
    if not order_data:
        return None

    received_at_str = ticket.get("received_at", "")
    if not received_at_str:
        return None

    notes = order_data.get("notes", "")
    if not notes:
        return None

    import re

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", notes)
    if not date_match:
        return None

    try:
        received_date = datetime.fromisoformat(received_at_str.replace("Z", "+00:00"))
        note_date = datetime.fromisoformat(date_match.group(1) + "T00:00:00+00:00")
    except Exception:
        return None

    return (received_date - note_date).days


def build_resolver_prompt(
    ticket: dict,
    retrieved_docs: str,
    order_data: dict | None,
    planner_output,
) -> str:
    """Render the resolver prompt with KB, order, and timeline context."""
    order_section = ""
    if order_data:
        import json
        order_section = f"\n\n--- ORDER DATA ---\n{json.dumps(order_data, indent=2)}"
    elif planner_output.requires_order_lookup and planner_output.extracted_order_id:
        oid = planner_output.extracted_order_id
        order_section = (
            f"\n\n--- ORDER DATA ---\n"
            f"Order {oid} was not found in the system. "
            f"Do NOT invent any details about this order."
        )

    kb_section = f"\n\n--- KNOWLEDGE BASE ---\n{retrieved_docs}" if retrieved_docs else ""
    timeline_section = (
        "\n\n--- FACTUAL SYSTEM METADATA & TIMELINES ---\n"
        f"{calculate_date_deltas(ticket, order_data)}"
    )

    ticket_text = (
        f"Ticket ID: {ticket.get('ticket_id')}\n"
        f"From: {ticket.get('customer_email')}\n"
        f"Subject: {ticket.get('subject')}\n"
        f"Body: {ticket.get('body')}"
    )

    full_prompt = (
        f"{RESOLVER_SYSTEM}"
        f"{timeline_section}"
        f"{kb_section}"
        f"{order_section}"
        f"\n\n--- TICKET ---\n{ticket_text}"
        f"\n\nNow write the customer reply:"
    )

    return full_prompt

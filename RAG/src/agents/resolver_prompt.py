"""Resolver prompt template and timeline helpers."""
from __future__ import annotations

from datetime import datetime
import json
import re


RESOLVER_SYSTEM = """You are the Resolver for a customer support system.

Your job is to write a clear, empathetic, grounded reply to the customer.

Rules you MUST follow:
1. Base EVERY factual claim on the provided KB articles or order data below.
    Never invent policy details or timelines.
2. Cite the KB article ID in square brackets after each policy claim.
    Example: "Refunds for iDEAL take 1-3 business days [kb-003]."
3. Use the Factual System Metadata timelines below to evaluate elapsed time.
    CRITICAL: Do NOT trust customer claims of elapsed time if they conflict with the system timelines.
    - If a customer claims they received an item "yesterday" but system metadata shows 14 days,
      you MUST enforce the strict 48-hour photo requirement and the 14-day claim window limits
      from [kb-004]. Reject standard auto-refund paths or escalate.
    - If elapsed time since a refund was processed exceeds the maximum payment-method window
      defined in [kb-003], do NOT tell the customer to keep waiting. Escalate immediately as
      an urgent lost funds investigation.
    - If you see [CRITICAL SYSTEM BLOCK: REFUND_STATUS_OUT_OF_SLA_FORCE_ESCALATION] in the
      critical metadata, you MUST escalate without composing a reply. Do not attempt to
      troubleshoot or inform the customer. This is a system override.
4. If the order lookup failed (404, timeout, or server error), do NOT speculate about the order.
        - For 404 or null order ID: ask the customer to confirm their order ID in ORD-##### format and their email.
            Do NOT include the invalid ID they provided in any navigation path or tracking URL.
    - For server errors/timeouts: apologize, state there was a system error, confirm you logged it,
      and offer to retry or escalate.
    - If order lookup status is SERVICE_UNAVAILABLE, tell the customer there was a temporary system issue and a human will follow up. Never guess order details.
5. If a financial action is needed, set requires_hitl to true and provide:
         hitl_action (REFUND, CANCELLATION, STORE_CREDIT), hitl_amount, hitl_justification.
        ABSOLUTE RULE - NO EXCEPTIONS:
        - NEVER use past tense for financial actions: forbidden phrases are
            "I have processed", "I have issued", "has been refunded", "refund was applied".
        - NEVER confirm the action happened before HITL approval.
        - The reply MUST end with exactly:
            "[PENDING HUMAN APPROVAL - reply will be sent after review]"
        - Use neutral future language only: "we would arrange", "we recommend",
            "a refund would be issued".
        WRONG: "I have processed a full refund of EUR 189. [PENDING HUMAN APPROVAL]"
        CORRECT: "We would arrange a full refund of EUR 189 [kb-004].
                            [PENDING HUMAN APPROVAL - reply will be sent after review]"
6. Do not make up information about orders or policies not in the provided data.
7. If the KB doesn't cover the question, say so honestly and offer to escalate.
8. Use order facts correctly. If the order is already delivered, do NOT say "once you receive it."
9. If order data conflicts with the customer's stated timeline (e.g., order status shows in_transit or placed recently,
     but customer claims months of ownership), do NOT proceed with warranty guidance. Set
     requires_hitl=true and escalate with reason:
     "Order data conflicts with customer's stated ownership timeline - requires manual verification before warranty claim can proceed."
10. For informational tickets with no order ID, answer directly from KB.
        Do NOT ask for an order ID or email unless the answer is genuinely impossible without live order data.
        Policy questions about refund times, shipping zones, promo codes, subscriptions, and customs never require an order ID.
11. For "forgot to apply gift card/promo code" tickets: check kb-015 first.
        If the order was placed within 48 hours, the customer is eligible for retroactive application as a partial refund [kb-015].
        Set requires_hitl=true with hitl_action=REFUND.
        Do NOT apply gift card non-refundability rules (kb-012) - those govern the gift card itself, not using it as payment.
12. NEVER unilaterally deny a return or refund. If a request falls OUTSIDE the 30-day return window
    or the 14-day damage claim window, you must escalate with the reason instead of telling the
    customer "your request cannot be processed." The decision to approve late returns requires
    human discretion [kb-002].
13. If a return or refund request involves policy constraints, validate against the KB windows
    (30 days for standard returns, 14 days for damage claims, 48 hours for damage photos).
    If the constraint is violated, escalate immediately rather than proposing an auto-action.
14. Citation map for common topics (use the correct KB ID):
    - Shipping times / delivery estimates: [kb-001]
    - Returns policy: [kb-002]
    - Refund processing times: [kb-003]
    - Damaged or faulty items: [kb-004]
    - Warranty claims: [kb-005]
        - Cancellations (including partial item cancellations): [kb-006]
        - Order changes (address, shipping method, size swaps): [kb-007]
            Note: partial cancellation = kb-006, not kb-007.
    - Payments: [kb-008]
    - Account/login: [kb-009]
    - Tracking: [kb-010]
    - International customs/duties: [kb-011]
    - Gift cards: [kb-012]
    - Sizing/fit: [kb-013]
    - Subscriptions: [kb-014]
    - Promo codes: [kb-015]
    - Product care: [kb-016]
    - Store credit: [kb-017]
    - Contact support: [kb-018]
15. Keep the tone warm, concise, and professional.
"""


def _parse_note_date(notes: str, pattern: str) -> datetime | None:
     if not notes:
          return None
     match = re.search(pattern, notes, re.IGNORECASE)
     if not match:
          return None
     date_str = match.group(1)
     try:
          return datetime.fromisoformat(date_str + "T00:00:00+00:00")
     except Exception:
          return None


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

    delivered_days: int | None = None
    delivered_date: datetime | None = None
    refund_date: datetime | None = None
    damage_claim_date: datetime | None = None

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
                    if event == "delivered_at":
                        delivered_days = delta.days
                        delivered_date = event_date
                except Exception:
                    continue

        notes = order_data.get("notes", "")
        if notes:
            refund_date = _parse_note_date(notes, r"refund processed (\d{4}-\d{2}-\d{2})")
            if refund_date:
                delta = received_date - refund_date
                lines.append(
                    "Elapsed calendar days since refund processed date "
                    f"({refund_date.strftime('%Y-%m-%d')}): {delta.days} days"
                )

            damage_claim_date = _parse_note_date(
                notes,
                r"damage claim filed (\d{4}-\d{2}-\d{2})",
            )
            if damage_claim_date:
                delta = received_date - damage_claim_date
                lines.append(
                    "Elapsed calendar days since damage claim filed date "
                    f"({damage_claim_date.strftime('%Y-%m-%d')}): {delta.days} days"
                )
                if delivered_date:
                    delivery_delta = damage_claim_date - delivered_date
                    lines.append(
                        "Elapsed calendar days between delivery and damage claim filed date "
                        f"({damage_claim_date.strftime('%Y-%m-%d')}): {delivery_delta.days} days"
                    )

    if delivered_days is not None:
        return_window = 30
        damage_window = 14
        lines.append("Policy Assessment Context:")
        lines.append(
            f"- Window for standard return: {return_window} days."
        )
        lines.append(
            f"- Current elapsed time since delivery: {delivered_days} days."
        )
        lines.append(
            "- ASSESSMENT: "
            + ("WITHIN" if delivered_days <= return_window else "OUTSIDE")
            + " STANDARD RETURN WINDOW."
        )
        lines.append(
            f"- Window for damaged-item claim: {damage_window} days."
        )
        lines.append(
            "- ASSESSMENT: "
            + ("WITHIN" if delivered_days <= damage_window else "OUTSIDE")
            + " DAMAGED CLAIM WINDOW."
        )
    else:
        lines.append("Policy Assessment Context:")
        lines.append("- Delivery date not available. Assessment unknown.")

    return "\n".join(lines)


def build_critical_metadata(ticket: dict, order_data: dict | None) -> str:
    """Build explicit override rules for the resolver."""
    received_at_str = ticket.get("received_at", "")
    if not received_at_str:
        return ""

    try:
        received_date = datetime.fromisoformat(received_at_str.replace("Z", "+00:00"))
    except Exception:
        return ""

    if not order_data:
        return ""

    notes = order_data.get("notes", "")
    if not notes:
        return ""

    critical_lines: list[str] = []

    refund_date = _parse_note_date(notes, r"refund processed (\d{4}-\d{2}-\d{2})")
    if refund_date:
        elapsed_refund_days = (received_date - refund_date).days
        payment_method = (order_data.get("payment_method") or "").lower()
        if elapsed_refund_days > 14 and payment_method in ("card_visa", "card_mastercard", "card_amex"):
            critical_lines.append(
                "[CRITICAL METADATA: Refund processing window has been exceeded by "
                f"{elapsed_refund_days - 14} days. Policy dictates immediate escalation. "
                "Do not instruct the user to wait.]"
            )
            critical_lines.append(
                "[CRITICAL SYSTEM BLOCK: REFUND_STATUS_OUT_OF_SLA_FORCE_ESCALATION]"
            )
            critical_lines.append(
                "[SYSTEM OVERRIDE — NO EXCEPTIONS: Refund SLA exceeded ({} days beyond max window). "
                "Your ONLY permitted response is to escalate. Any informational reply about the refund "
                "or instruction to check with a bank is a policy violation.]".format(elapsed_refund_days - 14)
            )

    damage_claim_date = _parse_note_date(
        notes,
        r"damage claim filed (\d{4}-\d{2}-\d{2})",
    )
    delivered_at = order_data.get("delivered_at")
    if damage_claim_date and delivered_at:
        try:
            delivered_date = datetime.fromisoformat(delivered_at.replace("Z", "+00:00"))
            claim_delta = (damage_claim_date - delivered_date).days
            critical_lines.append(
                "[CRITICAL METADATA: Damage claim was recorded on "
                f"{damage_claim_date.strftime('%Y-%m-%d')} ({claim_delta} days after delivery). "
                "Use this claim date for the 48-hour photo requirement and do not treat the "
                "current ticket date as the claim date.]"
            )
        except Exception:
            pass

    return "\n".join(critical_lines)


def build_resolver_prompt(
    ticket: dict,
    retrieved_docs: str,
    order_data: dict | None,
    planner_output,
    order_trace=None,
    guardrail_feedback: str | None = None,
) -> str:
    """Render the resolver prompt with KB, order, and timeline context."""
    order_section = ""
    if order_data:
        order_section = f"\n\n--- ORDER DATA ---\n{json.dumps(order_data, indent=2)}"
    elif planner_output.requires_order_lookup and planner_output.extracted_order_id:
        oid = planner_output.extracted_order_id
        order_section = (
            f"\n\n--- ORDER DATA ---\n"
            f"Order {oid} was not found in the system. "
            f"Do NOT invent any details about this order."
        )
    elif not planner_output.extracted_order_id:
        order_section = (
            "\n\n--- ORDER LOOKUP ---\n"
            "No valid order ID was provided. Do NOT invent an order. "
            "Ask the customer to confirm the correct order ID and email before proceeding."
        )

    if order_trace:
        order_section += (
            "\n\n--- ORDER LOOKUP STATUS ---\n"
            f"Status: {order_trace.status}\n"
            f"Reason: {order_trace.reason}\n"
            f"Summary: {order_trace.output_summary}"
        )

    kb_section = f"\n\n--- KNOWLEDGE BASE ---\n{retrieved_docs}" if retrieved_docs else ""
    timeline_section = (
        "\n\n--- FACTUAL SYSTEM METADATA & TIMELINES ---\n"
        f"{calculate_date_deltas(ticket, order_data)}"
    )

    critical_section = ""
    critical_meta = build_critical_metadata(ticket, order_data)
    if critical_meta:
        critical_section = f"\n\n--- CRITICAL METADATA ---\n{critical_meta}"

    ticket_text = (
        f"Ticket ID: {ticket.get('ticket_id')}\n"
        f"From: {ticket.get('customer_email')}\n"
        f"Subject: {ticket.get('subject')}\n"
        f"Body: {ticket.get('body')}"
    )

    feedback_section = ""
    if guardrail_feedback:
        feedback_section = (
            "\n\n--- GUARDRAIL FEEDBACK ---\n"
            f"{guardrail_feedback}"
        )

    full_prompt = (
        f"{RESOLVER_SYSTEM}"
        f"{timeline_section}"
        f"{critical_section}"
        f"{kb_section}"
        f"{order_section}"
        f"{feedback_section}"
        f"\n\n--- TICKET ---\n{ticket_text}"
        f"\n\nNow write the customer reply as structured output:"
    )

    return full_prompt

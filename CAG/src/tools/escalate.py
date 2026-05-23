"""Escalation tool."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.models import ToolCallTrace


ESCALATION_QUEUE_FILE = Path(__file__).parent.parent.parent / "runs" / "escalation_queue.jsonl"


def escalate(ticket: dict, reason: str) -> tuple[str, ToolCallTrace]:
    """Write an escalation record for human review."""
    ESCALATION_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "ticket_id": ticket.get("ticket_id"),
        "customer_email": ticket.get("customer_email"),
        "subject": ticket.get("subject"),
        "reason": reason,
        "escalated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PENDING_HUMAN_REVIEW",
    }

    with ESCALATION_QUEUE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    confirmation = (
        f"Ticket {ticket.get('ticket_id')} has been escalated to the human support queue. "
        f"Reason: {reason}"
    )

    trace = ToolCallTrace(
        tool_name="escalate",
        input_args={"ticket_id": ticket.get("ticket_id"), "reason": reason},
        status="SUCCESS",
        reason=reason,
        output_summary=f"Escalation record written for {ticket.get('ticket_id')}.",
    )
    return confirmation, trace

"""Extended pytest harness for stress-test support tickets."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.graph import compile_workflow

# ---------------------------------------------------------------------------
# Helpers — identical to test_harness.py so both files are self-contained
# ---------------------------------------------------------------------------

TICKETS_FILE = Path("tickets.json")


def load_ticket(ticket_id: str) -> dict:
    with TICKETS_FILE.open() as f:
        tickets = json.load(f)["tickets"]
    matched = [t for t in tickets if t["ticket_id"] == ticket_id]
    assert matched, f"Ticket {ticket_id} not found in tickets.json"
    return matched[0]


def run(ticket_id: str, auto_approve_hitl: bool = True) -> dict:
    ticket = load_ticket(ticket_id)
    app = compile_workflow()
    initial_state = {
        "ticket": ticket,
        "planner_output": None,
        "retrieved_docs": "",
        "order_data": None,
        "order_lookup_status": None,
        "order_lookup_error": None,
        "final_response": None,
        "requires_hitl": False,
        "execution_trace": None,
    }
    config = {"configurable": {"thread_id": f"test-{ticket_id}-{uuid.uuid4()}"}}
    try:
        return app.invoke(initial_state, config)
    except GraphInterrupt:
        approval = "approved" if auto_approve_hitl else "rejected"
        return app.invoke(Command(resume=approval), config)


# ---------------------------------------------------------------------------
# TKT-0015 — Promo code not working (informational, no order)
# EXPECT: informational, no HITL, kb-015 cited, no order lookup
# STRESS: agent must NOT ask for order ID when there is none
# ---------------------------------------------------------------------------
def test_tkt0015_promo_code_info():
    result = run("TKT-0015")
    trace = result["execution_trace"]

    assert trace.classification == "informational", f"Got: {trace.classification}"
    assert trace.workflow_status == "COMPLETED"
    assert not trace.hitl.triggered, "Pure info question — no HITL"
    assert "kb-015" in trace.citations_used, (
        f"Promo policy kb-015 must be cited, got: {trace.citations_used}"
    )
    response = (result.get("final_response") or "").lower()
    assert "minimum" in response or "threshold" in response or "€40" in response or "35" in response, (
        "Response must address why the code might be rejected (minimum spend likely)"
    )
    order_tools = [tc for tc in trace.tool_calls if tc.tool_name == "get_order_status"]
    assert not order_tools, "No order ID given — agent must not call get_order_status"


# ---------------------------------------------------------------------------
# TKT-0016 — Forgot promo/gift card within 48 hours (financial action)
# EXPECT: order_specific, HITL triggered, kb-015 cited
# STRESS: 48-hour retroactive window must be respected
# ---------------------------------------------------------------------------
def test_tkt0016_missed_giftcard_hitl():
    result = run("TKT-0016", auto_approve_hitl=True)
    trace = result["execution_trace"]

    assert trace.classification == "order_specific", f"Got: {trace.classification}"
    assert trace.hitl.triggered, "Retroactive gift card / refund is a financial action"
    assert trace.hitl.human_decision == "APPROVED"
    response = (result.get("final_response") or "").lower()
    assert "cannot" not in response or "48" in response, (
        "Agent must not deny a request within the 48-hour promo window"
    )
    assert "kb-015" in trace.citations_used, (
        f"kb-015 must be cited for retroactive promo/gift card policy"
    )


# ---------------------------------------------------------------------------
# TKT-0017 — Cancel subscription (informational, no order HITL)
# EXPECT: informational, completed, kb-014 cited
# STRESS: agent must give self-serve steps, not escalate
# ---------------------------------------------------------------------------
def test_tkt0017_subscription_cancel_info():
    result = run("TKT-0017")
    trace = result["execution_trace"]

    assert trace.workflow_status == "COMPLETED", f"Got: {trace.workflow_status}"
    assert not trace.hitl.triggered, "Cancellation instructions are informational"
    assert "kb-014" in trace.citations_used, (
        f"Subscription policy kb-014 must be cited, got: {trace.citations_used}"
    )
    response = (result.get("final_response") or "").lower()
    assert "account" in response and "subscription" in response, (
        "Response must direct customer to Account → Subscriptions"
    )


# ---------------------------------------------------------------------------
# TKT-0018 — Partial cancellation of multi-item order (order_specific)
# EXPECT: order_specific, completed or HITL, kb-006 cited
# STRESS: agent must mention partial cancellation is possible before shipping
# ---------------------------------------------------------------------------
def test_tkt0018_partial_cancellation():
    result = run("TKT-0018")
    trace = result["execution_trace"]

    assert trace.classification == "order_specific", f"Got: {trace.classification}"
    assert "kb-006" in trace.citations_used, (
        f"Cancellation policy kb-006 must be cited, got: {trace.citations_used}"
    )
    response = (result.get("final_response") or "").lower()
    assert "partial" in response or "individual" in response or "item" in response, (
        "Response must address partial cancellation capability"
    )


# ---------------------------------------------------------------------------
# TKT-0019 — Refusing customs fees, wants full refund incl. shipping
# EXPECT: escalation or completed, kb-011 cited, agent must NOT promise
#         refund of shipping or duties
# STRESS: agent must not hallucinate a refund policy that doesn't exist
# ---------------------------------------------------------------------------
def test_tkt0019_customs_refusal_no_hallucination():
    result = run("TKT-0019")
    trace = result["execution_trace"]

    assert trace.workflow_status in ("COMPLETED", "ESCALATED", "WAITING_HUMAN")
    assert "kb-011" in trace.citations_used, (
        f"International customs kb-011 must be cited, got: {trace.citations_used}"
    )
    response = (result.get("final_response") or "").lower()
    assert "duties" not in response or "not refundable" in response or "recipient" in response, (
        "Agent must not promise duty refunds — they are non-refundable per kb-011"
    )
    assert "shipping fee" not in response or "not refund" in response or "deduct" in response, (
        "Agent must not promise original shipping fee refund on refused parcels"
    )


# ---------------------------------------------------------------------------
# TKT-0020 — Size change before shipping (order_specific, no HITL)
# EXPECT: order_specific, completed, kb-007 cited
# STRESS: agent must offer address/method change path, not just cancel+reorder
# ---------------------------------------------------------------------------
def test_tkt0020_size_change_before_shipping():
    result = run("TKT-0020")
    trace = result["execution_trace"]

    assert trace.classification == "order_specific", f"Got: {trace.classification}"
    assert trace.workflow_status == "COMPLETED"
    assert "kb-007" in trace.citations_used, (
        f"Order changes kb-007 must be cited, got: {trace.citations_used}"
    )
    response = (result.get("final_response") or "").lower()
    assert "cancel" in response or "reorder" in response or "contact" in response, (
        "Agent must mention cancel-and-reorder as the path for item swaps"
    )


# ---------------------------------------------------------------------------
# TKT-0021 — No tracking number after 4 days (order_specific)
# EXPECT: order_specific, completed, kb-010 cited
# STRESS: agent must not invent a tracking number or shipping date
# ---------------------------------------------------------------------------
def test_tkt0021_no_tracking_number():
    result = run("TKT-0021")
    trace = result["execution_trace"]

    assert trace.classification == "order_specific", f"Got: {trace.classification}"
    assert trace.workflow_status in ("COMPLETED", "ESCALATED")
    assert "kb-010" in trace.citations_used, (
        f"Tracking kb-010 must be cited, got: {trace.citations_used}"
    )
    response = (result.get("final_response") or "").lower()
    assert "tracking number" not in response or "confirmation email" in response or "account" in response, (
        "Agent must not fabricate a tracking number"
    )


# ---------------------------------------------------------------------------
# TKT-0022 — Warranty claim, 18 months (within 2-year warranty)
# EXPECT: order_specific, completed, kb-005 cited, warranty path explained
# STRESS: agent must confirm coverage and give filing steps
# ---------------------------------------------------------------------------
def test_tkt0022_warranty_within_period():
    result = run("TKT-0022")
    trace = result["execution_trace"]

    assert trace.classification == "order_specific", f"Got: {trace.classification}"
    assert trace.workflow_status == "COMPLETED"
    assert "kb-005" in trace.citations_used, (
        f"Warranty kb-005 must be cited, got: {trace.citations_used}"
    )
    response = (result.get("final_response") or "").lower()
    assert "warrant" in response, "Response must mention warranty coverage"
    assert "photo" in response or "description" in response or "order number" in response, (
        "Agent must ask for required warranty claim info"
    )
    assert "not covered" not in response and "expired" not in response, (
        "18-month-old item is within 2-year warranty — agent must not deny it"
    )


# ---------------------------------------------------------------------------
# TKT-0023 — Refund to cancelled card (order_specific or escalation)
# EXPECT: completed or escalated, kb-017 cited (store credit fallback)
# STRESS: agent must mention store credit path for unreachable payment method
# ---------------------------------------------------------------------------
def test_tkt0023_refund_to_cancelled_card():
    result = run("TKT-0023")
    trace = result["execution_trace"]

    assert trace.workflow_status in ("COMPLETED", "ESCALATED")
    response = (result.get("final_response") or "").lower()
    assert "store credit" in response or "contact" in response or "bank" in response or "escalat" in response, (
        "Agent must address the cancelled card scenario — mention store credit or escalation"
    )


# ---------------------------------------------------------------------------
# TKT-0024 — 2FA lockout (account/security)
# EXPECT: escalation — cannot resolve 2FA lockout without human intervention
# STRESS: agent must escalate, not attempt to give a workaround that bypasses 2FA
# ---------------------------------------------------------------------------
def test_tkt0024_2fa_lockout_escalation():
    result = run("TKT-0024")
    trace = result["execution_trace"]

    assert trace.workflow_status == "ESCALATED", (
        f"2FA lockout requires human — got: {trace.workflow_status}"
    )
    assert trace.classification == "escalation", f"Got: {trace.classification}"
    response = (result.get("final_response") or "").lower()
    assert "bypass" not in response and "workaround" not in response, (
        "Agent must not suggest bypassing 2FA"
    )


# ---------------------------------------------------------------------------
# TKT-0025 — Final Sale item arrived damaged (order_specific, HITL)
# EXPECT: HITL triggered — damaged Final Sale items are still eligible for
#         refund/replacement per kb-004; Final Sale only blocks voluntary returns
# STRESS: agent must NOT deny the claim citing Final Sale status
# ---------------------------------------------------------------------------
def test_tkt0025_final_sale_damaged_not_denied():
    result = run("TKT-0025", auto_approve_hitl=True)
    trace = result["execution_trace"]

    response = (result.get("final_response") or "").lower()
    assert "final sale" not in response or "damaged" in response or "exception" in response, (
        "Agent must not block a damage claim just because item is Final Sale"
    )
    assert "kb-004" in trace.citations_used, (
        f"Damaged items kb-004 must be cited, got: {trace.citations_used}"
    )
    assert trace.hitl.triggered, "Damage refund/replacement must trigger HITL"


# ---------------------------------------------------------------------------
# TKT-0026 — Duplicate charge (escalation)
# EXPECT: escalated — duplicate billing is a financial/fraud issue needing human
# STRESS: agent must NOT attempt to resolve a duplicate charge autonomously
# ---------------------------------------------------------------------------
def test_tkt0026_duplicate_charge_escalation():
    result = run("TKT-0026")
    trace = result["execution_trace"]

    assert trace.workflow_status == "ESCALATED", (
        f"Duplicate charge must be escalated, got: {trace.workflow_status}"
    )
    response = (result.get("final_response") or "").lower()
    assert "escalat" in response or "team" in response or "review" in response, (
        "Escalation message must mention human review"
    )
    assert not (trace.hitl.triggered and trace.hitl.human_decision == "APPROVED" and
                "refund" in (trace.hitl.action_type or "").lower()), (
        "Duplicate charge must escalate to human, not auto-approve a refund"
    )

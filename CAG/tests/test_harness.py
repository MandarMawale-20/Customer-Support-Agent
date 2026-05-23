"""Pytest harness for the support agent."""
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

pytestmark = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from src.graph import compile_workflow

TICKETS_FILE = Path("tickets.json")


def load_ticket(ticket_id: str) -> dict:
    """Load a single ticket fixture by ID."""
    with TICKETS_FILE.open() as f:
        tickets = json.load(f)["tickets"]
    matched = [t for t in tickets if t["ticket_id"] == ticket_id]
    assert matched, f"Ticket {ticket_id} not found in tickets.json"
    return matched[0]


def run(ticket_id: str, auto_approve_hitl: bool = True) -> dict:
    """Run the workflow for one ticket and optionally auto-approve HITL."""
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
    config = {
        "configurable": {
            "thread_id": f"test-{ticket_id}-{uuid.uuid4()}"
        }
    }

    try:
        return app.invoke(initial_state, config)
    except GraphInterrupt:
        approval = "approved" if auto_approve_hitl else "rejected"
        return app.invoke(Command(resume=approval), config)


def test_tkt0001_refund_info_no_hitl():
    result = run("TKT-0001")
    trace = result["execution_trace"]

    assert trace.classification == "informational", f"Got: {trace.classification}"
    assert trace.workflow_status == "COMPLETED", f"Got: {trace.workflow_status}"
    assert not trace.hitl.triggered, "HITL should NOT trigger for an info question"
    assert any("kb-003" in c for c in trace.citations_used), (
        f"Expected kb-003 cited, got: {trace.citations_used}"
    )
    assert result["final_response"], "Should have a final response"


def test_tkt0002_order_in_transit():
    result = run("TKT-0002")
    trace = result["execution_trace"]

    assert trace.classification == "order_specific", f"Got: {trace.classification}"
    assert trace.workflow_status == "COMPLETED"
    assert any(tc.tool_name == "get_order_status" for tc in trace.tool_calls), (
        "get_order_status should have been called"
    )
    order_trace = next(tc for tc in trace.tool_calls if tc.tool_name == "get_order_status")
    assert order_trace.status == "SUCCESS", f"Expected SUCCESS, got: {order_trace.status}"
    assert not trace.hitl.triggered


def test_tkt0003_refund_triggers_hitl():
    result = run("TKT-0003", auto_approve_hitl=True)
    trace = result["execution_trace"]

    assert trace.hitl.triggered, "HITL MUST trigger for a refund request"
    assert trace.hitl.action_type in ("REFUND", "REFUND_REQUEST"), (
        f"Expected REFUND action, got: {trace.hitl.action_type}"
    )
    assert trace.hitl.human_decision == "APPROVED"
    assert trace.workflow_status == "COMPLETED"


def test_tkt0004_shipping_info():
    result = run("TKT-0004")
    trace = result["execution_trace"]

    assert trace.classification == "informational"
    assert trace.workflow_status == "COMPLETED"
    kb_cited = set(trace.citations_used)
    assert kb_cited & {"kb-001", "kb-009"}, (
        f"Expected shipping or customs KB cited, got: {kb_cited}"
    )


def test_tkt0008_unknown_order_404():
    result = run("TKT-0008")
    trace = result["execution_trace"]
    response = (result.get("final_response") or "").lower()

    order_traces = [tc for tc in trace.tool_calls if tc.tool_name == "get_order_status"]
    if order_traces:
        assert order_traces[0].status == "404_NOT_FOUND", (
            f"Unknown order must return 404, not: {order_traces[0].status}"
        )
    assert trace.workflow_status == "ESCALATED"
    assert "pkz-77" not in response or "not found" in response or "cannot" in response, (
        "Agent should not hallucinate details for unknown order PKZ-77"
    )
    if result.get("order_data") is None:
        assert "delivered" not in response
        assert "shipped" not in response


def test_tkt0014_security_escalation():
    result = run("TKT-0014")
    trace = result["execution_trace"]

    assert trace.classification == "escalation", f"Got: {trace.classification}"
    assert trace.workflow_status == "ESCALATED"
    assert trace.resolution_type in ("ESCALATED", "SECURITY_ESCALATION")
    non_escalate_tools = [
        tc for tc in trace.tool_calls if tc.tool_name != "escalate"
    ]
    assert not non_escalate_tools, (
        f"Security escalation should not call other tools: {non_escalate_tools}"
    )


def test_tkt0003_refund_hitl_rejected():
    result = run("TKT-0003", auto_approve_hitl=False)
    trace = result["execution_trace"]

    assert trace.hitl.triggered
    assert trace.hitl.human_decision == "REJECTED"
    response = result["final_response"] or ""
    assert len(response) > 10, "Should still send a response after HITL rejection"


def test_tkt0009_return_policy_gift():
    result = run("TKT-0009")
    trace = result["execution_trace"]

    assert trace.classification == "informational"
    assert trace.workflow_status == "COMPLETED"
    assert result["final_response"], "Must have a response"
    assert "kb-002" in trace.citations_used, (
        f"Return policy (kb-002) should be cited, got: {trace.citations_used}"
    )


def test_hallucination_guardrails_info_citations():
    result = run("TKT-0001")
    trace = result["execution_trace"]
    if trace.resolution_type == "INFO_RESPONSE":
        assert len(trace.citations_used) > 0, "INFO responses must cite KB sources"

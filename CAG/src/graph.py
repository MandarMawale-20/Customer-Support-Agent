"""LangGraph workflow for the support agent."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langgraph.types import interrupt

from src.models import (
    AgentState,
    HITLTrace,
    PlannerResponse,
    ResolverResponse,
    RunTrace,
    TicketExecutionTrace,
    ToolCallTrace,
)
from src.agents import build_planner_prompt, build_resolver_prompt
from src.agents.resolver_prompt import extract_notes_date_delta
from src.tools import kb_search, get_order_status, escalate

RUNS_DIR = Path(__file__).parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)

def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.1,
    )


def _escalate_for_llm_failure(ticket: dict, trace: TicketExecutionTrace | None, stage: str, error: Exception) -> dict:
    reason = f"{stage} LLM service unavailable. Human review required."
    confirmation, esc_trace = escalate(ticket, reason)

    if trace is None:
        trace = TicketExecutionTrace(
            ticket_id=ticket["ticket_id"],
            classification="escalation",
            sub_tasks=["Escalate because the AI service is unavailable."],
            decision_trace=[f"{stage} failed: {error}", f"Escalation triggered: {reason}"],
            tool_calls=[esc_trace],
            hitl=HITLTrace(triggered=False, action_type="NONE"),
            workflow_status="ESCALATED",
            resolution_type="ESCALATED",
        )
    else:
        trace = trace.model_copy(update={
            "tool_calls": list(trace.tool_calls) + [esc_trace],
            "decision_trace": list(trace.decision_trace) + [f"{stage} failed: {error}", f"Escalation triggered: {reason}"],
            "workflow_status": "ESCALATED",
            "resolution_type": "ESCALATED",
        })

    return {
        "final_response": confirmation,
        "execution_trace": trace,
        "requires_hitl": False,
    }


def planner_node(state: AgentState) -> dict:
    """Classify the ticket and seed the execution trace."""
    ticket = state["ticket"]
    llm = _get_llm()

    structured_llm = llm.with_structured_output(PlannerResponse)
    prompt = build_planner_prompt(ticket)
    try:
        planner_output = structured_llm.invoke(prompt)
    except Exception as error:
        return _escalate_for_llm_failure(state["ticket"], None, "Planner", error)

    trace = TicketExecutionTrace(
        ticket_id=ticket["ticket_id"],
        classification=planner_output.classification,
        sub_tasks=planner_output.sub_tasks,
        decision_trace=[f"Planner classified as: {planner_output.classification}"],
        tool_calls=[],
        hitl=HITLTrace(triggered=False, action_type="NONE"),
        workflow_status="FAILED",
        resolution_type="INFO_RESPONSE",
    )

    return {
        "planner_output": planner_output,
        "execution_trace": trace,
        "requires_hitl": False,
        "retrieved_docs": "",
        "order_data": None,
        "order_lookup_status": None,
        "order_lookup_error": None,
        "final_response": None,
    }


def escalation_node(state: AgentState) -> dict:
    """Handle direct escalations from the planner."""
    planner: PlannerResponse = state["planner_output"]
    ticket = state["ticket"]
    trace: TicketExecutionTrace = state["execution_trace"]
    tool_calls: list[ToolCallTrace] = list(trace.tool_calls)
    decision_trace = list(trace.decision_trace)

    reason = planner.immediate_escalation_reason or "Requires human review."
    confirmation, esc_trace = escalate(ticket, reason)
    tool_calls.append(esc_trace)
    decision_trace.append(f"Escalation triggered: {reason}")

    updated_trace = trace.model_copy(update={
        "tool_calls": tool_calls,
        "decision_trace": decision_trace,
        "workflow_status": "ESCALATED",
        "resolution_type": "ESCALATED",
    })

    return {
        "execution_trace": updated_trace,
        "final_response": confirmation,
        "retrieved_docs": "",
        "order_data": None,
    }


def kb_search_node(state: AgentState) -> dict:
    """Load the requested KB articles."""
    planner: PlannerResponse = state["planner_output"]
    trace: TicketExecutionTrace = state["execution_trace"]
    tool_calls: list[ToolCallTrace] = list(trace.tool_calls)
    decision_trace = list(trace.decision_trace)

    if not planner.kb_article_ids:
        decision_trace.append("KB search skipped: no article IDs provided.")
        updated_trace = trace.model_copy(update={
            "decision_trace": decision_trace,
        })
        return {"execution_trace": updated_trace, "retrieved_docs": ""}

    retrieved_text, kb_trace = kb_search(planner.kb_article_ids)
    tool_calls.append(kb_trace)
    decision_trace.append(f"KB search: loaded {planner.kb_article_ids}")

    updated_trace = trace.model_copy(update={
        "tool_calls": tool_calls,
        "decision_trace": decision_trace,
    })

    return {
        "execution_trace": updated_trace,
        "retrieved_docs": retrieved_text,
    }


def order_lookup_node(state: AgentState) -> dict:
    """Fetch live order data when required."""
    planner: PlannerResponse = state["planner_output"]
    trace: TicketExecutionTrace = state["execution_trace"]
    tool_calls: list[ToolCallTrace] = list(trace.tool_calls)
    decision_trace = list(trace.decision_trace)

    if not planner.extracted_order_id:
        decision_trace.append("Order lookup skipped: no order ID extracted.")
        updated_trace = trace.model_copy(update={
            "decision_trace": decision_trace,
        })
        return {"execution_trace": updated_trace, "order_data": None}

    live_order, order_trace = get_order_status(planner.extracted_order_id)
    tool_calls.append(order_trace)
    status_msg = f"Order {planner.extracted_order_id}: {order_trace.status}"
    decision_trace.append(status_msg)

    updated_trace = trace.model_copy(update={
        "tool_calls": tool_calls,
        "decision_trace": decision_trace,
    })

    return {
        "execution_trace": updated_trace,
        "order_data": live_order,
        "order_lookup_status": order_trace.status,
        "order_lookup_error": order_trace.output_summary,
    }


def invalid_order_node(state: AgentState) -> dict:
    """Escalate when an order lookup fails for a required order ID."""
    planner: PlannerResponse = state["planner_output"]
    ticket = state["ticket"]
    trace: TicketExecutionTrace = state["execution_trace"]
    tool_calls: list[ToolCallTrace] = list(trace.tool_calls)
    decision_trace = list(trace.decision_trace)

    reason = (
        f"Order {planner.extracted_order_id} not found in the system. "
        "Requires human review before responding."
    )
    confirmation, esc_trace = escalate(ticket, reason)
    tool_calls.append(esc_trace)
    decision_trace.append("Escalation triggered: order not found")

    updated_trace = trace.model_copy(update={
        "tool_calls": tool_calls,
        "decision_trace": decision_trace,
        "workflow_status": "ESCALATED",
        "resolution_type": "ESCALATED",
    })

    return {
        "execution_trace": updated_trace,
        "final_response": confirmation,
        "retrieved_docs": "",
        "order_data": None,
    }


def order_error_node(state: AgentState) -> dict:
    """Handle order lookup errors without hallucinating details."""
    ticket = state["ticket"]
    trace: TicketExecutionTrace = state["execution_trace"]
    tool_calls: list[ToolCallTrace] = list(trace.tool_calls)
    decision_trace = list(trace.decision_trace)

    reason = "Order lookup error (upstream service failure)."
    confirmation, esc_trace = escalate(ticket, reason)
    tool_calls.append(esc_trace)
    decision_trace.append("Escalation triggered: order lookup error")

    reply = (
        "We hit a temporary system issue while retrieving your order details. "
        "We have escalated this to our human support team and they will follow up shortly."
    )

    updated_trace = trace.model_copy(update={
        "tool_calls": tool_calls,
        "decision_trace": decision_trace,
        "workflow_status": "ESCALATED",
        "resolution_type": "ESCALATED",
    })

    return {
        "execution_trace": updated_trace,
        "final_response": reply,
        "retrieved_docs": "",
        "order_data": None,
        "order_lookup_error": confirmation,
    }


def resolver_node(state: AgentState) -> dict:
    """Draft the customer reply from KB and live order data."""
    trace: TicketExecutionTrace = state["execution_trace"]

    if trace.workflow_status == "ESCALATED":
        return {}

    notes_delta = extract_notes_date_delta(state["ticket"], state.get("order_data"))
    if notes_delta is not None and notes_delta > 14:
        reason = "Refund processing SLA violated."
        confirmation, esc_trace = escalate(state["ticket"], reason)
        tool_calls: list[ToolCallTrace] = list(trace.tool_calls)
        decision_trace = list(trace.decision_trace)
        tool_calls.append(esc_trace)
        decision_trace.append("Escalation triggered: refund SLA violated")

        updated_trace = trace.model_copy(update={
            "tool_calls": tool_calls,
            "decision_trace": decision_trace,
            "workflow_status": "ESCALATED",
            "resolution_type": "ESCALATED",
            "hitl": HITLTrace(triggered=False, action_type="NONE"),
        })

        return {
            "final_response": confirmation,
            "requires_hitl": False,
            "execution_trace": updated_trace,
        }

    llm = _get_llm()
    prompt = build_resolver_prompt(
        ticket=state["ticket"],
        retrieved_docs=state["retrieved_docs"],
        order_data=state["order_data"],
        planner_output=state["planner_output"],
    )
    structured_llm = llm.with_structured_output(ResolverResponse)
    try:
        response = structured_llm.invoke(prompt)
    except Exception as error:
        return _escalate_for_llm_failure(state["ticket"], trace, "Resolver", error)
    reply = response.customer_reply.strip()

    requires_hitl = response.requires_financial_action or response.hitl_action is not None
    hitl_action_type = response.hitl_action or "NONE"
    hitl_justification = response.hitl_justification

    if requires_hitl and hitl_action_type not in ["REFUND", "CANCELLATION", "STORE_CREDIT"]:
        hitl_action_type = "REFUND"
    if not requires_hitl:
        hitl_action_type = "NONE"

    citations = response.citations or []
    if not citations:
        citations = re.findall(r"\[kb-\d{3}\]", reply)
        citations = list(dict.fromkeys(c.strip("[]") for c in citations))

    planner: PlannerResponse = state["planner_output"]
    if requires_hitl:
        resolution_type = "REFUND_REQUEST"
    elif planner.classification == "order_specific":
        resolution_type = "ORDER_UPDATE"
    else:
        resolution_type = "INFO_RESPONSE"

    updated_trace = trace.model_copy(update={
        "citations_used": citations,
        "workflow_status": "WAITING_HUMAN" if requires_hitl else "COMPLETED",
        "resolution_type": resolution_type,
        "hitl": HITLTrace(
            triggered=requires_hitl,
            action_type=hitl_action_type,
            justification=hitl_justification,
        ),
    })

    return {
        "final_response": reply,
        "requires_hitl": requires_hitl,
        "execution_trace": updated_trace,
    }


def hitl_node(state: AgentState) -> dict:
    """Pause for human approval before sending financial actions."""
    trace: TicketExecutionTrace = state["execution_trace"]

    if not state.get("requires_hitl", False):
        return {}

    reply = state.get("final_response", "")

    decision = interrupt({
        "ticket_id": state["ticket"]["ticket_id"],
        "action": trace.hitl.action_type,
        "justification": trace.hitl.justification,
        "proposed_reply": reply,
    })

    if isinstance(decision, dict):
        decision_value = decision.get("decision") or decision.get("approved")
    else:
        decision_value = decision

    decision_text = str(decision_value or "").strip().lower()
    human_decision = "APPROVED" if decision_text in {"y", "yes", "approve", "approved", "true", "1"} else "REJECTED"

    if human_decision == "REJECTED":
        reply = (
            "Thank you for reaching out. Your request has been forwarded "
            "to our support team, who will contact you within 1 business day."
        )

    updated_trace = trace.model_copy(update={
        "workflow_status": "COMPLETED" if human_decision == "APPROVED" else "ESCALATED",
        "hitl": trace.hitl.model_copy(update={"human_decision": human_decision}),
    })

    return {
        "final_response": reply,
        "execution_trace": updated_trace,
    }


def observability_node(state: AgentState) -> dict:
    """Persist the run trace for audit and grading."""
    trace = state.get("execution_trace")
    ticket_id = state["ticket"]["ticket_id"]

    if trace is None:
        print(f"⚠️  No trace to save — pipeline failed early for ticket {ticket_id}.")
        return {}

    record = RunTrace(
        ticket_id=ticket_id,
        final_response=state.get("final_response"),
        execution_trace=trace,
        saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    outfile = RUNS_DIR / f"{ticket_id}.json"
    outfile.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n📁 Trace saved → runs/{ticket_id}.json")
    return {}


def compile_workflow() -> Pregel:
    """Build the linear workflow used by the CLI and tests."""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("escalate", escalation_node)
    builder.add_node("kb_search", kb_search_node)
    builder.add_node("order_lookup", order_lookup_node)
    builder.add_node("invalid_order", invalid_order_node)
    builder.add_node("order_error", order_error_node)
    builder.add_node("resolver", resolver_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("observability", observability_node)

    def route_from_planner(state: AgentState) -> str:
        planner: PlannerResponse = state["planner_output"]
        if planner.classification == "escalation":
            return "escalate"
        if planner.requires_kb_search:
            return "kb_search"
        if planner.requires_order_lookup:
            return "order_lookup"
        return "resolver"

    def route_from_kb_search(state: AgentState) -> str:
        planner: PlannerResponse = state["planner_output"]
        if planner.requires_order_lookup:
            return "order_lookup"
        return "resolver"

    def route_from_order_lookup(state: AgentState) -> str:
        planner: PlannerResponse = state["planner_output"]
        if state.get("order_lookup_status") == "404_NOT_FOUND":
            return "invalid_order"
        if state.get("order_lookup_status") == "ERROR":
            return "order_error"
        if planner.requires_order_lookup and state.get("order_data") is None:
            return "order_error"
        return "resolver"

    builder.set_entry_point("planner")
    builder.add_conditional_edges("planner", route_from_planner)
    builder.add_edge("escalate", "observability")
    builder.add_conditional_edges("kb_search", route_from_kb_search)
    builder.add_conditional_edges("order_lookup", route_from_order_lookup)
    builder.add_edge("invalid_order", "observability")
    builder.add_edge("order_error", "observability")
    builder.add_edge("resolver", "hitl")
    builder.add_edge("hitl", "observability")
    builder.add_edge("observability", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)

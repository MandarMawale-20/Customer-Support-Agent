"""LangGraph workflow for the RAG support agent."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel

from src.models import (
    AgentState,
    HITLTrace,
    PlannerResponse,
    ResolverResponse,
    TicketExecutionTrace,
    ToolCallTrace,
)
from src.agents import build_planner_prompt, build_resolver_prompt
from src.agents.resolver_prompt import build_critical_metadata
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


def planner_node(state: AgentState) -> dict:
    """Classify the ticket and seed the execution trace."""
    ticket = state["ticket"]
    llm = _get_llm()

    structured_llm = llm.with_structured_output(PlannerResponse)
    prompt = build_planner_prompt(ticket)
    planner_output = structured_llm.invoke(prompt)

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
        "order_tool_trace": None,
        "final_response": None,
        "guardrail_attempts": 0,
        "guardrail_feedback": None,
    }


def tool_executor_node(state: AgentState) -> dict:
    """Run the planner-selected tools and update the audit trail."""
    planner: PlannerResponse = state["planner_output"]
    ticket = state["ticket"]
    trace: TicketExecutionTrace = state["execution_trace"]
    tool_calls: list[ToolCallTrace] = list(trace.tool_calls)
    decision_trace = list(trace.decision_trace)

    retrieved_text = ""
    live_order = None
    order_trace = None

    if planner.classification == "escalation":
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

    if planner.requires_kb_search and planner.kb_queries:
        retrieved_text, kb_trace = kb_search(
            planner.kb_queries,
            mandatory_category=planner.kb_category,
        )
        tool_calls.append(kb_trace)
        decision_trace.append(
            "KB search: " + ", ".join(planner.kb_queries)
        )

    if planner.requires_order_lookup and planner.extracted_order_id:
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
        "retrieved_docs": retrieved_text,
        "order_data": live_order,
        "order_tool_trace": order_trace if planner.requires_order_lookup else None,
    }


def resolver_node(state: AgentState) -> dict:
    """Draft the customer reply from KB and live order data."""
    trace: TicketExecutionTrace = state["execution_trace"]

    if trace.workflow_status == "ESCALATED":
        return {}

    critical_meta = build_critical_metadata(state["ticket"], state["order_data"])
    if "REFUND_STATUS_OUT_OF_SLA_FORCE_ESCALATION" in critical_meta:
        reason = "Refund is beyond SLA window; immediate lost funds investigation required."
        confirmation, esc_trace = escalate(state["ticket"], reason)
        tool_calls = list(trace.tool_calls) + [esc_trace]
        decision_trace = list(trace.decision_trace) + [f"Escalation triggered: {reason}"]
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

    planner: PlannerResponse = state["planner_output"]
    order_trace = state.get("order_tool_trace")
    
    # Guard: if order lookup was required but failed or ID was invalid, escalate
    if planner.requires_order_lookup and not state.get("order_data"):
        if order_trace and order_trace.status in ("404_NOT_FOUND", "ERROR", "TIMEOUT"):
            reason = f"Order lookup failed: {order_trace.status}. Requires human review before composing response."
            confirmation, esc_trace = escalate(state["ticket"], reason)
            tool_calls = list(trace.tool_calls) + [esc_trace]
            decision_trace = list(trace.decision_trace) + [f"Escalation triggered: invalid order lookup"]
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
        elif not planner.extracted_order_id:
            reason = "Order ID could not be extracted or is invalid. Requires human review."
            confirmation, esc_trace = escalate(state["ticket"], reason)
            tool_calls = list(trace.tool_calls) + [esc_trace]
            decision_trace = list(trace.decision_trace) + [f"Escalation triggered: missing/invalid order ID"]
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
    structured_llm = llm.with_structured_output(ResolverResponse)
    prompt = build_resolver_prompt(
        ticket=state["ticket"],
        retrieved_docs=state["retrieved_docs"],
        order_data=state["order_data"],
        planner_output=state["planner_output"],
        order_trace=state.get("order_tool_trace"),
        guardrail_feedback=state.get("guardrail_feedback"),
    )
    res: ResolverResponse = structured_llm.invoke(prompt)
    reply = res.customer_reply.strip()
    requires_hitl = res.requires_hitl
    hitl_action_type = res.hitl_action or "NONE"
    hitl_justification = res.hitl_justification

    citations = list(dict.fromkeys(re.findall(r"kb-\d{3}", reply)))

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


def _validate_reply(
    reply: str,
    requires_kb_search: bool,
    order_data: dict | None,
    order_trace: ToolCallTrace | None,
    classification: str,
    requires_hitl: bool = False,
) -> tuple[bool, str]:
    if "HITL_" in reply or "<" in reply or ">" in reply:
        return False, "Remove any placeholder tokens or HITL markers from the reply."

    if requires_hitl:
        if "[PENDING HUMAN APPROVAL" not in reply:
            return False, "Financial action replies must end with [PENDING HUMAN APPROVAL — reply will be sent after review]."
        if re.search(r"\b(processed|issued|refunded|will be applied)\b", reply, re.IGNORECASE):
            return False, "Do not confirm that a financial action has been processed or issued. Use pending approval format instead."

    if requires_kb_search and not re.search(r"kb-\d{3}", reply):
        return False, "Include at least one kb-### citation for policy claims."

    keywords = (
        "refund",
        "return",
        "cancel",
        "warranty",
        "shipping",
        "tracking",
        "delivery",
        "store credit",
        "promo",
        "subscription",
        "payment",
        "account",
    )
    sentences = re.split(r"(?<=[.!?])\s+", reply)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(word in lowered for word in keywords) and not re.search(r"kb-\d{3}", sentence):
            return False, "Every policy-related sentence must include a kb-### citation."

        if any(word in lowered for word in ("damaged", "faulty")) and "kb-004" not in sentence:
            return False, "Damaged or faulty item claims must cite [kb-004]."

        if any(word in lowered for word in ("customs", "duties", "ddp", "ddu")) and "kb-011" not in sentence:
            return False, "Customs or duties statements must cite [kb-011]."

        if any(word in lowered for word in ("shipping", "delivery estimate", "business days", "rest of world")):
            if "kb-001" not in sentence:
                return False, "Shipping time or delivery estimate statements must cite [kb-001]."

        if "refund" in lowered and "business days" in lowered and "kb-003" not in sentence:
            return False, "Refund processing time statements must cite [kb-003]."

    if order_data and order_data.get("delivered_at"):
        if re.search(r"\b(once|when) you receive\b", reply, re.IGNORECASE):
            return False, "Order was already delivered; avoid 'once you receive it' phrasing."

    if order_trace and order_trace.status == "404_NOT_FOUND":
        if not re.search(r"(not found|could not find|cannot find|unable to find)", reply, re.IGNORECASE):
            return False, "Order lookup returned 404; reply must state the order was not found and request confirmation."

    if order_trace and order_trace.status in ("ERROR", "TIMEOUT"):
        if not re.search(r"(system|technical|error|issue)", reply, re.IGNORECASE):
            return False, "Order lookup failed; reply must acknowledge a system error and avoid speculation."

    if classification == "informational":
        if re.search(r"order (id|number)", reply, re.IGNORECASE):
            return False, "Informational tickets should not request an order ID."

    # Prevent bot from unilaterally denying returns/refunds; these must be escalated
    denial_patterns = (
        r"cannot (process|handle|approve)",
        r"(cannot|no longer|no)\s+(accept|process)\s+.{0,20}(return|refund)",
        r"(unfortunately|sorry).{0,30}(cannot|unable).{0,20}(return|refund)",
    )
    escalation_keywords = (
        "escalat", "human support", "review", "manager", "team"
    )
    for pattern in denial_patterns:
        if re.search(pattern, reply, re.IGNORECASE):
            if not any(kw in reply.lower() for kw in escalation_keywords):
                return False, "Return/refund denials must be escalated to human review, not communicated as bot decisions."

    return True, ""


def guardrail_node(state: AgentState) -> dict:
    """Validate groundedness and structure before HITL or final output."""
    reply = state.get("final_response", "") or ""
    planner: PlannerResponse = state["planner_output"]
    ok, feedback = _validate_reply(
        reply,
        planner.requires_kb_search,
        state.get("order_data"),
        state.get("order_tool_trace"),
        planner.classification,
        requires_hitl=state.get("requires_hitl", False),
    )

    if ok:
        return {"guardrail_feedback": None}

    attempts = state.get("guardrail_attempts", 0)
    attempts += 1
    trace: TicketExecutionTrace = state["execution_trace"]
    decision_trace = list(trace.decision_trace)
    decision_trace.append(f"Guardrail failed: {feedback}")
    updated_trace = trace.model_copy(update={"decision_trace": decision_trace})

    return {
        "guardrail_attempts": attempts,
        "guardrail_feedback": feedback,
        "execution_trace": updated_trace,
    }


def hitl_node(state: AgentState) -> dict:
    """Pause for human approval before sending financial actions."""
    trace: TicketExecutionTrace = state["execution_trace"]

    if not state.get("requires_hitl", False):
        return {}

    reply = state.get("final_response", "")

    print("\n" + "=" * 60)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print(f"Ticket:  {state['ticket']['ticket_id']}")
    print(f"Action:  {trace.hitl.action_type}")
    print(f"Reason:  {trace.hitl.justification}")
    print("\n-- Proposed customer reply --")
    print(reply)
    print("-" * 60)

    while True:
        decision = input("\nApprove this action? [y/n]: ").strip().lower()
        if decision in ("y", "n"):
            break
        print("Please enter y or n.")

    human_decision = "APPROVED" if decision == "y" else "REJECTED"

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
        print(f"No trace to save - pipeline failed early for ticket {ticket_id}.")
        return {}

    output = {
        "ticket_id": ticket_id,
        "final_response": state.get("final_response"),
        "execution_trace": trace.model_dump(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    outfile = RUNS_DIR / f"{ticket_id}.json"
    outfile.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Trace saved -> runs/{ticket_id}.json")

    return {}


def compile_workflow() -> Pregel:
    """Build and compile the workflow graph."""
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tools", tool_executor_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("observe", observability_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tools")
    graph.add_edge("tools", "resolver")
    graph.add_edge("resolver", "guardrail")

    def route_guardrail(state: AgentState) -> str:
        attempts = state.get("guardrail_attempts", 0)
        feedback = state.get("guardrail_feedback")
        if feedback and attempts < 2:
            return "resolver"
        return "hitl"

    graph.add_conditional_edges("guardrail", route_guardrail, {
        "resolver": "resolver",
        "hitl": "hitl",
    })
    graph.add_edge("hitl", "observe")
    graph.add_edge("observe", END)

    return graph.compile()


from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("GEMINI_API_KEY not set. Add it to your .env file or export it.")
    sys.exit(1)

from src.graph import compile_workflow

TICKETS_FILE = Path("tickets.json")


def load_tickets() -> list[dict]:
    """Load the ticket fixture used by the CLI."""
    with TICKETS_FILE.open() as f:
        return json.load(f)["tickets"]


def run_ticket(ticket: dict, app) -> None:
    """Run one ticket through the compiled graph and print the reply."""
    print("\n" + "=" * 60)
    print(f"Processing {ticket['ticket_id']}: {ticket['subject']}")
    print("=" * 60)

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
            "thread_id": f"cli-{ticket['ticket_id']}-{uuid.uuid4()}"
        }
    }

    result = app.invoke(initial_state, config)
    final = result.get("final_response", "")
    trace = result.get("execution_trace")

    print(f"\n Resolution: {trace.workflow_status if trace else 'N/A'}")
    print("\n── Customer Reply ──")
    print(final)


def main():
    """Process one ticket or the full ticket set from the command line."""
    parser = argparse.ArgumentParser(description="Customer Support Resolution Agent")
    parser.add_argument("--ticket", help="Process a single ticket ID (e.g. TKT-0001)")
    args = parser.parse_args()

    tickets = load_tickets()
    app = compile_workflow()

    if args.ticket:
        matched = [t for t in tickets if t["ticket_id"] == args.ticket]
        if not matched:
            print(f" Ticket {args.ticket} not found in tickets.json")
            sys.exit(1)
        run_ticket(matched[0], app)
    else:
        for ticket in tickets:
            run_ticket(ticket, app)
            input("\n[Press Enter to continue to next ticket...]")


if __name__ == "__main__":
    main()

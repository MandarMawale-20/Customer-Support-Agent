# Customer-Support Resolution Agent — Materials Pack

This folder contains everything you need for the take-home. Read the main brief (`Customer_Support_Resolution_Agent_TakeHome.docx`) first; the files here are the inputs that brief refers to.

## Files

| File / folder | Description |
|---|---|
| `knowledge_base/` | 18 Markdown articles covering shipping, returns, refunds, warranty, accounts, product care, etc. This is your RAG corpus. |
| `orders.json` | Mock order dataset — 35 orders with varied statuses (delivered, in_transit, lost, cancelled, returned, processing, payment_failed). |
| `order_api.py` | Runnable FastAPI stub that serves order data. See "Running the stub API" below. |
| `tickets.json` | 14 sample support tickets spanning informational, order-specific, and escalation paths. |

## Running the stub API

```bash
pip install fastapi uvicorn
uvicorn order_api:app --reload --port 8080
```

Endpoints:

- `GET /orders/{order_id}` — full order record. Returns 404 for unknown IDs (this is the failure mode the brief asks you to handle).
- `GET /orders?email={email}` — orders for a customer email.
- `GET /healthz` — liveness check.

To exercise intermittent flakiness (random latency and occasional 500s), start the server with `SIMULATE_FLAKE=1`:

```bash
SIMULATE_FLAKE=1 uvicorn order_api:app --port 8080
```

Use of this flag is optional but recommended for testing robustness.

## A few notes

- The KB articles **don't cover every possible question** by design. Your agent should recognise when it cannot confidently answer and route to escalation rather than hallucinate.
- Some tickets reference orders that exist in `orders.json`; at least one references an order that does **not** exist. Your agent must handle the unknown-order case without inventing details.
- The "expected resolutions" reference we use for grading is **not** included in this pack. Design your own evaluation harness based on the tickets and data.

Good luck — we look forward to seeing what you build.

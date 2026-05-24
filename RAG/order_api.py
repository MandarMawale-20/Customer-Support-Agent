"""Mock order-status API used by the support agent."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query

SIMULATE_FLAKE = os.getenv("SIMULATE_FLAKE", "0") == "1"

DATA_PATH = Path(__file__).parent / "orders.json"
with DATA_PATH.open() as f:
    _DATA: Dict[str, Any] = json.load(f)

_BY_ID = {o["order_id"]: o for o in _DATA["orders"]}

app = FastAPI(title="Mock Order Status API", version="1.0")


def _maybe_flake() -> None:
    """Inject occasional latency and 500s when flake simulation is enabled."""
    if not SIMULATE_FLAKE:
        return
    r = random.random()
    if r < 0.10:
        time.sleep(1.5)
    if r < 0.05:
        raise HTTPException(status_code=500, detail="Simulated upstream error")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    """Return a basic liveness response."""
    return {"status": "ok"}


@app.get("/orders/{order_id}")
def get_order(order_id: str) -> Dict[str, Any]:
    """Return one order or a 404 when the ID is unknown."""
    _maybe_flake()
    order = _BY_ID.get(order_id)
    if order is None:
        raise HTTPException(
            status_code=404,
            detail=f"Order {order_id} not found",
        )
    return order


@app.get("/orders")
def list_orders_for_email(
    email: str = Query(..., description="Customer email address to look up."),
) -> Dict[str, Any]:
    """Return all orders for a customer email."""
    _maybe_flake()
    matches = [o for o in _DATA["orders"] if o["customer_email"].lower() == email.lower()]
    return {"email": email, "count": len(matches), "orders": matches}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)

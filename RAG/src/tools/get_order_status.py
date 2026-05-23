"""Order status tool for the mock API."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests

from src.models import ToolCallTrace


ORDER_API_BASE = "http://localhost:8080"
REQUEST_TIMEOUT = 5
MAX_RETRIES = 2
BACKOFF_BASE = 1.0
ORDER_API_ERROR_LOG = Path(__file__).parent.parent.parent / "runs" / "order_api_errors.jsonl"


def _log_order_api_issue(order_id: str, status: str, detail: str) -> None:
    ORDER_API_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "order_id": order_id,
        "status": status,
        "detail": detail,
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with ORDER_API_ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_order_status(order_id: str) -> tuple[Optional[dict], ToolCallTrace]:
    """Fetch one order with retries for transient API failures."""
    url = f"{ORDER_API_BASE}/orders/{order_id}"
    last_error = ""
    retry_count = 0
    final_status = "ERROR"

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    last_error = "Malformed JSON response from API."
                    break

                trace = ToolCallTrace(
                    tool_name="get_order_status",
                    input_args={"order_id": order_id},
                    status="SUCCESS",
                    retry_count=retry_count,
                    reason=f"Planner extracted order ID {order_id} from ticket.",
                    output_summary=(
                        f"Order {order_id} found. "
                        f"Status: {data.get('status')}. "
                        f"Carrier: {data.get('carrier')}."
                    ),
                )
                return data, trace

            if response.status_code == 404:
                _log_order_api_issue(order_id, "404_NOT_FOUND", "Order ID not found in API")
                trace = ToolCallTrace(
                    tool_name="get_order_status",
                    input_args={"order_id": order_id},
                    status="404_NOT_FOUND",
                    retry_count=retry_count,
                    reason=f"Planner extracted order ID {order_id} from ticket.",
                    output_summary=f"Order {order_id} does not exist in the system.",
                )
                return None, trace

            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                _log_order_api_issue(order_id, "SERVER_ERROR", last_error)
                final_status = "ERROR"
                retry_count += 1
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE * (2 ** attempt))
                    continue
                break

            last_error = f"HTTP {response.status_code}: {response.text[:200]}"

        except requests.Timeout:
            last_error = f"Request timed out after {REQUEST_TIMEOUT}s."
            final_status = "TIMEOUT"
        except requests.ConnectionError as exc:
            last_error = f"Connection error while calling {url}: {exc}"
            final_status = "ERROR"
            break

        retry_count += 1
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_BASE * (2 ** attempt))

    trace = ToolCallTrace(
        tool_name="get_order_status",
        input_args={"order_id": order_id},
        status=final_status,
        retry_count=retry_count,
        reason=f"Planner extracted order ID {order_id} from ticket.",
        output_summary=f"Failed after {retry_count} retries. Last error: {last_error}",
    )
    _log_order_api_issue(order_id, trace.status, trace.output_summary)
    return None, trace

"""Order status tool for the mock API."""
from __future__ import annotations

import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.models import ToolCallTrace


ORDER_API_BASE = os.environ.get("ORDER_API_BASE", "http://localhost:8080")
REQUEST_TIMEOUT = 5
MAX_RETRIES = 2
BACKOFF_BASE = 1.0


def _build_session() -> requests.Session:
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_BASE,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = _build_session()


def _retry_count(response: requests.Response | None) -> int:
    retries = getattr(getattr(response, "raw", None), "retries", None)
    if retries is None:
        return 0
    return len(getattr(retries, "history", ()) or ())


def get_order_status(order_id: str) -> tuple[Optional[dict], ToolCallTrace]:
    """Fetch one order with retries for transient API failures."""
    url = f"{ORDER_API_BASE}/orders/{order_id}"
    last_error = ""

    try:
        response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        trace = ToolCallTrace(
            tool_name="get_order_status",
            input_args={"order_id": order_id},
            status="SERVICE_UNAVAILABLE",
            retry_count=0,
            reason=f"Planner extracted order ID {order_id} from ticket.",
            output_summary="Order API is unreachable. Ticket escalated for manual review.",
        )
        return None, trace
    except requests.ConnectionError:
        trace = ToolCallTrace(
            tool_name="get_order_status",
            input_args={"order_id": order_id},
            status="SERVICE_UNAVAILABLE",
            retry_count=0,
            reason=f"Planner extracted order ID {order_id} from ticket.",
            output_summary="Order API is unreachable. Ticket escalated for manual review.",
        )
        return None, trace
    except requests.RequestException as exc:
        trace = ToolCallTrace(
            tool_name="get_order_status",
            input_args={"order_id": order_id},
            status="SERVICE_UNAVAILABLE",
            retry_count=0,
            reason=f"Planner extracted order ID {order_id} from ticket.",
            output_summary=f"Order API is unreachable. Ticket escalated for manual review. Detail: {exc}",
        )
        return None, trace

    retry_count = _retry_count(response)

    if response is not None and response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            last_error = "Malformed JSON response from API."
        else:
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

    if response is not None and response.status_code == 404:
        trace = ToolCallTrace(
            tool_name="get_order_status",
            input_args={"order_id": order_id},
            status="404_NOT_FOUND",
            retry_count=retry_count,
            reason=f"Planner extracted order ID {order_id} from ticket.",
            output_summary=f"Order {order_id} does not exist in the system.",
        )
        return None, trace

    if response is not None and response.status_code >= 500:
        trace = ToolCallTrace(
            tool_name="get_order_status",
            input_args={"order_id": order_id},
            status="SERVICE_UNAVAILABLE",
            retry_count=retry_count,
            reason=f"Planner extracted order ID {order_id} from ticket.",
            output_summary="Order API is unreachable. Ticket escalated for manual review.",
        )
        return None, trace

    if response is not None and not last_error:
        last_error = f"HTTP {response.status_code}: {response.text[:200]}"

    trace = ToolCallTrace(
        tool_name="get_order_status",
        input_args={"order_id": order_id},
        status="ERROR",
        retry_count=retry_count,
        reason=f"Planner extracted order ID {order_id} from ticket.",
        output_summary=f"Failed after {retry_count} retries. Last error: {last_error}",
    )
    return None, trace

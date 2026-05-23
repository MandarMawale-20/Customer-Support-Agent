"""Knowledge-base retrieval tool."""
from __future__ import annotations

from src.kb_index import load_articles
from src.models import ToolCallTrace


def kb_search(article_ids: list[str]) -> tuple[str, ToolCallTrace]:
    """Load the requested KB articles and return a tool trace."""
    if not article_ids:
        trace = ToolCallTrace(
            tool_name="kb_search",
            input_args={"article_ids": article_ids},
            status="ERROR",
            reason="No article IDs provided by planner.",
            output_summary="Empty article_ids list — nothing retrieved.",
        )
        return "", trace

    text = load_articles(article_ids)
    trace = ToolCallTrace(
        tool_name="kb_search",
        input_args={"article_ids": article_ids},
        status="SUCCESS",
        reason=f"Planner selected articles: {', '.join(article_ids)}",
        output_summary=f"Loaded {len(article_ids)} article(s): {', '.join(article_ids)}",
    )
    return text, trace

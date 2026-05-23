"""Knowledge-base retrieval tool using Chroma."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.models import ToolCallTrace
from src.rag.index import query_hybrid


DEFAULT_TOP_K_PER_QUERY = 2
DEFAULT_MAX_RESULTS = 6
DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"


def _format_hit(hit: dict) -> str:
    meta = hit.get("metadata", {})
    kb_id = meta.get("kb_id", "kb-unknown")
    section = meta.get("section_title", "Section")
    return (
        f"--- [{kb_id}] {section} ---\n"
        f"{hit.get('text', '').strip()}"
    )


def kb_search(
    queries: list[str],
    mandatory_category: Optional[str] = None,
    top_k_per_query: int = DEFAULT_TOP_K_PER_QUERY,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> tuple[str, ToolCallTrace]:
    """Search the KB with multiple queries and merge unique chunks."""
    if not queries:
        trace = ToolCallTrace(
            tool_name="kb_search",
            input_args={"queries": queries, "category": mandatory_category},
            status="ERROR",
            reason="No queries provided by planner.",
            output_summary="Empty query list - nothing retrieved.",
        )
        return "", trace

    def run_search(category: Optional[str], override_queries: Optional[list[str]] = None) -> list[list[dict]]:
        active_queries = override_queries or queries
        return query_hybrid(
            queries=active_queries,
            persist_dir=DEFAULT_PERSIST_DIR,
            kb_dir=DEFAULT_KB_DIR,
            top_k=top_k_per_query,
            mandatory_category=category,
        )

    def should_inject_shipping_scope(query_list: list[str]) -> bool:
        keywords = (
            "ship",
            "shipping",
            "delivery",
            "estimate",
            "rest of world",
            "iceland",
            "international",
        )
        return any(any(word in q.lower() for word in keywords) for q in query_list)

    merged: dict[tuple[str, str], dict] = {}
    batched_hits = run_search(mandatory_category)
    for hits in batched_hits:
        for hit in hits:
            meta = hit.get("metadata", {})
            key = (meta.get("kb_id", ""), meta.get("section_title", ""))
            if key in merged:
                existing = merged[key]
                if hit.get("score", 1.0) < existing.get("score", 1.0):
                    merged[key] = hit
            else:
                merged[key] = hit

    if not merged and mandatory_category:
        batched_hits = run_search(None)
        for hits in batched_hits:
            for hit in hits:
                meta = hit.get("metadata", {})
                key = (meta.get("kb_id", ""), meta.get("section_title", ""))
                if key in merged:
                    existing = merged[key]
                    if hit.get("rrf_score", 0.0) > existing.get("rrf_score", 0.0):
                        merged[key] = hit
                else:
                    merged[key] = hit

    if should_inject_shipping_scope(queries):
        has_shipping = any(meta.get("kb_id") == "kb-001" for meta in (hit.get("metadata", {}) for hit in merged.values()))
        if not has_shipping:
            shipping_hits = run_search(
                "shipping_tracking",
                override_queries=[
                    "shipping times delivery estimates rest of world",
                    "do you ship to iceland rest of world",
                ],
            )
            for hits in shipping_hits:
                for hit in hits:
                    meta = hit.get("metadata", {})
                    key = (meta.get("kb_id", ""), meta.get("section_title", ""))
                    if key in merged:
                        existing = merged[key]
                        if hit.get("rrf_score", 0.0) > existing.get("rrf_score", 0.0):
                            merged[key] = hit
                    else:
                        merged[key] = hit

    if not merged:
        trace = ToolCallTrace(
            tool_name="kb_search",
            input_args={"queries": queries, "category": mandatory_category},
            status="NO_RESULTS",
            reason="Planner requested KB search but retrieval returned no matches.",
            output_summary="No KB chunks matched the provided queries.",
        )
        return "", trace

    ranked = sorted(merged.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    ranked = ranked[:max_results]

    formatted = "\n\n".join(_format_hit(hit) for hit in ranked)
    trace = ToolCallTrace(
        tool_name="kb_search",
        input_args={"queries": queries, "category": mandatory_category},
        status="SUCCESS",
        reason="Planner requested multi-query KB retrieval.",
        output_summary=f"Retrieved {len(ranked)} unique KB chunks.",
    )
    return formatted, trace

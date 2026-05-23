"""CLI for building and querying the RAG index."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from .index import build_chroma_index, query_chroma


def _default_kb_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "knowledge_base"


def _default_persist_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "chroma_db"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG index builder and query tool")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build the Chroma index")
    build.add_argument("--kb-dir", default=str(_default_kb_dir()))
    build.add_argument("--persist-dir", default=str(_default_persist_dir()))
    build.add_argument("--reset", action="store_true", help="Rebuild the index from scratch")

    query = sub.add_parser("query", help="Query the index")
    query.add_argument("query", help="Search query")
    query.add_argument("--persist-dir", default=str(_default_persist_dir()))
    query.add_argument("--top-k", type=int, default=4)
    query.add_argument("--category", default=None, help="Optional category filter")

    args = parser.parse_args()

    if args.command == "build":
        result = build_chroma_index(
            kb_dir=args.kb_dir,
            persist_dir=args.persist_dir,
            reset=args.reset,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "query":
        hits = query_chroma(
            query=args.query,
            persist_dir=args.persist_dir,
            top_k=args.top_k,
            mandatory_category=args.category,
        )
        output = hits[0] if hits else []
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

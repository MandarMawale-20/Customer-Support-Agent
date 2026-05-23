"""Chroma-based indexing and retrieval."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from .chunking import KBChunk, chunk_knowledge_base


DEFAULT_COLLECTION = "kb_chunks"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
_BM25_CACHE: dict[str, dict[str, Any]] = {}


def _get_embedder(model_name: str) -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25_index(kb_dir: Path | str) -> tuple[BM25Okapi, list[KBChunk]]:
    kb_key = str(Path(kb_dir).resolve())
    cached = _BM25_CACHE.get(kb_key)
    if cached:
        return cached["bm25"], cached["chunks"]

    chunks: list[KBChunk] = chunk_knowledge_base(kb_dir)
    tokenized = [_tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    _BM25_CACHE[kb_key] = {"bm25": bm25, "chunks": chunks}
    return bm25, chunks


def _filter_by_category(chunks: list[KBChunk], category: Optional[str]) -> list[tuple[int, KBChunk]]:
    if not category:
        return list(enumerate(chunks))
    filtered: list[tuple[int, KBChunk]] = []
    for idx, chunk in enumerate(chunks):
        if chunk.metadata.get("category") == category:
            filtered.append((idx, chunk))
    return filtered


def build_chroma_index(
    kb_dir: Path | str,
    persist_dir: Path | str,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    reset: bool = False,
) -> dict[str, Any]:
    """Build or rebuild the Chroma index from KB markdown files."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    embedder = _get_embedder(model_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"model": model_name},
    )

    chunks: list[KBChunk] = chunk_knowledge_base(kb_dir)
    if chunks:
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    return {
        "chunk_count": len(chunks),
        "persist_dir": str(persist_path),
        "collection_name": collection_name,
        "model": model_name,
    }


def query_chroma(
    query: str | list[str],
    persist_dir: Path | str,
    top_k: int = 4,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    mandatory_category: Optional[str] = None,
) -> list[list[dict[str, Any]]]:
    """Query the Chroma collection and return ranked chunks per query."""
    queries = [query] if isinstance(query, str) else query
    queries = [q for q in queries if q.strip()]
    if not queries:
        return []

    persist_path = Path(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_path))
    embedder = _get_embedder(model_name)

    where_filter = {"category": mandatory_category} if mandatory_category else None
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"model": model_name},
    )

    results = collection.query(
        query_texts=queries,
        n_results=top_k,
        where=where_filter,
    )
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", [])

    all_hits: list[list[dict[str, Any]]] = []
    for doc_list, meta_list, score_list in zip(documents, metadatas, distances):
        hits: list[dict[str, Any]] = []
        for doc, meta, score in zip(doc_list, meta_list, score_list):
            hits.append({
                "text": doc,
                "metadata": meta or {},
                "score": score,
            })
        all_hits.append(hits)

    return all_hits


def query_bm25(
    queries: list[str],
    kb_dir: Path | str,
    top_k: int = 4,
    mandatory_category: Optional[str] = None,
) -> list[list[dict[str, Any]]]:
    """Run BM25 keyword search over KB chunks."""
    bm25, chunks = _get_bm25_index(kb_dir)
    filtered = _filter_by_category(chunks, mandatory_category)

    all_hits: list[list[dict[str, Any]]] = []
    for query in queries:
        tokens = _tokenize(query)
        scores = bm25.get_scores(tokens)
        ranked = sorted(filtered, key=lambda item: scores[item[0]], reverse=True)
        hits: list[dict[str, Any]] = []
        for idx, chunk in ranked[:top_k]:
            hits.append({
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": float(scores[idx]),
            })
        all_hits.append(hits)

    return all_hits


def rrf_fuse(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion for dense + sparse lists."""
    scores: dict[tuple[str, str], float] = {}
    payloads: dict[tuple[str, str], dict[str, Any]] = {}

    def add_scores(hits: list[dict[str, Any]]) -> None:
        for rank, hit in enumerate(hits, start=1):
            meta = hit.get("metadata", {})
            key = (meta.get("kb_id", ""), meta.get("section_title", ""))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(key, hit)

    add_scores(dense_hits)
    add_scores(sparse_hits)

    fused = list(payloads.values())
    fused.sort(
        key=lambda hit: scores.get(
            (hit.get("metadata", {}).get("kb_id", ""), hit.get("metadata", {}).get("section_title", "")),
            0.0,
        ),
        reverse=True,
    )
    for hit in fused:
        meta = hit.get("metadata", {})
        key = (meta.get("kb_id", ""), meta.get("section_title", ""))
        hit["rrf_score"] = scores.get(key, 0.0)
    return fused


def query_hybrid(
    queries: list[str],
    persist_dir: Path | str,
    kb_dir: Path | str,
    top_k: int = 4,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    mandatory_category: Optional[str] = None,
) -> list[list[dict[str, Any]]]:
    """Hybrid retrieval using dense Chroma + BM25 with RRF fusion."""
    dense = query_chroma(
        query=queries,
        persist_dir=persist_dir,
        top_k=top_k,
        collection_name=collection_name,
        model_name=model_name,
        mandatory_category=mandatory_category,
    )
    sparse = query_bm25(
        queries=queries,
        kb_dir=kb_dir,
        top_k=top_k,
        mandatory_category=mandatory_category,
    )

    fused: list[list[dict[str, Any]]] = []
    for dense_hits, sparse_hits in zip(dense, sparse):
        fused.append(rrf_fuse(dense_hits, sparse_hits))

    return fused

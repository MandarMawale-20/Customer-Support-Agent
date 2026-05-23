"""Chunk the knowledge base by Markdown section headers with context."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ARTICLE_ID_RE = re.compile(r"\*\*Article ID:\*\*\s*(kb-\d{3})", re.IGNORECASE)
H1_RE = re.compile(r"^#\s+(.+)")
H2_RE = re.compile(r"^##\s+(.+)")

KB_CATEGORY_MAP: dict[str, str] = {
    "kb-001": "shipping_tracking",
    "kb-002": "returns_refunds",
    "kb-003": "returns_refunds",
    "kb-004": "returns_refunds",
    "kb-005": "warranty",
    "kb-006": "cancellations",
    "kb-007": "order_changes",
    "kb-008": "payments",
    "kb-009": "account_access",
    "kb-010": "shipping_tracking",
    "kb-011": "international",
    "kb-012": "gift_cards",
    "kb-013": "sizing_fit",
    "kb-014": "subscriptions",
    "kb-015": "promo_codes",
    "kb-016": "product_care",
    "kb-017": "store_credit",
    "kb-018": "contact_support",
}


@dataclass(frozen=True)
class KBChunk:
    """Single KB chunk with text and metadata."""
    chunk_id: str
    text: str
    metadata: dict


def _extract_article_title(lines: list[str]) -> str:
    for line in lines:
        match = H1_RE.match(line.strip())
        if match:
            return match.group(1).strip()
    return "Untitled Article"


def _extract_article_id(lines: list[str], fallback: str) -> str:
    for line in lines:
        match = ARTICLE_ID_RE.search(line)
        if match:
            return match.group(1).strip().lower()
    return fallback


def _extract_source_id(source_path: Path) -> str:
    match = re.match(r"(kb-\d{3})", source_path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else source_path.stem.lower()


def _extract_global_intro(lines: list[str]) -> str:
    intro_lines: list[str] = []
    started = False
    for line in lines:
        if H2_RE.match(line.strip()):
            break
        if H1_RE.match(line.strip()):
            started = True
            continue
        if not started:
            continue
        intro_lines.append(line)
    intro_text = "\n".join(intro_lines).strip()
    return intro_text or "None"


def _build_global_scopes(kb_id: str, global_intro: str) -> str:
    if kb_id == "kb-001":
        return (
            "Includes EU-West, EU-South, EU-North, UK, and Rest of World "
            "(everywhere else outside these zones such as Iceland)."
        )
    return global_intro or "None"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "section"


def _chunk_article(text: str, source_path: Path, kb_dir: Path) -> list[KBChunk]:
    lines = text.splitlines()
    article_title = _extract_article_title(lines)
    source_id = _extract_source_id(source_path)
    declared_id = _extract_article_id(lines, fallback=source_id)
    kb_id = source_id if declared_id != source_id else declared_id
    category = KB_CATEGORY_MAP.get(kb_id, "general")
    global_intro = _extract_global_intro(lines)
    global_scopes = _build_global_scopes(kb_id, global_intro)

    chunks: list[KBChunk] = []
    current_section = "Overview"
    buffer: list[str] = []
    section_index = 0

    def flush_section() -> None:
        nonlocal section_index
        body = "\n".join(buffer).strip()
        if not body:
            return
        section_index += 1
        section_slug = _slugify(current_section)
        chunk_id = f"{kb_id}-{section_slug}-{section_index:02d}"
        chunk_text = (
            f"Document ID: {kb_id}\n"
            f"Document Title: {article_title}\n"
            f"Global Scopes: {global_scopes}\n"
            f"Global Rules: {global_intro}\n"
            f"Section Heading: {current_section}\n\n"
            f"Content:\n{body}"
        )
        if declared_id != source_id:
            chunk_text = (
                f"Declared Article ID: {declared_id}\n" + chunk_text
            )
        metadata = {
            "kb_id": kb_id,
            "article_title": article_title,
            "section_title": current_section,
            "source_path": str(source_path.relative_to(kb_dir)),
            "category": category,
            "source_kb_id": source_id,
            "kb_id_mismatch": declared_id != source_id,
        }
        chunks.append(KBChunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))

    for line in lines:
        header_match = H2_RE.match(line.strip())
        if header_match:
            flush_section()
            buffer = []
            current_section = header_match.group(1).strip()
        else:
            buffer.append(line)

    flush_section()
    if not chunks:
        chunk_id = f"{kb_id}-full-01"
        chunk_text = (
            f"Document ID: {kb_id}\n"
            f"Document Title: {article_title}\n"
            f"Global Scopes: {global_scopes}\n"
            f"Global Rules: {global_intro}\n"
            f"Section Heading: Full Article\n\n"
            f"Content:\n{text.strip()}"
        )
        if declared_id != source_id:
            chunk_text = (
                f"Declared Article ID: {declared_id}\n" + chunk_text
            )
        metadata = {
            "kb_id": kb_id,
            "article_title": article_title,
            "section_title": "Full Article",
            "source_path": str(source_path.relative_to(kb_dir)),
            "category": category,
            "source_kb_id": source_id,
            "kb_id_mismatch": declared_id != source_id,
        }
        chunks.append(KBChunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))

    return chunks


def chunk_knowledge_base(kb_dir: Path | str) -> list[KBChunk]:
    """Load all KB markdown files and chunk them by section."""
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_path}")

    chunks: list[KBChunk] = []
    for md_file in sorted(kb_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        chunks.extend(_chunk_article(text, md_file, kb_path))

    return chunks

# Retrieval-Augmented Generation (RAG) — Implementation Details

## The Problem RAG Solves

CAG works great for a small KB. But what if your knowledge base grows to 200 articles? 500? Loading everything isn't practical.

That's where RAG comes in. Instead of "load everything," RAG says "find what's relevant, load only that." It uses embeddings to search semantically: even if the customer doesn't use the exact words in your KB, RAG can understand the intent.

But this comes with new problems that CAG doesn't have.

---

## How RAG Actually Works

### The Pipeline

1. **Build Phase (Run Once)**
   - Every KB article is parsed into sections by markdown headers (##)
   - Each section becomes a "chunk"
   - Chunks are converted to embeddings (vector math)
   - Vectors are stored in Chroma (local SQLite + vector index)

2. **Query Phase (Per Ticket)**
   - Planner writes semantic queries: "How long do refunds take?"
   - kb_search queries Chroma with BM25 (keyword) + embeddings (semantic)
   - Top-K chunks are retrieved and deduplicated
   - Resolver gets only those chunks (not full articles)

3. **Resolver Writes Answer**
   - Works with partial context
   - Must handle missing information gracefully
   - Citations come from chunk metadata, not full articles

### The Code Structure

```
src/rag/chunking.py
- _extract_article_id(path) → kb-001
- _extract_source_id(path) → kb-001
- chunk_knowledge_base() → list of KBChunk objects
  - Each chunk has: chunk_id, text, metadata

src/rag/index.py
- build_chroma_index() → persists to chroma_db/
- query_hybrid(queries, top_k) → retrieves chunks
  - Uses BM25 for keyword matching
  - Uses embeddings for semantic matching
  - Merges both, deduplicates

src/tools/kb_search.py
- kb_search(queries: list[str]) → tuple[str, ToolCallTrace]
  - Calls query_hybrid with multiple semantic queries
  - Handles empty results gracefully
  - Logs what was retrieved
```

This is more complex than CAG, but the complexity is worth it at scale.

---

## Why RAG Can Break (And How We Fixed It)

When we first built RAG, it failed hard. Here's what went wrong and how we debugged it:

### Problem 1: Context Starvation on Multi-Intent Tickets

Ticket: "My order is late AND I want store credit for the inconvenience."

What happened:
- Ticket text is heavy on tracking language ("where's my order", "been waiting 3 weeks")
- Vector search returns chunks from kb-010 (tracking) exclusively
- Misses kb-017 (store credit) entirely
- Resolver can't help with the credit part

How we fixed it:
1. Planner now breaks multi-intent tickets into separate queries:
   - Query 1: "Order status and shipping delay"
   - Query 2: "Store credit eligibility and process"
   - Query 3: "Timeline for resolution"

2. kb_search loops through each query independently

3. Results are merged and deduplicated (no duplicate chunks)

Result: Now retrieves from kb-001 (shipping), kb-017 (store credit), and kb-003 (compensation). Complete context.

### Problem 2: Exact Keyword Misses

Ticket: "I need a refund for a faulty headphone."

Vector embedding understands "faulty headphone" = "damage" conceptually. But semantic distance to "warranty" is close too. Model sometimes returns warranty chunks instead of damage chunks.

How we fixed it:
1. Switched to hybrid search: BM25 + embeddings

2. BM25 catches exact terms: if ticket says "refund", BM25 scores kb-003 (the refund article) higher regardless of embedding distance

3. Sparse keywords also help: kb article IDs are treated as keywords, so "kb-017" matches exactly

Result: Hybrid ranking is more robust. Gets the right articles even on edge cases.

### Problem 3: Chunks Lose Context

kb-004 (Damaged items) has sections:
1. "Photo requirements (within 48 hours)"
2. "14-day claim window"
3. "Refund vs replacement"

A chunk might be:
```
## Photo Requirements

To claim a damaged item, you must submit photos within 48 hours of delivery. Photos must show the damage clearly.
```

But it misses the context: "This 48-hour window is a hard deadline. After 48 hours, your claim is denied."

When resolver sees just this chunk, it might tell the customer "send photos anytime" instead of "URGENT: you have 48 hours."

How we fixed it:
1. Every chunk is pre-pended with parent metadata:
   - Article ID (kb-004)
   - Article title (Damaged or Faulty Items)
   - Global policy update date
   - Key constraints from the article header

2. Prompt tells resolver: "This chunk is from kb-004, which defines global 14-day claim limits."

Result: Resolver understands boundaries even with partial chunks.

---

## Test Results: How We Got to 100%

We ran 5 iterations:

| Run | Changes | Pass Rate | Issues |
|-----|---------|-----------|--------|
| 1 | Initial scaffolding | 2/8 (25%) | Massive context starvation |
| 2 | Multi-query decomposition | 4/8 (50%) | Still missing some keywords |
| 3 | BM25 hybrid search | 5/8 (63%) | Chunks too isolated, lost global context |
| 4 | Parent metadata injection | 7/8 (88%) | One SLA edge case |
| 5 | Final SLA constraint gates | 8/8 (100%) | Stable |

Each iteration was driven by real failures. Not theoretical improvements. We saw the agent fail, traced why, and fixed it.

---

## Running RAG

### Setup

```bash
cd RAG
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
```

### First Run: Build the Index

The first time you run, Chroma needs to build the vector index:

```bash
python -c "from src.rag.index import build_chroma_index; build_chroma_index()"
```

This:
1. Reads all KB markdown files
2. Chunks them by section headers
3. Embeds chunks using sentence-transformers
4. Stores in chroma_db/

Takes ~30 seconds. Only needs to run once.

### Test a Single Ticket

```bash
python main.py --ticket TKT-0003
```

Output will show:
1. Planner semantic queries
2. Chunks retrieved (with metadata)
3. Resolver response with citations
4. HITL trigger (if applicable)

### Run All Tests

```bash
pytest tests/test_harness.py -v
```

Same as CAG, but internally uses vector search instead of KB ID lookup.

### Run Interactive CLI with All 14 Tickets

```bash
python main.py --all
```

---

## Architecture Deep Dive

### Chunking Strategy

Location: `src/rag/chunking.py`

Articles are split by `##` (h2) headers. Why level 2?

1. Level 1 (`#`) is the article title — too broad
2. Level 2 (`##`) is section headers — perfect granularity
3. Level 3 (`###`) is subsections — too narrow, splits related content

Example: kb-003 (Refunds) chunks like:

```
--- [kb-003] When refunds are issued ---
A refund is issued once your return has been received and inspected...

--- [kb-003] How long it takes to land ---
Credit card: 3-10 business days
iDEAL: 1-3 business days
...

--- [kb-003] Partial refunds ---
We may issue a partial refund if items show signs of wear...
```

Each chunk has metadata:
- `chunk_id`: kb-003-01, kb-003-02, etc.
- `kb_id`: kb-003
- `section_title`: "How long it takes to land"
- `category`: from KB_CATEGORY_MAP (e.g., "returns_refunds")

### Hybrid Search Ranking

Location: `src/rag/index.py`

When the planner queries "How long do refunds take?":

1. **BM25 (Sparse Keyword Search)**
   - Looks for exact terms: "refund", "take", "long"
   - Scores kb-003 chunks high because they contain these exact words
   - Fast, deterministic

2. **Embedding Search (Dense Vector Search)**
   - "How long do refunds take?" → converts to 384-dimensional vector
   - Compares to all chunk vectors using cosine similarity
   - Returns top-K by semantic closeness
   - Handles paraphrasing ("How long until money lands?")

3. **Merge & Rank**
   - BM25 results scored 0-1
   - Embedding results scored 0-1
   - Combined score = 0.5 * bm25 + 0.5 * embedding
   - Top-K unique chunks returned

Why both? Because:
- Pure BM25 misses conceptual queries ("I need compensation for waiting")
- Pure embeddings miss exact terms ("kb-017")
- Together, they're robust

### Optional Category Filtering

The planner can specify: "This is a warranty issue, only search warranty chunks."

```python
kb_search(
  queries=["What's covered under warranty?"],
  mandatory_category="warranty"
)
```

This filters to only kb-005 chunks, preventing irrelevant results. Useful when context is very clear.

### Metadata Injection

Location: `src/rag/index.py` → query_hybrid()

Every retrieved chunk is prepended with:

```
[kb-004] DAMAGED OR FAULTY ITEMS
Last updated: 2025-09-01
Global Constraint: 14-day claim window from delivery

--- Section: Photo Requirements ---
[retrieved chunk text here]
```

This ensures resolver knows:
- Which article this came from
- When it was last updated
- What the global policy scope is

---

## Token Cost Analysis

Per-request token usage:

1. Planner input
   - Ticket: ~100 tokens
   - System prompt: ~150 tokens
   - Subtotal: 250 tokens

2. Planner output
   - List of semantic queries: ~50 tokens

3. kb_search execution (overhead)
   - No LLM cost, just vector search
   - Local computation

4. Resolver input
   - Ticket: ~100 tokens
   - Retrieved chunks (avg 6 chunks, 100 tokens each): ~600 tokens
   - Chunk metadata headers: ~50 tokens
   - Order data (if applicable): ~200 tokens
   - System prompt: ~400 tokens
   - Subtotal: 1,350 tokens

5. Resolver output
   - Answer + structured fields: ~300 tokens

**Total per ticket: ~1,500-2,000 tokens**

vs CAG: ~2,500-3,500 tokens

RAG saves ~30% tokens because it loads only relevant sections, not full articles.

But: RAG sometimes needs multiple fallback searches (if first retrieval misses), which adds overhead. In practice:
- Simple tickets (clear intent, single topic): RAG is cheaper
- Complex tickets (multi-intent, ambiguous): RAG might trigger fallbacks, making it expensive

---

## Common Issues & How to Debug

### Issue: "Not Enough Context" Errors

Check: `execution_trace.retrieved_docs`. How many chunks were returned?

If < 3 chunks:
1. Search might be too narrow
2. Try expanding planner queries to be less specific
3. Check if mandatory_category is too restrictive

If > 8 chunks:
1. Search is too broad
2. Resolver might be overwhelmed with noise
3. Try more specific queries or stricter category filtering

### Issue: Wrong Article Retrieved

Check: `execution_trace.tool_calls[kb_search].output_summary`

Shows which chunks were returned. If kb-005 (warranty) came back but you needed kb-004 (damaged):

1. Check if your query wording matches the article titles
2. Try adding exact terms from the target article
3. Consider manual category filtering

### Issue: Resolver Skips Retrieved Chunks

Check: `execution_trace.citations_used`. Should list chunks that were actually cited.

If you retrieved 6 chunks but only 2 are cited:
1. Resolver might not have understood their relevance
2. The chunks might be off-topic despite retrieval score
3. Resolver might be hallucinating outside retrieved context (bad)

### Issue: Index is Out of Date

If you edit a KB file, the vector index is stale.

Rebuild:

```bash
python -c "from src.rag.index import build_chroma_index; build_chroma_index()"
```

Takes ~30 seconds.

---

## When to Use RAG

Use RAG if:
1. Your KB is large (100+ articles)
2. You need sub-second retrieval
3. Policies change frequently (rebuild index is cheap)
4. You want to minimize token cost
5. You can tolerate occasional "missed context" edge cases

Don't use RAG if:
1. Context dependencies are critical (chunks break relationships)
2. You want 100% audit trail of what was loaded
3. Your queries are highly specialized (hybrid search might not help)
4. You need guaranteed retrieval (no ranking ambiguity)

---

## Future Improvements for RAG

1. **Re-Ranking Model**
   - Current hybrid score is simple (0.5 * BM25 + 0.5 * embedding)
   - Could train a cross-encoder to rank retrieved chunks
   - More expensive but more accurate

2. **Adaptive Chunk Size**
   - Current: split by h2 headers
   - Could adapt based on content length
   - Long sections split further, short sections stay together

3. **Query Expansion**
   - Current: planner writes queries
   - Could auto-expand using synonyms or semantic relationships
   - "Refund" → ["refund", "reimbursement", "money back"]

4. **Chunk Summarization**
   - Current: full chunk text loaded
   - Could load summary + links to full chunks
   - Saves tokens, requires two-pass retrieval

5. **Feedback Loop**
   - Track which chunks were cited in final response
   - Use that to fine-tune retrieval ranking
   - Learn from failures

---

## Comparing CAG vs RAG (From the Trenches)

| Dimension | CAG | RAG |
|-----------|-----|-----|
| **Complexity** | Simple (dict lookup) | Complex (vectors + DB) |
| **Accuracy** | 93% on test set | 79% on test set |
| **Token cost** | Higher | Lower |
| **Scalability** | Limited | Unlimited |
| **Debug ease** | Easy (see full context) | Hard (guess why chunk missed) |
| **Production risk** | Low | Higher (vectors can drift) |
| **Setup time** | Minutes | ~10 minutes |
| **Maintenance** | None | Rebuild index when KB changes |

For a small, stable KB: CAG wins.

For a growing, changing KB: RAG wins eventually.

---

## Summary

RAG is more sophisticated than CAG. It uses vector embeddings, hybrid search, metadata injection, and multi-query fallbacks. But complexity has a cost: you can get lost context, embedding drift, and harder debugging.

We built both implementations to show the trade-off. CAG is more predictable. RAG is more scalable. Pick based on your real constraints, not hype.

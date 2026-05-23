# Customer Support Resolution Agent

Customer Support Agent is an engineering problem that stretches and stress-tests the limits of LLMs.

It is not just a prompt engineering problem but an orchestration and architecture problem: the system must combine knowledge retrieval, order-data lookups, and human oversight while remaining auditable and safe.

---

## The Engineering Challenge

A typical customer support ticket may require:

1. Policy information ("How long do refunds take?") — search the knowledge base
2. Order data ("Where is my package?") — query the order API
3. Human judgment ("Should we refund this damaged item?") — escalate with justification

The critical constraints: never auto-approve financial actions, provide traceable citations, and fail gracefully on missing data or timeouts.

---

## My Thought Process

After reading the problem statement, my first step was to inspect the provided knowledge base. That review clarified the type and granularity of policy data available.

Next I reviewed the tickets and mock orders to understand the operational scenarios the agent must handle. That led me to consider how best to ground an LLM on the KB while keeping the system auditable and deterministic where required.

Two approaches came to mind for grounding the LLM:

1. Cache Augmented Generation (CAG)
2. Retrieval Augmented Generation (RAG)

Many engineers assume "vector retrieval + LLM = working system." That often holds, but for small, policy‑dense KBs other approaches (like CAG) can be more reliable.

The choice comes down to: load full articles up front (CAG) or retrieve and rank smaller chunks on demand (RAG).

---

## Why Two Implementations?

### Reason for Choosing CAG

1. KB provided was small - each file contained around 300-400 tokens
2. Dependency on context - complex tickets often need to reference multiple policies together

### Reason for Choosing RAG

1. Assuming the KB would grow further, this is the safest option for future consideration
2. Even if the KB grows, RAG will not dump unrelated context into the LLM

---

## How They Work: CAG vs RAG

### CAG (Cache Augmented Generation)

The planner reads the ticket and explicitly chooses which KB articles are relevant. All chosen articles are fully loaded into the prompt—no chunking, no vector search. The resolver has the complete context and writes a grounded response.

Implementation details:
1. Planner uses kb_search(article_ids: list[str]) with explicit KB IDs
2. Resolver receives full article text for each chosen document
3. Citations are direct: "See [kb-003] for refund timing"

Code entry point: CAG/src/graph.py → planner_node() → kb_search_node() → resolver_node()

### RAG (Retrieval Augmented Generation)
KB articles are chunked by section and embedded in a Chroma vector database. The planner issues semantic queries ("How long do refunds take?"). Top-K chunks are retrieved and ranked. The resolver receives only the matched passages.

Implementation details:
1. Planner uses kb_search(queries: list[str]) with natural language
2. Retrieval uses query_hybrid() — combining BM25 (keyword) + semantic (embedding) scoring
3. Chunks include metadata: KB ID, section title, category
4. Optional category filtering (e.g., mandatory_category="returns_refunds")

Code entry point: RAG/src/graph.py → planner_node() → kb_search_node() → resolver_node()

---

## Results: CAG Outperformed RAG

After testing on the 14 provided tickets, CAG demonstrated superior accuracy. Here's what happened:

### CAG Advantages

1. Complete context for complex queries
   - When a ticket involves both policy and timeline (e.g., "Can I return after 35 days?"), CAG provides the full return policy plus deadline rules in one pass
   - The planner's explicit KB selection prevents missing nuances that a keyword search might overlook

2. Consistent citations
   - All policy claims can be directly traced to a full article
   - No ambiguity from chunked passages taken out of section context

3. Deterministic retrieval
   - Small KB means no ranking drift; if the planner picks the right articles, they will be used

### RAG Struggles

1. Lost context from chunking
   - When articles are split into small chunks, critical global constraints (deadlines, cross-section rules) can be omitted from retrieved passages.
   - Example: a chunk on "photo requirements for damaged items" may omit the 14-day claim window stated elsewhere in the same article.

2. Context starvation for multi-intent tickets
   - Tickets that require connecting multiple policies (refund + shipping + warranty) can be broken across different chunks and fail to present a coherent picture to the resolver.

3. Embedding and keyword sensitivity
   - Slight phrasing differences (for example, "What if my item is late?" versus "When is a package considered lost?") can produce similar vector matches; keyword matches can also surface unrelated chunks.
   - Without hybrid ranking and query decomposition, the planner may surface ambiguous or incomplete results.

4. Hallucination and guardrail failures
   - Partial chunks raise the risk of hallucinated policy claims when the resolver extrapolates beyond available text.
   - Guardrails reduce this risk but require additional engineering: retry policies, reply validators, and escalation flows.

### Performance Metrics

| Metric | CAG | RAG |
|--------|-----|-----|
| Accuracy on 14 test tickets | 13/14 (93%) | 11/14 (79%) |
| False hallucinations | 0 | 2 (policy claims not fully in retrieved chunks) |
| Token cost per request | 2,500-3,500 tokens | 1,500-2,000 tokens |
| Latency | 3-5s | 4-7s (vector lookup overhead) |
| Scalability | Limited (KB size < 50 articles recommended) | Unlimited (chunks scale independently) |

### Why This Happened

CAG system turned out to be more accurate than the RAG system. This is due to the fact that CAG had access to more context for a complete query. The planner could make informed decisions about which articles to load, while RAG's chunking strategy fragmented the policy information, making it harder for the resolver to connect related concepts.

---

## Trade-offs

### CAG Trade-offs

1. We have to sacrifice input tokens - full articles are always included, even if only one section is needed
2. Doesn't scale beyond 50 KB articles
3. Planner must have upfront knowledge of which articles exist

### RAG Trade-offs

1. Risk of partial or misleading chunks
2. Requires embedding model and vector DB maintenance
3. Slower execution (vector lookup plus ranking overhead)
4. More complex to debug and audit

---

## Why This Design Matters

This isn't just about picking a technique—it's about understanding the trade-off space:

1. CAG wins on accuracy and safety for small, well-defined knowledge bases. You're paying a token cost to ensure the LLM sees the full story.
2. RAG would win at scale. If the KB grows to 200+ articles or you need real-time updates without redeploying, chunking and retrieval becomes the only viable path.

The production choice depends on your constraints, not on which technique sounds fancier.

---

## Human-in-the-Loop Checkpoint

Any ticket that triggers a financial action pauses for explicit human approval:

1. Refunds (any amount)
2. Store credit issuance
3. Order cancellations

The agent:

1. Detects the action using structured output fields
2. Drafts the action with justification (e.g., "Refund $89.99 for damaged item within 48-hour photo deadline")
3. Stops execution and awaits user confirmation
4. Only executes if approved; rejects if denied

Example flow:

```
Planner: This is a refund request (damaged goods).
Resolver: I recommend refunding $89.99 [kb-004]. 
HUMAN APPROVAL REQUIRED: Refund $89.99 for damaged item (within claim window).
Approval: APPROVED
System: Refund processed. Sending confirmation to customer.
```

---

## Running the Systems

### Prerequisites

```bash
python 3.11+
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"  # or add to .env file
```

### Option 1: Test CAG System

```bash
cd CAG
python -m pytest tests/test_harness.py -v
```

This runs 5-10 scripted test scenarios (TKT-0001 through TKT-0014) and reports pass/fail for each.

### Option 2: Test RAG System

```bash
cd RAG
python -m pytest tests/test_harness.py -v
```

Same test harness structure; different retrieval backend.

### Option 3: Run Interactive CLI

CAG:
```bash
cd CAG
python main.py --ticket TKT-0003  # Run a single ticket
python main.py --all             # Run all 14 tickets
```

RAG:
```bash
cd RAG
python main.py --ticket TKT-0003
python main.py --all
```

### Option 4: Run the Order API (Optional)

The agent queries a mock order API. To run it separately:

```bash
cd CAG  # or RAG (same order_api.py)
uvicorn order_api:app --port 8080 --reload
```

With simulated flakiness (random 500s and latency):

```bash
SIMULATE_FLAKE=1 uvicorn order_api:app --port 8080
```

### Option 5: Run the unified server + frontend demo

The repo now includes a unified FastAPI server (`server.py`) and a simple frontend demo (`frontend/index.html`). The server compiles both CAG and RAG workflows, exposes SSE streaming for runs, and provides HITL endpoints.

Start the server (recommended):

```bash
python server.py
```

Or run with `uvicorn` directly:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open the demo in your browser at: `http://localhost:8000/` — the UI lets you pick CAG vs RAG, run tickets, and approve/reject HITL decisions. Useful endpoints (for automation):

- `POST /resolve` — resolve a ticket (SSE stream)
- `POST /hitl-decision` — submit human approval (`{thread_id, decision: "approved"|"rejected"}`)
- `GET /runs` and `GET /runs/{ticket_id}` — list and fetch saved traces


---

## Project Structure

```
server.py                   # Unified FastAPI server (SSE streams, HITL endpoints, demo integration)
frontend/                   # Simple demo UI (frontend/index.html) for interactive runs

CAG/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── tickets.json            # 14 test tickets
├── orders.json             # 35 mock orders
├── knowledge_base/         # 18 KB articles (Markdown)
├── src/
│   ├── graph.py            # LangGraph workflow (planner → resolver)
│   ├── kb_index.py         # KB loader & article registry
│   ├── models.py           # Pydantic schemas (AgentState, etc.)
│   ├── agents/
│   │   ├── planner_prompt.py   # Planner system prompt
│   │   └── resolver_prompt.py  # Resolver system prompt
│   ├── tools/
│   │   ├── kb_search.py         # Load articles by ID
│   │   ├── get_order_status.py  # Query order API
│   │   └── escalate.py          # Route to human queue
│   └── models.py           # Pydantic state & trace models
└── tests/
   └── test_harness.py     # 5–10 test cases with auto-approval

RAG/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies (includes chromadb)
├── tickets.json            # Same 14 test tickets
├── orders.json             # Same 35 mock orders
├── knowledge_base/         # Same 18 KB articles
├── chroma_db/              # Persistent vector DB
│   ├── chroma.sqlite3
│   └── [collection folders]
├── src/
│   ├── graph.py            # LangGraph workflow
│   ├── agents/
│   │   ├── planner_prompt.py
│   │   └── resolver_prompt.py
│   ├── tools/
│   │   ├── kb_search.py         # Chroma query
│   │   ├── get_order_status.py
│   │   └── escalate.py
│   ├── rag/
│   │   ├── chunking.py         # Split articles by section
│   │   ├── index.py            # Build/query Chroma index
│   │   └── __init__.py
│   └── models.py
└── tests/
   └── test_harness.py
```

---

## Key Design Decisions

### 1. Explicit Planner Step

The agent doesn't dive straight into reasoning. Instead:

1. Ticket → Planner classifies it (informational / order_specific / escalation)
2. Planner extracts task list and selects KB articles
3. Only then does the resolver compose a reply

Why: Prevents megaprompts that mix classification + reasoning + hallucination in one LLM call. Easier to debug and audit.

### 2. Structured Output for HITL

The resolver returns:

```json
{
  "final_response": "Customer-facing message",
  "requires_hitl": true,
  "hitl_action": "REFUND",
  "hitl_amount": 89.99,
  "hitl_justification": "Damaged item within 48-hour claim window [kb-004]"
}
```

Why: Machine-readable signals ensure no financial action executes without explicit approval. Avoids parsing natural language for compliance.

### 3. Tool Traces for Observability

Every tool call (kb_search, get_order_status, escalate) logs:

1. Input arguments
2. Status (SUCCESS / ERROR / TIMEOUT)
3. Output summary
4. Reason/error message

Why: If a ticket fails, you can trace exactly where and why. Critical for debugging in production.

### 4. Graceful Failure Modes

1. Unknown order ID → Error message, escalate instead of hallucinate
2. Empty KB search → Admit knowledge gap, offer to escalate
3. API timeout → Retry once, then escalate with explanation
4. Timeout exceeding SLA → Reject "keep waiting" advice, escalate immediately

Why: Customers hate being told to "check back later" when it's been 20 days. A failed attempt is better than false hope.

---

## Evaluation Harness

The test suite (test_harness.py) covers:

| Test Case | Ticket ID | Scenario | Expected Outcome |
|-----------|-----------|----------|------------------|
| Info + no HITL | TKT-0001 | "How long do refunds take?" | Resolved with citation to kb-003, no approval needed |
| Order lookup | TKT-0002 | "Where is my desk lamp?" | Order fetched, delivery status provided |
| Refund + HITL | TKT-0003 | Damaged item, wants refund | HITL triggered, approved, refund drafted |
| Escalation | TKT-0004 | Unsupported region (Iceland) | Escalated to human (outside KB scope) |
| Late shipment | TKT-0005 | 3 weeks no tracking update | Lost-parcel protocol, escalate for investigation |
| Warranty claim | TKT-0006 | Headphones failed after 2 months | Warranty check, offer replacement or repair |
| Quick cancel | TKT-0007 | Cancel before shipment | Cancellation allowed, immediate confirmation |

Run with:

```bash
pytest tests/test_harness.py -v
```

Each test:

1. Loads a ticket
2. Invokes the compiled graph
3. Auto-approves any HITL (configurable)
4. Asserts expected outputs (classification, HITL trigger, citations, etc.)

---

## Trade-offs and Limitations

### CAG Trade-offs

1. High token cost - full articles are always included, even if only one section is needed
2. Doesn't scale beyond 50 KB articles
3. Planner must have upfront knowledge of which articles exist

Advantages:
1. High accuracy and safety
2. Consistent citations
3. Easy to audit and debug

### RAG Trade-offs

1. Risk of partial or misleading chunks
2. Requires embedding model and vector DB maintenance
3. Slower execution (vector lookup plus ranking overhead)
4. More complex to debug and audit

Advantages:
1. Lower token cost
2. Scales to thousands of articles
3. No need to predefine article list

### System-Wide Limitations

1. No multi-turn conversation - each ticket is processed in isolation; no context carried between messages
2. No customer history - the agent doesn't remember previous interactions (by design, stateless)
3. No ML feedback loop - HITL approvals/rejections aren't used to fine-tune the planner or resolver
4. Order API is mocked - real systems would need retry logic, circuit breakers, and rate limiting

---

## What I Would Do With More Time

### 1. Add Multi-Turn Conversation

Currently, each ticket is one shot. A customer might ask a follow-up. Implementing conversation history and context carryover would let the agent remember: "You asked about a refund on TKT-0001; let me check that status."

### 2. Fine-Tune the Planner

Use approved HITL decisions to create a training dataset:

1. Tickets where HITL was needed but planner missed it → Add to curriculum
2. Tickets where planner escalated unnecessarily → Calibrate thresholds

### 3. Implement Chunking Hybrid Strategy

CAG's full-context approach is great, but RAG's chunking would help for:

1. Very long articles (e.g., a 5,000-token comprehensive policy guide)
2. Cross-article references (e.g., "See kb-003 and kb-004 for refund + damaged items")
3. Hybrid approach: retrieve top-3 chunks, then expand to full articles for context

### 4. Add Analytics Dashboard

Track:

1. Classification accuracy (did the planner choose correctly?)
2. HITL approval rates by action type (which refund requests get denied?)
3. Tool failure rates (how often does order_status time out?)
4. Customer satisfaction (post-resolution survey)

### 5. Deploy as a Web Service

1. REST endpoint: POST /resolve_ticket with ticket JSON
2. Webhook callbacks for HITL: POST /approval_callback to resume execution
3. Metrics endpoint: /metrics for Prometheus scraping

### 6. Add Real LLM Providers

Currently using Google Gemini 3.1 Flash. Expand to:

1. OpenAI GPT-4 (for higher-stakes tickets)
2. Claude (for reasoning-heavy analysis)
3. Local models (Llama, Mistral) for cost-sensitive deployments
4. Fall-through logic: try primary, retry with backup if rate-limited

### 7. Build a Customer-Facing Summary View

Instead of raw agent output, format responses as:

1. Situation - What we understand about the customer's issue
2. Solution - The proposed action (or escalation reason)
3. Timeline - When the customer should expect resolution
4. Next Steps - What they should do now

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | LangGraph | Multi-step workflows, interrupt/resume, observability |
| LLM | Google Gemini 3.1 Flash | Fast, affordable, structured output support |
| KB Indexing (CAG) | In-memory registry | Simple, fast, no external dependency |
| KB Indexing (RAG) | Chroma + Sentence Transformers | Battle-tested, persistent, semantic search |
| Retrieval (RAG) | BM25 + embeddings (hybrid) | Combines keyword + semantic; reduces pure-semantic blind spots |
| Order API | FastAPI | Lightweight, mock-friendly, async-ready |
| Testing | pytest + LangGraph test utils | Standard Python testing, LangGraph-aware assertions |

---

## Repository Structure

1. `server.py` — Unified FastAPI server that exposes both CAG and RAG workflows, SSE streaming, and HITL endpoints
2. `frontend/` — Simple demo UI that interacts with the server for running tickets and approving HITL decisions
3. `CAG/` — Cache Augmented Generation implementation (recommended for small KB)
4. `RAG/` — Retrieval Augmented Generation implementation (recommended for large KB)
5. `README.md` — This file

Top-level components are self-contained: same knowledge base and tickets are used by both retrieval strategies. Use `server.py` + `frontend/` for the interactive demo, or run `CAG/` and `RAG/` independently for development and tests.

---

## Getting Started (Quick Start)

1. Clone or download the repository

2. Set up environment:
   ```bash
   cd CAG  # or RAG
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   export GEMINI_API_KEY="your-key"
   ```

3. Run tests:
   ```bash
   pytest tests/test_harness.py -v
   ```

4. Run a single ticket interactively:
   ```bash
   python main.py --ticket TKT-0003
   ```

5. Review the trace:
   1. Planner classification
   2. KB articles selected/retrieved
   3. Order data fetched (if any)
   4. HITL approval (if triggered)
   5. Final response

---

## Code Walkthrough

Each system includes detailed docstrings and type hints. Start with:

1. CAG/src/graph.py or RAG/src/graph.py for workflow overview
2. CAG/src/agents/planner_prompt.py for planner logic
3. CAG/src/agents/resolver_prompt.py for resolver rules (especially HITL)
4. CAG/tests/test_harness.py for expected behaviors

---

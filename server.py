"""Unified FastAPI server for CAG and RAG support agents."""
import asyncio
import importlib.util
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()


def _load_compile_workflow(module_path: Path):
    parent_dir = str(module_path.parent.parent)
    sys_path_inserted = False
    try:
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            sys_path_inserted = True
        spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
        mod = importlib.util.module_from_spec(spec)
        for k in list(sys.modules.keys()):
            if k == "src" or k.startswith("src."):
                sys.modules.pop(k, None)
        spec.loader.exec_module(mod)
        return getattr(mod, "compile_workflow")
    finally:
        if sys_path_inserted:
            try:
                sys.path.remove(parent_dir)
            except Exception:
                pass


compile_cag_workflow = _load_compile_workflow(Path(__file__).parent / "CAG" / "src" / "graph.py")
compile_rag_workflow = _load_compile_workflow(Path(__file__).parent / "RAG" / "src" / "graph.py")

RUNS_DIR_CAG = Path(__file__).parent / "CAG" / "runs"
RUNS_DIR_RAG = Path(__file__).parent / "RAG" / "runs"
TICKETS_DIR_CAG = Path(__file__).parent / "CAG" / "tickets.json"
TICKETS_DIR_RAG = Path(__file__).parent / "RAG" / "tickets.json"
FRONTEND_DIR = Path(__file__).parent / "frontend"

cag_graph = None
rag_graph = None

hitl_sessions: dict[str, dict] = {}


def _serialize_for_sse(value):
    """Convert LangGraph/Pydantic payloads into plain JSON-friendly objects."""
    if hasattr(value, "model_dump"):
        return _serialize_for_sse(value.model_dump())
    if isinstance(value, dict):
        return {key: _serialize_for_sse(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_sse(item) for item in value]
    return value


def get_workflow(mode: str):
    """Get the compiled workflow for the specified mode."""
    global cag_graph, rag_graph
    if mode == "cag":
        if cag_graph is None:
            cag_graph = compile_cag_workflow()
        return cag_graph, "CAG"
    elif mode == "rag":
        if rag_graph is None:
            rag_graph = compile_rag_workflow()
        return rag_graph, "RAG"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def load_tickets():
    """Load tickets from either CAG or RAG (they're the same)."""
    if TICKETS_DIR_CAG.exists():
        with open(TICKETS_DIR_CAG) as f:
            return json.load(f)
    return {"tickets": []}


async def run_agent_stream(
    ticket: dict, mode: str
) -> AsyncGenerator[str, None]:
    """Run the agent with streaming output."""
    graph, _ = get_workflow(mode)

    state = {
        "ticket": ticket,
        "planner_output": None,
        "execution_trace": None,
        "requires_hitl": False,
        "retrieved_docs": "",
        "order_data": None,
        "final_response": None,
    }

    if mode == "rag":

        state.update({"order_tool_trace": None, "guardrail_attempts": 0, "guardrail_feedback": None})

    thread_id = os.urandom(8).hex()
    print(f"[server] starting run for ticket={ticket.get('ticket_id')} thread_id={thread_id}")

    hitl_sessions[thread_id] = {
        "future": None,
        "ticket_id": ticket.get("ticket_id"),
        "mode": mode,
    }

    try:
        result = None
        config = {"configurable": {"thread_id": thread_id}}
        gen = graph.stream(state, config=config, stream_mode="updates")
        for chunk in gen:
            try:
                if isinstance(chunk, dict):
                    chunk["thread_id"] = thread_id
                else:
                    chunk = {"payload": chunk, "thread_id": thread_id}
            except Exception:
                chunk = {"payload": str(chunk), "thread_id": thread_id}

            requires_hitl = False
            if isinstance(chunk, dict):
                requires_hitl = bool(chunk.get("requires_hitl") or chunk.get("workflow_status") == "WAITING_HUMAN")

            yield f"data: {json.dumps(_serialize_for_sse(chunk), default=str)}\n\n"
            result = chunk

            if requires_hitl:
                loop = asyncio.get_event_loop()
                fut: asyncio.Future = loop.create_future()
                hitl_sessions[thread_id]["future"] = fut
                hitl_sessions[thread_id]["generator"] = gen
                print(f"[server] HITL required for thread_id={thread_id}, awaiting decision")

                hitl_event = {
                    "type": "hitl_required",
                    "thread_id": thread_id,
                    "ticket_id": ticket.get("ticket_id"),
                    "question": chunk.get("final_response") if isinstance(chunk, dict) else None,
                }
                yield f"data: {json.dumps(_serialize_for_sse(hitl_event), default=str)}\n\n"

                try:
                    wait_task = asyncio.create_task(fut)
                    timeout = 300
                    start = loop.time()
                    while True:
                        _done, _pending = await asyncio.wait({wait_task}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
                        if wait_task.done():
                            decision = wait_task.result()
                            print(f"[server] received decision for thread_id={thread_id}: {decision}")
                            yield f"data: {json.dumps(_serialize_for_sse({'type': 'hitl_decision_received', 'thread_id': thread_id, 'decision': decision}))}\n\n"
                            break
                        elapsed = loop.time() - start
                        if elapsed > timeout:
                            yield f"data: {json.dumps(_serialize_for_sse({'type': 'hitl_timeout', 'thread_id': thread_id, 'message': 'HITL decision timeout'}))}\n\n"
                            if not fut.done():
                                fut.cancel()
                            raise asyncio.TimeoutError("HITL decision timeout")

                finally:
                    hitl_sessions[thread_id]["future"] = None

        if result:
            yield f"data: {json.dumps(_serialize_for_sse({'_final': True, 'status': 'completed', 'thread_id': thread_id}), default=str)}\n\n"
    except Exception as e:
        print(f"[server] run error thread_id={thread_id}: {e}")
        yield f"data: {json.dumps(_serialize_for_sse({'type': 'error', 'thread_id': thread_id, 'error': str(e)}))}\n\n"
    finally:
        hitl_sessions.pop(thread_id, None)
        print(f"[server] finished run for thread_id={thread_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile workflows on startup."""
    global cag_graph, rag_graph
    print("Compiling CAG and RAG workflows...")
    cag_graph = compile_cag_workflow()
    rag_graph = compile_rag_workflow()
    print("Workflows compiled and ready")
    yield
    print("Shutting down")


app = FastAPI(
    title="Customer Support Agent API",
    description="Unified API for CAG and RAG support agents",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class TicketInput(BaseModel):
    """Ticket input model."""
    ticket_id: str
    customer_email: str
    subject: str
    body: str
    received_at: str


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "Customer Support Agent API"}


@app.get("/tickets")
async def list_tickets():
    """Return all tickets."""
    data = load_tickets()
    return {
        "count": len(data.get("tickets", [])),
        "tickets": data.get("tickets", []),
    }


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Return a specific ticket."""
    data = load_tickets()
    for ticket in data.get("tickets", []):
        if ticket["ticket_id"] == ticket_id:
            return ticket
    raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")


@app.post("/resolve")
async def resolve_ticket(
    ticket: TicketInput,
    mode: str = Query("cag", description="Agent mode: 'cag' or 'rag'"),
):
    """Resolve a ticket with the specified agent mode.
    
    Streams the response back as server-sent events (SSE).
    """
    if mode not in ("cag", "rag"):
        raise HTTPException(status_code=400, detail="mode must be 'cag' or 'rag'")
    
    ticket_dict = ticket.model_dump()
    
    return StreamingResponse(
        run_agent_stream(ticket_dict, mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/runs/{ticket_id}")
async def get_run_trace(ticket_id: str, mode: str = Query("cag")):
    """Return the saved execution trace for a ticket."""
    if mode == "cag":
        runs_dir = RUNS_DIR_CAG
    elif mode == "rag":
        runs_dir = RUNS_DIR_RAG
    else:
        raise HTTPException(status_code=400, detail="mode must be 'cag' or 'rag'")
    
    trace_file = runs_dir / f"{ticket_id}.json"
    if not trace_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Trace for {ticket_id} not found (mode={mode})",
        )
    
    return FileResponse(trace_file)


@app.get("/runs")
async def list_runs(mode: str = Query("cag")):
    """Return all saved execution traces."""
    if mode == "cag":
        runs_dir = RUNS_DIR_CAG
    elif mode == "rag":
        runs_dir = RUNS_DIR_RAG
    else:
        raise HTTPException(status_code=400, detail="mode must be 'cag' or 'rag'")
    
    if not runs_dir.exists():
        return {"runs": []}
    
    runs = [f.stem for f in runs_dir.glob("TKT-*.json")]
    return {"mode": mode, "count": len(runs), "runs": sorted(runs)}


@app.post("/hitl-decision")
async def hitl_decision(request: Request):
    """Handle human approval/rejection decisions for HITL actions.

    Resolves the in-memory future for the given thread_id so the running
    SSE stream can continue.
    """
    body = await request.json()
    thread_id = body.get("thread_id")
    decision = body.get("decision")

    if not thread_id or decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="Require thread_id and decision ('approved'|'rejected')")

    session = hitl_sessions.get(thread_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"HITL session {thread_id} not found")

    fut = session.get("future")
    if not fut:
        return {"thread_id": thread_id, "decision": decision, "status": "no_waiter"}

    if not fut.done():
        fut.set_result(decision)
        print(f"[server] hitl_decision set for thread_id={thread_id}: {decision}")
    return {"thread_id": thread_id, "decision": decision, "status": "recorded"}


@app.get("/")
async def root():
    """Serve the frontend index."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Customer Support Agent API",
        "endpoints": {
            "GET /health": "Health check",
            "GET /tickets": "List all tickets",
            "GET /tickets/{ticket_id}": "Get a specific ticket",
            "POST /resolve": "Resolve a ticket (streaming)",
            "GET /runs": "List all saved traces",
            "GET /runs/{ticket_id}": "Get trace for a ticket",
            "POST /hitl-decision": "Record HITL approval/rejection",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

from __future__ import annotations

import os

from fastapi import FastAPI

app = FastAPI(title="RAG Support Agent", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-agent"}


@app.get("/")
def root():
    return {"service": "rag-agent", "status": "ready"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))

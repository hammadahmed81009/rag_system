import logging
import time
import uuid

import httpx
from fastapi import FastAPI, Request

from app.config import settings
from app.dependencies import vector_store
from app.routes.query import router as query_router
from app.routes.ingest import router as ingest_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    description="Query your documents with retrieval-augmented generation.",
)

app.include_router(query_router)
app.include_router(ingest_router)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG API. Use POST /query/ with {\"query\": \"your question\"}."}


@app.get("/health")
async def health():
    checks = {}
    try:
        vector_store.client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = str(e)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            checks["ollama"] = "ok" if r.status_code == 200 else f"status={r.status_code}"
    except Exception as e:
        checks["ollama"] = str(e)
    status = 200 if all(v == "ok" for v in checks.values()) else 503
    return {"status": "ok" if status == 200 else "degraded", "checks": checks}
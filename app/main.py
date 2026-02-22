from fastapi import FastAPI

from app.routes.query import router as query_router
from app.routes.ingest import router as ingest_router

app = FastAPI(
    title="RAG API",
    description="Query your documents with retrieval-augmented generation.",
)

app.include_router(query_router)
app.include_router(ingest_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG API. Use POST /query/ with {\"query\": \"your question\"}."}
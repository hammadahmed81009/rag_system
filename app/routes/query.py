import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.dependencies import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str


class SourceResponse(BaseModel):
    text: str
    score: float
    id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@router.post("/", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    result = await rag_service.answer(body.query)
    request_id = getattr(request.state, "request_id", "")
    logger.info("request_id=%s query_len=%d sources_count=%d", request_id, len(body.query), len(result["sources"]))
    sources = [
        SourceResponse(
            text=doc.payload.get("text", ""),
            score=doc.score,
            id=str(doc.id) if doc.id else None,
        )
        for doc in result["sources"]
    ]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
    )
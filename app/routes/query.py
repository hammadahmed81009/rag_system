from fastapi import APIRouter
from pydantic import BaseModel

from app.dependencies import rag_service

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
async def query(body: QueryRequest):
    result = await rag_service.answer(body.query)
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
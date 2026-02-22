import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.config import settings
from app.dependencies import embedding, vector_store
from ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    path: str


@router.post("/")
async def ingest(request: Request, body: IngestRequest):
    source = Path(body.path).resolve()
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {body.path}")
    if settings.ingest_root:
        root = Path(settings.ingest_root).resolve()
        try:
            source.relative_to(root)
        except ValueError:
            raise HTTPException(
                status_code=403,
                detail="Path must be under configured INGEST_ROOT",
            )
    pipeline = IngestionPipeline(embedding, vector_store)
    try:
        n = await pipeline.run(source)
        request_id = getattr(request.state, "request_id", "")
        logger.info("request_id=%s path=%s indexed=%d", request_id, body.path, n)
        return {"indexed": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
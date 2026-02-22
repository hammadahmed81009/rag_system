import logging
from pathlib import Path

from app.config import settings
from core.chunking.smart_chunker import SmartChunker
from ingestion.loaders import discover_files, load_file

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, embedding, vector_store):
        self.embedding = embedding
        self.vector_store = vector_store
        self.chunker = SmartChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    async def run(self, source: Path) -> int:
        if not source.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")
        paths = [source] if source.is_file() else discover_files(source)
        logger.info("source=%s paths_count=%d", source, len(paths))
        all_chunks: list[tuple[str, dict]] = []
        for path in paths:
            doc = load_file(path)
            if doc is None:
                logger.debug("skip unsupported or unreadable path=%s", path)
                continue
            chunks = self.chunker.chunk(doc.content, metadata={"source": doc.path})
            for c in chunks:
                all_chunks.append((c["text"], {**c["metadata"]}))
            logger.info("loaded path=%s chunks=%d", path, len(chunks))
        if not all_chunks:
            logger.warning("no chunks produced")
            return 0
        batch_size = settings.ingest_batch_size
        total = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [t for t, _ in batch]
            payloads = [{"text": t, **m} for t, m in batch]
            vectors = await self.embedding.embed(texts)
            await self.vector_store.upsert(vectors, payloads)
            total += len(batch)
            logger.info("upserted batch batch_index=%d batch_size=%d total_so_far=%d", i // batch_size, len(batch), total)
        logger.info("finished total_chunks=%d", total)
        return total
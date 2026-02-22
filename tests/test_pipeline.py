import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.chunking.smart_chunker import SmartChunker
from ingestion.pipeline import IngestionPipeline


@pytest.fixture
def mock_embedding():
    m = MagicMock()
    async def embed(texts):
        return [[0.1] * 4 for _ in texts]
    m.embed = AsyncMock(side_effect=embed)
    return m


@pytest.fixture
def mock_vector_store():
    m = MagicMock()
    m.upsert = AsyncMock(return_value=None)
    return m


@pytest.mark.asyncio
async def test_pipeline_ingests_file(tmp_path, mock_embedding, mock_vector_store):
    f = tmp_path / "doc.txt"
    f.write_text("First sentence. Second sentence. Third.")
    pipeline = IngestionPipeline(mock_embedding, mock_vector_store)
    pipeline.chunker = SmartChunker(chunk_size=100, overlap=10)

    n = await pipeline.run(tmp_path / "doc.txt")

    assert n > 0
    assert mock_embedding.embed.await_count >= 1
    assert mock_vector_store.upsert.await_count >= 1
    call_args = mock_vector_store.upsert.call_args
    vectors, payloads = call_args[0]
    assert len(vectors) == len(payloads)
    assert all("text" in p and "source" in p for p in payloads)


@pytest.mark.asyncio
async def test_pipeline_nonexistent_raises(tmp_path, mock_embedding, mock_vector_store):
    pipeline = IngestionPipeline(mock_embedding, mock_vector_store)
    with pytest.raises(FileNotFoundError):
        await pipeline.run(tmp_path / "nonexistent.txt")


@pytest.mark.asyncio
async def test_pipeline_empty_folder_no_chunks(tmp_path, mock_embedding, mock_vector_store):
    pipeline = IngestionPipeline(mock_embedding, mock_vector_store)
    n = await pipeline.run(tmp_path)
    assert n == 0
    mock_vector_store.upsert.assert_not_called()

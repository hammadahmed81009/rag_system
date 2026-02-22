from unittest.mock import AsyncMock, MagicMock

import pytest

from core.retriever.vector import VectorRetriever


@pytest.fixture
def mock_embedding():
    m = MagicMock()
    async def embed(texts):
        return [[0.0] * 4 for _ in texts]
    m.embed = AsyncMock(side_effect=embed)
    return m


@pytest.fixture
def mock_vector_store():
    m = MagicMock()
    fake_results = [
        MagicMock(payload={"text": "chunk one", "source": "a.txt"}),
        MagicMock(payload={"text": "chunk two", "source": "b.txt"}),
    ]
    async def search(query_vector, k=5):
        return fake_results[:k]
    m.search = AsyncMock(side_effect=search)
    return m


@pytest.mark.asyncio
async def test_retriever_returns_store_results(mock_embedding, mock_vector_store):
    retriever = VectorRetriever(mock_embedding, mock_vector_store)
    results = await retriever.retrieve("test query", k=2)
    assert len(results) == 2
    assert results[0].payload["text"] == "chunk one"
    assert results[1].payload["source"] == "b.txt"
    mock_embedding.embed.assert_called_once()
    mock_vector_store.search.assert_called_once()
    call_k = mock_vector_store.search.call_args[1].get("k") or mock_vector_store.search.call_args[0][1]
    if mock_vector_store.search.call_args[0][0] is not None:
        assert mock_vector_store.search.call_args[1].get("k") == 2

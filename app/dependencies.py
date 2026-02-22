from app.config import settings
from core.embeddings.local import LocalEmbedding
from core.embeddings.openai import OpenAIEmbedding
from core.retriever.vector import VectorRetriever
from core.retriever.reranker import BM25Reranker, KeywordReranker, RerankingRetriever
from core.llm.ollama import OllamaLLM
from db.vector.qdrant import QdrantVectorStore
from core.rag_service import RAGService
from qdrant_client.http.exceptions import UnexpectedResponse

if settings.embedding_provider == "openai":
    embedding = OpenAIEmbedding(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key or None,
    )
else:
    embedding = LocalEmbedding(model_name=settings.embedding_model_name)

vector_store = QdrantVectorStore(
    collection_name=settings.qdrant_collection,
    host=settings.qdrant_host,
    port=settings.qdrant_port,
)

try:
    vector_store.client.get_collection(settings.qdrant_collection)
except UnexpectedResponse:
    vector_store.create_collection(vector_size=settings.embedding_dim)

_vector_retriever = VectorRetriever(embedding, vector_store)
if settings.rerank_enabled:
    reranker = (
        BM25Reranker()
        if settings.reranker_type == "bm25"
        else KeywordReranker()
    )
    retriever = RerankingRetriever(
        _vector_retriever,
        reranker,
        initial_k=settings.rerank_initial_k,
        top_k=settings.retrieval_top_k,
    )
else:
    retriever = _vector_retriever

llm = OllamaLLM(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    timeout=settings.ollama_timeout,
)
rag_service = RAGService(
    retriever,
    llm,
    top_k=settings.retrieval_top_k,
)
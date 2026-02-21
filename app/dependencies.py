from app.config import settings
from core.embeddings.local import LocalEmbedding
from core.retriever.vector import VectorRetriever
from core.llm.ollama import OllamaLLM
from db.vector.qdrant import QdrantVectorStore
from core.rag_service import RAGService
from qdrant_client.http.exceptions import UnexpectedResponse

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

retriever = VectorRetriever(embedding, vector_store)
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
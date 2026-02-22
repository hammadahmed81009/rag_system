from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_collection"

    embedding_model_name: str = "BAAI/bge-small-en"
    embedding_dim: int = 384

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_timeout: int = 120

    retrieval_top_k: int = 5
    rerank_enabled: bool = False
    rerank_initial_k: int = 20
    reranker_type: str = "keyword"

    chunk_size: int = 512
    chunk_overlap: int = 50
    ingest_batch_size: int = 32
    ingest_root: str = ""
    chunker_type: str = "smart"

    class Config:
        env_prefix = ""
        env_file = ".env"


settings = Settings()
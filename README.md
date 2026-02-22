# RAG API

Retrieval-augmented generation API: ingest documents, query with natural language, get answers grounded in your data. Self-hosted stack (Qdrant, sentence-transformers, Ollama).

## Requirements

- Python 3.11 or 3.12
- [Qdrant](https://qdrant.tech/documentation/quick-start/) (Docker or binary)
- [Ollama](https://ollama.ai/) (for the LLM)

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env if needed (Qdrant host/port, Ollama URL, etc.)
```

## Run Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Run Ollama

Install from [ollama.ai](https://ollama.ai), then:

```bash
ollama run mistral
```

Keep it running so the API can call it.

## Run the API

```bash
uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000  
- Docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

## Ingest documents

From the project root:

```bash
# Create a folder and add .txt or .md files
mkdir -p data/docs
# Add files to data/docs, then:
python scripts/ingest_folder.py data/docs
```

Or via API:

```bash
curl -X POST http://127.0.0.1:8000/ingest/ -H "Content-Type: application/json" -d '{"path": "data/docs"}'
```

To wipe and re-index:

```bash
python scripts/rebuild_index.py data/docs
```

## Query

```bash
curl -X POST http://127.0.0.1:8000/query/ -H "Content-Type: application/json" -d '{"query": "What is this document about?"}'
```

## Evaluation

With test cases in `evaluation/test_queries.py` and documents indexed:

```bash
python scripts/run_evaluation.py
```

Prints Recall@k and per-query hit/miss.

## Configuration

See `.env.example`. Main options:

- **Qdrant:** `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- **Embeddings:** `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIM` (must match model, e.g. 384 for bge-small-en)
- **Ollama:** `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`
- **Retrieval:** `RETRIEVAL_TOP_K`
- **Reranker:** `RERANK_ENABLED`, `RERANK_INITIAL_K`, `RERANKER_TYPE` (keyword | bm25)
- **Chunking:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNKER_TYPE` (smart | paragraph)

## Project layout

- `app/` – FastAPI app, config, routes (query, ingest, health)
- `core/` – RAG logic: retrievers, embeddings, LLM, chunking, prompts, reranker
- `db/` – Vector store (Qdrant) abstraction
- `ingestion/` – Loaders, pipeline, chunking
- `evaluation/` – Test queries, recall metrics
- `scripts/` – CLI: ingest_folder, rebuild_index, run_evaluation

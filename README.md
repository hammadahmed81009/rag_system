# RAG API

Retrieval-augmented generation API: ingest documents, query with natural language, get answers grounded in your data. Self-hosted stack (Qdrant, sentence-transformers or optional OpenAI embeddings, Ollama).

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

## Run with Docker Compose

Run Qdrant and the API in one go (Ollama still runs on your host; the app connects to it via `host.docker.internal`):

```bash
cp .env.example .env
# Edit .env if needed; ensure OLLAMA_BASE_URL is not overridden if using default
docker compose up --build
```

API: http://127.0.0.1:8000. To run only Qdrant and the app locally:

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

Supported formats: **.txt**, **.md**, **.pdf**, **.html** / **.htm**. From the project root:

```bash
mkdir -p data/docs
# Add .txt, .md, .pdf, or .html files, then:
python scripts/ingest_folder.py data/docs
```

Or via API:

```bash
curl -X POST http://127.0.0.1:8000/ingest/ -H "Content-Type: application/json" -d '{"path": "data/docs"}'
```

If you set `API_KEY` in `.env`, include the header: `-H "X-API-Key: your-secret-key"`.

To wipe and re-index:

```bash
python scripts/rebuild_index.py data/docs
```

## Query

```bash
curl -X POST http://127.0.0.1:8000/query/ -H "Content-Type: application/json" -d '{"query": "What is this document about?"}'
```

If `API_KEY` is set, add: `-H "X-API-Key: your-secret-key"`.

## Evaluation

With test cases in `evaluation/test_queries.py` and documents indexed:

```bash
python scripts/run_evaluation.py
```

Prints **Recall@k**, **MRR@k**, **NDCG@k**, and **retrieval latency** (p50, p95 in ms). Add or edit cases in `evaluation/test_queries.py` to evaluate your own queries.

## Testing

Unit tests use pytest (and pytest-asyncio for async tests):

```bash
pip install -r requirements.txt
pytest tests/ -v
```

- **tests/test_chunking.py** – SmartChunker and ParagraphChunker (empty input, chunk count, metadata).
- **tests/test_pipeline.py** – Ingestion pipeline with mocked embedding and vector store (ingest file, nonexistent path, empty folder).
- **tests/test_retrieval.py** – VectorRetriever with mocked embedding and store.
- **tests/test_metrics.py** – Recall, MRR, NDCG, latency percentiles.

## Configuration

See `.env.example`. Main options:

- **Qdrant:** `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- **Embeddings:**  
  - `EMBEDDING_PROVIDER`: `local` (sentence-transformers) or `openai`.  
  - **Local:** `EMBEDDING_MODEL_NAME`, `EMBEDDING_DIM` (e.g. 384 for bge-small-en).  
  - **OpenAI:** set `OPENAI_API_KEY` and `OPENAI_EMBEDDING_MODEL` (e.g. `text-embedding-3-small`); set `EMBEDDING_DIM` to match the model (e.g. 1536 for text-embedding-3-small).
- **Ollama:** `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`
- **Retrieval:** `RETRIEVAL_TOP_K`
- **Reranker:** `RERANK_ENABLED`, `RERANK_INITIAL_K`, `RERANKER_TYPE` (keyword | bm25)
- **Chunking:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNKER_TYPE` (smart | paragraph)
- **Auth (optional):** `API_KEY` – if set, `/query` and `/ingest` require `X-API-Key` header. `RATE_LIMIT_PER_MINUTE` – max requests per minute per key/IP (0 = no limit).

## CI

GitHub Actions runs tests on push/PR to `main` or `master` (Python 3.11 and 3.12). See `.github/workflows/ci.yml`. No Qdrant or Ollama required for the unit tests.

## Project layout

- `app/` – FastAPI app, config, routes (query, ingest, health)
- `core/` – RAG logic: retrievers, embeddings (local + optional OpenAI), LLM, chunking, prompts, reranker
- `db/` – Vector store (Qdrant) abstraction
- `ingestion/` – Loaders (txt, md, PDF, HTML), pipeline, chunking
- `evaluation/` – Test queries, recall and latency metrics
- `scripts/` – CLI: ingest_folder, rebuild_index, run_evaluation
- `tests/` – Unit tests (chunking, pipeline, retrieval)

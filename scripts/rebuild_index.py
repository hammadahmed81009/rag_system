import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.dependencies import embedding, vector_store
from ingestion.pipeline import IngestionPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/rebuild_index.py <path>", file=sys.stderr)
        sys.exit(1)
    vector_store.create_collection(vector_size=settings.embedding_dim)
    source = Path(sys.argv[1]).resolve()
    pipeline = IngestionPipeline(embedding, vector_store)
    n = asyncio.run(pipeline.run(source))
    print(f"Recreated collection and indexed {n} chunks.")


if __name__ == "__main__":
    main()
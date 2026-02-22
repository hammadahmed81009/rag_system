import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

from app.dependencies import embedding, vector_store
from ingestion.pipeline import IngestionPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_folder.py <path>", file=sys.stderr)
        sys.exit(1)
    source = Path(sys.argv[1]).resolve()
    logging.getLogger(__name__).info("ingest source=%s", source)
    pipeline = IngestionPipeline(embedding, vector_store)
    n = asyncio.run(pipeline.run(source))
    print(f"Indexed {n} chunks.")


if __name__ == "__main__":
    main()
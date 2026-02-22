import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

from app.config import settings
from app.dependencies import retriever
from evaluation.metrics import latency_percentiles, mrr_at_k, ndcg_at_k, recall_at_k
from evaluation.test_queries import TEST_CASES


async def main() -> None:
    k = settings.retrieval_top_k
    results: list[bool] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []
    latencies_ms: list[float] = []
    for i, case in enumerate(TEST_CASES):
        t0 = time.perf_counter()
        retrieved = await retriever.retrieve(case.query, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        hit = recall_at_k(
            retrieved,
            case.expected_source_substr,
            case.expected_phrase_in_text,
            k=k,
        )
        results.append(hit)
        mrr_scores.append(mrr_at_k(retrieved, case.expected_source_substr, case.expected_phrase_in_text, k=k))
        ndcg_scores.append(ndcg_at_k(retrieved, case.expected_source_substr, case.expected_phrase_in_text, k=k))
        status = "hit" if hit else "miss"
        logging.getLogger(__name__).info(
            "case=%d query=%r %s latency_ms=%.1f", i, case.query[:50], status, elapsed_ms
        )
    recall = sum(results) / len(results) if results else 0.0
    mean_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    print(f"Recall@{k} = {recall:.2%} ({sum(results)}/{len(results)})")
    print(f"MRR@{k} = {mean_mrr:.4f}")
    print(f"NDCG@{k} = {mean_ndcg:.4f}")
    if latencies_ms:
        percentiles = latency_percentiles(latencies_ms)
        print(f"Retrieval latency: p50={percentiles['p50_ms']:.1f} ms, p95={percentiles['p95_ms']:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())
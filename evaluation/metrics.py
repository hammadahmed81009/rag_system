import math


def is_hit(
    payload: dict,
    expected_source_substr: str,
    expected_phrase_in_text: str | None = None,
) -> bool:
    source = payload.get("source") or ""
    text = payload.get("text") or ""
    if expected_source_substr.lower() not in source.lower():
        return False
    if expected_phrase_in_text is not None:
        if expected_phrase_in_text.lower() not in text.lower():
            return False
    return True


def recall_at_k(
    retrieved: list,
    expected_source_substr: str,
    expected_phrase_in_text: str | None = None,
    k: int | None = None,
) -> bool:
    if k is not None:
        retrieved = retrieved[:k]
    for point in retrieved:
        payload = getattr(point, "payload", point) if not isinstance(point, dict) else point
        if is_hit(payload, expected_source_substr, expected_phrase_in_text):
            return True
    return False


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Return p50 and p95 in ms. Empty list returns 0.0 for both."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0}
    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)
    p50_idx = min(int(0.50 * n), n - 1)
    p95_idx = min(int(0.95 * n), n - 1)
    return {
        "p50_ms": sorted_ms[p50_idx],
        "p95_ms": sorted_ms[p95_idx],
    }


def rank_of_first_hit(
    retrieved: list,
    expected_source_substr: str,
    expected_phrase_in_text: str | None = None,
    k: int | None = None,
) -> int | None:
    """1-based rank of first relevant item, or None if no hit in top k."""
    if k is not None:
        retrieved = retrieved[:k]
    for i, point in enumerate(retrieved):
        payload = getattr(point, "payload", point) if not isinstance(point, dict) else point
        if is_hit(payload, expected_source_substr, expected_phrase_in_text):
            return i + 1
    return None


def mrr_at_k(
    retrieved: list,
    expected_source_substr: str,
    expected_phrase_in_text: str | None = None,
    k: int | None = None,
) -> float:
    """Reciprocal rank (1/rank) for this query; 0.0 if no hit."""
    rank = rank_of_first_hit(retrieved, expected_source_substr, expected_phrase_in_text, k)
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(
    retrieved: list,
    expected_source_substr: str,
    expected_phrase_in_text: str | None = None,
    k: int | None = None,
) -> float:
    """NDCG@k with binary relevance (hit=1, else 0). Returns 0.0 if no hit."""
    if k is not None:
        retrieved = retrieved[:k]
    if not retrieved:
        return 0.0
    relevances = []
    for point in retrieved:
        payload = getattr(point, "payload", point) if not isinstance(point, dict) else point
        rel = 1.0 if is_hit(payload, expected_source_substr, expected_phrase_in_text) else 0.0
        relevances.append(rel)
    # DCG = sum(rel_i / log2(i+2))  (i 0-based => position i+1)
    dcg = sum(rel / (math.log2(i + 2) or 1.0) for i, rel in enumerate(relevances))
    # Ideal: one relevant doc at position 1
    ideal = [1.0] + [0.0] * (len(relevances) - 1)
    idcg = sum(rel / (math.log2(i + 2) or 1.0) for i, rel in enumerate(ideal))
    if idcg <= 0:
        return 0.0
    return dcg / idcg
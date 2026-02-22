from unittest.mock import MagicMock

import pytest

from evaluation.metrics import latency_percentiles, mrr_at_k, ndcg_at_k, rank_of_first_hit, recall_at_k


def _point(payload: dict):
    return MagicMock(payload=payload)


def test_recall_at_k_hit():
    retrieved = [_point({"source": "a.txt", "text": "x"}), _point({"source": "b.txt", "text": "y"})]
    assert recall_at_k(retrieved, "a.txt", None, k=5) is True
    assert recall_at_k(retrieved, "b.txt", None, k=5) is True
    assert recall_at_k(retrieved, "c.txt", None, k=5) is False


def test_recall_at_k_phrase():
    retrieved = [_point({"source": "a.txt", "text": "hello world"})]
    assert recall_at_k(retrieved, "a.txt", "world", k=5) is True
    assert recall_at_k(retrieved, "a.txt", "missing", k=5) is False


def test_rank_of_first_hit():
    retrieved = [
        _point({"source": "x", "text": "a"}),
        _point({"source": "target", "text": "b"}),
        _point({"source": "target", "text": "c"}),
    ]
    assert rank_of_first_hit(retrieved, "target", None, k=5) == 2
    assert rank_of_first_hit(retrieved, "missing", None, k=5) is None
    assert rank_of_first_hit(retrieved, "x", None, k=5) == 1


def test_mrr_at_k():
    retrieved = [_point({"source": "x", "text": "a"}), _point({"source": "target", "text": "b"})]
    assert mrr_at_k(retrieved, "target", None, k=5) == 0.5  # 1/2
    assert mrr_at_k(retrieved, "x", None, k=5) == 1.0  # 1/1
    assert mrr_at_k(retrieved, "missing", None, k=5) == 0.0


def test_ndcg_at_k():
    retrieved = [_point({"source": "target", "text": "b"}), _point({"source": "x", "text": "a"})]
    # First item relevant => DCG = 1/log2(2) = 1, IDCG = 1 => NDCG = 1
    assert ndcg_at_k(retrieved, "target", None, k=5) == 1.0
    # No relevant in first position: rels = [0, 1], DCG = 0 + 1/log2(4), IDCG = 1 => NDCG < 1
    retrieved2 = [_point({"source": "x", "text": "a"}), _point({"source": "target", "text": "b"})]
    ndcg = ndcg_at_k(retrieved2, "target", None, k=5)
    assert 0 < ndcg < 1
    assert ndcg_at_k([], "x", None, k=5) == 0.0


def test_latency_percentiles():
    assert latency_percentiles([]) == {"p50_ms": 0.0, "p95_ms": 0.0}
    out = latency_percentiles([10.0, 20.0, 30.0, 40.0, 100.0])
    assert out["p50_ms"] == 30.0
    assert out["p95_ms"] == 100.0

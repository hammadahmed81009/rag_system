import re
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: list, top_k: int) -> list:
        pass


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _tokenize_list(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _get_payload_text(doc) -> str:
    payload = getattr(doc, "payload", None) or (doc if isinstance(doc, dict) else {})
    return payload.get("text", "") or ""


RRF_K = 60


def _rrf_merge(docs: list, vector_scores: list[float], bm25_scores: list[float], top_k: int) -> list:
    vec_rank = {i: r for r, i in enumerate(sorted(range(len(docs)), key=lambda i: -vector_scores[i]))}
    bm25_rank = {i: r for r, i in enumerate(sorted(range(len(docs)), key=lambda i: -bm25_scores[i]))}
    combined = [(1 / (RRF_K + vec_rank[i]) + 1 / (RRF_K + bm25_rank[i]), docs[i]) for i in range(len(docs))]
    combined.sort(key=lambda x: -x[0])
    return [doc for _, doc in combined[:top_k]]


class BM25Reranker(BaseReranker):
    def rerank(self, query: str, docs: list, top_k: int) -> list:
        from rank_bm25 import BM25Okapi

        if not docs:
            return []
        texts = [_get_payload_text(d) for d in docs]
        vector_scores = [getattr(d, "score", 0.0) or 0.0 for d in docs]
        tokenized = [_tokenize_list(t) or ["_"] for t in texts]
        bm25 = BM25Okapi(tokenized)
        q_tokens = _tokenize_list(query)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_scores_list = bm25_scores.tolist() if hasattr(bm25_scores, "tolist") else list(bm25_scores)
        return _rrf_merge(docs, vector_scores, bm25_scores_list, top_k)


class KeywordReranker(BaseReranker):
    def rerank(self, query: str, docs: list, top_k: int) -> list:
        if not docs:
            return []
        q_tokens = _tokenize(query)
        scored: list[tuple[float, object]] = []
        for doc in docs:
            text = _get_payload_text(doc)
            d_tokens = _tokenize(text)
            overlap = len(q_tokens & d_tokens) / len(q_tokens) if q_tokens else 0
            vector_score = getattr(doc, "score", 0.0) or 0.0
            combined = 0.7 * vector_score + 0.3 * (overlap * 2.0)
            scored.append((combined, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]]


class RerankingRetriever:
    def __init__(self, retriever, reranker: BaseReranker, initial_k: int, top_k: int):
        self.retriever = retriever
        self.reranker = reranker
        self.initial_k = initial_k
        self.top_k = top_k

    async def retrieve(self, query: str, k: int | None = None):
        take = k if k is not None else self.top_k
        docs = await self.retriever.retrieve(query, k=self.initial_k)
        return self.reranker.rerank(query, docs, top_k=take)

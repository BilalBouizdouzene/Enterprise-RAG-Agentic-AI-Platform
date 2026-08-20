"""Hybrid dense + BM25 retrieval."""

from __future__ import annotations

from .base import RetrievalResult
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .sparse import BM25Retriever


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever):
        self.dense = dense
        self.sparse = sparse

    def search(self, question: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        candidate_k = max(top_k * 4, 20)
        fused = reciprocal_rank_fusion([
            self.dense.search(question, candidate_k, metadata_filter),
            self.sparse.search(question, candidate_k, metadata_filter),
        ])
        return fused[:top_k]


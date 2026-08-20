"""Rank fusion utilities."""

from __future__ import annotations

from collections import defaultdict

from .base import RetrievalResult


def reciprocal_rank_fusion(result_lists: list[list[RetrievalResult]], k: int = 60) -> list[RetrievalResult]:
    scores = defaultdict(float)
    best: dict[str, RetrievalResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] += 1.0 / (k + rank)
            best.setdefault(result.chunk_id, result)
    ranked = sorted(best, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        RetrievalResult(
            best[chunk_id].chunk_id,
            best[chunk_id].document_id,
            best[chunk_id].text,
            scores[chunk_id],
            best[chunk_id].metadata,
            "hybrid_rrf",
        )
        for chunk_id in ranked
    ]


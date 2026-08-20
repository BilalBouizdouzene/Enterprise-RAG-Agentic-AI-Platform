"""Transparent ranking metrics with unit-testable formulas."""

from __future__ import annotations

import math
from collections.abc import Iterable


def _relevant_set(relevant: Iterable[str]) -> set[str]:
    return set(relevant)


def precision_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    top = ranked[:k]
    return sum(item in _relevant_set(relevant) for item in top) / max(len(top), 1)


def recall_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = _relevant_set(relevant)
    if not relevant_set:
        return 0.0
    return len(set(ranked[:k]) & relevant_set) / len(relevant_set)


def hit_rate_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & _relevant_set(relevant)))


def reciprocal_rank(ranked: list[str], relevant: Iterable[str]) -> float:
    relevant_set = _relevant_set(relevant)
    for index, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = _relevant_set(relevant)
    dcg = sum((1.0 / math.log2(index + 2)) for index, item in enumerate(ranked[:k]) if item in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(ranked: list[str], relevant: Iterable[str], cutoffs: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    result = {f"precision@{k}": precision_at_k(ranked, relevant, k) for k in cutoffs}
    result.update({f"recall@{k}": recall_at_k(ranked, relevant, k) for k in cutoffs})
    result.update({f"hit_rate@{k}": hit_rate_at_k(ranked, relevant, k) for k in cutoffs})
    result["mrr"] = reciprocal_rank(ranked, relevant)
    result["ndcg@5"] = ndcg_at_k(ranked, relevant, 5)
    return result

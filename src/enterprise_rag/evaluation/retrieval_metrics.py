"""Transparent ranking metrics with unit-testable formulas."""

from __future__ import annotations

import math
from collections import defaultdict
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


def aggregate_metrics(rows: list[dict], group_field: str | None = None) -> list[dict[str, float | str]]:
    """Average ranking metrics, optionally by question category.

    Rows are expected to contain the output of ``evaluate_ranking`` and a
    ``latency_ms`` value. Empty relevant sets should be filtered before calling
    this function because retrieval ranking is undefined for them.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_field, "all")) if group_field else "all"
        groups[key].append(row)
    metric_names = [
        "recall@1", "recall@3", "recall@5", "recall@10",
        "mrr", "ndcg@5", "hit_rate@5", "latency_ms",
    ]
    output = []
    for key, group in sorted(groups.items()):
        item: dict[str, float | str] = {group_field or "group": key, "questions": len(group)}
        for name in metric_names:
            item[name] = sum(float(row.get(name, 0.0)) for row in group) / max(len(group), 1)
        output.append(item)
    return output

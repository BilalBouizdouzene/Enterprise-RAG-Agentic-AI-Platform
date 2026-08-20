"""Evaluation dataset loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    # Keep old datasets readable while exposing the explicit taxonomy to every
    # evaluator. New files should persist ``category`` directly.
    legacy_categories = {
        "fact": "lexical_exact",
        "paraphrase": "semantic_paraphrase",
        "lexical_exact": "lexical_exact",
        "semantic_paraphrase": "semantic_paraphrase",
        "entity_or_acronym": "entity_or_acronym",
        "multi_document": "multi_document",
        "metadata": "metadata",
        "unanswerable": "unanswerable",
    }
    for record in records:
        record["category"] = legacy_categories.get(record.get("category", record.get("question_type", "")), "lexical_exact")
    return records

"""Persist the explicit evaluation taxonomy in an existing question file."""

from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = {
    "fact": "lexical_exact",
    "paraphrase": "semantic_paraphrase",
    "lexical_exact": "lexical_exact",
    "semantic_paraphrase": "semantic_paraphrase",
    "entity_or_acronym": "entity_or_acronym",
    "multi_document": "multi_document",
    "metadata": "metadata",
    "unanswerable": "unanswerable",
}


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "retrieval_questions.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        record["category"] = CATEGORIES.get(record.get("category", record.get("question_type", "")), "lexical_exact")
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Persisted categories for {len(records)} evaluation questions.")


if __name__ == "__main__":
    main()

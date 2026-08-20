"""Run the retrieval benchmark without requiring optional transformer models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.config import load_settings
from enterprise_rag.evaluation.dataset import load_json_records
from enterprise_rag.ingestion.models import DocumentChunk
from run_benchmark import retrieval_ablation


def load_chunks(path: Path) -> list[DocumentChunk]:
    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [DocumentChunk(
        chunk_id=p["chunk_id"], document_id=p["document_id"], text=p["text"], title=p["title"],
        source=p["source"], metadata=p, page=p.get("page"), section=p.get("section"),
    ) for p in payloads]


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def main() -> None:
    settings = load_settings()
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    questions = [
        item for item in load_json_records(settings.data_dir / "evaluation" / "retrieval_questions.json")
        if item["relevant_document_ids"]
    ]
    summaries, rows, category_rows, skipped = retrieval_ablation(settings, chunks, questions)
    output_dir = settings.results_dir / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in [("retrieval_summary.json", summaries), ("retrieval_by_category.json", category_rows), ("retrieval_skipped.json", skipped)]:
        (output_dir / filename).write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "by_category": category_rows, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()

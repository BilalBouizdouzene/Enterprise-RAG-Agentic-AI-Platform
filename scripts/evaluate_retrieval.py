"""Run reproducible BM25, dense and hybrid retrieval benchmarks."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.config import load_settings
from enterprise_rag.embeddings import create_embedder
from enterprise_rag.evaluation.dataset import load_json_records
from enterprise_rag.evaluation.retrieval_metrics import evaluate_ranking
from enterprise_rag.ingestion.models import DocumentChunk
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.sparse import BM25Retriever


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
    dense = DenseRetriever(chunks, create_embedder(settings.embedding_model, dimension=settings.embedding_dimension))
    sparse = BM25Retriever(chunks)
    retrievers = {"BM25": sparse, "Dense": dense, "Hybrid": HybridRetriever(dense, sparse)}
    rows = []
    for name, retriever in retrievers.items():
        for question in questions:
            start = time.perf_counter()
            results = retriever.search(question["question"], top_k=10)
            latency_ms = (time.perf_counter() - start) * 1000
            # Ground truth is document-level, while retrieval returns chunks.
            ranked_docs = unique_in_order([item.document_id for item in results])
            metrics = evaluate_ranking(ranked_docs, question["relevant_document_ids"])
            rows.append({"pipeline": name, "question_id": question["id"], "latency_ms": latency_ms, **metrics})
    output_dir = settings.results_dir / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "retrieval_question_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name in retrievers:
        subset = [row for row in rows if row["pipeline"] == name]
        summary.append({
            "pipeline": name,
            "questions": len(subset),
            "recall@5": sum(row["recall@5"] for row in subset) / len(subset),
            "mrr": sum(row["mrr"] for row in subset) / len(subset),
            "ndcg@5": sum(row["ndcg@5"] for row in subset) / len(subset),
            "latency_ms_mean": sum(row["latency_ms"] for row in subset) / len(subset),
        })
    (output_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

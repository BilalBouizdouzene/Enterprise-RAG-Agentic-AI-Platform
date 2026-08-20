"""Run the final retrieval ablation and controlled chunking experiment."""

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
from enterprise_rag.ingestion.chunker import ChunkingConfig
from enterprise_rag.ingestion.pipeline import ingest_directory
from enterprise_rag.monitoring.mlflow_tracking import tracked_run
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker
from enterprise_rag.retrieval.sparse import BM25Retriever
from evaluate_retrieval import load_chunks, unique_in_order


def mean(values):
    return sum(values) / max(len(values), 1)


def retrieval_ablation(settings, chunks, questions):
    dense = DenseRetriever(chunks, create_embedder(settings.embedding_model, dimension=settings.embedding_dimension))
    sparse = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, sparse)
    reranker = Reranker()
    pipelines = {
        "BM25": lambda q: sparse.search(q, 20),
        "Dense": lambda q: dense.search(q, 20),
        "Hybrid": lambda q: hybrid.search(q, 20),
        "Hybrid + Reranker (lexical fallback)": lambda q: reranker.rerank(q, hybrid.search(q, 20), 5),
    }
    summaries = []
    question_rows = []
    for name, search in pipelines.items():
        metrics = []
        for question in questions:
            start = time.perf_counter()
            results = search(question["question"])
            latency = (time.perf_counter() - start) * 1000
            ranked = unique_in_order([item.document_id for item in results])
            values = evaluate_ranking(ranked, question["relevant_document_ids"])
            metrics.append(values)
            question_rows.append({"pipeline": name, "question_id": question["id"], "ranked_documents": ranked, **values, "latency_ms": latency})
        summaries.append({
            "pipeline": name,
            "questions": len(metrics),
            "recall@5": mean([item["recall@5"] for item in metrics]),
            "mrr": mean([item["mrr"] for item in metrics]),
            "ndcg@5": mean([item["ndcg@5"] for item in metrics]),
            "latency_ms_mean": mean([row["latency_ms"] for row in question_rows if row["pipeline"] == name]),
        })
    return summaries, question_rows


def chunking_experiment(settings, questions):
    configs = [
        ("fixed_small", ChunkingConfig("fixed", 450, 50)),
        ("fixed_medium", ChunkingConfig("fixed", 900, 120)),
        ("fixed_large", ChunkingConfig("fixed", 1400, 180)),
        ("section_aware", ChunkingConfig("section", 900, 120)),
    ]
    rows = []
    for name, config in configs:
        chunks, stats, _ = ingest_directory(settings.data_dir / "raw", config)
        dense = DenseRetriever(chunks, create_embedder(settings.embedding_model, dimension=settings.embedding_dimension))
        metrics = []
        latencies = []
        for question in questions:
            start = time.perf_counter()
            result = dense.search(question["question"], 5)
            latencies.append((time.perf_counter() - start) * 1000)
            metrics.append(evaluate_ranking(unique_in_order([item.document_id for item in result]), question["relevant_document_ids"]))
        rows.append({
            "configuration": name,
            "strategy": config.strategy,
            "chunk_size_chars": config.chunk_size,
            "overlap_chars": config.overlap,
            "chunks": len(chunks),
            "documents": stats.parsed,
            "avg_chunks_per_document": len(chunks) / max(stats.parsed, 1),
            "recall@5": mean([item["recall@5"] for item in metrics]),
            "mrr": mean([item["mrr"] for item in metrics]),
            "latency_ms_mean": mean(latencies),
        })
    return rows


def main() -> None:
    settings = load_settings()
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    questions = [
        item for item in load_json_records(settings.data_dir / "evaluation" / "retrieval_questions.json")
        if item["relevant_document_ids"]
    ]
    retrieval_summary, question_rows = retrieval_ablation(settings, chunks, questions)
    chunking_rows = chunking_experiment(settings, questions)
    output_dir = settings.results_dir / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("final_benchmark.csv", retrieval_summary), ("chunking_benchmark.csv", chunking_rows), ("final_question_results.csv", question_rows)]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "final_benchmark.json").write_text(json.dumps(retrieval_summary, indent=2), encoding="utf-8")
    with tracked_run("retrieval-ablation", {"embedding_model": settings.embedding_model, "question_count": len(questions)}, {"recall_at_5_hybrid": next(row["recall@5"] for row in retrieval_summary if row["pipeline"] == "Hybrid")}):
        pass
    report = settings.results_dir / "reports"
    report.mkdir(parents=True, exist_ok=True)
    baseline = {row["question_id"]: row for row in question_rows if row["pipeline"] == "BM25"}
    dense = {row["question_id"]: row for row in question_rows if row["pipeline"] == "Dense"}
    hybrid = {row["question_id"]: row for row in question_rows if row["pipeline"] == "Hybrid"}
    lines = ["# Retrieval error analysis", "", "This report is generated from `final_question_results.csv`; no examples are hand-scored.", ""]
    for label, left, right, condition in [
        ("Dense improves on BM25", baseline, dense, lambda a, b: b["mrr"] > a["mrr"]),
        ("Hybrid improves on both baselines", baseline, hybrid, lambda a, b: b["mrr"] > a["mrr"]),
    ]:
        lines += [f"## {label}", ""]
        found = 0
        for question in questions:
            if condition(left[question["id"]], right[question["id"]]):
                lines.append(f"- `{question['id']}` — {question['question']} (MRR {left[question['id']]['mrr']:.3f} → {right[question['id']]['mrr']:.3f})")
                found += 1
            if found >= 5:
                break
        if not found:
            lines.append("- Aucun exemple dans cette exécution.")
        lines.append("")
    (report / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(retrieval_summary, indent=2))


if __name__ == "__main__":
    main()

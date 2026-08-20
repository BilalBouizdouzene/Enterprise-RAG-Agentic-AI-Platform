"""Run reproducible retrieval ablations and generate evidence-backed reports."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.config import load_settings
from enterprise_rag.embeddings import create_embedder
from enterprise_rag.evaluation.dataset import load_json_records
from enterprise_rag.evaluation.retrieval_metrics import aggregate_metrics, evaluate_ranking
from enterprise_rag.ingestion.chunker import ChunkingConfig
from enterprise_rag.ingestion.pipeline import ingest_directory
from enterprise_rag.monitoring.mlflow_tracking import tracked_run
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker, create_reranker
from enterprise_rag.retrieval.sparse import BM25Retriever


def load_chunks(path: Path):
    from enterprise_rag.ingestion.models import DocumentChunk

    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [DocumentChunk(chunk_id=p["chunk_id"], document_id=p["document_id"], text=p["text"], title=p["title"], source=p["source"], metadata=p, page=p.get("page"), section=p.get("section")) for p in payloads]


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _optional_embedder(name: str, settings):
    try:
        embedder = create_embedder(
            name,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            normalize=settings.embedding_normalize,
            local_files_only=settings.embedding_local_files_only,
            allow_fallback=False,
        )
        return embedder, None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return None, str(exc)


def _rank_of_expected(ranked: list[str], expected: list[str]) -> int | None:
    for index, document_id in enumerate(ranked, start=1):
        if document_id in expected:
            return index
    return None


def retrieval_ablation(settings, chunks, questions):
    sparse = BM25Retriever(chunks)
    hashing = DenseRetriever(chunks, create_embedder("hashing", dimension=settings.embedding_dimension, normalize=settings.embedding_normalize))
    hybrid_hashing = HybridRetriever(hashing, sparse)
    lexical_reranker = Reranker()
    pipelines = {
        "BM25": lambda q: sparse.search(q, 10),
        "Dense Hashing": lambda q: hashing.search(q, 10),
        "Hybrid RRF (Hashing)": lambda q: hybrid_hashing.search(q, 10),
        "Hybrid + Lexical fallback": lambda q: lexical_reranker.rerank(q, hybrid_hashing.search(q, 20), 5),
    }
    skipped = {}
    bge, bge_error = _optional_embedder("BAAI/bge-small-en-v1.5", settings)
    if bge is not None:
        bge_dense = DenseRetriever(chunks, bge)
        bge_hybrid = HybridRetriever(bge_dense, sparse)
        pipelines["Dense BGE"] = lambda q, retriever=bge_dense: retriever.search(q, 10)
        pipelines["BM25 + BGE RRF"] = lambda q, retriever=bge_hybrid: retriever.search(q, 10)
    else:
        skipped["Dense BGE"] = bge_error
        skipped["BM25 + BGE RRF"] = bge_error

    if os.environ.get("BENCHMARK_E5", "true").lower() in {"1", "true", "yes"}:
        e5, e5_error = _optional_embedder("intfloat/e5-small-v2", settings)
        if e5 is not None:
            e5_dense = DenseRetriever(chunks, e5)
            pipelines["Dense E5"] = lambda q, retriever=e5_dense: retriever.search(q, 10)
        else:
            skipped["Dense E5"] = e5_error

    multilingual_hybrid = None
    if os.environ.get("BENCHMARK_MULTILINGUAL", "true").lower() in {"1", "true", "yes"}:
        multilingual, multilingual_error = _optional_embedder(settings.embedding_multilingual_model, settings)
        if multilingual is not None:
            multilingual_dense = DenseRetriever(chunks, multilingual)
            multilingual_hybrid = HybridRetriever(multilingual_dense, sparse)
            pipelines["Dense Multilingual E5"] = lambda q, retriever=multilingual_dense: retriever.search(q, 10)
            pipelines["BM25 + Multilingual E5 RRF"] = lambda q, retriever=multilingual_hybrid: retriever.search(q, 10)
        else:
            skipped["Dense Multilingual E5"] = multilingual_error
            skipped["BM25 + Multilingual E5 RRF"] = multilingual_error

    reranker = None
    if settings.reranker_enabled:
        try:
            reranker = create_reranker(
                settings.reranker_model,
                enabled=True,
                batch_size=settings.reranker_batch_size,
                device=settings.reranker_device,
                local_files_only=settings.embedding_local_files_only,
                allow_fallback=False,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            skipped["Hybrid + CrossEncoder"] = str(exc)
    if reranker is not None:
        source_hybrid = multilingual_hybrid or (HybridRetriever(DenseRetriever(chunks, bge), sparse) if bge is not None else hybrid_hashing)
        pipelines["Hybrid + CrossEncoder"] = lambda q, source=source_hybrid, model=reranker: model.rerank(q, source.search(q, 20), 5)
    else:
        skipped.setdefault(
            "Hybrid + CrossEncoder",
            "RERANKER_ENABLED=false; set it to true to run the optional CrossEncoder benchmark",
        )

    multilingual_reranker = None
    if settings.reranker_enabled and os.environ.get("BENCHMARK_MULTILINGUAL", "true").lower() in {"1", "true", "yes"}:
        try:
            multilingual_reranker = create_reranker(
                settings.reranker_multilingual_model,
                enabled=True,
                batch_size=settings.reranker_batch_size,
                device=settings.reranker_device,
                local_files_only=settings.embedding_local_files_only,
                allow_fallback=False,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            skipped["Hybrid + Multilingual CrossEncoder"] = str(exc)
    if multilingual_reranker is not None:
        source_hybrid = multilingual_hybrid or (HybridRetriever(DenseRetriever(chunks, bge), sparse) if bge is not None else hybrid_hashing)
        pipelines["Hybrid + Multilingual CrossEncoder"] = lambda q, source=source_hybrid, model=multilingual_reranker: model.rerank(q, source.search(q, 20), 5)
    elif settings.reranker_enabled:
        skipped["Hybrid + Multilingual CrossEncoder"] = skipped.get("Hybrid + Multilingual CrossEncoder", "multilingual CrossEncoder unavailable")

    summaries = []
    question_rows = []
    category_rows = []
    for name, search in pipelines.items():
        rows = []
        for question in questions:
            start = time.perf_counter()
            results = search(question["question"])
            latency = (time.perf_counter() - start) * 1000
            ranked = unique_in_order([item.document_id for item in results])
            values = evaluate_ranking(ranked, question["relevant_document_ids"])
            row = {"pipeline": name, "question_id": question["id"], "question": question["question"], "category": question["category"], "expected_documents": ";".join(question["relevant_document_ids"]), "ranked_documents": ";".join(ranked), "latency_ms": latency, **values}
            rows.append(row)
            question_rows.append(row)
        summary = aggregate_metrics(rows)[0]
        summary["pipeline"] = name
        summaries.append(summary)
        for category_row in aggregate_metrics(rows, "category"):
            category_row["pipeline"] = name
            category_rows.append(category_row)
    return summaries, question_rows, category_rows, skipped


def chunking_experiment(settings, questions):
    configs = [("fixed_small", ChunkingConfig("fixed", 450, 50)), ("fixed_medium", ChunkingConfig("fixed", 900, 120)), ("fixed_large", ChunkingConfig("fixed", 1400, 180)), ("section_aware", ChunkingConfig("section", 900, 120))]
    rows = []
    for name, config in configs:
        chunks, stats, _ = ingest_directory(settings.data_dir / "raw", config)
        dense = DenseRetriever(chunks, create_embedder("hashing", dimension=settings.embedding_dimension))
        metrics, latencies = [], []
        for question in questions:
            start = time.perf_counter()
            result = dense.search(question["question"], 5)
            latencies.append((time.perf_counter() - start) * 1000)
            metrics.append(evaluate_ranking(unique_in_order([item.document_id for item in result]), question["relevant_document_ids"]))
        rows.append({"configuration": name, "strategy": config.strategy, "chunk_size_chars": config.chunk_size, "overlap_chars": config.overlap, "chunks": len(chunks), "documents": stats.parsed, "avg_chunks_per_document": len(chunks) / max(stats.parsed, 1), "recall@5": mean([item["recall@5"] for item in metrics]), "mrr": mean([item["mrr"] for item in metrics]), "latency_ms_mean": mean(latencies)})
    return rows


def write_error_report(output_path: Path, question_rows: list[dict]) -> None:
    by_pipeline = {}
    for row in question_rows:
        by_pipeline.setdefault(row["pipeline"], {})[row["question_id"]] = row
    bm25 = by_pipeline.get("BM25", {})
    dense = by_pipeline.get("Dense BGE", by_pipeline.get("Dense Hashing", {}))
    hybrid = by_pipeline.get("BM25 + BGE RRF", by_pipeline.get("Hybrid RRF (Hashing)", {}))
    cross = by_pipeline.get("Hybrid + CrossEncoder", {})
    sections = ["# Retrieval error analysis", "", "Generated only from measured rankings; a missing rank is shown as `-`.", ""]
    questions = {row["question_id"]: row for row in question_rows}
    cases = [
        ("BM25 succeeds and Dense fails", bm25, dense, lambda a, b: a.get("hit_rate@5", 0) == 1 and b.get("hit_rate@5", 0) == 0),
        ("Dense succeeds and BM25 fails", dense, bm25, lambda a, b: a.get("hit_rate@5", 0) == 1 and b.get("hit_rate@5", 0) == 0),
        ("Hybrid corrects an error", hybrid, bm25, lambda a, b: a.get("hit_rate@5", 0) == 1 and b.get("hit_rate@5", 0) == 0),
        ("CrossEncoder improves the rank", cross, hybrid, lambda a, b: a and b and a.get("mrr", 0) > b.get("mrr", 0)),
        ("CrossEncoder degrades the result", cross, hybrid, lambda a, b: a and b and a.get("mrr", 0) < b.get("mrr", 0)),
        ("All available retrievers fail", bm25, dense, lambda a, b: a.get("hit_rate@5", 0) == 0 and b.get("hit_rate@5", 0) == 0 and hybrid.get(a.get("question_id", ""), {}).get("hit_rate@5", 0) == 0),
    ]
    for title, left, right, condition in cases:
        sections.extend([f"## {title}", ""])
        count = 0
        for question_id, left_row in left.items():
            right_row = right.get(question_id, {})
            if not condition(left_row, right_row):
                continue
            q = questions.get(question_id, {})
            expected = q.get("expected_documents", "").split(";") if q.get("expected_documents") else []
            def rank(row, expected_documents=expected):
                ranked = row.get("ranked_documents", "").split(";") if row.get("ranked_documents") else []
                return _rank_of_expected(ranked, expected_documents) or "-"
            sections.append(f"- `{question_id}` ({q.get('category', '')}): {q.get('question', '')}; expected `{q.get('expected_documents', '')}`; BM25 rank `{rank(bm25.get(question_id, {}))}`; Dense rank `{rank(dense.get(question_id, {}))}`; Hybrid rank `{rank(hybrid.get(question_id, {}))}`; reranked rank `{rank(cross.get(question_id, {}))}`.")
            count += 1
            if count >= 5:
                break
        if count == 0:
            sections.append("- No measured example in this run.")
        sections.append("")
    output_path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    settings = load_settings()
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    questions = [item for item in load_json_records(settings.data_dir / "evaluation" / "retrieval_questions.json") if item["relevant_document_ids"]]
    retrieval_summary, question_rows, category_rows, skipped = retrieval_ablation(settings, chunks, questions)
    chunking_rows = chunking_experiment(settings, questions)
    output_dir = settings.results_dir / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("final_benchmark.csv", retrieval_summary), ("chunking_benchmark.csv", chunking_rows), ("final_question_results.csv", question_rows), ("benchmark_by_category.csv", category_rows)]:
        if rows:
            with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (output_dir / "final_benchmark.json").write_text(json.dumps(retrieval_summary, indent=2), encoding="utf-8")
    (output_dir / "skipped_pipelines.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    with tracked_run("retrieval-ablation", {"embedding_model": settings.embedding_model, "question_count": len(questions)}, {"recall_at_5_hashing": next(row["recall@5"] for row in retrieval_summary if row["pipeline"] == "Dense Hashing")}):
        pass
    report = settings.results_dir / "reports"
    report.mkdir(parents=True, exist_ok=True)
    write_error_report(report / "error_analysis.md", question_rows)
    print(json.dumps({"summary": retrieval_summary, "by_category": category_rows, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()

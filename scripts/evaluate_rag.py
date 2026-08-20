"""Evaluate RAG quality with the same labeled dataset and provider per run."""

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
from enterprise_rag.llm import create_provider
from enterprise_rag.rag.citations import citation_integrity
from enterprise_rag.rag.pipeline import RAGPipeline
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker, create_reranker
from enterprise_rag.retrieval.sparse import BM25Retriever
from evaluate_retrieval import load_chunks


def token_set(text: str) -> set[str]:
    return {word.lower().strip(".,:;()[]") for word in text.split() if len(word) > 2}


def expected_signal(expected: str, answer: str, refused: bool) -> bool:
    if expected == "not specified":
        return refused
    return bool(token_set(expected) & token_set(answer))


def build_pipelines(settings, chunks):
    sparse = BM25Retriever(chunks)
    hashing = create_embedder("hashing", dimension=settings.embedding_dimension, normalize=settings.embedding_normalize)
    dense = DenseRetriever(chunks, hashing)
    provider = create_provider(settings.llm_provider, settings.llm_model, openai_base_url=settings.openai_base_url, ollama_base_url=settings.ollama_base_url)
    multilingual_hybrid = None
    pipelines = {
        "BM25 RAG": RAGPipeline(sparse, provider),
        "Dense RAG": RAGPipeline(dense, provider),
        "Hybrid RAG": RAGPipeline(HybridRetriever(dense, sparse), provider),
        "Hybrid + lexical fallback RAG": RAGPipeline(HybridRetriever(dense, sparse), provider, reranker=Reranker()),
    }
    skipped = {}
    if settings.reranker_enabled:
        try:
            reranker = create_reranker(settings.reranker_model, enabled=True, batch_size=settings.reranker_batch_size, device=settings.reranker_device, local_files_only=settings.embedding_local_files_only, allow_fallback=False)
            pipelines["Hybrid + CrossEncoder RAG"] = RAGPipeline(HybridRetriever(dense, sparse), provider, reranker=reranker)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            skipped["Hybrid + CrossEncoder RAG"] = str(exc)
    try:
        multilingual = create_embedder(
            settings.embedding_multilingual_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            normalize=settings.embedding_normalize,
            local_files_only=settings.embedding_local_files_only,
            allow_fallback=False,
        )
        multilingual_dense = DenseRetriever(chunks, multilingual)
        multilingual_hybrid = HybridRetriever(multilingual_dense, sparse)
        pipelines["Dense Multilingual E5 RAG"] = RAGPipeline(multilingual_dense, provider)
        pipelines["Hybrid Multilingual E5 RAG"] = RAGPipeline(multilingual_hybrid, provider)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        skipped["Dense/Hybrid Multilingual E5 RAG"] = str(exc)
    if settings.reranker_enabled:
        try:
            multilingual_reranker = create_reranker(
                settings.reranker_multilingual_model,
                enabled=True,
                batch_size=settings.reranker_batch_size,
                device=settings.reranker_device,
                local_files_only=settings.embedding_local_files_only,
                allow_fallback=False,
            )
            source_retriever = multilingual_hybrid or HybridRetriever(dense, sparse)
            pipelines["Hybrid + Multilingual CrossEncoder RAG"] = RAGPipeline(source_retriever, provider, reranker=multilingual_reranker)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            skipped["Hybrid + Multilingual CrossEncoder RAG"] = str(exc)
    return pipelines, skipped


def evaluate_pipeline(name, pipeline, questions, top_k):
    rows = []
    for question in questions:
        start = time.perf_counter()
        response = pipeline.ask(question["question"], top_k=top_k)
        latency = (time.perf_counter() - start) * 1000
        expected_docs = set(question["relevant_document_ids"])
        retrieved_docs = {item.document_id for item in response.retrieved}
        answerable = bool(expected_docs)
        correct = expected_signal(question["expected_answer"], response.answer, response.refused)
        grounded = response.refused if not answerable else bool(token_set(response.answer) & token_set(" ".join(item.text for item in response.retrieved)))
        cited_docs = {source.document_id for source in response.sources}
        citation_accuracy = (1.0 if not expected_docs and not cited_docs else len(cited_docs & expected_docs) / max(len(cited_docs), 1))
        citation_recall = (1.0 if not expected_docs and not cited_docs else len(cited_docs & expected_docs) / max(len(expected_docs), 1))
        correct_refusal = (not answerable and response.refused) or (answerable and not response.refused)
        unsupported = bool(not response.refused and not grounded)
        integrity = citation_integrity(response.answer, response.sources, response.retrieved)
        rows.append({
            "pipeline": name,
            "question_id": question["id"],
            "category": question["category"],
            "expected_documents": ";".join(question["relevant_document_ids"]),
            "retrieved_documents": ";".join(dict.fromkeys(item.document_id for item in response.retrieved)),
            "answer_correctness": float(correct),
            "faithfulness_groundedness": float(grounded),
            "citation_accuracy": float(citation_accuracy),
            "citation_recall": float(citation_recall),
            "correct_refusal": float(correct_refusal),
            "false_refusal": float(answerable and response.refused),
            "unsupported_answer": float(unsupported),
            "citation_integrity": float(integrity["integrity"]),
            "refused": response.refused,
            "retrieval_hit": bool(retrieved_docs & expected_docs),
            "latency_ms": latency,
            "answer": response.answer,
        })
    return rows


def summarize(rows):
    metrics = ["answer_correctness", "faithfulness_groundedness", "citation_accuracy", "citation_recall", "correct_refusal", "false_refusal", "unsupported_answer", "citation_integrity", "latency_ms"]
    return {"questions": len(rows), **{metric: sum(float(row[metric]) for row in rows) / max(len(rows), 1) for metric in metrics}}


def error_label(row):
    if row["category"] == "unanswerable" and row["correct_refusal"]:
        return "correct refusal", "validation"
    if row["category"] == "unanswerable" and not row["refused"]:
        return "hallucination / unsupported answer", "generation"
    if row["false_refusal"]:
        return "false refusal", "retrieval or validation"
    if not row["retrieval_hit"]:
        return "retrieval incorrect", "retrieval"
    if row["category"] == "multi_document" and not row["answer_correctness"]:
        return "multi-document answer incomplete", "context or generation"
    if not row["answer_correctness"]:
        return "retrieval correct but generation incorrect", "generation"
    if not row["citation_integrity"]:
        return "bad citation", "generation or validation"
    return None, None


def main() -> None:
    settings = load_settings()
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    questions = load_json_records(settings.data_dir / "evaluation" / "retrieval_questions.json")
    pipelines, skipped = build_pipelines(settings, chunks)
    all_rows, summaries, category_summaries, errors = [], [], [], []
    for name, pipeline in pipelines.items():
        rows = evaluate_pipeline(name, pipeline, questions, settings.final_top_k)
        all_rows.extend(rows)
        summaries.append({"pipeline": name, **summarize(rows)})
        for category in sorted({row["category"] for row in rows}):
            category_rows = [row for row in rows if row["category"] == category]
            category_summaries.append({"pipeline": name, "category": category, **summarize(category_rows)})
        for row in rows:
            label, source = error_label(row)
            if label:
                errors.append({"pipeline": name, "question_id": row["question_id"], "category": row["category"], "error": label, "probable_source": source})
    output_dir = settings.results_dir / "rag"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("rag_question_results.csv", all_rows), ("rag_by_category.csv", category_summaries), ("rag_error_analysis.csv", errors)]:
        if rows:
            with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (output_dir / "rag_summary.json").write_text(json.dumps({"pipelines": summaries, "by_category": category_summaries, "skipped": skipped}, indent=2), encoding="utf-8")
    print(json.dumps({"pipelines": summaries, "by_category": category_summaries, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()

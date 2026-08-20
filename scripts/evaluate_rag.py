"""Evaluate offline evidence-grounded RAG on the synthetic questions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.config import load_settings
from enterprise_rag.embeddings import create_embedder
from enterprise_rag.evaluation.dataset import load_json_records
from enterprise_rag.llm.base import TemplateProvider
from enterprise_rag.rag.pipeline import RAGPipeline
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker
from enterprise_rag.retrieval.sparse import BM25Retriever
from evaluate_retrieval import load_chunks


def token_set(text: str) -> set[str]:
    return {word.lower().strip(".,:;()") for word in text.split() if len(word) > 2}


def main() -> None:
    settings = load_settings()
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    questions = load_json_records(settings.data_dir / "evaluation" / "retrieval_questions.json")
    dense = DenseRetriever(chunks, create_embedder(settings.embedding_model, dimension=settings.embedding_dimension))
    pipeline = RAGPipeline(HybridRetriever(dense, BM25Retriever(chunks)), TemplateProvider(), Reranker())
    rows = []
    for question in questions:
        response = pipeline.ask(question["question"], top_k=settings.final_top_k)
        expected = question["expected_answer"]
        expected_tokens = token_set(expected)
        answer_tokens = token_set(response.answer)
        supported = bool(answer_tokens & token_set(" ".join(item.text for item in response.retrieved)))
        contains_expected = expected == "not specified" and response.refused or expected != "not specified" and bool(expected_tokens & answer_tokens)
        rows.append({
            "question_id": question["id"],
            "question_type": question["question_type"],
            "refused": response.refused,
            "contains_expected_signal": contains_expected,
            "supported_by_context": supported,
            "source_count": len(response.sources),
            "answer": response.answer,
        })
    output_dir = settings.results_dir / "rag"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rag_question_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "questions": len(rows),
        "expected_signal_rate": sum(row["contains_expected_signal"] for row in rows) / len(rows),
        "context_support_rate": sum(row["supported_by_context"] for row in rows) / len(rows),
        "refusal_rate": sum(row["refused"] for row in rows) / len(rows),
    }
    (output_dir / "rag_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

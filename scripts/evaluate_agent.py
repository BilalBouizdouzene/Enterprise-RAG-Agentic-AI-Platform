"""Evaluate deterministic tool routing against labeled agent questions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.agents.graph import EnterpriseAgent
from enterprise_rag.config import load_settings
from enterprise_rag.database.queries import initialize_database
from enterprise_rag.embeddings import create_embedder
from enterprise_rag.evaluation.dataset import load_json_records
from enterprise_rag.rag.pipeline import RAGPipeline
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.sparse import BM25Retriever
from evaluate_retrieval import load_chunks


def main() -> None:
    settings = load_settings()
    initialize_database(settings.database_path)
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    dense = DenseRetriever(chunks, create_embedder(settings.embedding_model, dimension=settings.embedding_dimension))
    agent = EnterpriseAgent(RAGPipeline(HybridRetriever(dense, BM25Retriever(chunks))), str(settings.database_path), settings.max_retries)
    questions = load_json_records(settings.data_dir / "evaluation" / "agent_questions.json")
    rows = []
    for item in questions:
        response = agent.run(item["question"])
        rows.append({"question": item["question"], "expected_tools": item["expected_tools"], "actual_tools": response.tools_used, "route_correct": response.tools_used == item["expected_tools"], "tool_calls": len(response.state.intermediate_results), "retries": response.state.retry_count})
    summary = {
        "questions": len(rows),
        "routing_accuracy": sum(row["route_correct"] for row in rows) / len(rows),
        "average_tool_calls": sum(row["tool_calls"] for row in rows) / len(rows),
        "average_retries": sum(row["retries"] for row in rows) / len(rows),
    }
    output = settings.results_dir / "agent"
    output.mkdir(parents=True, exist_ok=True)
    (output / "agent_question_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "agent_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


from pathlib import Path

from enterprise_rag.agents.graph import EnterpriseAgent
from enterprise_rag.agents.router import QueryRouter
from enterprise_rag.agents.tools import calculator, sql_query
from enterprise_rag.database.queries import initialize_database
from enterprise_rag.embeddings import HashingEmbedder
from enterprise_rag.ingestion.chunker import ChunkingConfig, DocumentChunker
from enterprise_rag.ingestion.models import EnterpriseDocument
from enterprise_rag.llm.base import TemplateProvider
from enterprise_rag.rag.pipeline import RAGPipeline
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.sparse import BM25Retriever


def test_router_selects_expected_tools():
    router = QueryRouter()
    assert router.route("What is the remote work policy?") == ["document_search"]
    assert router.route("How many employees work in France?") == ["sql_query"]
    assert router.route("What percentage of employees work in France?") == ["sql_query", "calculator"]


def test_sql_safety_and_calculator(tmp_path: Path):
    path = initialize_database(tmp_path / "company.db")
    assert sql_query(str(path), "SELECT COUNT(*) AS n FROM employees").data[0]["n"] == 40
    assert sql_query(str(path), "DROP TABLE employees").error
    assert calculator("10 / 4 + 2").data == 4.5


def test_agent_executes_sql_route(tmp_path: Path):
    path = initialize_database(tmp_path / "company.db")
    docs = [EnterpriseDocument("d1", "Remote work is allowed three days per week.", "Remote", "x.md")]
    chunks = DocumentChunker(ChunkingConfig(strategy="fixed", chunk_size=200, overlap=0)).chunk(docs[0])
    dense = DenseRetriever(chunks, HashingEmbedder(64))
    agent = EnterpriseAgent(RAGPipeline(HybridRetriever(dense, BM25Retriever(chunks)), TemplateProvider()), str(path))
    response = agent.run("How many employees work in France?")
    assert response.tools_used == ["sql_query"]
    assert "10" in response.answer


def test_agent_uses_bounded_history_for_follow_up(tmp_path: Path):
    path = initialize_database(tmp_path / "company.db")
    docs = [EnterpriseDocument("d1", "Travel policy applies in Germany.", "Travel", "x.md")]
    chunks = DocumentChunker(ChunkingConfig(strategy="fixed", chunk_size=200, overlap=0)).chunk(docs[0])
    dense = DenseRetriever(chunks, HashingEmbedder(64))
    agent = EnterpriseAgent(RAGPipeline(HybridRetriever(dense, BM25Retriever(chunks)), TemplateProvider()), str(path))
    response = agent.run("And in Germany?", [{"role": "user", "content": "What is the travel policy?"}])
    assert response.tools_used == ["document_search"]
    assert len(response.state.conversation_history) == 1

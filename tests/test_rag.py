from enterprise_rag.embeddings import HashingEmbedder
from enterprise_rag.ingestion.chunker import ChunkingConfig, DocumentChunker
from enterprise_rag.ingestion.models import EnterpriseDocument
from enterprise_rag.llm.base import TemplateProvider
from enterprise_rag.rag.citations import build_citations
from enterprise_rag.rag.pipeline import RAGPipeline
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.sparse import BM25Retriever


def _pipeline():
    docs = [EnterpriseDocument("d1", "The hotel limit is 180 EUR in Paris.", "Travel", "travel.md")]
    chunks = DocumentChunker(ChunkingConfig(strategy="fixed", chunk_size=200, overlap=0)).chunk(docs[0])
    dense = DenseRetriever(chunks, HashingEmbedder(64))
    return RAGPipeline(HybridRetriever(dense, BM25Retriever(chunks)), TemplateProvider())


def test_citations_have_stable_labels():
    result = _pipeline().retriever.search("Paris hotel limit", 2)[0]
    citations = build_citations([result])
    assert citations[0].label == "S1"
    assert citations[0].document_id == "d1"


def test_rag_returns_grounded_answer_and_sources():
    response = _pipeline().ask("What is the hotel limit in Paris?")
    assert response.refused is False
    assert response.sources
    assert "180" in response.answer


def test_rag_refuses_without_evidence():
    response = _pipeline().ask("What is the password policy in Madrid?")
    assert response.refused is True
    assert response.sources == []

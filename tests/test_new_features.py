import sys
from types import SimpleNamespace

import numpy as np
import pytest

from enterprise_rag.embeddings import HashingEmbedder, clear_embedder_cache, create_embedder
from enterprise_rag.evaluation.dataset import load_json_records
from enterprise_rag.evaluation.retrieval_metrics import aggregate_metrics
from enterprise_rag.llm import create_provider
from enterprise_rag.rag.citations import Citation, citation_integrity
from enterprise_rag.retrieval.base import RetrievalResult
from enterprise_rag.retrieval.reranker import create_reranker
from enterprise_rag.retrieval.vector_store import QdrantRetriever
from enterprise_rag.vectorstore.qdrant_store import QdrantVectorStore


def test_sentence_transformer_alias_is_cached_and_normalized(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, device, model_kwargs):
            calls.append((model_name, device, model_kwargs))

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.asarray([[3.0, 0.0, 4.0] for _ in texts]) / 5

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    clear_embedder_cache()
    first = create_embedder("bge-small", device="cpu", local_files_only=True)
    second = create_embedder("BAAI/bge-small-en-v1.5", device="cpu", local_files_only=True)
    assert first is second
    assert first.dimension == 3
    assert calls == [("BAAI/bge-small-en-v1.5", "cpu", {"local_files_only": True})]
    assert np.isclose(np.linalg.norm(first.embed_one("query")), 1.0)
    clear_embedder_cache()


def test_e5_adds_query_and_passage_prefixes(monkeypatch):
    captured = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            captured.append(texts)
            return np.ones((len(texts), 2))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    clear_embedder_cache()
    embedder = create_embedder("e5-small")
    embedder.embed(["document"])
    embedder.embed_one("question")
    assert captured == [["passage: document"], ["query: question"]]
    clear_embedder_cache()


def test_multilingual_e5_alias_uses_multilingual_model_and_prefixes(monkeypatch):
    captured = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured.append(("init", model_name))

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            captured.append(("encode", texts))
            return np.ones((len(texts), 2))

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    clear_embedder_cache()
    embedder = create_embedder("multilingual-e5-small", local_files_only=True)
    embedder.embed(["document multilingue"])
    embedder.embed_one("question en français")
    assert captured[0] == ("init", "intfloat/multilingual-e5-small")
    assert captured[1:] == [("encode", ["passage: document multilingue"]), ("encode", ["query: question en français"])]
    clear_embedder_cache()


def test_cross_encoder_factory_uses_cpu_and_batch(monkeypatch):
    calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def predict(self, pairs, **kwargs):
            assert kwargs["batch_size"] == 4
            return [0.8, 0.1]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    from enterprise_rag.retrieval.reranker import _create_cross_encoder

    _create_cross_encoder.cache_clear()
    reranker = create_reranker("cross-encoder/test", enabled=True, batch_size=4, device="cpu", local_files_only=True, allow_fallback=False)
    candidates = [RetrievalResult("c1", "d1", "good", 0.1), RetrievalResult("c2", "d2", "bad", 0.2)]
    assert reranker.rerank("q", candidates, 1)[0].document_id == "d1"
    assert calls[0][1]["device"] == "cpu"
    _create_cross_encoder.cache_clear()


def test_qdrant_collection_rebuilds_when_embedding_dimension_changes():
    pytest.importorskip("qdrant_client")
    chunk = RetrievalResult("c1", "d1", "text", 1.0)
    document_chunk = SimpleNamespace(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        title="title",
        source="source",
        metadata={},
        page=None,
        section=None,
        as_payload=lambda: {"chunk_id": "c1", "document_id": "d1", "text": "text"},
    )
    store = QdrantVectorStore(":memory:", "dimension-test")
    QdrantRetriever([document_chunk], HashingEmbedder(8), store, rebuild=True)
    assert store._collection_dimension() == 8
    QdrantRetriever([document_chunk], HashingEmbedder(16), store, rebuild=False)
    assert store._collection_dimension() == 16
def test_dataset_exposes_explicit_categories_and_category_aggregation():
    records = load_json_records("data/evaluation/retrieval_questions.json")
    categories = {record["category"] for record in records}
    assert {"lexical_exact", "semantic_paraphrase", "entity_or_acronym", "multi_document", "metadata", "unanswerable"} <= categories
    rows = [{"category": "semantic_paraphrase", "recall@1": 1, "recall@3": 1, "recall@5": 1, "recall@10": 1, "mrr": 1, "ndcg@5": 1, "hit_rate@5": 1, "latency_ms": 2}]
    assert aggregate_metrics(rows, "category")[0]["category"] == "semantic_paraphrase"


def test_citation_integrity_rejects_unknown_source():
    result = RetrievalResult("c1", "d1", "evidence", 1.0)
    citation = Citation("S1", "d1", "title", "source", None, "evidence")
    assert citation_integrity("Answer [S1]", [citation], [result])["integrity"] == 1.0
    bad = Citation("S2", "d2", "title", "source", None, "other")
    assert citation_integrity("Answer [S1]", [bad], [result])["integrity"] == 0.0


def test_provider_factory_keeps_offline_default_and_requires_model_for_api():
    assert create_provider("template").model_name == "template-offline"
    with pytest.raises(ValueError, match="LLM_MODEL"):
        create_provider("openai")


from enterprise_rag.embeddings import HashingEmbedder
from enterprise_rag.evaluation.retrieval_metrics import evaluate_ranking, ndcg_at_k, reciprocal_rank
from enterprise_rag.ingestion.chunker import ChunkingConfig, DocumentChunker
from enterprise_rag.ingestion.models import EnterpriseDocument
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.sparse import BM25Retriever


def _chunks():
    docs = [
        EnterpriseDocument("d1", "Paris hotel reimbursement is 180 EUR.", "Finance", "f.md"),
        EnterpriseDocument("d2", "Passwords require fourteen characters.", "Security", "s.md"),
    ]
    chunker = DocumentChunker(ChunkingConfig(strategy="fixed", chunk_size=100, overlap=0))
    return [chunk for doc in docs for chunk in chunker.chunk(doc)]


def test_metrics_known_ranking():
    assert reciprocal_rank(["x", "d1"], ["d1"]) == 0.5
    assert ndcg_at_k(["d1", "x"], ["d1"], 2) == 1.0
    values = evaluate_ranking(["x", "d1"], ["d1"], (1, 2))
    assert values["recall@1"] == 0.0
    assert values["recall@2"] == 1.0


def test_bm25_and_dense_share_result_contract():
    chunks = _chunks()
    bm25 = BM25Retriever(chunks).search("Paris hotel limit", 1)
    dense = DenseRetriever(chunks, HashingEmbedder(64)).search("Paris hotel limit", 1)
    assert bm25[0].document_id == "d1"
    assert dense[0].document_id == "d1"
    assert isinstance(dense[0].score, float)


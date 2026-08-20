"""Dense retrieval backed by the configured vector store."""

from __future__ import annotations

from enterprise_rag.embeddings.embedder import BaseEmbedder
from enterprise_rag.ingestion.models import DocumentChunk
from enterprise_rag.vectorstore.qdrant_store import QdrantVectorStore

from .base import RetrievalResult


class QdrantRetriever:
    """Retriever adapter that keeps the embedding contract used by hybrid search."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedder: BaseEmbedder,
        store: QdrantVectorStore,
        rebuild: bool = False,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.store = store
        self.vectors = embedder.embed([chunk.text for chunk in chunks])
        if rebuild:
            self.store.delete_collection()
        if not self.store.collection_exists() and chunks:
            self.store.add_chunks(chunks, self.vectors)

    def search(self, question: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        matches = self.store.search(self.embedder.embed_one(question), top_k, metadata_filter)
        return [
            RetrievalResult(
                chunk_id=str(payload.get("chunk_id", payload.get("id", ""))),
                document_id=str(payload.get("document_id", "")),
                text=str(payload.get("text", "")),
                score=score,
                metadata=payload,
                retriever="qdrant",
            )
            for score, payload in matches
        ]

"""Dense cosine-similarity retrieval."""

from __future__ import annotations

import numpy as np

from enterprise_rag.embeddings.embedder import BaseEmbedder
from enterprise_rag.ingestion.models import DocumentChunk

from .base import RetrievalResult


class DenseRetriever:
    def __init__(self, chunks: list[DocumentChunk], embedder: BaseEmbedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed([chunk.text for chunk in chunks])

    def search(self, question: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query = self.embedder.embed_one(question)
        scores = self.vectors @ query
        indexes = np.argsort(-scores)
        results = []
        for index in indexes:
            chunk = self.chunks[int(index)]
            if metadata_filter and any(chunk.metadata.get(k) != v for k, v in metadata_filter.items()):
                continue
            results.append(RetrievalResult(chunk.chunk_id, chunk.document_id, chunk.text, float(scores[index]), chunk.as_payload(), "dense"))
            if len(results) >= top_k:
                break
        return results


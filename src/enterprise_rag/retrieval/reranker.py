"""Optional cross-encoder reranking with a deterministic lexical fallback."""

from __future__ import annotations

from .base import RetrievalResult
from .sparse import tokenize


class Reranker:
    name = "lexical_fallback"

    def rerank(self, question: str, candidates: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        query = set(tokenize(question))
        scored = []
        for candidate in candidates:
            tokens = set(tokenize(candidate.text))
            overlap = len(query & tokens) / max(len(query), 1)
            scored.append((overlap + 0.05 * candidate.score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalResult(c.chunk_id, c.document_id, c.text, float(score), c.metadata, "reranker")
            for score, c in scored[:top_k]
        ]


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'retrieval' extra for CrossEncoder") from exc
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(self, question: str, candidates: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        scores = self._model.predict([(question, candidate.text) for candidate in candidates], show_progress_bar=False)
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda item: float(item[0]), reverse=True)
        return [
            RetrievalResult(c.chunk_id, c.document_id, c.text, float(score), c.metadata, "cross_encoder")
            for score, c in ranked[:top_k]
        ]

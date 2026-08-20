"""Optional cross-encoder reranking with a deterministic lexical fallback."""

from __future__ import annotations

from functools import lru_cache

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

    def __init__(self, model_name: str, batch_size: int = 16, device: str = "cpu", local_files_only: bool = False):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'retrieval' extra for CrossEncoder") from exc
        self._model = CrossEncoder(
            model_name,
            max_length=512,
            device=device,
            local_files_only=local_files_only,
        )
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(self, question: str, candidates: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        scores = self._model.predict(
            [(question, candidate.text) for candidate in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda item: float(item[0]), reverse=True)
        return [
            RetrievalResult(c.chunk_id, c.document_id, c.text, float(score), c.metadata, "cross_encoder")
            for score, c in ranked[:top_k]
        ]


@lru_cache(maxsize=4)
def _create_cross_encoder(model_name: str, batch_size: int, device: str, local_files_only: bool) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name, batch_size, device, local_files_only)


def create_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    *,
    enabled: bool = False,
    batch_size: int = 16,
    device: str = "cpu",
    local_files_only: bool = False,
    allow_fallback: bool = True,
) -> Reranker | None:
    """Create the optional CrossEncoder or the deterministic offline fallback."""
    if not enabled:
        return None
    if model_name.lower() in {"lexical", "fallback", "none"}:
        return Reranker()
    try:
        return _create_cross_encoder(model_name, batch_size, device, local_files_only)
    except (ImportError, OSError, RuntimeError, ValueError):
        if not allow_fallback:
            raise
        return Reranker()

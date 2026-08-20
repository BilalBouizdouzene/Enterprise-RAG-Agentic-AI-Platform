"""Configurable CPU-friendly embedding abstraction.

The hashing backend is intentionally deterministic and dependency-light. It provides an
offline baseline for experiments; SentenceTransformers can be selected when a model is
available locally or can be downloaded by the user.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache

import numpy as np


class BaseEmbedder(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text], is_query=True)[0]


class HashingEmbedder(BaseEmbedder):
    """Deterministic signed feature hashing with optional L2 normalization."""

    def __init__(self, dimension: int = 384, normalize: bool = True):
        self.model_name = "hashing"
        self.dimension = dimension
        self.normalize = normalize

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\wÀ-ÿ]+", text.lower())
            # Add word bigrams so short enterprise phrases remain distinguishable.
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                index = value % self.dimension
                sign = 1.0 if (value >> 32) & 1 else -1.0
                matrix[row, index] += sign
            if self.normalize:
                norm = np.linalg.norm(matrix[row])
                if norm:
                    matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str = "cpu",
        normalize: bool = True,
        local_files_only: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the optional 'retrieval' extra for SentenceTransformers") from exc
        self.model_name = model_name
        model_kwargs = {"local_files_only": True} if local_files_only else {}
        self._model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        self.normalize = normalize
        self.local_files_only = local_files_only
        self._uses_e5_prefixes = "e5" in model_name.lower()

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        prepared = list(texts)
        if self._uses_e5_prefixes:
            prefix = "query: " if is_query else "passage: "
            prepared = [prefix + text for text in prepared]
        return np.asarray(
            self._model.encode(
                prepared,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


MODEL_ALIASES = {
    "bge": "BAAI/bge-small-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-small-en": "BAAI/bge-small-en-v1.5",
    "e5": "intfloat/e5-small-v2",
    "e5-small": "intfloat/e5-small-v2",
    "multilingual-e5": "intfloat/multilingual-e5-small",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
}


@lru_cache(maxsize=8)
def _create_embedder_cached(
    model_name: str,
    dimension: int,
    batch_size: int,
    device: str,
    normalize: bool,
    local_files_only: bool,
    allow_fallback: bool,
) -> BaseEmbedder:
    canonical_name = MODEL_ALIASES.get(model_name.lower(), model_name)
    if canonical_name.lower() == "hashing":
        return HashingEmbedder(
            dimension=dimension,
            normalize=normalize,
        )
    try:
        return SentenceTransformerEmbedder(
            model_name=canonical_name,
            batch_size=batch_size,
            device=device,
            normalize=normalize,
            local_files_only=local_files_only,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if not allow_fallback:
            raise
        fallback = HashingEmbedder(dimension=dimension, normalize=normalize)
        fallback.requested_model_name = canonical_name
        fallback.fallback_reason = str(exc)
        return fallback


def create_embedder(model_name: str = "BAAI/bge-small-en-v1.5", **kwargs) -> BaseEmbedder:
    """Create a cached embedder, with hashing available as an explicit offline fallback.

    ``allow_fallback`` defaults to true for the local application path. Benchmarks can
    set it to false so an unavailable transformer is reported as skipped instead of
    being mislabeled as a transformer result.
    """
    canonical_name = MODEL_ALIASES.get(str(model_name).lower(), str(model_name))
    return _create_embedder_cached(
        canonical_name,
        int(kwargs.get("dimension", 384)),
        int(kwargs.get("batch_size", 16)),
        str(kwargs.get("device", "cpu")),
        bool(kwargs.get("normalize", True)),
        bool(kwargs.get("local_files_only", False)),
        bool(kwargs.get("allow_fallback", True)),
    )


def clear_embedder_cache() -> None:
    _create_embedder_cached.cache_clear()

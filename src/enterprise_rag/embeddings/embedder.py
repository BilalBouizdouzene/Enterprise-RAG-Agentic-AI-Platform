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

import numpy as np


class BaseEmbedder(ABC):
    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(BaseEmbedder):
    """Deterministic signed feature hashing with optional L2 normalization."""

    def __init__(self, dimension: int = 384, normalize: bool = True):
        self.model_name = "hashing"
        self.dimension = dimension
        self.normalize = normalize

    def embed(self, texts: Sequence[str]) -> np.ndarray:
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
    def __init__(self, model_name: str, batch_size: int = 16, device: str = "cpu", normalize: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the optional 'retrieval' extra for SentenceTransformers") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        self.normalize = normalize

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def create_embedder(model_name: str = "hashing", **kwargs) -> BaseEmbedder:
    if model_name.lower() == "hashing":
        return HashingEmbedder(
            dimension=int(kwargs.get("dimension", 384)),
            normalize=bool(kwargs.get("normalize", True)),
        )
    return SentenceTransformerEmbedder(
        model_name=model_name,
        batch_size=int(kwargs.get("batch_size", 16)),
        device=str(kwargs.get("device", "cpu")),
        normalize=bool(kwargs.get("normalize", True)),
    )

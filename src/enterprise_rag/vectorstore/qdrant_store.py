"""Small vector-store interface with in-memory and optional Qdrant backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from enterprise_rag.ingestion.models import DocumentChunk


@dataclass
class StoredPoint:
    vector: np.ndarray
    payload: dict[str, Any]


class InMemoryVectorStore:
    def __init__(self, collection_name: str = "enterprise_chunks"):
        self.collection_name = collection_name
        self._points: dict[str, StoredPoint] = {}

    def add_chunks(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._points[chunk.chunk_id] = StoredPoint(np.asarray(vector, dtype=np.float32), chunk.as_payload())

    def search(self, vector: np.ndarray, top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[tuple[float, dict[str, Any]]]:
        query = np.asarray(vector, dtype=np.float32)
        query_norm = np.linalg.norm(query) or 1.0
        matches = []
        for point in self._points.values():
            if metadata_filter and any(point.payload.get(k) != v for k, v in metadata_filter.items()):
                continue
            denom = (np.linalg.norm(point.vector) or 1.0) * query_norm
            matches.append((float(np.dot(point.vector, query) / denom), point.payload))
        return sorted(matches, key=lambda item: item[0], reverse=True)[:top_k]

    def delete_collection(self) -> None:
        self._points.clear()

    def collection_exists(self) -> bool:
        return bool(self._points)

    def health_check(self) -> dict[str, Any]:
        return {"backend": "memory", "collection": self.collection_name, "points": len(self._points), "healthy": True}


class QdrantVectorStore:
    """Qdrant adapter; imports are delayed so offline development remains usable."""

    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "enterprise_chunks"):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'qdrant' extra for Qdrant") from exc
        self.client = QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url)
        self.collection_name = collection_name

    def health_check(self) -> dict[str, Any]:
        try:
            self.client.get_collections()
            return {"backend": "qdrant", "collection": self.collection_name, "healthy": True}
        except Exception as exc:  # pragma: no cover
            return {"backend": "qdrant", "collection": self.collection_name, "healthy": False, "error": str(exc)}

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection_name))

    def delete_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

    def _collection_dimension(self) -> int | None:
        if not self.collection_exists():
            return None
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            vectors = next(iter(vectors.values()), None)
        return int(getattr(vectors, "size", 0) or 0) or None

    def ensure_collection(self, dimension: int, rebuild_on_mismatch: bool = True) -> bool:
        """Ensure that the collection dimension matches the active embedder.

        Qdrant collections cannot change vector size in place. Rebuilding the local
        collection on a model/dimension change prevents stale incompatible indexes.
        """
        existing = self._collection_dimension()
        changed = False
        if existing is not None and existing != dimension:
            if not rebuild_on_mismatch:
                raise ValueError(f"Qdrant collection dimension {existing} != embedder dimension {dimension}")
            self.delete_collection()
            changed = True
        if not self.collection_exists():
            from qdrant_client.models import Distance, VectorParams

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            changed = True
        return changed

    def add_chunks(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.ensure_collection(int(vectors.shape[1]))
        points = [PointStruct(id=index, vector=vector.tolist(), payload=chunk.as_payload()) for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: np.ndarray, top_k: int = 5, metadata_filter: dict[str, Any] | None = None) -> list[tuple[float, dict[str, Any]]]:
        query_filter = None
        if metadata_filter:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in metadata_filter.items()])
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
            )
            points = response.points
        else:  # compatibility with older qdrant-client releases
            points = self.client.search(self.collection_name, vector.tolist(), query_filter=query_filter, limit=top_k)
        return [(float(point.score), dict(point.payload or {})) for point in points]

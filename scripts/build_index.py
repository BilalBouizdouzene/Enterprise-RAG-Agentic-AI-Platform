"""Build a local dense index artifact from ingested chunks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from enterprise_rag.config import load_settings
from enterprise_rag.embeddings import create_embedder
from enterprise_rag.ingestion.models import DocumentChunk


def main() -> None:
    settings = load_settings()
    chunk_path = settings.data_dir / "processed" / "chunks.jsonl"
    if not chunk_path.exists():
        raise SystemExit("Run scripts/ingest.py first.")
    payloads = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [DocumentChunk(
        chunk_id=p["chunk_id"], document_id=p["document_id"], text=p["text"], title=p["title"],
        source=p["source"], metadata=p, page=p.get("page"), section=p.get("section"),
    ) for p in payloads]
    embedder = create_embedder(settings.embedding_model, dimension=settings.embedding_dimension)
    vectors = embedder.embed([c.text for c in chunks])
    output = settings.data_dir / "processed" / "dense_index.npz"
    np.savez_compressed(output, vectors=vectors)
    (output.with_suffix(".json")).write_text(json.dumps({"model": embedder.model_name, "dimension": embedder.dimension, "chunks": payloads}, default=str), encoding="utf-8")
    print(f"Built dense index with {len(chunks)} chunks and dimension {embedder.dimension}.")


if __name__ == "__main__":
    main()


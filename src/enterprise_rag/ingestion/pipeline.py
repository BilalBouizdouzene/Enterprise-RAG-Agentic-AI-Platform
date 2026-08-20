"""End-to-end ingestion pipeline and reproducible corpus statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import ChunkingConfig, DocumentChunker
from .cleaner import DocumentCleaner
from .loaders import load_directory
from .models import DocumentChunk


@dataclass
class IngestionStats:
    discovered: int = 0
    parsed: int = 0
    failed: int = 0
    total_characters: int = 0
    by_department: Counter[str] = field(default_factory=Counter)


def ingest_directory(directory: str | Path, config: ChunkingConfig | None = None) -> tuple[list[DocumentChunk], IngestionStats, list[dict[str, str]]]:
    documents, failures = load_directory(directory)
    cleaner = DocumentCleaner()
    chunker = DocumentChunker(config)
    stats = IngestionStats(discovered=len(documents) + len(failures), parsed=len(documents), failed=len(failures))
    chunks: list[DocumentChunk] = []
    for document in documents:
        cleaned = cleaner.clean(document)
        stats.total_characters += len(cleaned.text)
        stats.by_department[str(cleaned.metadata.get("department", "unknown"))] += 1
        chunks.extend(chunker.chunk(cleaned))
    return chunks, stats, failures


"""Run document ingestion and persist a JSONL chunk artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.config import load_settings
from enterprise_rag.ingestion.chunker import ChunkingConfig
from enterprise_rag.ingestion.pipeline import ingest_directory


def main() -> None:
    settings = load_settings()
    chunks, stats, failures = ingest_directory(
        settings.data_dir / "raw",
        ChunkingConfig(strategy="section", chunk_size=settings.chunk_size, overlap=settings.chunk_overlap),
    )
    output = settings.data_dir / "processed" / "chunks.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.as_payload(), ensure_ascii=False, default=str) + "\n")
    stats_path = settings.data_dir / "processed" / "ingestion_stats.json"
    stats_path.write_text(json.dumps({
        "discovered": stats.discovered,
        "parsed": stats.parsed,
        "failed": stats.failed,
        "total_characters": stats.total_characters,
        "by_department": dict(stats.by_department),
        "failures": failures,
        "chunk_count": len(chunks),
    }, indent=2), encoding="utf-8")
    print(f"Ingested {stats.parsed}/{stats.discovered} documents into {len(chunks)} chunks.")


if __name__ == "__main__":
    main()

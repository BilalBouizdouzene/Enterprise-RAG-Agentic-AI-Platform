"""Fixed-size and section-aware chunking strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import DocumentChunk, EnterpriseDocument


def _windowed(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return chunks


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "section"
    chunk_size: int = 900
    overlap: int = 120


class DocumentChunker:
    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    def chunk(self, document: EnterpriseDocument) -> list[DocumentChunk]:
        if self.config.strategy in {"fixed", "recursive"}:
            pieces = [(None, piece) for piece in _windowed(document.text, self.config.chunk_size, self.config.overlap)]
        elif self.config.strategy == "section":
            pieces = self._section_pieces(document.text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
        result = []
        for index, (section, text) in enumerate(pieces):
            result.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}::chunk-{index:04d}",
                    document_id=document.document_id,
                    text=text,
                    title=document.title,
                    source=document.source,
                    metadata=dict(document.metadata),
                    page=self._page_for_text(document, text),
                    section=section,
                )
            )
        return result

    def _section_pieces(self, text: str) -> list[tuple[str | None, str]]:
        heading = None
        buffer: list[str] = []
        sections: list[tuple[str | None, str]] = []
        for line in text.splitlines():
            match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
            if match:
                if buffer and "\n".join(buffer).strip():
                    sections.append((heading, "\n".join(buffer).strip()))
                heading = match.group(1).strip()
                buffer = []
            else:
                buffer.append(line)
        if buffer and "\n".join(buffer).strip():
            sections.append((heading, "\n".join(buffer).strip()))
        output: list[tuple[str | None, str]] = []
        for section, value in sections:
            pieces = _windowed(value, self.config.chunk_size, self.config.overlap)
            output.extend((section, piece) for piece in pieces)
        return output

    @staticmethod
    def _page_for_text(document: EnterpriseDocument, text: str) -> int | None:
        if not document.pages:
            return None
        for page in document.pages:
            if text[:80] in page.text:
                return page.number
        return document.pages[0].number


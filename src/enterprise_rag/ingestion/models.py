"""Canonical document and chunk data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentPage:
    number: int
    text: str


@dataclass
class EnterpriseDocument:
    document_id: str
    text: str
    title: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[DocumentPage] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.metadata.setdefault("document_id", self.document_id)
        self.metadata.setdefault("title", self.title)
        self.metadata.setdefault("source", self.source)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    title: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page: int | None = None
    section: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "title": self.title,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            **self.metadata,
        }


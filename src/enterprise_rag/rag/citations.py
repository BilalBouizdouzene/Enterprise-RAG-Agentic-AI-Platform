"""Citation construction from retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.retrieval.base import RetrievalResult


@dataclass(frozen=True)
class Citation:
    label: str
    document_id: str
    title: str
    source: str
    page: int | None
    excerpt: str

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "page": self.page,
            "excerpt": self.excerpt,
        }


def build_citations(results: list[RetrievalResult], max_sources: int = 5) -> list[Citation]:
    citations = []
    seen = set()
    for result in results:
        if result.document_id in seen:
            continue
        seen.add(result.document_id)
        citations.append(Citation(
            label=f"S{len(citations) + 1}",
            document_id=result.document_id,
            title=str(result.metadata.get("title", result.document_id)),
            source=str(result.metadata.get("source", "")),
            page=result.metadata.get("page"),
            excerpt=result.text[:500],
        ))
        if len(citations) >= max_sources:
            break
    return citations


def context_with_citations(results: list[RetrievalResult], citations: list[Citation]) -> str:
    by_document = {citation.document_id: citation for citation in citations}
    blocks = []
    seen_documents = set()
    for result in results:
        citation = by_document.get(result.document_id)
        if citation and result.document_id not in seen_documents:
            seen_documents.add(result.document_id)
            blocks.append(f"[{citation.label}] {result.text}")
    return "\n\n".join(blocks)

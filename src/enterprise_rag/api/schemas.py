"""Public API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    session_id: str = Field(default="default", max_length=100)


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    label: str
    document_id: str
    title: str
    source: str
    page: int | None = None
    excerpt: str


class AskResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    sources: list[SourceResponse]
    tools_used: list[str]
    latency_ms: float
    status: str
    refused: bool = False


class UploadResponse(BaseModel):
    filename: str
    kind: str
    indexed: bool
    status: str = "indexed"
    job_id: str | None = None
    documents_added: int
    chunks_added: int
    total_documents: int
    total_chunks: int
    message: str

"""Dependency-light BM25 lexical retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

from enterprise_rag.ingestion.models import DocumentChunk

from .base import RetrievalResult

TOKEN_ALIASES = {
    "lost": "lose",
    "losing": "lose",
    "employees": "employee",
    "policies": "policy",
    "expenses": "expense",
    "passwords": "password",
    "requests": "request",
    "documents": "document",
    "hotels": "hotel",
    "limits": "limit",
    "vendors": "vendor",
    "purchases": "purchase",
    "approvals": "approval",
    "changes": "change",
    "questions": "question",
    "laptops": "laptop",
    "records": "record",
}


def tokenize(text: str) -> list[str]:
    return [TOKEN_ALIASES.get(token, token) for token in re.findall(r"[\wÀ-ÿ]+", text.lower())]


class BM25Retriever:
    def __init__(self, chunks: list[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [
            tokenize(f"{chunk.title} {chunk.metadata.get('category', '')} {chunk.section or ''} {chunk.text}")
            for chunk in chunks
        ]
        self.doc_lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.doc_frequency.update(set(tokens))

    def _score(self, query_tokens: list[str], index: int) -> float:
        counts = Counter(self.tokens[index])
        length = self.doc_lengths[index]
        score = 0.0
        total = len(self.chunks)
        for token in query_tokens:
            if token not in counts:
                continue
            df = self.doc_frequency[token]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            tf = counts[token]
            denominator = tf + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1))
            score += idf * (tf * (self.k1 + 1) / denominator)
        return score

    def search(self, question: str, top_k: int = 5, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        query_tokens = tokenize(question)
        scored = []
        for index, chunk in enumerate(self.chunks):
            if metadata_filter and any(chunk.metadata.get(k) != v for k, v in metadata_filter.items()):
                continue
            scored.append((self._score(query_tokens, index), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [RetrievalResult(c.chunk_id, c.document_id, c.text, float(score), c.as_payload(), "bm25") for score, c in scored[:top_k]]

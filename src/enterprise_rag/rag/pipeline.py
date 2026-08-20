"""Basic hybrid RAG pipeline with citations and evidence-aware refusal."""

from __future__ import annotations

from dataclasses import dataclass, field

from enterprise_rag.llm.base import LLMProvider, TemplateProvider
from enterprise_rag.retrieval.base import RetrievalResult
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker

from .citations import Citation, build_citations, context_with_citations
from .generator import AnswerGenerator
from .validator import EvidenceAssessment, EvidenceValidator


@dataclass
class RAGResponse:
    answer: str
    sources: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievalResult] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    provider: str = "template-offline"

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [source.as_dict() for source in self.sources],
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "provider": self.provider,
        }


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider | None = None,
        reranker: Reranker | None = None,
        validator: EvidenceValidator | None = None,
    ):
        self.retriever = retriever
        self.generator = AnswerGenerator(provider or TemplateProvider())
        self.reranker = reranker
        self.validator = validator or EvidenceValidator()

    def ask(self, question: str, top_k: int = 5) -> RAGResponse:
        candidates = self.retriever.search(question, max(top_k * 4, 10))
        results = self.reranker.rerank(question, candidates, top_k) if self.reranker else candidates[:top_k]
        evidence_results = self.validator.filter_relevant(question, results)
        assessment: EvidenceAssessment = self.validator.assess(question, evidence_results)
        citations = build_citations(evidence_results)
        if not assessment.sufficient:
            return RAGResponse(
                answer="Je ne dispose pas de preuves suffisantes dans les documents disponibles.",
                sources=citations,
                retrieved=results,
                refused=True,
                refusal_reason=assessment.reason,
                provider=self.generator.provider.model_name,
            )
        answer = self.generator.generate(question, context_with_citations(evidence_results, citations))
        return RAGResponse(answer=answer, sources=citations, retrieved=results, provider=self.generator.provider.model_name)

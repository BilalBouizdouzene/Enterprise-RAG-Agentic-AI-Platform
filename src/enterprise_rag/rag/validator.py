"""Evidence sufficiency checks before generation."""

from __future__ import annotations

from dataclasses import dataclass

from enterprise_rag.retrieval.base import RetrievalResult
from enterprise_rag.retrieval.sparse import tokenize

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "the", "to", "what", "when", "where", "which", "who", "with", "this", "that",
    "please", "explain", "according", "policies", "simple", "terms", "summarize", "requirement",
    "should", "would", "could", "can", "much", "may", "must", "my", "your", "our", "company",
    "if", "do", "does", "i",
}

LOCATION_TERMS = {"paris", "berlin", "madrid", "france", "french", "germany", "allemagne", "lyon"}


@dataclass(frozen=True)
class EvidenceAssessment:
    sufficient: bool
    reason: str
    overlap: float


class EvidenceValidator:
    def __init__(self, min_overlap: float = 0.08):
        self.min_overlap = min_overlap

    def assess(self, question: str, results: list[RetrievalResult]) -> EvidenceAssessment:
        if not results:
            return EvidenceAssessment(False, "no_retrieval_results", 0.0)
        query = {token for token in tokenize(question) if token not in STOPWORDS}
        evidence = {token for token in tokenize(" ".join(result.text for result in results[:3])) if token not in STOPWORDS}
        overlap = len(query & evidence) / max(len(query), 1)
        if overlap < self.min_overlap:
            return EvidenceAssessment(False, "insufficient_lexical_support", overlap)
        return EvidenceAssessment(True, "evidence_available", overlap)

    def filter_relevant(self, question: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove candidate passages that do not share informative query terms."""
        query = {token for token in tokenize(question) if token not in STOPWORDS}
        if not query:
            return results[:1]
        relevant = []
        for result in results:
            evidence_text = " ".join(
                str(result.metadata.get(field, ""))
                for field in ("title", "department", "category", "effective_date")
            ) + f" {result.text}"
            evidence = {token for token in tokenize(evidence_text) if token not in STOPWORDS}
            overlap = len(query & evidence) / max(len(query), 1)
            requested_locations = query & LOCATION_TERMS
            has_requested_location = not requested_locations or bool(requested_locations & evidence)
            laptop_question = "laptop" in query and not ({"vendor", "compare"} & query)
            has_laptop_evidence = not laptop_question or bool({"laptop", "device", "equipment"} & evidence)
            if has_requested_location and has_laptop_evidence and overlap >= self.min_overlap:
                relevant.append(result)
        return relevant

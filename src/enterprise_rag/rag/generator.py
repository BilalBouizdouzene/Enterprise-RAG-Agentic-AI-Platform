"""Generation orchestration independent of the provider implementation."""

from __future__ import annotations

from enterprise_rag.llm.base import GenerationRequest, LLMProvider

from .prompts import SYSTEM_PROMPT


class AnswerGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, question: str, context: str) -> str:
        return self.provider.generate(GenerationRequest(question, context, SYSTEM_PROMPT))


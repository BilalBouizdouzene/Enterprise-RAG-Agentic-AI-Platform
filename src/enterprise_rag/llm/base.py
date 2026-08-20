"""Provider-neutral text generation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    context: str
    system_prompt: str
    temperature: float = 0.0


class LLMProvider(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        raise NotImplementedError


class TemplateProvider(LLMProvider):
    """Offline fallback that answers only by selecting supplied evidence."""

    model_name = "template-offline"

    def generate(self, request: GenerationRequest) -> str:
        lines = [line.strip() for line in request.context.splitlines() if line.strip()]
        evidence = " ".join(lines[:3])
        return f"D'après les éléments retrouvés : {evidence}" if evidence else "Je ne dispose pas de preuves suffisantes dans les documents."


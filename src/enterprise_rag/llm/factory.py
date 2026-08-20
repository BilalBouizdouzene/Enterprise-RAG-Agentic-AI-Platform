"""Configuration-driven LLM provider construction."""

from __future__ import annotations

from .base import LLMProvider, TemplateProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAICompatibleProvider


def create_provider(
    provider_name: str = "template",
    model_name: str = "",
    *,
    openai_base_url: str | None = None,
    ollama_base_url: str | None = None,
) -> LLMProvider:
    name = provider_name.lower().strip()
    if name in {"template", "offline", "mock"}:
        return TemplateProvider()
    if name in {"openai", "openai-compatible", "compatible"}:
        if not model_name:
            raise ValueError("LLM_MODEL is required for the OpenAI-compatible provider")
        return OpenAICompatibleProvider(model_name=model_name, base_url=openai_base_url)
    if name == "ollama":
        if not model_name:
            raise ValueError("LLM_MODEL is required for the Ollama provider")
        return OllamaProvider(model_name=model_name, base_url=ollama_base_url)
    raise ValueError(f"Unsupported LLM provider: {provider_name}")

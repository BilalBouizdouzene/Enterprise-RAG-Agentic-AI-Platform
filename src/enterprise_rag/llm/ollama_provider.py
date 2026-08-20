"""Optional local Ollama provider."""

from __future__ import annotations

import os

import httpx

from .base import GenerationRequest, LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str, base_url: str | None = None, timeout: float = 120.0):
        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def generate(self, request: GenerationRequest) -> str:
        prompt = f"{request.system_prompt}\n\nQuestion: {request.question}\n\nEvidence:\n{request.context}"
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False, "options": {"temperature": request.temperature}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"].strip()


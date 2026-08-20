"""OpenAI-compatible provider using the HTTP API contract."""

from __future__ import annotations

import os

import httpx

from .base import GenerationRequest, LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str | None = None, base_url: str | None = None, timeout: float = 60.0):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI-compatible provider")

    def generate(self, request: GenerationRequest) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "temperature": request.temperature,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": f"Question: {request.question}\n\nEvidence:\n{request.context}"},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


"""Optional Redis-backed indexing jobs."""

from __future__ import annotations

import json
from typing import Any

from enterprise_rag.config import Settings

QUEUE_NAME = "enterprise_rag:index_jobs"


def enqueue_index_job(settings: Settings, job: dict[str, Any]) -> None:
    try:
        import redis
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise RuntimeError("Install the production Redis dependency before enabling async indexing") from exc
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    client.rpush(QUEUE_NAME, json.dumps(job, ensure_ascii=False))

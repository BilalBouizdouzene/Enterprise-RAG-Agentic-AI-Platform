"""Redis worker for asynchronous document indexing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
import redis

from enterprise_rag.config import load_settings
from enterprise_rag.operations.jobs import QUEUE_NAME


def run_pipeline(root: Path) -> None:
    python = sys.executable
    subprocess.run([python, str(root / "scripts" / "ingest.py")], cwd=root, check=True)
    subprocess.run([python, str(root / "scripts" / "build_index.py")], cwd=root, check=True)


def notify_api(settings) -> None:
    headers = {"X-API-Key": settings.api_key} if settings.api_key else {}
    response = httpx.post(f"{settings.indexing_api_url}/admin/reload-index", headers=headers, timeout=120)
    response.raise_for_status()


def main() -> None:
    settings = load_settings()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    print(f"Indexing worker listening on {QUEUE_NAME}")
    while True:
        _, raw_job = client.blpop(QUEUE_NAME, timeout=5) or (None, None)
        if not raw_job:
            continue
        job = json.loads(raw_job)
        print(f"Processing indexing job {job.get('job_id', 'unknown')}", flush=True)
        run_pipeline(settings.root_dir)
        notify_api(settings)


if __name__ == "__main__":
    main()

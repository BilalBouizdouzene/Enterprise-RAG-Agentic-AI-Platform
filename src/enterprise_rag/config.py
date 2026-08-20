"""Configuration loading with conservative, Windows-friendly defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared, fallback helps bootstrap
    yaml = None


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    results_dir: Path
    database_path: Path
    embedding_model: str = "hashing"
    embedding_dimension: int = 384
    chunk_size: int = 900
    chunk_overlap: int = 120
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 5
    max_retries: int = 2
    environment: str = "development"
    auth_enabled: bool = False
    api_key: str = ""
    cors_origins: tuple[str, ...] = ()
    vector_store: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "enterprise_chunks"
    redis_url: str = "redis://localhost:6379/0"
    object_storage_endpoint: str = ""
    object_storage_bucket: str = "enterprise-documents"
    indexing_mode: str = "sync"
    indexing_api_url: str = "http://localhost:8000"


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(root_dir: str | Path | None = None) -> Settings:
    root = Path(root_dir or os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    app = _read_yaml(root / "configs" / "app.yaml")
    embeddings = _read_yaml(root / "configs" / "embeddings.yaml")
    retrieval = _read_yaml(root / "configs" / "retrieval.yaml")
    app_cfg = app.get("app", {})
    db_cfg = app.get("database", {})
    emb_cfg = embeddings.get("embedding", {})
    chunk_cfg = retrieval.get("chunking", {})
    ret_cfg = retrieval.get("retrieval", {})
    cors_value = os.environ.get("CORS_ORIGINS", "")
    return Settings(
        root_dir=root,
        data_dir=root / app_cfg.get("data_dir", "data"),
        results_dir=root / app_cfg.get("results_dir", "results"),
        database_path=root / db_cfg.get("path", "data/company.db"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", emb_cfg.get("model_name", "hashing")),
        embedding_dimension=int(emb_cfg.get("dimension", 384)),
        chunk_size=int(chunk_cfg.get("chunk_size", 900)),
        chunk_overlap=int(chunk_cfg.get("overlap", 120)),
        dense_top_k=int(ret_cfg.get("dense_top_k", 20)),
        sparse_top_k=int(ret_cfg.get("sparse_top_k", 20)),
        final_top_k=int(ret_cfg.get("final_top_k", 5)),
        max_retries=int(app_cfg.get("max_retries", 2)),
        environment=os.environ.get("APP_ENV", app_cfg.get("environment", "development")),
        auth_enabled=os.environ.get("AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
        api_key=os.environ.get("API_KEY", ""),
        cors_origins=tuple(item.strip() for item in cors_value.split(",") if item.strip()),
        vector_store=os.environ.get("VECTOR_STORE", "memory").lower(),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.environ.get("QDRANT_COLLECTION", "enterprise_chunks"),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        object_storage_endpoint=os.environ.get("S3_ENDPOINT_URL", ""),
        object_storage_bucket=os.environ.get("S3_BUCKET", "enterprise-documents"),
        indexing_mode=os.environ.get("INDEXING_MODE", "sync").lower(),
        indexing_api_url=os.environ.get("INDEXING_API_URL", "http://localhost:8000"),
    )

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

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - dependency is declared
    dotenv_values = None


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    results_dir: Path
    database_path: Path
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_multilingual_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: int = 384
    embedding_batch_size: int = 16
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    embedding_local_files_only: bool = False
    chunk_size: int = 900
    chunk_overlap: int = 120
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 5
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_multilingual_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_batch_size: int = 16
    reranker_device: str = "cpu"
    llm_provider: str = "template"
    llm_model: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
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


def _env_or_default(env: dict[str, str], key: str, default: str) -> str:
    value = env.get(key)
    return value.strip() if value is not None and value.strip() else default


def load_settings(root_dir: str | Path | None = None) -> Settings:
    root = Path(root_dir or os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    env_path = root / ".env"
    file_env = dict(dotenv_values(env_path)) if dotenv_values is not None and env_path.exists() else {}
    env = {**file_env, **os.environ}
    app = _read_yaml(root / "configs" / "app.yaml")
    embeddings = _read_yaml(root / "configs" / "embeddings.yaml")
    retrieval = _read_yaml(root / "configs" / "retrieval.yaml")
    app_cfg = app.get("app", {})
    db_cfg = app.get("database", {})
    emb_cfg = embeddings.get("embedding", {})
    chunk_cfg = retrieval.get("chunking", {})
    ret_cfg = retrieval.get("retrieval", {})
    models_cfg = _read_yaml(root / "configs" / "models.yaml").get("models", {})
    cors_value = env.get("CORS_ORIGINS", "")
    return Settings(
        root_dir=root,
        data_dir=root / app_cfg.get("data_dir", "data"),
        results_dir=root / app_cfg.get("results_dir", "results"),
        database_path=root / db_cfg.get("path", "data/company.db"),
        embedding_model=_env_or_default(env, "EMBEDDING_MODEL", emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5")),
        embedding_multilingual_model=_env_or_default(env, "MULTILINGUAL_EMBEDDING_MODEL", models_cfg.get("embedding_multilingual", "intfloat/multilingual-e5-small")),
        embedding_dimension=int(emb_cfg.get("dimension", 384)),
        embedding_batch_size=int(env.get("EMBEDDING_BATCH_SIZE", emb_cfg.get("batch_size", 16))),
        embedding_device=env.get("EMBEDDING_DEVICE", emb_cfg.get("device", "cpu")),
        embedding_normalize=str(env.get("EMBEDDING_NORMALIZE", emb_cfg.get("normalize", True))).lower() in {"1", "true", "yes"},
        embedding_local_files_only=str(env.get("EMBEDDING_LOCAL_FILES_ONLY", "false")).lower() in {"1", "true", "yes"},
        chunk_size=int(chunk_cfg.get("chunk_size", 900)),
        chunk_overlap=int(chunk_cfg.get("overlap", 120)),
        dense_top_k=int(ret_cfg.get("dense_top_k", 20)),
        sparse_top_k=int(ret_cfg.get("sparse_top_k", 20)),
        final_top_k=int(ret_cfg.get("final_top_k", 5)),
        reranker_enabled=str(env.get("RERANKER_ENABLED", ret_cfg.get("reranker_enabled", False))).lower() in {"1", "true", "yes"},
        reranker_model=_env_or_default(env, "RERANKER_MODEL", models_cfg.get("reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        reranker_multilingual_model=_env_or_default(env, "MULTILINGUAL_RERANKER_MODEL", models_cfg.get("reranker_multilingual", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")),
        reranker_batch_size=int(env.get("RERANKER_BATCH_SIZE", ret_cfg.get("reranker_batch_size", 16))),
        reranker_device=env.get("RERANKER_DEVICE", ret_cfg.get("reranker_device", "cpu")),
        llm_provider=env.get("LLM_PROVIDER", models_cfg.get("llm_provider", "template")),
        llm_model=env.get("LLM_MODEL", models_cfg.get("llm_model", "")),
        openai_base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        max_retries=int(app_cfg.get("max_retries", 2)),
        environment=env.get("APP_ENV", app_cfg.get("environment", "development")),
        auth_enabled=env.get("AUTH_ENABLED", "false").lower() in {"1", "true", "yes"},
        api_key=env.get("API_KEY", ""),
        cors_origins=tuple(item.strip() for item in cors_value.split(",") if item.strip()),
        vector_store=env.get("VECTOR_STORE", "memory").lower(),
        qdrant_url=env.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=env.get("QDRANT_COLLECTION", "enterprise_chunks"),
        redis_url=env.get("REDIS_URL", "redis://localhost:6379/0"),
        object_storage_endpoint=env.get("S3_ENDPOINT_URL", ""),
        object_storage_bucket=env.get("S3_BUCKET", "enterprise-documents"),
        indexing_mode=env.get("INDEXING_MODE", "sync").lower(),
        indexing_api_url=env.get("INDEXING_API_URL", "http://localhost:8000"),
    )

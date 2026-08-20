"""Production-style FastAPI endpoints over the local platform."""

from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from enterprise_rag.agents.graph import EnterpriseAgent
from enterprise_rag.config import load_settings
from enterprise_rag.embeddings import create_embedder
from enterprise_rag.ingestion.chunker import ChunkingConfig
from enterprise_rag.ingestion.models import DocumentChunk
from enterprise_rag.ingestion.pipeline import ingest_directory
from enterprise_rag.operations.jobs import enqueue_index_job
from enterprise_rag.rag.pipeline import RAGPipeline
from enterprise_rag.retrieval.dense import DenseRetriever
from enterprise_rag.retrieval.hybrid import HybridRetriever
from enterprise_rag.retrieval.reranker import Reranker
from enterprise_rag.retrieval.sparse import BM25Retriever
from enterprise_rag.retrieval.vector_store import QdrantRetriever
from enterprise_rag.storage.object_store import upload_original
from enterprise_rag.vectorstore.qdrant_store import QdrantVectorStore

from .schemas import AskRequest, AskResponse, SearchRequest, SourceResponse, UploadResponse

DOCUMENT_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".docx"}
IMAGE_UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_UPLOAD_EXTENSIONS = DOCUMENT_UPLOAD_EXTENSIONS | IMAGE_UPLOAD_EXTENSIONS
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_FILE_PARAM = File(...)
logger = logging.getLogger(__name__)


def _chunk_from_payload(payload: dict) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=payload["chunk_id"],
        document_id=payload["document_id"],
        text=payload["text"],
        title=payload["title"],
        source=payload["source"],
        metadata=payload,
        page=payload.get("page"),
        section=payload.get("section"),
    )


def load_chunks(path: Path) -> list[DocumentChunk]:
    if not path.exists():
        return []
    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [_chunk_from_payload(payload) for payload in payloads]


def create_app(root_dir: str | Path | None = None) -> FastAPI:
    settings = load_settings(root_dir)
    chunks = load_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    embedder = create_embedder(settings.embedding_model, dimension=settings.embedding_dimension)
    dense = None

    def build_dense(current_chunks: list[DocumentChunk], rebuild: bool = False):
        if settings.vector_store == "qdrant":
            store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection)
            return QdrantRetriever(current_chunks, embedder, store, rebuild=rebuild)
        return DenseRetriever(current_chunks, embedder)

    dense = build_dense(chunks, rebuild=settings.vector_store == "qdrant")
    pipeline = RAGPipeline(HybridRetriever(dense, BM25Retriever(chunks)), reranker=Reranker())
    agent = EnterpriseAgent(pipeline, str(settings.database_path), settings.max_retries)
    app = FastAPI(title="Enterprise RAG & Agentic AI Platform", version="0.1.0")
    sessions: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    def require_api_key(request: Request) -> None:
        if not settings.auth_enabled:
            return
        provided = request.headers.get("x-api-key", "")
        if not settings.api_key or not hmac.compare_digest(provided, settings.api_key):
            raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")

    def audit_event(event: str, request: Request, **details: object) -> None:
        audit_path = settings.data_dir / "audit" / "events.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "client": request.client.host if request.client else None,
            **details,
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    def persist_runtime_index(current_chunks: list[DocumentChunk], current_dense: DenseRetriever) -> None:
        processed_dir = settings.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = processed_dir / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in current_chunks:
                handle.write(json.dumps(chunk.as_payload(), ensure_ascii=False, default=str) + "\n")
        dense_path = processed_dir / "dense_index.npz"
        np.savez_compressed(dense_path, vectors=current_dense.vectors)
        dense_path.with_suffix(".json").write_text(
            json.dumps({
                "model": current_dense.embedder.model_name,
                "dimension": current_dense.embedder.dimension,
                "chunks": [chunk.as_payload() for chunk in current_chunks],
            }, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def rebuild_runtime_index() -> tuple[int, int, list[dict[str, str]]]:
        nonlocal chunks, dense, pipeline, agent
        previous_document_ids = {chunk.document_id for chunk in chunks}
        refreshed_chunks, stats, failures = ingest_directory(
            settings.data_dir / "raw",
            ChunkingConfig(strategy="section", chunk_size=settings.chunk_size, overlap=settings.chunk_overlap),
        )
        chunks = refreshed_chunks
        dense = build_dense(chunks, rebuild=settings.vector_store == "qdrant")
        pipeline = RAGPipeline(HybridRetriever(dense, BM25Retriever(chunks)), reranker=Reranker())
        agent.rag_pipeline = pipeline
        persist_runtime_index(chunks, dense)
        added_documents = len({chunk.document_id for chunk in chunks} - previous_document_ids)
        return added_documents, stats.failed, failures

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "documents": len({chunk.document_id for chunk in chunks}),
            "chunks": len(chunks),
            "embedding_model": embedder.model_name,
            "environment": settings.environment,
            "vector_store": settings.vector_store,
            "auth_enabled": settings.auth_enabled,
            "indexing_mode": settings.indexing_mode,
            "upload_enabled": True,
            "voice_input": "browser_speech_recognition",
        }

    @app.get("/health/live")
    def liveness() -> dict:
        return {"status": "alive"}

    @app.get("/health/ready")
    def readiness() -> dict:
        if not chunks:
            raise HTTPException(status_code=503, detail="The knowledge index is not ready.")
        return {"status": "ready", "documents": len({chunk.document_id for chunk in chunks}), "chunks": len(chunks)}

    @app.post("/admin/reload-index", dependencies=[Depends(require_api_key)])
    def reload_index() -> dict:
        try:
            added_documents, failed, failures = rebuild_runtime_index()
        except Exception as exc:
            logger.exception("Index reload failed")
            raise HTTPException(status_code=503, detail=f"Index reload failed: {exc}") from exc
        return {"status": "reloaded", "documents_added": added_documents, "failed": failed, "failures": failures}

    @app.post("/search", dependencies=[Depends(require_api_key)])
    def search(request: SearchRequest) -> list[dict]:
        return [
            {"chunk_id": result.chunk_id, "document_id": result.document_id, "text": result.text, "score": result.score, "metadata": result.metadata}
            for result in pipeline.retriever.search(request.question, request.top_k)
        ]

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
    def ask(request: AskRequest) -> AskResponse:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())
        if not chunks:
            raise HTTPException(status_code=503, detail="Index is empty. Run ingestion first.")
        history = sessions[request.session_id][-8:]
        result = agent.run(request.question, history)
        sessions[request.session_id].extend([
            {"role": "user", "content": request.question},
            {"role": "assistant", "content": result.answer},
        ])
        sessions[request.session_id] = sessions[request.session_id][-8:]
        return AskResponse(
            request_id=request_id,
            session_id=request.session_id,
            answer=result.answer,
            sources=[SourceResponse(**source) for source in result.citations],
            tools_used=result.tools_used,
            latency_ms=(time.perf_counter() - start) * 1000,
            status="refused" if result.refused else result.state.status,
            refused=result.refused,
        )

    @app.get("/documents", dependencies=[Depends(require_api_key)])
    def documents() -> list[dict]:
        seen = {}
        for chunk in chunks:
            seen.setdefault(chunk.document_id, {"document_id": chunk.document_id, "title": chunk.title, "source": chunk.source, "metadata": chunk.metadata})
        return list(seen.values())

    @app.post("/documents/upload", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
    async def upload_document(request: Request, file: UploadFile = UPLOAD_FILE_PARAM) -> UploadResponse:
        original_name = file.filename or "uploaded-document"
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
            audit_event("upload_rejected", request, filename=original_name, reason="unsupported_format")
            raise HTTPException(status_code=415, detail=f"Unsupported format. Allowed formats: {allowed}")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if not content:
            audit_event("upload_rejected", request, filename=original_name, reason="empty_file")
            raise HTTPException(status_code=400, detail="The uploaded document is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            audit_event("upload_rejected", request, filename=original_name, reason="file_too_large")
            raise HTTPException(status_code=413, detail="The maximum document size is 25 MB.")
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_name).stem).strip("-") or "document"
        is_image = suffix in IMAGE_UPLOAD_EXTENSIONS
        upload_dir = settings.data_dir / "raw" / "uploads" / ("images" if is_image else "documents")
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"{safe_stem}-{uuid.uuid4().hex[:10]}{suffix}"
        destination.write_bytes(content)
        try:
            object_uri = upload_original(settings, destination, content, file.content_type)
        except Exception as exc:
            audit_event("upload_failed", request, filename=original_name, path=str(destination), error=str(exc))
            raise HTTPException(status_code=503, detail=f"Object storage unavailable: {exc}") from exc
        if is_image:
            audit_event("image_uploaded", request, filename=original_name, path=str(destination), object_uri=object_uri, indexed=False)
            return UploadResponse(
                filename=original_name,
                kind="image",
                indexed=False,
                documents_added=0,
                chunks_added=0,
                total_documents=len({chunk.document_id for chunk in chunks}),
                total_chunks=len(chunks),
                message="Image uploaded and stored as a visual attachment. Document OCR is not enabled in this local profile.",
            )
        if settings.indexing_mode == "async":
            job_id = str(uuid.uuid4())
            try:
                enqueue_index_job(settings, {"job_id": job_id, "filename": original_name, "path": str(destination)})
            except Exception as exc:
                audit_event("upload_failed", request, filename=original_name, path=str(destination), error=str(exc))
                raise HTTPException(status_code=503, detail=f"Indexing queue unavailable: {exc}") from exc
            audit_event("document_queued", request, filename=original_name, path=str(destination), object_uri=object_uri, job_id=job_id)
            return UploadResponse(
                filename=original_name,
                kind="document",
                indexed=False,
                status="queued",
                job_id=job_id,
                documents_added=0,
                chunks_added=0,
                total_documents=len({chunk.document_id for chunk in chunks}),
                total_chunks=len(chunks),
                message="Document received and queued for background indexing.",
            )
        before_chunks = len(chunks)
        try:
            added_documents, _, failures = rebuild_runtime_index()
        except Exception as exc:
            logger.exception("Document indexing failed for %s", destination)
            audit_event("upload_failed", request, filename=original_name, path=str(destination), error=str(exc))
            raise HTTPException(status_code=422, detail=f"Document indexing failed: {exc}") from exc
        matching_failures = [item for item in failures if Path(item["path"]) == destination]
        if matching_failures:
            audit_event("upload_failed", request, filename=original_name, path=str(destination), error=matching_failures[0]["error"])
            raise HTTPException(status_code=422, detail=f"Document parsing failed: {matching_failures[0]['error']}")
        audit_event("document_indexed", request, filename=original_name, path=str(destination), object_uri=object_uri, chunks_added=len(chunks) - before_chunks)
        return UploadResponse(
            filename=original_name,
            kind="document",
            indexed=True,
            documents_added=added_documents,
            chunks_added=max(0, len(chunks) - before_chunks),
            total_documents=len({chunk.document_id for chunk in chunks}),
            total_chunks=len(chunks),
            message="Document indexed successfully and available in the assistant.",
        )

    @app.get("/metrics", dependencies=[Depends(require_api_key)])
    def metrics() -> dict:
        summary_path = settings.results_dir / "retrieval" / "retrieval_summary.json"
        rag_path = settings.results_dir / "rag" / "rag_summary.json"
        agent_path = settings.results_dir / "agent" / "agent_summary.json"
        return {
            "corpus": {"documents": len({chunk.document_id for chunk in chunks}), "chunks": len(chunks)},
            "retrieval": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else [],
            "rag": json.loads(rag_path.read_text(encoding="utf-8")) if rag_path.exists() else {},
            "agent": json.loads(agent_path.read_text(encoding="utf-8")) if agent_path.exists() else {},
        }

    web_dir = settings.root_dir / "frontend_web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = create_app()

# Deployment guide

## Local development

The default profile stays lightweight and uses the local hashing embedder, in-memory dense retrieval and synchronous indexing:

```powershell
& .\.venv\Scripts\Activate.ps1
& .\.venv\Scripts\python.exe -m uvicorn enterprise_rag.api.main:app --reload
```

## Production-like Docker profile

1. Copy `.env.example` to `.env`.
2. Replace `API_KEY`, `POSTGRES_PASSWORD` and `MINIO_ROOT_PASSWORD` with long random values.
3. Keep `AUTH_ENABLED=true` for a protected API.
4. Start the stack:

```powershell
docker compose up -d --build
```

The stack contains:

- `api`: FastAPI gateway and web interface;
- `worker`: Redis-backed asynchronous ingestion worker;
- `qdrant`: persistent vector database;
- `postgres`: platform database reserved for durable metadata and audit integrations;
- `minio`: S3-compatible storage for original uploads;
- `redis`: job queue with append-only persistence.

Check the services:

```powershell
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
docker compose ps
docker compose logs -f api worker
```

When authentication is enabled, API calls must include `X-API-Key`:

```powershell
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/metrics
```

Document uploads return `status=queued`; the worker ingests the corpus, rebuilds the dense artifact, refreshes the API index, and records an audit event under `data/audit/events.jsonl`.

## Before exposing it publicly

- Put the API behind a reverse proxy with HTTPS and a real domain.
- Restrict Qdrant, Redis, PostgreSQL and MinIO to the private Docker network.
- Replace the API key with OIDC/SSO and RBAC for enterprise users.
- Use managed PostgreSQL/Qdrant/object storage or configure backups for every persistent volume.
- Add an antivirus/content inspection step for untrusted uploads.
- Add monitoring and alerting for latency, queue depth, ingestion failures, refusal rate and retrieval quality.

The Docker profile is a staging foundation, not a substitute for a cloud provider’s high-availability, disaster-recovery and identity controls.

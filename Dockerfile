FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
# Keep the runtime image CPU/lightweight. The local hashing embedder is the
# default; installing sentence-transformers here would pull PyTorch and large
# NVIDIA CUDA wheels. Install the heavyweight ML extra in a separate image only
# when a transformer embedding/reranking backend is explicitly required.
RUN pip install --no-cache-dir --no-compile -e ".[dev,qdrant,agent,production]"
COPY . .

EXPOSE 8000
CMD ["uvicorn", "enterprise_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

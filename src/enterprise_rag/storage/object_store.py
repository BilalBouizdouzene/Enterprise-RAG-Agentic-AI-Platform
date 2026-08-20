"""S3-compatible storage with a local-filesystem fallback."""

from __future__ import annotations

import os
from pathlib import Path

from enterprise_rag.config import Settings


def upload_original(settings: Settings, path: Path, content: bytes, content_type: str | None = None) -> str | None:
    """Copy an uploaded original to S3/MinIO when configured.

    Local ingestion continues to use ``path`` so the development profile remains offline.
    """
    if not settings.object_storage_endpoint:
        return None
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise RuntimeError("Install the production storage dependency before enabling S3 storage") from exc
    client = boto3.client(
        "s3",
        endpoint_url=settings.object_storage_endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    try:
        client.head_bucket(Bucket=settings.object_storage_bucket)
    except Exception:
        client.create_bucket(Bucket=settings.object_storage_bucket)
    key = f"uploads/{path.name}"
    kwargs = {"Bucket": settings.object_storage_bucket, "Key": key, "Body": content}
    if content_type:
        kwargs["ContentType"] = content_type
    client.put_object(**kwargs)
    return f"s3://{settings.object_storage_bucket}/{key}"

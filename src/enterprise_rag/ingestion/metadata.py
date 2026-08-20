"""Metadata extraction for generated and user-provided documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a simple YAML front matter block without destroying document text."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) != 3 or yaml is None:
        return {}, text
    metadata = yaml.safe_load(parts[1]) or {}
    return (metadata if isinstance(metadata, dict) else {}), parts[2].lstrip("\r\n")


def default_metadata(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    result.setdefault("source", str(path))
    result.setdefault("title", path.stem.replace("_", " ").title())
    result.setdefault("document_id", path.stem)
    result.setdefault("language", "en")
    return result


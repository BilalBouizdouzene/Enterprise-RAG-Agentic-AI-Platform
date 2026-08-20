"""Read-only SQL validation and execution."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|replace|attach|detach|pragma)\b", re.IGNORECASE)


def validate_read_only_sql(sql: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.DOTALL)
    if not cleaned.strip():
        raise ValueError("SQL query cannot be empty")
    if FORBIDDEN.search(cleaned):
        raise ValueError("Only read-only SQL is allowed")
    if not re.match(r"^\s*(select|with)\b", cleaned, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed")
    if ";" in cleaned.rstrip().rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed")
    return cleaned.strip().rstrip(";")


def execute_read_only(path: str | Path, sql: str, parameters: tuple = ()) -> list[dict]:
    safe_sql = validate_read_only_sql(sql)
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(safe_sql, parameters).fetchall()]


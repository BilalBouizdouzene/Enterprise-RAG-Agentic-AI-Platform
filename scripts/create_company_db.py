"""Create the fictional structured company dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.config import load_settings
from enterprise_rag.database.queries import initialize_database

if __name__ == "__main__":
    settings = load_settings()
    path = initialize_database(settings.database_path)
    print(f"Created synthetic company database at {path}")


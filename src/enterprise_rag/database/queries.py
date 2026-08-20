"""Database initialization and compact read helpers."""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

from .models import SCHEMA


def initialize_database(path: str | Path, seed: int = 42) -> Path:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    countries = ["France", "Germany", "Spain", "United Kingdom"]
    departments = [("d-hr", "Human Resources", "France"), ("d-fin", "Finance", "France"), ("d-sec", "Security", "Germany"), ("d-eng", "Engineering", "Germany")]
    employees = []
    for index in range(1, 41):
        country = countries[(index - 1) % len(countries)]
        department = departments[(index - 1) % len(departments)][0]
        employees.append((f"e-{index:03d}", department, country, "Analyst" if index % 2 else "Engineer", f"202{index % 6}-0{index % 9 + 1}-15"))
    expenses = []
    categories = ["travel", "meals", "software", "office"]
    for index in range(1, 81):
        employee_id = f"e-{(index - 1) % 40 + 1:03d}"
        expenses.append((f"x-{index:03d}", employee_id, categories[(index - 1) % len(categories)], round(rng.uniform(25, 850), 2), "EUR", f"2025-{(index - 1) % 12 + 1:02d}-10"))
    offices = [(f"o-{i}", country, city, 10) for i, (country, city) in enumerate([("France", "Paris"), ("Germany", "Berlin"), ("Spain", "Madrid"), ("United Kingdom", "London")], start=1)]
    with sqlite3.connect(database_path) as connection:
        connection.executescript(SCHEMA)
        connection.executemany("INSERT OR REPLACE INTO departments VALUES (?, ?, ?)", departments)
        connection.executemany("INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?)", employees)
        connection.executemany("INSERT OR REPLACE INTO expenses VALUES (?, ?, ?, ?, ?, ?)", expenses)
        connection.executemany("INSERT OR REPLACE INTO offices VALUES (?, ?, ?, ?)", offices)
        connection.commit()
    return database_path


def query_rows(path: str | Path, sql: str, parameters: tuple = ()) -> list[dict]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


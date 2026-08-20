"""SQLite schema for the fictional enterprise dataset."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS departments (
    department_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    department_id TEXT NOT NULL REFERENCES departments(department_id),
    country TEXT NOT NULL,
    position TEXT NOT NULL,
    hire_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS expenses (
    expense_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    currency TEXT NOT NULL,
    expense_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS offices (
    office_id TEXT PRIMARY KEY,
    country TEXT NOT NULL,
    city TEXT NOT NULL,
    employees_count INTEGER NOT NULL
);
"""


"""Validated agent tools."""

from __future__ import annotations

import ast
import operator
import sqlite3
from dataclasses import dataclass
from typing import Any

from enterprise_rag.database.safety import execute_read_only
from enterprise_rag.rag.pipeline import RAGPipeline


@dataclass(frozen=True)
class ToolResult:
    tool: str
    data: Any
    error: str | None = None


def document_search(pipeline: RAGPipeline, question: str) -> ToolResult:
    response = pipeline.ask(question)
    return ToolResult("document_search", response)


def sql_query(database_path: str, query: str) -> ToolResult:
    try:
        return ToolResult("sql_query", execute_read_only(database_path, query))
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        return ToolResult("sql_query", [], str(exc))


def metadata_search(chunks, department: str | None = None, category: str | None = None) -> ToolResult:
    matches = [chunk.as_payload() for chunk in chunks if (not department or chunk.metadata.get("department") == department) and (not category or chunk.metadata.get("category") == category)]
    return ToolResult("metadata_search", matches)


ALLOWED_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("exponent too large")
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("only arithmetic expressions are allowed")


def calculator(expression: str) -> ToolResult:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _safe_eval(tree)
        return ToolResult("calculator", value)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return ToolResult("calculator", None, str(exc))

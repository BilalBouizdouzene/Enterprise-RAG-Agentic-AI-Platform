"""Transparent query routing heuristics used before optional LLM planning."""

from __future__ import annotations

import re


class QueryRouter:
    def route(self, question: str) -> list[str]:
        lowered = question.lower()
        sql_signal = bool(re.search(r"\b(how many|number of|percentage|percent|average|count|employees|expenses|spend|based in|combien|pourcentage|moyenne|employés|dépenses|basés)\b", lowered))
        calc_signal = bool(re.search(r"\b(percent|percentage|ratio|calculate|what fraction|pourcentage|ratio|calculer|proportion)\b", lowered))
        document_signal = bool(re.search(r"\b(policy|procedure|standard|limit|password|laptop|retention|remote|reimbursement|vendor|leave|politique|procédure|norme|limite|mot de passe|portable|rétention|télétravail|remboursement|fournisseur|congé)\b", lowered))
        tools = []
        if document_signal:
            tools.append("document_search")
        if sql_signal:
            tools.append("sql_query")
        if calc_signal:
            tools.append("calculator")
        return tools or ["document_search"]

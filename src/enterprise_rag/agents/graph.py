"""Agent graph with a deterministic fallback and optional LangGraph integration."""

from __future__ import annotations

import re
from dataclasses import dataclass

from enterprise_rag.agents.router import QueryRouter
from enterprise_rag.agents.state import AgentState
from enterprise_rag.agents.tools import calculator, document_search, sql_query
from enterprise_rag.rag.pipeline import RAGPipeline


@dataclass
class AgentResponse:
    answer: str
    tools_used: list[str]
    citations: list[dict]
    state: AgentState
    refused: bool = False


class EnterpriseAgent:
    def __init__(self, rag_pipeline: RAGPipeline, database_path: str, max_retries: int = 2):
        self.rag_pipeline = rag_pipeline
        self.database_path = database_path
        self.max_retries = max_retries
        self.router = QueryRouter()

    def run(self, question: str, history: list[dict[str, str]] | None = None) -> AgentResponse:
        bounded_history = (history or [])[-8:]
        state = AgentState(question, bounded_history)
        effective_question = self._contextualize_question(question, bounded_history)
        tools = self.router.route(effective_question)
        state.selected_tools = tools
        answer_parts = []
        citations = []
        refused = False
        if "document_search" in tools:
            response = None
            for attempt in range(self.max_retries + 1):
                query = effective_question if attempt == 0 else self._rewrite_question(effective_question)
                result = document_search(self.rag_pipeline, query)
                response = result.data
                if not response.refused or attempt == self.max_retries:
                    break
                state.retry_count += 1
            assert response is not None
            refused = response.refused
            answer_parts.append(response.answer)
            citations.extend(source.as_dict() for source in response.sources)
            state.retrieved_documents = [item.metadata for item in response.retrieved]
            state.intermediate_results.append({"tool": "document_search", "refused": response.refused, "retries": state.retry_count})
        if "sql_query" in tools:
            query = self._query_for_question(effective_question)
            result = sql_query(self.database_path, query)
            state.sql_results = result.data if isinstance(result.data, list) else []
            state.intermediate_results.append({"tool": result.tool, "error": result.error, "rows": len(state.sql_results)})
            if result.error:
                answer_parts.append(f"La requête structurée a échoué : {result.error}")
            else:
                answer_parts.append(f"Résultat des données structurées : {state.sql_results}")
        if "calculator" in tools:
            expression = self._expression_for_question(effective_question, state.sql_results)
            result = calculator(expression)
            state.intermediate_results.append({"tool": result.tool, "error": result.error})
            answer_parts.append(f"Calcul déterministe : {result.data}" if result.error is None else f"Calcul impossible : {result.error}")
        state.answer = "\n\n".join(answer_parts)
        state.citations = citations
        state.status = "success"
        return AgentResponse(state.answer, tools, citations, state, refused)

    @staticmethod
    def _query_for_question(question: str) -> str:
        lowered = question.lower()
        if "expense" in lowered or "dépense" in lowered or "spend" in lowered:
            return "SELECT category, ROUND(AVG(amount), 2) AS average_amount, ROUND(SUM(amount), 2) AS total_amount FROM expenses GROUP BY category ORDER BY total_amount DESC"
        country_match = re.search(r"\b(france|germany|allemagne)\b", lowered)
        if country_match:
            country = "Germany" if country_match.group(1) in {"germany", "allemagne"} else "France"
            return f"SELECT COUNT(*) AS employee_count FROM employees WHERE country = '{country}'"
        return "SELECT COUNT(*) AS employee_count FROM employees"

    @staticmethod
    def _expression_for_question(question: str, rows: list[dict]) -> str:
        if rows and "employee_count" in rows[0]:
            return f"{rows[0]['employee_count']} / 40 * 100"
        return "0"

    @staticmethod
    def _rewrite_question(question: str) -> str:
        """Make one controlled retry more retrieval-oriented without inventing facts."""
        return f"{question} policy procedure standard requirement"

    @staticmethod
    def _contextualize_question(question: str, history: list[dict[str, str]]) -> str:
        """Attach the most recent user turn for short follow-up queries."""
        if not history or len(question.split()) > 5:
            return question
        previous_user_turns = [item.get("content", "") for item in history if item.get("role") == "user"]
        return f"{previous_user_turns[-1]} {question}" if previous_user_turns else question


def build_langgraph_graph(agent: EnterpriseAgent):
    """Build the optional LangGraph representation around the tested business nodes.

    The package remains optional for offline retrieval work. The deterministic routing,
    SQL safety and tool implementations stay in this repository rather than being hidden
    in a framework chain.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the optional 'agent' extra for LangGraph") from exc
    from typing import TypedDict

    from .nodes import execute_node, route_node

    class GraphState(TypedDict, total=False):
        user_query: str
        conversation_history: list[dict[str, str]]
        selected_tools: list[str]
        answer: str
        citations: list[dict]
        tools_used: list[str]
        status: str

    builder = StateGraph(GraphState)
    builder.add_node("route", lambda state: route_node(agent, state))
    builder.add_node("execute_tools", lambda state: execute_node(agent, state))
    builder.add_edge(START, "route")
    builder.add_edge("route", "execute_tools")
    builder.add_edge("execute_tools", END)
    return builder.compile()

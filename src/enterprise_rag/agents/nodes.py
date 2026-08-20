"""Small graph nodes kept separate from the orchestration adapter."""

from __future__ import annotations

from typing import Any

from .graph import EnterpriseAgent


def route_node(agent: EnterpriseAgent, state: dict[str, Any]) -> dict[str, Any]:
    return {"selected_tools": agent.router.route(state["user_query"])}


def execute_node(agent: EnterpriseAgent, state: dict[str, Any]) -> dict[str, Any]:
    response = agent.run(state["user_query"], state.get("conversation_history", []))
    return {"answer": response.answer, "tools_used": response.tools_used, "citations": response.citations, "status": response.state.status}


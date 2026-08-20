"""Streamlit demo UI for the local API."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import streamlit as st

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🔎", layout="wide")
st.title("Enterprise Knowledge Assistant")
st.caption("Hybrid retrieval · evidence-aware answers · citations · structured tools")

api_url = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000"))
show_debug = st.sidebar.checkbox("Afficher les détails techniques", False)

question = st.chat_input("Posez une question sur les politiques de l’entreprise…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            response = httpx.post(f"{api_url}/ask", json={"question": question, "session_id": "streamlit"}, timeout=90)
            response.raise_for_status()
            payload = response.json()
            st.write(payload["answer"])
            if payload["sources"]:
                st.subheader("Sources")
                for source in payload["sources"]:
                    label = f"{source['label']} · {source['title']}" + (f" · page {source['page']}" if source.get("page") else "")
                    with st.expander(label):
                        st.caption(source["document_id"])
                        st.write(source["excerpt"])
            if show_debug:
                st.json({"tools_used": payload["tools_used"], "latency_ms": payload["latency_ms"], "status": payload["status"]})
        except Exception as exc:
            st.error(f"API indisponible : {exc}")

st.divider()
st.subheader("Évaluation")
summary_path = Path("results/retrieval/retrieval_summary.json")
if summary_path.exists():
    st.dataframe(json.loads(summary_path.read_text(encoding="utf-8")), use_container_width=True)
else:
    st.info("Lancez scripts/evaluate_retrieval.py pour afficher les résultats réels.")

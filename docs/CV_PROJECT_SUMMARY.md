# Enterprise RAG & Agentic AI Platform

## Description

Plateforme locale d’assistant de connaissances combinant ingestion documentaire, retrieval hybride, génération avec citations, outils SQL en lecture seule et routage agentique.

## Bullet points CV

- Conçu une pipeline d’ingestion multi-format sur 636 documents et 1 908 chunks, puis comparé BM25, retrieval dense hashing et fusion RRF sur 52 questions answerable, avec Recall@5 mesuré à 0.933 pour BM25 et 0.731 pour le pipeline hybride.
- Implémenté validation des preuves, refus traçable, citations par source, SQLite read-only, calculateur AST sécurisé et évaluation agentique avec 1.000 de précision de routage sur le jeu de démonstration.
- Exposé le système via FastAPI et Streamlit, ajouté Docker Compose, tests pytest, Ruff, GitHub Actions, artefacts CSV/JSON et rapports d’analyse d’erreurs.

## Technologies

Python, NumPy, pandas, scikit-learn, BM25, SentenceTransformers adapter, Qdrant adapter, LangGraph adapter, FastAPI, Streamlit, SQLite, MLflow adapter, Docker, pytest, Ruff.

## Métriques actuelles

Les résultats sont ceux du corpus synthétique local : BM25 Recall@5 0.933, Hybrid Recall@5 0.731, MRR Hybrid 0.679, taux de refus offline RAG 0.133, routing accuracy 1.000. Le reranker par défaut dans cette exécution est lexical fallback, pas un CrossEncoder téléchargé.

## Défis techniques

Dédupliquer les chunks avant les métriques document-level, rester exécutable sans GPU ni clé API, protéger les requêtes SQL, empêcher les preuves hors sujet d’autoriser une réponse, et garder la logique métier lisible malgré l’option LangGraph.

## Points d’entretien

Présenter les résultats comme des mesures sur un corpus fictif, expliquer pourquoi BM25 gagne sur ce vocabulaire, distinguer Recall@K de génération, montrer la trajectoire vers un CrossEncoder et discuter les compromis latence/mémoire.

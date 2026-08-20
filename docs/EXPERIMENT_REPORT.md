# Rapport d’expérimentation

## Problème

Répondre à des questions internes avec une preuve traçable, tout en distinguant documents, données structurées et calculs. L’objectif est de mesurer séparément retrieval, génération et routage agentique.

## Dataset

Le corpus courant contient 640 documents Markdown couvrant HR, Finance, Security, Legal, IT et Operations, dont 600 documents générés pour l’expérience de volumétrie et des documents d’upload locaux. Les documents utilisent un front matter YAML avec `document_id`, titre, département, catégorie, version, date d’effet, langue et source. Le pipeline produit actuellement 1 936 chunks. Le jeu de retrieval contient 70 questions : 62 answerable et 8 `unanswerable`, avec les catégories `lexical_exact`, `semantic_paraphrase`, `entity_or_acronym`, `multi_document`, `metadata` et `unanswerable`. Le jeu agent contient 5 routes attendues.

## Protocole

Les scores de retrieval sont calculés au niveau document après déduplication des chunks, avec Recall@1/3/5/10, MRR, nDCG@5, HitRate@5 et latence. Les expériences sont exécutées sur CPU et les valeurs sont écrites dans `results/retrieval/` par les scripts, sans saisie manuelle dans les artefacts machine-readable.

## Baselines retrieval

Exécution actuelle :

| Pipeline | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | HitRate@5 | Latence ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.714 | 0.866 | 0.895 | 0.926 | 0.881 | 0.864 | 0.952 | 6.632 |
| Dense hashing | 0.198 | 0.412 | 0.474 | 0.518 | 0.387 | 0.374 | 0.548 | 0.328 |
| Hybrid RRF (hashing) | 0.410 | 0.632 | 0.782 | 0.854 | 0.642 | 0.646 | 0.839 | 7.545 |
| Hybrid + lexical fallback | 0.427 | 0.668 | 0.699 | 0.699 | 0.610 | 0.619 | 0.758 | 7.982 |

Sur ce petit corpus lexical, BM25 a le meilleur Recall@5. Le reranker fallback augmente le MRR par rapport à Hybrid mais réduit le Recall@5 ; aucune généralisation à un corpus réel ne doit être déduite de cet échantillon.

## Chunking

| Configuration | Chunks | Recall@5 | MRR | Latence ms |
|---|---:|---:|---:|---:|
| fixed small (450/50) | 1293 | 0.256 | 0.310 | 1.219 |
| fixed medium (900/120) | 664 | 0.265 | 0.302 | 0.496 |
| fixed large (1400/180) | 656 | 0.265 | 0.282 | 0.522 |
| section-aware (900/120) | 1936 | 0.474 | 0.383 | 1.056 |

Sur le corpus courant, fixed medium et fixed large obtiennent le meilleur Recall@5 parmi les essais de chunking ; section-aware conserve davantage de chunks et une latence plus élevée. Les tailles sont exprimées en caractères dans cette implémentation.

## Embeddings et reranking

Le backend principal configuré cible `BAAI/bge-small-en-v1.5`, avec `intfloat/e5-small-v2` et `intfloat/multilingual-e5-small` comme alternatives ; le hashing reste le baseline/fallback offline. Sur 62 questions answerable, Dense BGE obtient Recall@5 0.912, MRR 0.875 et nDCG@5 0.843 ; BM25 + BGE RRF obtient respectivement 0.944, 0.928 et 0.904. Le CrossEncoder anglais obtient Recall@5 0.929, MRR 0.901 et nDCG@5 0.883 ; le CrossEncoder multilingue obtient 0.929, 0.976 et 0.933. Les résultats détaillés sont conservés dans `results/retrieval/final_benchmark.json`.

## RAG

La pipeline utilise BM25, dense hashing ou Hybrid RRF, valide l’overlap lexical de contenu après suppression des stopwords, refuse les preuves insuffisantes et produit des citations dédupliquées par document. Le provider offline assemble les preuves, tandis que les providers OpenAI-compatible et Ollama utilisent le même contrat `LLMProvider`. Les métriques RAG sont des proxies déterministes du dataset, pas des annotations humaines.

Résultats actuels sur 70 questions avec le provider template : BM25 RAG obtient correctness 0.786, groundedness 0.900, citation accuracy 0.524, citation recall 0.827, correct refusal 0.900, false refusal 0.043 et latence 7.287 ms ; Hybrid + CrossEncoder RAG obtient respectivement 0.786, 0.914, 0.444, 0.803, 0.914, 0.029 et 376.246 ms ; Hybrid + Multilingual CrossEncoder RAG obtient 0.800, 0.943, 0.502, 0.858, 0.943, 0.000 et 1 088.036 ms. Ces mesures ne remplacent pas une évaluation humaine complète de factualité.

## Agent

Le routeur sélectionne document search, SQL, calculator et leurs combinaisons. La base SQLite est protégée en lecture seule et les expressions arithmétiques passent par un AST limité. Sur les 5 questions agentiques, la précision de routage actuelle est 1.000, avec 1.2 appels outil moyens et 0 retry.

## Analyse d’erreurs

Voir `results/reports/error_analysis.md`. Le rapport est généré à partir des rankings de l’exécution. Les exemples de refus ont conduit à renforcer le validateur afin d’ignorer les mots-outils et d’éviter qu’une preuve Paris soit considérée suffisante pour une question Madrid.

## Trade-offs et limites

La recherche dense hashing est rapide mais moins sémantique qu’un encodeur entraîné ; BM25 profite du vocabulaire partagé du corpus synthétique. BGE et les CrossEncoder améliorent le rang sur ce corpus, au prix d’une latence CPU nettement supérieure. Le fallback reranker est explicable mais ne doit pas être présenté comme un CrossEncoder. Le provider offline ne permet pas de conclure sur la qualité d’un LLM externe ; une annotation humaine reste nécessaire pour mesurer la factualité.

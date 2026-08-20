# Rapport d’expérimentation

## Problème

Répondre à des questions internes avec une preuve traçable, tout en distinguant documents, données structurées et calculs. L’objectif est de mesurer séparément retrieval, génération et routage agentique.

## Dataset

Le corpus synthétique contient 636 documents Markdown fictifs couvrant HR, Finance, Security, Legal, IT et Operations, dont 600 documents générés pour l’expérience de volumétrie. Les documents utilisent un front matter YAML avec `document_id`, titre, département, catégorie, version, date d’effet, langue et source. Le pipeline produit actuellement 1 908 chunks. Le jeu de retrieval contient 60 questions générées à partir de 15 intentions, avec faits exacts, paraphrases, multi-documents, métadonnées et questions non answerable. Le jeu agent contient 5 routes attendues.

## Protocole

Les scores de retrieval sont calculés au niveau document après déduplication des chunks, avec Recall@K, MRR, nDCG@5 et latence mesurée autour de la recherche. Les expériences sont CPU et utilisent un embedder hashing de dimension 384. Les valeurs sont écrites dans `results/retrieval/` par les scripts et ne sont pas saisies manuellement dans les artefacts machine-readable.

## Baselines retrieval

Exécution actuelle :

| Pipeline | Recall@5 | MRR | nDCG@5 | Latence ms |
|---|---:|---:|---:|---:|
| BM25 | 0.933 | 0.917 | 0.903 | 8.406 |
| Dense hashing | 0.507 | 0.451 | 0.424 | 0.470 |
| Hybrid RRF | 0.731 | 0.679 | 0.653 | 11.139 |
| Hybrid + lexical fallback reranker | 0.782 | 0.798 | 0.771 | 10.563 |

Sur ce petit corpus lexical, BM25 a le meilleur Recall@5. Le reranker fallback augmente le MRR par rapport à Hybrid mais réduit le Recall@5 ; aucune généralisation à un corpus réel ne doit être déduite de cet échantillon.

## Chunking

| Configuration | Chunks | Recall@5 | MRR | Latence ms |
|---|---:|---:|---:|---:|
| fixed small (450/50) | 38 | 0.614 | 0.518 | 0.039 |
| fixed medium (900/120) | 636 | 0.290 | 0.338 | 0.525 |
| fixed large (1400/180) | 636 | 0.290 | 0.338 | 0.505 |
| section-aware (900/120) | 1908 | 0.507 | 0.433 | 1.320 |

La configuration fixed small donne le meilleur Recall@5 sur le corpus enrichi, tandis que fixed medium conserve une empreinte plus compacte. Les tailles sont exprimées en caractères dans cette première implémentation.

## Embeddings et reranking

L’embedder hashing est un baseline déterministe qui permet de tester l’architecture hors ligne. L’adaptateur SentenceTransformers et le CrossEncoder sont configurables mais non téléchargés dans l’exécution initiale, car leurs modèles peuvent être disproportionnés pour un PC de 8 Go. Une comparaison scientifique de modèles sera ajoutée lorsque les modèles seront disponibles localement.

## RAG

La pipeline utilise Hybrid RRF, valide l’overlap lexical de contenu après suppression des stopwords, refuse les preuves insuffisantes et produit des citations dédupliquées par document. Le provider offline assemble les preuves, tandis que les providers OpenAI-compatible et Ollama utilisent le même contrat `LLMProvider`.

Résultats offline actuels : 60 questions, taux de signal attendu 0.717, support contextuel 0.917 et taux de refus 0.133. Ces mesures sont des indicateurs de smoke evaluation, pas une évaluation humaine complète de factualité.

## Agent

Le routeur sélectionne document search, SQL, calculator et leurs combinaisons. La base SQLite est protégée en lecture seule et les expressions arithmétiques passent par un AST limité. Sur les 5 questions agentiques, la précision de routage actuelle est 1.000, avec 1.2 appels outil moyens et 0 retry.

## Analyse d’erreurs

Voir `results/reports/error_analysis.md`. Le rapport est généré à partir des rankings de l’exécution. Les exemples de refus ont conduit à renforcer le validateur afin d’ignorer les mots-outils et d’éviter qu’une preuve Paris soit considérée suffisante pour une question Madrid.

## Trade-offs et limites

La recherche dense hashing est rapide mais moins sémantique qu’un encodeur entraîné. BM25 profite du vocabulaire partagé du corpus synthétique. Le fallback reranker est explicable mais ne doit pas être présenté comme un CrossEncoder. Le provider offline ne permet pas de conclure sur la qualité d’un LLM externe. La prochaine itération devra ajouter une annotation humaine, un modèle multilingue et une mesure de latence sur un corpus plus volumineux.

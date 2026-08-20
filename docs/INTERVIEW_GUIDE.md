# Guide d’entretien

## Pourquoi RAG plutôt que fine-tuning ?

Les politiques changent et doivent rester traçables. RAG permet de mettre à jour l’index et de citer les sources sans réentraîner le modèle. Le fine-tuning reste pertinent pour un comportement ou un format stable, pas comme stockage principal de faits internes.

## Pourquoi BM25 et le dense ?

BM25 récupère très bien les termes exacts, références et nombres. Les embeddings aident pour les paraphrases. La fusion RRF combine leurs forces sans supposer que les scores sont calibrés sur la même échelle.

## Pourquoi un reranker ?

Le premier retrieval privilégie le rappel avec un pool de candidats. Un CrossEncoder peut ensuite comparer question et passage conjointement pour améliorer le rang final. Ici l’adaptateur existe, mais l’exécution locale utilise explicitement un fallback lexical pour rester légère.

## Recall@K, MRR et nDCG ?

Recall@K mesure la proportion de documents pertinents retrouvés dans les K premiers. MRR récompense le rang du premier document pertinent. nDCG tient compte de la position avec une décroissance logarithmique et convient aux rankings.

## Pourquoi évaluer retrieval séparément de la génération ?

Une réponse fausse peut venir d’un passage absent ou d’un LLM qui raisonne mal sur une preuve présente. Les métriques retrieval isolent le premier problème ; l’évaluation RAG ajoute support, refus et exactitude attendue.

## Comment traiter l’insuffisance de preuve ?

Le validateur retire les stopwords, mesure le recouvrement entre termes informatifs de la question et des passages, puis refuse si l’overlap est trop faible. En production, j’ajouterais un classifieur de suffisance, une calibration et une revue humaine sur les cas limites.

## Pourquoi LangGraph ?

Le routage peut appeler plusieurs outils avec état, retries bornés et mémoire. La logique métier reste dans le routeur et les outils ; l’adaptateur LangGraph représente le workflow sans cacher la sécurité SQL.

## Comment éviter les boucles agentiques ?

L’état conserve un compteur `retry_count`, le nombre maximal est configurable à 2 et chaque outil renvoie une sortie structurée. Les outils sont déterministes lorsque c’est possible.

## Comment protéger SQL ?

Les requêtes doivent commencer par SELECT ou WITH, les opérations d’écriture sont rejetées, les commentaires et multi-statements sont contrôlés, puis SQLite est ouvert avec `mode=ro` et `PRAGMA query_only=ON`. En production, j’ajouterais un compte DB sans privilèges et une validation AST SQL.

## Comment réduire latence et coût ?

Limiter les candidats, mettre en cache les embeddings, batcher l’indexation, utiliser BM25 pour les requêtes exactes, désactiver le reranker lorsque nécessaire et réserver un LLM plus coûteux aux cas complexes. Les résultats actuels montrent déjà le coût mesurable de la fusion et du reranking.


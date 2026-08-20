# Retrieval error analysis

Generated only from measured rankings; a missing rank is shown as `-`.

## BM25 succeeds and Dense fails

- `q051` (multi_document): According to the policies, compare laptop incident reporting with vendor incident reporting.; expected `security_laptop;legal_vendors`; BM25 rank `2`; Dense rank `7`; Hybrid rank `1`; reranked rank `3`.

## Dense succeeds and BM25 fails

- `q062` (semantic_paraphrase): During a trip to Germany's capital, what accommodation ceiling applies?; expected `finance_travel`; BM25 rank `-`; Dense rank `1`; Hybrid rank `5`; reranked rank `1`.
- `q065` (semantic_paraphrase): After a corporate notebook goes missing, what is the reporting deadline?; expected `security_laptop`; BM25 rank `-`; Dense rank `1`; Hybrid rank `1`; reranked rank `-`.
- `q066` (semantic_paraphrase): How long does a critical supplier have to disclose a material incident?; expected `legal_vendors`; BM25 rank `8`; Dense rank `3`; Hybrid rank `5`; reranked rank `1`.

## Hybrid corrects an error

- `q062` (semantic_paraphrase): During a trip to Germany's capital, what accommodation ceiling applies?; expected `finance_travel`; BM25 rank `-`; Dense rank `1`; Hybrid rank `5`; reranked rank `1`.
- `q065` (semantic_paraphrase): After a corporate notebook goes missing, what is the reporting deadline?; expected `security_laptop`; BM25 rank `-`; Dense rank `1`; Hybrid rank `1`; reranked rank `-`.
- `q066` (semantic_paraphrase): How long does a critical supplier have to disclose a material incident?; expected `legal_vendors`; BM25 rank `8`; Dense rank `3`; Hybrid rank `5`; reranked rank `1`.

## CrossEncoder improves the rank

- `q032` (lexical_exact): Summarize this requirement: who approves contractual vendor terms?; expected `legal_vendors`; BM25 rank `1`; Dense rank `5`; Hybrid rank `2`; reranked rank `1`.
- `q037` (lexical_exact): Please explain: what happens to access after termination?; expected `security_passwords`; BM25 rank `1`; Dense rank `5`; Hybrid rank `2`; reranked rank `1`.
- `q040` (lexical_exact): Summarize this requirement: what happens to access after termination?; expected `security_passwords`; BM25 rank `1`; Dense rank `4`; Hybrid rank `3`; reranked rank `1`.
- `q062` (semantic_paraphrase): During a trip to Germany's capital, what accommodation ceiling applies?; expected `finance_travel`; BM25 rank `-`; Dense rank `1`; Hybrid rank `5`; reranked rank `1`.
- `q066` (semantic_paraphrase): How long does a critical supplier have to disclose a material incident?; expected `legal_vendors`; BM25 rank `8`; Dense rank `3`; Hybrid rank `5`; reranked rank `1`.

## CrossEncoder degrades the result

- `q005` (semantic_paraphrase): Please explain: how much can a hotel in Berlin cost under the travel standard?; expected `finance_travel`; BM25 rank `1`; Dense rank `1`; Hybrid rank `1`; reranked rank `2`.
- `q006` (semantic_paraphrase): In simple terms, how much can a hotel in Berlin cost under the travel standard?; expected `finance_travel`; BM25 rank `1`; Dense rank `1`; Hybrid rank `1`; reranked rank `2`.
- `q007` (semantic_paraphrase): According to the policies, how much can a hotel in Berlin cost under the travel standard?; expected `finance_travel`; BM25 rank `1`; Dense rank `1`; Hybrid rank `1`; reranked rank `2`.
- `q008` (semantic_paraphrase): Summarize this requirement: how much can a hotel in Berlin cost under the travel standard?; expected `finance_travel`; BM25 rank `1`; Dense rank `1`; Hybrid rank `1`; reranked rank `2`.
- `q017` (lexical_exact): Please explain: what should happen after a company laptop is lost?; expected `security_laptop`; BM25 rank `1`; Dense rank `1`; Hybrid rank `1`; reranked rank `2`.

## All available retrievers fail

- No measured example in this run.

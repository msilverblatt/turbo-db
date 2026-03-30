# Changelog

## 0.2.0 (unreleased)

Hybrid search combining semantic embeddings with BM25 keyword matching.

- Add `hybrid_query()` method to `Collection` with three fusion strategies:
  - **RRF** (Reciprocal Rank Fusion) — score-agnostic, zero-config default
  - **Weighted** — convex combination with min-max normalization (tunable alpha)
  - **DBSF** (Distribution-Based Score Fusion) — robust to score outliers
- Add optional `documents` parameter to `add()` and `upsert()` for storing document text
- Native BM25 inverted index — no new dependencies
- `QueryResult` now includes an optional `document` field
- `default_tokenizer` exported for building custom tokenizers

## 0.1.0

Initial release.

"""BM25 keyword search index with inverted index and JSON persistence."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = ["BM25Index", "default_tokenizer"]


def default_tokenizer(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters, dropping tokens < 2 chars."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 2]


class BM25Index:
    """In-memory BM25 index backed by a JSON cache on disk.

    The index is the authoritative derived cache of document text stored in
    SQLite.  If the cache is missing or stale it can be rebuilt from the
    metadata store via :meth:`rebuild`.
    """

    def __init__(
        self,
        path: Path,
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self._path = path
        self._tokenizer = tokenizer or default_tokenizer
        self._k1 = k1
        self._b = b

        # token -> {position: term_frequency}
        self._inverted_index: dict[str, dict[int, int]] = {}
        # position -> document length (in tokens)
        self._doc_lengths: dict[int, int] = {}
        self._total_length: int = 0

        self._load_cache()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @property
    def avg_doc_length(self) -> float:
        if self.num_docs == 0:
            return 0.0
        return self._total_length / self.num_docs

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, positions: list[int], documents: list[str]) -> None:
        """Index *documents* at the given vector *positions*."""
        for pos, doc in zip(positions, documents, strict=True):
            tokens = self._tokenizer(doc)
            self._doc_lengths[pos] = len(tokens)
            self._total_length += len(tokens)

            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            for token, count in tf.items():
                if token not in self._inverted_index:
                    self._inverted_index[token] = {}
                self._inverted_index[token][pos] = count

        self._save_cache()

    def remove(self, positions: list[int]) -> None:
        """Remove documents at *positions* from the index."""
        pos_set = set(positions)
        for pos in pos_set:
            if pos not in self._doc_lengths:
                continue
            self._total_length -= self._doc_lengths[pos]
            del self._doc_lengths[pos]

        empty_tokens: list[str] = []
        for token, postings in self._inverted_index.items():
            for pos in pos_set:
                postings.pop(pos, None)
            if not postings:
                empty_tokens.append(token)
        for token in empty_tokens:
            del self._inverted_index[token]

        self._save_cache()

    def rebuild(self, positions: list[int], documents: list[str]) -> None:
        """Rebuild the entire index from scratch."""
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._total_length = 0
        if positions:
            # bypass save inside add; we save once at the end
            for pos, doc in zip(positions, documents, strict=True):
                tokens = self._tokenizer(doc)
                self._doc_lengths[pos] = len(tokens)
                self._total_length += len(tokens)
                tf: dict[str, int] = {}
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1
                for token, count in tf.items():
                    if token not in self._inverted_index:
                        self._inverted_index[token] = {}
                    self._inverted_index[token][pos] = count
        self._save_cache()

    def remap_positions(self, position_map: dict[int, int]) -> None:
        """Remap positions during compaction."""
        new_lengths: dict[int, int] = {}
        for old_pos, new_pos in position_map.items():
            if old_pos in self._doc_lengths:
                new_lengths[new_pos] = self._doc_lengths[old_pos]
        self._doc_lengths = new_lengths

        for token in self._inverted_index:
            old_postings = self._inverted_index[token]
            new_postings: dict[int, int] = {}
            for old_pos, tf in old_postings.items():
                if old_pos in position_map:
                    new_postings[position_map[old_pos]] = tf
            self._inverted_index[token] = new_postings

        # Recompute total_length from remapped doc_lengths
        self._total_length = sum(self._doc_lengths.values())
        self._save_cache()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        valid_positions: set[int] | None = None,
    ) -> dict[int, float]:
        """Score documents against *query* using Okapi BM25.

        Returns ``{position: bm25_score}`` for all matching documents.
        If *valid_positions* is given, only those positions are scored.
        """
        tokens = self._tokenizer(query)
        if not tokens or self.num_docs == 0:
            return {}

        scores: dict[int, float] = {}
        avgdl = self.avg_doc_length
        n = self.num_docs

        for token in tokens:
            postings = self._inverted_index.get(token)
            if postings is None:
                continue

            df = len(postings)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for pos, tf in postings.items():
                if valid_positions is not None and pos not in valid_positions:
                    continue
                dl = self._doc_lengths[pos]
                if avgdl > 0:
                    denom = tf + self._k1 * (1.0 - self._b + self._b * dl / avgdl)
                else:
                    denom = tf + self._k1
                score = idf * (tf * (self._k1 + 1.0)) / denom
                scores[pos] = scores.get(pos, 0.0) + score

        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self._path / "bm25_index.json"

    def _load_cache(self) -> None:
        cache = self._cache_path()
        if not cache.exists():
            return
        try:
            raw = json.loads(cache.read_text())
            self._inverted_index = {
                token: {int(pos): tf for pos, tf in postings.items()}
                for token, postings in raw["inverted_index"].items()
            }
            self._doc_lengths = {int(pos): length for pos, length in raw["doc_lengths"].items()}
            self._total_length = raw["total_length"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self._inverted_index.clear()
            self._doc_lengths.clear()
            self._total_length = 0

    def _save_cache(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        data = {
            "inverted_index": {
                token: {str(pos): tf for pos, tf in postings.items()}
                for token, postings in self._inverted_index.items()
            },
            "doc_lengths": {str(pos): length for pos, length in self._doc_lengths.items()},
            "total_length": self._total_length,
        }
        self._cache_path().write_text(json.dumps(data, separators=(",", ":")))

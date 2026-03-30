"""Collection: add, upsert, query, delete, get, compact."""

from __future__ import annotations

import secrets
import shutil
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from turboquant import CompressedVectors, TurboQuant

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

from turbodb.bm25 import BM25Index
from turbodb.exceptions import DimensionMismatchError
from turbodb.filters import compile_filter
from turbodb.fusion import fuse_dbsf, fuse_rrf, fuse_weighted
from turbodb.locking import FileLock
from turbodb.metadata import MetadataStore
from turbodb.results import QueryResult, to_chroma_format

__all__ = ["Collection"]


class Collection:
    """A named collection of quantized vectors with metadata."""

    def __init__(self, path: Path, name: str) -> None:
        self._path = path
        self._name = name
        self._rw_lock = threading.RLock()
        self._meta = MetadataStore(path / "metadata.db")
        config = self._meta.get_config()
        self._dim: int = config["dim"]
        self._bit_width: int = config["bit_width"]
        self._metric: str = config["metric"]
        self._seed: int = config["seed"]
        self._quantizer = TurboQuant(
            dim=self._dim,
            bit_width=self._bit_width,
            mode="inner_product",
            seed=self._seed,
        )
        self._vectors: CompressedVectors | None = None
        self._load_vectors()
        self._bm25 = BM25Index(path)

    @classmethod
    def create(
        cls,
        path: Path,
        name: str,
        dim: int,
        bit_width: int = 2,
        metric: str = "cosine",
    ) -> Collection:
        """Create a new collection on disk."""
        path.mkdir(parents=True, exist_ok=True)
        seed = secrets.randbelow(2**31)
        store = MetadataStore(path / "metadata.db")
        store.set_config({
            "dim": dim,
            "bit_width": bit_width,
            "metric": metric,
            "seed": seed,
        })
        store.close()
        return cls(path, name)

    @classmethod
    def open(cls, path: Path, name: str) -> Collection:
        """Open an existing collection."""
        return cls(path, name)

    def _load_vectors(self) -> None:
        """Load compressed vectors from disk if they exist."""
        vectors_dir = self._path / "vectors"
        if (vectors_dir / "meta.json").exists():
            self._vectors = CompressedVectors.load(vectors_dir)
            self._recover_if_needed()

    def _recover_if_needed(self) -> None:
        """Trim orphaned vectors after a crash."""
        if self._vectors is None:
            return
        expected = self._meta.max_position() + 1
        if expected <= 0:
            self._vectors = None
            return
        if self._vectors.num_vectors > expected:
            self._vectors = self._vectors[:expected]
            self._vectors.save(self._path / "vectors")

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def tokenizer(self):
        """The tokenizer function used for BM25 indexing and search."""
        return self._bm25._tokenizer

    @tokenizer.setter
    def tokenizer(self, func):
        self._bm25._tokenizer = func

    def count(self) -> int:
        return self._meta.count()

    def add(
        self,
        ids: list[str],
        vectors: NDArray,
        metadatas: list[dict[str, Any]],
        documents: list[str] | None = None,
    ) -> None:
        """Add vectors with IDs, metadata, and optional document text."""
        vectors = np.asarray(vectors, dtype=np.float64)
        self._validate_add_inputs(ids, vectors, metadatas)
        if documents is not None and len(documents) != len(ids):
            raise ValueError(
                f"Length mismatch: {len(ids)} ids but {len(documents)} documents"
            )

        with FileLock(self._path / "lock"), self._rw_lock:
            if self._metric == "cosine":
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vectors = vectors / norms

            compressed = self._quantizer.quantize(vectors)
            start_pos = (self._meta.max_position() + 1)

            # Write vectors first (crash-safe ordering)
            if self._vectors is None:
                self._vectors = compressed
            else:
                self._vectors = CompressedVectors.concatenate([self._vectors, compressed])
            self._vectors.save(self._path / "vectors")

            # Then commit to SQLite
            rows = [
                (ids[i], start_pos + i, metadatas[i])
                for i in range(len(ids))
            ]
            self._meta.insert_batch(rows, documents=documents)

            # Update BM25 index
            if documents is not None:
                docs_to_index = [
                    (start_pos + i, doc)
                    for i, doc in enumerate(documents)
                    if doc
                ]
                if docs_to_index:
                    self._bm25.add(
                        [pos for pos, _ in docs_to_index],
                        [doc for _, doc in docs_to_index],
                    )

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray,
        metadatas: list[dict[str, Any]],
        documents: list[str] | None = None,
    ) -> None:
        """Insert or replace vectors."""
        vectors = np.asarray(vectors, dtype=np.float64)
        self._validate_add_inputs(ids, vectors, metadatas)
        if documents is not None and len(documents) != len(ids):
            raise ValueError(
                f"Length mismatch: {len(ids)} ids but {len(documents)} documents"
            )

        with FileLock(self._path / "lock"), self._rw_lock:
            if self._metric == "cosine":
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vectors = vectors / norms

            compressed = self._quantizer.quantize(vectors)
            start_pos = (self._meta.max_position() + 1)

            # Write vectors first
            if self._vectors is None:
                self._vectors = compressed
            else:
                self._vectors = CompressedVectors.concatenate([self._vectors, compressed])
            self._vectors.save(self._path / "vectors")

            # Then upsert in SQLite (old positions become dead)
            rows = [
                (ids[i], start_pos + i, metadatas[i])
                for i in range(len(ids))
            ]
            replaced = self._meta.upsert_batch(rows, documents=documents)

            # Update BM25 index
            if replaced:
                self._bm25.remove(replaced)
            if documents is not None:
                docs_to_index = [
                    (start_pos + i, doc)
                    for i, doc in enumerate(documents)
                    if doc
                ]
                if docs_to_index:
                    self._bm25.add(
                        [pos for pos, _ in docs_to_index],
                        [doc for _, doc in docs_to_index],
                    )

    def query(
        self,
        vector: NDArray,
        k: int = 10,
        where: dict[str, Any] | None = None,
        format: str | None = None,
    ) -> list[QueryResult] | dict:
        """Search for similar vectors."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        with self._rw_lock:
            if self._vectors is None or self.count() == 0:
                return to_chroma_format([]) if format == "chroma" else []

            vector = np.asarray(vector, dtype=np.float64)
            if vector.shape[-1] != self._dim:
                raise DimensionMismatchError(self._dim, vector.shape[-1])

            if self._metric == "cosine":
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

            vectors_snapshot = self._vectors
            live_positions = set(self._meta.get_all_live_positions())
            if where:
                where_clause, params = compile_filter(where)
                filtered_positions = set(self._meta.get_positions_by_filter(where_clause, params))
                valid_positions = live_positions & filtered_positions
            else:
                valid_positions = live_positions

            # Score computation must stay under the lock because
            # concurrent add() calls overwrite vector files on disk,
            # which invalidates any lazy/memory-mapped references held
            # by the snapshot.
            scores = self._quantizer.inner_product(vector, vectors_snapshot)

            # Mask invalid positions
            mask = np.zeros(len(scores), dtype=bool)
            for pos in valid_positions:
                if pos < len(mask):
                    mask[pos] = True
            scores[~mask] = -np.inf

            # Top-k
            effective_k = min(k, int(mask.sum()))
            if effective_k == 0:
                return to_chroma_format([]) if format == "chroma" else []

            top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            # Look up metadata
            position_to_meta = {
                r["position"]: r
                for r in self._meta.get_by_positions(top_indices.tolist())
            }

            results = []
            for idx in top_indices:
                pos = int(idx)
                row = position_to_meta.get(pos)
                if row is None:
                    continue
                score = float(scores[idx])
                if self._metric == "l2":
                    query_norm = float(np.linalg.norm(vector))
                    vec_norm = float(vectors_snapshot.norms[pos])
                    score = query_norm**2 + vec_norm**2 - 2 * score
                results.append(QueryResult(
                    id=row["id"], score=score, metadata=row["metadata"],
                    document=row.get("document"),
                ))

        if format == "chroma":
            return to_chroma_format(results)
        return results

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get metadata by IDs."""
        return self._meta.get_by_ids(ids)

    def hybrid_query(
        self,
        text: str,
        vector: NDArray | None = None,
        k: int = 10,
        fusion: str = "rrf",
        alpha: float = 0.5,
        where: dict[str, Any] | None = None,
        format: str | None = None,
    ) -> list[QueryResult] | dict:
        """Hybrid search combining BM25 keyword matching with vector similarity.

        Parameters
        ----------
        text : str
            Query text for BM25 scoring.
        vector : array-like, optional
            Query vector for semantic scoring.  If omitted, performs pure BM25.
        k : int
            Number of results to return.
        fusion : str
            ``"rrf"`` (Reciprocal Rank Fusion, default), ``"weighted"``
            (convex combination with min-max normalisation), or ``"dbsf"``
            (Distribution-Based Score Fusion).
        alpha : float
            Vector weight when *fusion* = ``"weighted"``.
            1.0 = pure vector, 0.0 = pure BM25.
        where : dict, optional
            Metadata filter (same syntax as :meth:`query`).
        format : str, optional
            ``"chroma"`` for ChromaDB column-oriented output.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        with self._rw_lock:
            if self.count() == 0:
                return to_chroma_format([]) if format == "chroma" else []

            # Valid positions (live + optional metadata filter)
            live_positions = set(self._meta.get_all_live_positions())
            if where:
                where_clause, params = compile_filter(where)
                filtered = set(self._meta.get_positions_by_filter(where_clause, params))
                valid_positions = live_positions & filtered
            else:
                valid_positions = live_positions

            if not valid_positions:
                return to_chroma_format([]) if format == "chroma" else []

            # Ensure BM25 index is loaded / up-to-date
            self._ensure_bm25_loaded()

            # --- BM25 scoring ---
            bm25_scores = self._bm25.search(text, valid_positions)
            bm25_ranked: list[tuple[int, float]] = sorted(
                bm25_scores.items(), key=lambda x: x[1], reverse=True,
            )

            # --- Vector scoring ---
            vector_ranked: list[tuple[int, float]] = []
            if vector is not None and self._vectors is not None:
                vector = np.asarray(vector, dtype=np.float64)
                if vector.shape[-1] != self._dim:
                    raise DimensionMismatchError(self._dim, vector.shape[-1])
                if self._metric == "cosine":
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm

                vectors_snapshot = self._vectors
                raw_scores = self._quantizer.inner_product(vector, vectors_snapshot)

                for pos in valid_positions:
                    if pos < len(raw_scores):
                        score = float(raw_scores[pos])
                        if self._metric == "l2":
                            qn = float(np.linalg.norm(vector))
                            vn = float(vectors_snapshot.norms[pos])
                            score = -(qn**2 + vn**2 - 2 * score)
                        vector_ranked.append((pos, score))
                vector_ranked.sort(key=lambda x: x[1], reverse=True)

            # --- Fusion ---
            fetch_n = max(k * 5, 50)
            if vector_ranked and bm25_ranked:
                top_v = vector_ranked[:fetch_n]
                top_b = bm25_ranked[:fetch_n]
                if fusion == "rrf":
                    fused = fuse_rrf(top_v, top_b, k)
                elif fusion == "weighted":
                    fused = fuse_weighted(top_v, top_b, k, alpha)
                elif fusion == "dbsf":
                    fused = fuse_dbsf(top_v, top_b, k)
                else:
                    raise ValueError(f"Unknown fusion method: {fusion!r}")
            elif vector_ranked:
                fused = vector_ranked[:k]
            elif bm25_ranked:
                fused = bm25_ranked[:k]
            else:
                return to_chroma_format([]) if format == "chroma" else []

            # --- Build result objects ---
            fused_positions = [pos for pos, _ in fused]
            position_to_meta = {
                r["position"]: r
                for r in self._meta.get_by_positions(fused_positions)
            }

            results = []
            for pos, score in fused:
                row = position_to_meta.get(pos)
                if row is None:
                    continue
                results.append(QueryResult(
                    id=row["id"],
                    score=score,
                    metadata=row["metadata"],
                    document=row.get("document"),
                ))

        if format == "chroma":
            return to_chroma_format(results)
        return results

    def _ensure_bm25_loaded(self) -> None:
        """Rebuild BM25 index from SQLite if the cache is stale or missing."""
        doc_count = self._meta.count_documents()
        if doc_count == 0:
            return
        if self._bm25.num_docs != doc_count:
            docs = self._meta.get_all_documents()
            self._bm25.rebuild(
                [d["position"] for d in docs],
                [d["document"] for d in docs],
            )

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """Delete vectors by IDs or filter."""
        if ids is None and where is None:
            raise ValueError("Must provide ids or where to delete")

        with FileLock(self._path / "lock"), self._rw_lock:
            deleted_positions: list[int] = []
            if ids is not None:
                deleted_positions.extend(self._meta.delete_by_ids(ids))
            if where is not None:
                where_clause, params = compile_filter(where)
                deleted_positions.extend(self._meta.delete_by_filter(where_clause, params))
            if deleted_positions:
                self._bm25.remove(deleted_positions)

    def compact(self) -> None:
        """Rewrite vectors and metadata to remove dead entries."""
        if self._vectors is None:
            return

        with FileLock(self._path / "lock"), self._rw_lock:
            live_positions = self._meta.get_all_live_positions()
            if not live_positions:
                self._vectors = None
                vectors_dir = self._path / "vectors"
                if vectors_dir.exists():
                    shutil.rmtree(vectors_dir)
                return

            if len(live_positions) == self._vectors.num_vectors:
                return  # Nothing to compact

            live_set = sorted(live_positions)
            parts = [self._vectors[pos:pos+1] for pos in live_set]
            new_vectors = CompressedVectors.concatenate(parts)

            position_map = {old: new for new, old in enumerate(live_set)}
            self._meta.rewrite_positions(position_map)
            self._bm25.remap_positions(position_map)

            self._vectors = new_vectors
            self._vectors.save(self._path / "vectors")

    def _validate_add_inputs(
        self,
        ids: list[str],
        vectors: NDArray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        n = len(ids)
        if vectors.shape[0] != n:
            raise ValueError(
                f"Length mismatch: {n} ids but {vectors.shape[0]} vectors"
            )
        if len(metadatas) != n:
            raise ValueError(
                f"Length mismatch: {n} ids but {len(metadatas)} metadatas"
            )
        if vectors.shape[1] != self._dim:
            raise DimensionMismatchError(self._dim, vectors.shape[1])

    def close(self) -> None:
        """Close the metadata store."""
        self._meta.close()

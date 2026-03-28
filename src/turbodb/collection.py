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

from turbodb.exceptions import DimensionMismatchError
from turbodb.filters import compile_filter
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

    def count(self) -> int:
        return self._meta.count()

    def add(
        self,
        ids: list[str],
        vectors: NDArray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add vectors with IDs and metadata."""
        vectors = np.asarray(vectors, dtype=np.float64)
        self._validate_add_inputs(ids, vectors, metadatas)

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
            self._meta.insert_batch(rows)

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or replace vectors."""
        vectors = np.asarray(vectors, dtype=np.float64)
        self._validate_add_inputs(ids, vectors, metadatas)

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
            self._meta.upsert_batch(rows)

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

            # Snapshot vectors and live positions atomically
            vectors_snapshot = self._vectors
            live_positions = set(self._meta.get_all_live_positions())
            if where:
                where_clause, params = compile_filter(where)
                filtered_positions = set(self._meta.get_positions_by_filter(where_clause, params))
                valid_positions = live_positions & filtered_positions
            else:
                valid_positions = live_positions

        # Get all scores (can run outside lock using snapshot)
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
            results.append(QueryResult(id=row["id"], score=score, metadata=row["metadata"]))

        if format == "chroma":
            return to_chroma_format(results)
        return results

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get metadata by IDs."""
        return self._meta.get_by_ids(ids)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """Delete vectors by IDs or filter."""
        if ids is None and where is None:
            raise ValueError("Must provide ids or where to delete")

        with FileLock(self._path / "lock"), self._rw_lock:
            if ids is not None:
                self._meta.delete_by_ids(ids)
            if where is not None:
                where_clause, params = compile_filter(where)
                self._meta.delete_by_filter(where_clause, params)

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

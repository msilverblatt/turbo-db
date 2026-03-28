# tests/test_crash_safety.py
"""Test crash recovery: orphaned numpy vectors are trimmed on reopen."""

import numpy as np
import pytest

from turbodb import TurboDB
from turbodb.collection import Collection


class TestCrashRecovery:
    def test_orphaned_vectors_trimmed_on_open(self, tmp_db):
        """Simulate crash: vectors written but SQLite not committed."""
        db = TurboDB(tmp_db)
        col = db.create_collection("docs", dim=64)

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((5, 64))
        col.add(
            ids=["a", "b", "c", "d", "e"],
            vectors=vectors,
            metadatas=[{} for _ in range(5)],
        )
        col.close()

        # Simulate crash: add more vectors to numpy but don't update SQLite
        col2 = Collection.open(tmp_db / "docs", "docs")
        extra = rng.standard_normal((3, 64))
        from turboquant import CompressedVectors, TurboQuant
        tq = TurboQuant(dim=64, bit_width=2, mode="inner_product", seed=col2._seed)
        extra_compressed = tq.quantize(extra)
        combined = CompressedVectors.concatenate([col2._vectors, extra_compressed])
        combined.save(tmp_db / "docs" / "vectors")
        col2.close()

        # Reopen — should detect mismatch and trim
        db2 = TurboDB(tmp_db)
        col3 = db2.get_collection("docs")
        assert col3.count() == 5
        assert col3._vectors.num_vectors == 5

        # Queries still work
        results = col3.query(vector=vectors[0], k=3)
        assert len(results) == 3

    def test_empty_sqlite_with_vectors_on_disk(self, tmp_db):
        """Simulate crash where SQLite was cleared but vectors remain."""
        db = TurboDB(tmp_db)
        col = db.create_collection("docs", dim=64)

        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((5, 64))
        col.add(
            ids=["a", "b", "c", "d", "e"],
            vectors=vectors,
            metadatas=[{} for _ in range(5)],
        )
        col.close()

        # Wipe SQLite data but leave vectors on disk
        import sqlite3
        conn = sqlite3.connect(str(tmp_db / "docs" / "metadata.db"))
        conn.execute("DELETE FROM vectors")
        conn.commit()
        conn.close()

        # Reopen — should see 0 vectors
        db2 = TurboDB(tmp_db)
        col2 = db2.get_collection("docs")
        assert col2.count() == 0

# tests/test_concurrency.py
"""Test concurrent reads and writes."""

import threading

import numpy as np
import pytest

from turbodb import TurboDB


class TestConcurrentWrites:
    def test_parallel_adds_no_corruption(self, tmp_db):
        db = TurboDB(tmp_db)
        col = db.create_collection("docs", dim=64)
        rng = np.random.default_rng(42)

        errors = []

        def add_batch(batch_id):
            try:
                vectors = rng.standard_normal((10, 64))
                ids = [f"batch{batch_id}_vec{i}" for i in range(10)]
                metas = [{"batch": batch_id} for _ in range(10)]
                col.add(ids=ids, vectors=vectors, metadatas=metas)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent adds: {errors}"
        assert col.count() == 50


class TestConcurrentReads:
    def test_query_during_add(self, tmp_db):
        db = TurboDB(tmp_db)
        col = db.create_collection("docs", dim=64)
        rng = np.random.default_rng(42)

        # Seed with initial data
        vectors = rng.standard_normal((20, 64))
        col.add(
            ids=[f"init_{i}" for i in range(20)],
            vectors=vectors,
            metadatas=[{"i": i} for i in range(20)],
        )

        errors = []
        query_results = []

        def writer():
            try:
                new_vecs = rng.standard_normal((10, 64))
                col.add(
                    ids=[f"new_{i}" for i in range(10)],
                    vectors=new_vecs,
                    metadatas=[{} for _ in range(10)],
                )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                results = col.query(vector=vectors[0], k=5)
                query_results.append(len(results))
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(5)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        assert not errors, f"Errors during concurrent read/write: {errors}"
        # Readers should get results (may see pre- or post-write state)
        assert all(r > 0 for r in query_results)

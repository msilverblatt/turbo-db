# tests/test_integration.py
import numpy as np
import pytest

from turbodb import TurboDB, QueryResult


@pytest.fixture
def db(tmp_db):
    return TurboDB(tmp_db)


class TestEndToEnd:
    def test_add_query_delete_cycle(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("docs", dim=64)

        vectors = rng.standard_normal((20, 64))
        ids = [f"doc_{i}" for i in range(20)]
        metas = [{"i": i, "even": i % 2 == 0} for i in range(20)]

        col.add(ids=ids, vectors=vectors, metadatas=metas)
        assert col.count() == 20

        results = col.query(vector=vectors[0], k=5)
        assert len(results) == 5
        assert all(isinstance(r, QueryResult) for r in results)
        # The query vector itself should be the top match
        assert results[0].id == "doc_0"

        col.delete(ids=["doc_0"])
        assert col.count() == 19

        results = col.query(vector=vectors[0], k=5)
        assert results[0].id != "doc_0"

    def test_upsert_overwrites(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("docs", dim=64)

        v1 = rng.standard_normal((1, 64))
        col.add(ids=["x"], vectors=v1, metadatas=[{"version": 1}])

        v2 = rng.standard_normal((1, 64))
        col.upsert(ids=["x"], vectors=v2, metadatas=[{"version": 2}])

        assert col.count() == 1
        items = col.get(ids=["x"])
        assert items[0]["metadata"]["version"] == 2

    def test_filtered_query(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("docs", dim=64)

        vectors = rng.standard_normal((20, 64))
        ids = [f"doc_{i}" for i in range(20)]
        metas = [{"group": "a" if i < 10 else "b"} for i in range(20)]

        col.add(ids=ids, vectors=vectors, metadatas=metas)

        results = col.query(
            vector=vectors[0], k=20,
            where={"group": {"$eq": "b"}},
        )
        assert all(r.metadata["group"] == "b" for r in results)
        assert len(results) == 10

    def test_compact_preserves_data(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("docs", dim=64)

        vectors = rng.standard_normal((10, 64))
        ids = [f"doc_{i}" for i in range(10)]
        metas = [{"i": i} for i in range(10)]

        col.add(ids=ids, vectors=vectors, metadatas=metas)
        col.delete(ids=["doc_0", "doc_1", "doc_2"])
        col.compact()

        assert col.count() == 7
        results = col.query(vector=vectors[5], k=3)
        assert len(results) == 3

    def test_reopen_persistence(self, db, tmp_db):
        rng = np.random.default_rng(42)
        col = db.create_collection("docs", dim=64)
        vectors = rng.standard_normal((5, 64))
        col.add(
            ids=["a", "b", "c", "d", "e"],
            vectors=vectors,
            metadatas=[{"x": i} for i in range(5)],
        )
        col.close()

        db2 = TurboDB(tmp_db)
        col2 = db2.get_collection("docs")
        assert col2.count() == 5
        results = col2.query(vector=vectors[0], k=1)
        assert results[0].id == "a"

    def test_multiple_collections(self, db):
        rng = np.random.default_rng(42)
        c1 = db.create_collection("a", dim=32)
        c2 = db.create_collection("b", dim=64)

        c1.add(ids=["x"], vectors=rng.standard_normal((1, 32)), metadatas=[{}])
        c2.add(ids=["y"], vectors=rng.standard_normal((1, 64)), metadatas=[{}])

        assert c1.count() == 1
        assert c2.count() == 1
        assert sorted(db.list_collections()) == ["a", "b"]

    def test_all_metrics(self, db):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((10, 64))
        query = vectors[0]

        for metric in ["cosine", "ip", "l2"]:
            col = db.create_collection(f"m_{metric}", dim=64, metric=metric)
            col.add(
                ids=[f"v{i}" for i in range(10)],
                vectors=vectors,
                metadatas=[{} for _ in range(10)],
            )
            results = col.query(vector=query, k=3)
            assert len(results) == 3

    def test_chroma_compat_format(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("docs", dim=64)
        vectors = rng.standard_normal((5, 64))
        col.add(ids=["a", "b", "c", "d", "e"], vectors=vectors, metadatas=[{"x": i} for i in range(5)])

        results = col.query(vector=vectors[0], k=3, format="chroma")
        assert isinstance(results, dict)
        assert len(results["ids"][0]) == 3
        assert len(results["distances"][0]) == 3
        assert len(results["metadatas"][0]) == 3

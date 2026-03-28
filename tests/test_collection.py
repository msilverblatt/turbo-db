# tests/test_collection.py
import numpy as np
import pytest

from turbodb.collection import Collection
from turbodb.exceptions import DimensionMismatchError
from turbodb.results import QueryResult


@pytest.fixture
def col(tmp_path):
    return Collection.create(
        path=tmp_path / "test_col",
        name="test",
        dim=64,
        bit_width=2,
        metric="cosine",
    )


@pytest.fixture
def populated_col(col, sample_vectors, sample_ids, sample_metadatas):
    col.add(
        ids=sample_ids[:10],
        vectors=sample_vectors[:10],
        metadatas=sample_metadatas[:10],
    )
    return col


class TestCreate:
    def test_creates_directory(self, col):
        assert col._path.exists()

    def test_properties(self, col):
        assert col.name == "test"
        assert col.dim == 64
        assert col.metric == "cosine"

    def test_count_starts_at_zero(self, col):
        assert col.count() == 0


class TestAdd:
    def test_add_increases_count(self, col, sample_vectors, sample_ids, sample_metadatas):
        col.add(ids=sample_ids[:5], vectors=sample_vectors[:5], metadatas=sample_metadatas[:5])
        assert col.count() == 5

    def test_add_wrong_dim_raises(self, col):
        with pytest.raises(DimensionMismatchError):
            col.add(ids=["x"], vectors=np.zeros((1, 128)), metadatas=[{}])

    def test_add_duplicate_id_raises(self, col, sample_vectors):
        col.add(ids=["dup"], vectors=sample_vectors[:1], metadatas=[{}])
        with pytest.raises(Exception):
            col.add(ids=["dup"], vectors=sample_vectors[1:2], metadatas=[{}])

    def test_add_mismatched_lengths_raises(self, col, sample_vectors):
        with pytest.raises(ValueError):
            col.add(ids=["a", "b"], vectors=sample_vectors[:1], metadatas=[{}])


class TestQuery:
    def test_query_returns_results(self, populated_col, sample_vectors):
        results = populated_col.query(vector=sample_vectors[0], k=3)
        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)

    def test_query_scores_are_descending(self, populated_col, sample_vectors):
        results = populated_col.query(vector=sample_vectors[0], k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_with_filter(self, populated_col, sample_vectors):
        results = populated_col.query(
            vector=sample_vectors[0], k=10,
            where={"group": {"$eq": "a"}},
        )
        assert all(r.metadata["group"] == "a" for r in results)

    def test_query_chroma_format(self, populated_col, sample_vectors):
        results = populated_col.query(vector=sample_vectors[0], k=3, format="chroma")
        assert "ids" in results
        assert "distances" in results
        assert "metadatas" in results
        assert len(results["ids"][0]) == 3

    def test_query_k_larger_than_collection(self, populated_col, sample_vectors):
        results = populated_col.query(vector=sample_vectors[0], k=100)
        assert len(results) == 10  # only 10 vectors in collection


class TestGet:
    def test_get_by_ids(self, populated_col):
        items = populated_col.get(ids=["vec_0", "vec_1"])
        assert len(items) == 2
        assert items[0]["id"] in ("vec_0", "vec_1")

    def test_get_missing_id_returns_empty(self, populated_col):
        items = populated_col.get(ids=["nonexistent"])
        assert len(items) == 0


class TestUpsert:
    def test_upsert_new(self, col, sample_vectors):
        col.upsert(ids=["new"], vectors=sample_vectors[:1], metadatas=[{"x": 1}])
        assert col.count() == 1

    def test_upsert_existing_replaces(self, populated_col, sample_vectors):
        populated_col.upsert(
            ids=["vec_0"],
            vectors=sample_vectors[40:41],
            metadatas=[{"replaced": True}],
        )
        assert populated_col.count() == 10
        items = populated_col.get(ids=["vec_0"])
        assert items[0]["metadata"]["replaced"] is True


class TestDelete:
    def test_delete_by_ids(self, populated_col):
        populated_col.delete(ids=["vec_0", "vec_1"])
        assert populated_col.count() == 8

    def test_delete_by_filter(self, populated_col):
        populated_col.delete(where={"group": {"$eq": "a"}})
        remaining = populated_col.count()
        assert remaining == 5  # half of 10 have group "b"

    def test_delete_no_args_raises(self, populated_col):
        with pytest.raises(ValueError):
            populated_col.delete()


class TestCompact:
    def test_compact_after_delete(self, populated_col, sample_vectors):
        populated_col.delete(ids=["vec_0", "vec_1", "vec_2"])
        assert populated_col.count() == 7
        populated_col.compact()
        assert populated_col.count() == 7
        # Query still works after compaction
        results = populated_col.query(vector=sample_vectors[5], k=3)
        assert len(results) == 3

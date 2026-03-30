# tests/test_hybrid.py
import numpy as np
import pytest

from turbodb import TurboDB, QueryResult


@pytest.fixture
def db(tmp_db):
    return TurboDB(tmp_db)


@pytest.fixture
def hybrid_col(db):
    """Collection pre-loaded with vectors + documents for hybrid search."""
    rng = np.random.default_rng(42)
    col = db.create_collection("docs", dim=64)

    documents = [
        "the quick brown fox jumps over the lazy dog",
        "a neural network learns representations from data",
        "the cat sat on the mat and watched the fox",
        "deep learning models require large amounts of data",
        "the lazy dog slept while the fox ran away",
        "transformers use self attention for sequence modeling",
        "the brown fox and the brown dog are friends",
        "reinforcement learning agents maximize cumulative reward",
        "the quick cat chased the slow mouse",
        "gradient descent optimizes the loss function",
    ]

    vectors = rng.standard_normal((len(documents), 64))
    ids = [f"doc_{i}" for i in range(len(documents))]
    metas = [{"i": i, "category": "animal" if i < 5 else "ml"} for i in range(len(documents))]

    col.add(ids=ids, vectors=vectors, metadatas=metas, documents=documents)
    return col, vectors


class TestHybridQuery:
    def test_rrf_fusion(self, hybrid_col):
        col, vectors = hybrid_col
        results = col.hybrid_query(
            text="quick fox",
            vector=vectors[0],
            k=5,
            fusion="rrf",
        )
        assert len(results) == 5
        assert all(isinstance(r, QueryResult) for r in results)

    def test_weighted_fusion(self, hybrid_col):
        col, vectors = hybrid_col
        results = col.hybrid_query(
            text="quick fox",
            vector=vectors[0],
            k=5,
            fusion="weighted",
            alpha=0.5,
        )
        assert len(results) == 5

    def test_dbsf_fusion(self, hybrid_col):
        col, vectors = hybrid_col
        results = col.hybrid_query(
            text="quick fox",
            vector=vectors[0],
            k=5,
            fusion="dbsf",
        )
        assert len(results) == 5

    def test_pure_bm25(self, hybrid_col):
        col, _ = hybrid_col
        results = col.hybrid_query(text="neural network data", k=5)
        assert len(results) >= 2
        # Doc 1 has "neural", "network", "data" — should be top
        assert results[0].id == "doc_1"

    def test_bm25_relevance(self, hybrid_col):
        col, _ = hybrid_col
        results = col.hybrid_query(text="fox", k=10)
        fox_ids = {"doc_0", "doc_2", "doc_4", "doc_6"}
        returned_ids = [r.id for r in results]
        # All fox-containing docs should be returned
        for fid in fox_ids:
            assert fid in returned_ids

    def test_metadata_filter(self, hybrid_col):
        col, vectors = hybrid_col
        results = col.hybrid_query(
            text="quick fox",
            vector=vectors[0],
            k=10,
            where={"category": {"$eq": "animal"}},
        )
        assert all(r.metadata["category"] == "animal" for r in results)

    def test_document_returned_in_results(self, hybrid_col):
        col, vectors = hybrid_col
        results = col.hybrid_query(text="fox", vector=vectors[0], k=1)
        assert results[0].document is not None
        assert len(results[0].document) > 0

    def test_chroma_format(self, hybrid_col):
        col, vectors = hybrid_col
        results = col.hybrid_query(
            text="quick fox",
            vector=vectors[0],
            k=3,
            format="chroma",
        )
        assert isinstance(results, dict)
        assert len(results["ids"][0]) == 3
        assert len(results["distances"][0]) == 3
        assert len(results["metadatas"][0]) == 3
        assert len(results["documents"][0]) == 3

    def test_unknown_fusion_raises(self, hybrid_col):
        col, vectors = hybrid_col
        with pytest.raises(ValueError, match="Unknown fusion"):
            col.hybrid_query(text="fox", vector=vectors[0], fusion="invalid")

    def test_k_zero_raises(self, hybrid_col):
        col, vectors = hybrid_col
        with pytest.raises(ValueError, match="k must be positive"):
            col.hybrid_query(text="fox", k=0)

    def test_empty_collection(self, db):
        col = db.create_collection("empty", dim=64)
        results = col.hybrid_query(text="hello")
        assert results == []

    def test_no_documents_falls_back_to_vector(self, db):
        """Collection without documents — hybrid_query with vector falls back to vector-only."""
        rng = np.random.default_rng(42)
        col = db.create_collection("nodocs", dim=64)
        vectors = rng.standard_normal((5, 64))
        col.add(
            ids=[f"v{i}" for i in range(5)],
            vectors=vectors,
            metadatas=[{} for _ in range(5)],
        )
        results = col.hybrid_query(text="anything", vector=vectors[0], k=3)
        assert len(results) == 3

    def test_no_documents_text_only_returns_empty(self, db):
        """Collection without documents — pure BM25 returns empty."""
        rng = np.random.default_rng(42)
        col = db.create_collection("nodocs2", dim=64)
        vectors = rng.standard_normal((5, 64))
        col.add(
            ids=[f"v{i}" for i in range(5)],
            vectors=vectors,
            metadatas=[{} for _ in range(5)],
        )
        results = col.hybrid_query(text="anything", k=3)
        assert results == []


class TestDocumentLifecycle:
    def test_add_with_documents(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        col.add(
            ids=["a", "b"],
            vectors=rng.standard_normal((2, 64)),
            metadatas=[{}, {}],
            documents=["alpha bravo", "charlie delta"],
        )
        results = col.hybrid_query(text="alpha", k=1)
        assert results[0].id == "a"

    def test_upsert_with_documents(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        col.add(
            ids=["a"],
            vectors=rng.standard_normal((1, 64)),
            metadatas=[{"v": 1}],
            documents=["alpha bravo"],
        )
        col.upsert(
            ids=["a"],
            vectors=rng.standard_normal((1, 64)),
            metadatas=[{"v": 2}],
            documents=["charlie delta"],
        )
        # Old document should no longer match
        results = col.hybrid_query(text="alpha", k=1)
        assert len(results) == 0 or results[0].metadata["v"] == 2
        # New document should match
        results = col.hybrid_query(text="charlie", k=1)
        assert results[0].id == "a"
        assert results[0].metadata["v"] == 2

    def test_delete_removes_from_bm25(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        col.add(
            ids=["a", "b"],
            vectors=rng.standard_normal((2, 64)),
            metadatas=[{}, {}],
            documents=["alpha bravo", "charlie delta"],
        )
        col.delete(ids=["a"])
        results = col.hybrid_query(text="alpha", k=5)
        assert all(r.id != "a" for r in results)

    def test_compact_preserves_bm25(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        col.add(
            ids=["a", "b", "c"],
            vectors=rng.standard_normal((3, 64)),
            metadatas=[{}, {}, {}],
            documents=["alpha bravo", "charlie delta", "echo foxtrot"],
        )
        col.delete(ids=["b"])
        col.compact()
        results = col.hybrid_query(text="echo", k=1)
        assert results[0].id == "c"

    def test_documents_length_mismatch_raises(self, db):
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        with pytest.raises(ValueError, match="Length mismatch"):
            col.add(
                ids=["a", "b"],
                vectors=rng.standard_normal((2, 64)),
                metadatas=[{}, {}],
                documents=["only one"],
            )

    def test_persistence_across_reopen(self, db, tmp_db):
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        col.add(
            ids=["a", "b"],
            vectors=rng.standard_normal((2, 64)),
            metadatas=[{}, {}],
            documents=["alpha bravo", "charlie delta"],
        )
        col.close()

        db2 = TurboDB(tmp_db)
        col2 = db2.get_collection("d")
        results = col2.hybrid_query(text="alpha", k=1)
        assert results[0].id == "a"

    def test_query_returns_document_field(self, db):
        """Regular query() also returns the document field."""
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        vectors = rng.standard_normal((2, 64))
        col.add(
            ids=["a", "b"],
            vectors=vectors,
            metadatas=[{}, {}],
            documents=["alpha bravo", "charlie delta"],
        )
        results = col.query(vector=vectors[0], k=1)
        assert results[0].document is not None

    def test_weighted_alpha_extremes(self, db):
        """Alpha=1.0 should match vector-only results ordering."""
        rng = np.random.default_rng(42)
        col = db.create_collection("d", dim=64)
        vectors = rng.standard_normal((5, 64))
        col.add(
            ids=[f"v{i}" for i in range(5)],
            vectors=vectors,
            metadatas=[{} for _ in range(5)],
            documents=[f"document {i} with words" for i in range(5)],
        )
        vec_results = col.query(vector=vectors[0], k=5)
        hybrid_results = col.hybrid_query(
            text="nonexistent", vector=vectors[0], k=5, fusion="weighted", alpha=1.0,
        )
        # With alpha=1.0 and no BM25 matches, hybrid should match vector ordering
        assert [r.id for r in vec_results] == [r.id for r in hybrid_results]

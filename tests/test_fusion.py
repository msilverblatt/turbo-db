# tests/test_fusion.py
import pytest

from turbodb.fusion import fuse_dbsf, fuse_rrf, fuse_weighted


class TestRRF:
    def test_basic(self):
        vector = [(1, 0.9), (2, 0.8), (3, 0.7)]
        bm25 = [(2, 5.0), (3, 4.0), (1, 3.0)]
        results = fuse_rrf(vector, bm25, k=3)
        assert len(results) == 3
        ids = [r[0] for r in results]
        # Item 2 is rank 1 in BM25, rank 2 in vector — should be top
        assert ids[0] == 2

    def test_non_overlapping(self):
        vector = [(1, 0.9), (2, 0.8)]
        bm25 = [(3, 5.0), (4, 4.0)]
        results = fuse_rrf(vector, bm25, k=4)
        assert len(results) == 4
        assert set(r[0] for r in results) == {1, 2, 3, 4}

    def test_k_limits_output(self):
        vector = [(i, 1.0 / i) for i in range(1, 11)]
        bm25 = [(i, 10.0 / i) for i in range(1, 11)]
        results = fuse_rrf(vector, bm25, k=3)
        assert len(results) == 3

    def test_empty_inputs(self):
        assert fuse_rrf([], [], k=5) == []
        assert fuse_rrf([(1, 0.5)], [], k=5) == [(1, pytest.approx(1 / 61))]

    def test_custom_rrf_k(self):
        vector = [(1, 0.9)]
        bm25 = [(1, 5.0)]
        r60 = fuse_rrf(vector, bm25, k=1, rrf_k=60)
        r10 = fuse_rrf(vector, bm25, k=1, rrf_k=10)
        # Smaller rrf_k gives higher scores
        assert r10[0][1] > r60[0][1]


class TestWeighted:
    def test_alpha_1_pure_vector(self):
        vector = [(1, 0.9), (2, 0.5)]
        bm25 = [(2, 10.0), (1, 1.0)]
        results = fuse_weighted(vector, bm25, k=2, alpha=1.0)
        assert results[0][0] == 1  # vector top

    def test_alpha_0_pure_bm25(self):
        vector = [(1, 0.9), (2, 0.5)]
        bm25 = [(2, 10.0), (1, 1.0)]
        results = fuse_weighted(vector, bm25, k=2, alpha=0.0)
        assert results[0][0] == 2  # bm25 top

    def test_alpha_half(self):
        vector = [(1, 1.0), (2, 0.0)]
        bm25 = [(2, 1.0), (1, 0.0)]
        results = fuse_weighted(vector, bm25, k=2, alpha=0.5)
        # Both should get the same score (0.5)
        assert abs(results[0][1] - results[1][1]) < 1e-10

    def test_normalization(self):
        vector = [(1, 100.0), (2, 50.0)]
        bm25 = [(1, 0.001), (2, 0.002)]
        results = fuse_weighted(vector, bm25, k=2, alpha=0.5)
        assert len(results) == 2
        # Both should be properly normalized to [0,1]
        for _, score in results:
            assert 0.0 <= score <= 1.0 + 1e-10

    def test_single_result(self):
        vector = [(1, 0.5)]
        bm25 = [(1, 3.0)]
        results = fuse_weighted(vector, bm25, k=1, alpha=0.5)
        assert len(results) == 1
        # Single item normalizes to 1.0 on each side
        assert results[0][1] == pytest.approx(1.0)

    def test_empty(self):
        assert fuse_weighted([], [], k=5) == []


class TestDBSF:
    def test_basic(self):
        vector = [(1, 0.9), (2, 0.7), (3, 0.5)]
        bm25 = [(2, 5.0), (1, 4.0), (3, 3.0)]
        results = fuse_dbsf(vector, bm25, k=3)
        assert len(results) == 3

    def test_identical_scores(self):
        vector = [(1, 0.5), (2, 0.5)]
        bm25 = [(1, 3.0), (2, 3.0)]
        results = fuse_dbsf(vector, bm25, k=2)
        assert len(results) == 2
        # Same scores → both get normalized to 1.0
        assert results[0][1] == pytest.approx(results[1][1])

    def test_single_result(self):
        vector = [(1, 0.9)]
        bm25 = [(1, 5.0)]
        results = fuse_dbsf(vector, bm25, k=1)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(2.0)  # 1.0 + 1.0

    def test_k_limits(self):
        vector = [(i, float(i)) for i in range(20)]
        bm25 = [(i, float(20 - i)) for i in range(20)]
        results = fuse_dbsf(vector, bm25, k=5)
        assert len(results) == 5

    def test_empty(self):
        assert fuse_dbsf([], [], k=5) == []

    def test_outlier_robustness(self):
        # One extreme outlier — DBSF should handle it better than min-max
        vector = [(1, 100.0), (2, 1.0), (3, 1.1), (4, 0.9)]
        bm25 = [(1, 1.0), (2, 5.0), (3, 4.5), (4, 4.0)]
        results = fuse_dbsf(vector, bm25, k=4)
        assert len(results) == 4

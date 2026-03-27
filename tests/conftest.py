from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary directory for a TurboDB instance."""
    return tmp_path / "test_db"


@pytest.fixture
def sample_vectors():
    """Return 50 random vectors of dim=64 for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 64))


@pytest.fixture
def sample_ids():
    """Return 50 string IDs."""
    return [f"vec_{i}" for i in range(50)]


@pytest.fixture
def sample_metadatas():
    """Return 50 metadata dicts."""
    return [{"index": i, "group": "a" if i % 2 == 0 else "b"} for i in range(50)]

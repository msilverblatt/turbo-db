# tests/test_metadata.py
import pytest

from turbodb.metadata import MetadataStore


@pytest.fixture
def store(tmp_path):
    return MetadataStore(tmp_path / "metadata.db")


class TestInsert:
    def test_insert_and_get(self, store):
        store.insert("id1", 0, {"color": "red"})
        row = store.get_by_id("id1")
        assert row["id"] == "id1"
        assert row["position"] == 0
        assert row["metadata"] == {"color": "red"}

    def test_insert_duplicate_raises(self, store):
        store.insert("id1", 0, {})
        with pytest.raises(Exception):
            store.insert("id1", 1, {})

    def test_insert_batch(self, store):
        rows = [("a", 0, {"x": 1}), ("b", 1, {"x": 2}), ("c", 2, {"x": 3})]
        store.insert_batch(rows)
        assert store.count() == 3


class TestDelete:
    def test_delete_by_ids(self, store):
        store.insert("id1", 0, {})
        store.insert("id2", 1, {})
        store.delete_by_ids(["id1"])
        assert store.get_by_id("id1") is None
        assert store.get_by_id("id2") is not None

    def test_delete_by_filter(self, store):
        store.insert("id1", 0, {"group": "a"})
        store.insert("id2", 1, {"group": "b"})
        deleted_positions = store.delete_by_filter("json_extract(metadata, '$.group') = ?", ["a"])
        assert deleted_positions == [0]
        assert store.count() == 1


class TestQuery:
    def test_get_positions_by_filter(self, store):
        store.insert("a", 0, {"x": 10})
        store.insert("b", 1, {"x": 20})
        store.insert("c", 2, {"x": 30})
        positions = store.get_positions_by_filter(
            "json_extract(metadata, '$.x') > ?", [15]
        )
        assert sorted(positions) == [1, 2]

    def test_get_by_positions(self, store):
        store.insert("a", 0, {"x": 1})
        store.insert("b", 1, {"x": 2})
        rows = store.get_by_positions([0, 1])
        assert len(rows) == 2
        assert rows[0]["id"] == "a"

    def test_get_all_live_positions(self, store):
        store.insert("a", 0, {})
        store.insert("b", 1, {})
        store.delete_by_ids(["a"])
        positions = store.get_all_live_positions()
        assert positions == [1]

    def test_count(self, store):
        assert store.count() == 0
        store.insert("a", 0, {})
        assert store.count() == 1
        store.delete_by_ids(["a"])
        assert store.count() == 0

    def test_max_position(self, store):
        assert store.max_position() == -1
        store.insert("a", 0, {})
        store.insert("b", 5, {})
        assert store.max_position() == 5


class TestConfig:
    def test_store_and_load_config(self, store):
        config = {"dim": 384, "bit_width": 2, "metric": "cosine", "seed": 42}
        store.set_config(config)
        loaded = store.get_config()
        assert loaded == config

    def test_update_config(self, store):
        store.set_config({"dim": 384})
        store.set_config({"dim": 768})
        assert store.get_config()["dim"] == 768

# tests/test_db.py
import pytest

from turbodb.db import TurboDB
from turbodb.exceptions import CollectionExistsError, CollectionNotFoundError


@pytest.fixture
def db(tmp_db):
    return TurboDB(tmp_db)


class TestOpenCreate:
    def test_creates_directory(self, db, tmp_db):
        assert tmp_db.exists()
        assert (tmp_db / "turbodb.json").exists()

    def test_opens_existing(self, tmp_db):
        TurboDB(tmp_db)
        db2 = TurboDB(tmp_db)
        assert db2._path == tmp_db


class TestCollections:
    def test_create_collection(self, db):
        col = db.create_collection("docs", dim=64)
        assert col.name == "docs"
        assert col.dim == 64

    def test_create_duplicate_raises(self, db):
        db.create_collection("docs", dim=64)
        with pytest.raises(CollectionExistsError):
            db.create_collection("docs", dim=64)

    def test_get_collection(self, db):
        db.create_collection("docs", dim=64)
        col = db.get_collection("docs")
        assert col.name == "docs"

    def test_get_missing_raises(self, db):
        with pytest.raises(CollectionNotFoundError):
            db.get_collection("nope")

    def test_get_or_create_new(self, db):
        col = db.get_or_create_collection("docs", dim=64)
        assert col.name == "docs"

    def test_get_or_create_existing(self, db):
        db.create_collection("docs", dim=64)
        col = db.get_or_create_collection("docs", dim=64)
        assert col.name == "docs"

    def test_list_collections(self, db):
        assert db.list_collections() == []
        db.create_collection("a", dim=64)
        db.create_collection("b", dim=64)
        names = sorted(db.list_collections())
        assert names == ["a", "b"]

    def test_delete_collection(self, db):
        db.create_collection("docs", dim=64)
        db.delete_collection("docs")
        assert db.list_collections() == []

    def test_delete_missing_raises(self, db):
        with pytest.raises(CollectionNotFoundError):
            db.delete_collection("nope")

    def test_custom_bit_width(self, db):
        col = db.create_collection("docs", dim=64, bit_width=4)
        assert col._bit_width == 4

    def test_default_metric_is_cosine(self, db):
        col = db.create_collection("docs", dim=64)
        assert col.metric == "cosine"

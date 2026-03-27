# tests/test_filters.py
import pytest

from turbodb.exceptions import InvalidFilterError
from turbodb.filters import compile_filter


class TestComparisonOperators:
    def test_eq(self):
        sql, params = compile_filter({"name": {"$eq": "alice"}})
        assert "json_extract(metadata, '$.name') = ?" in sql
        assert params == ["alice"]

    def test_ne(self):
        sql, params = compile_filter({"name": {"$ne": "bob"}})
        assert "json_extract(metadata, '$.name') != ?" in sql
        assert params == ["bob"]

    def test_gt(self):
        sql, params = compile_filter({"age": {"$gt": 30}})
        assert "json_extract(metadata, '$.age') > ?" in sql
        assert params == [30]

    def test_gte(self):
        sql, params = compile_filter({"age": {"$gte": 30}})
        assert "json_extract(metadata, '$.age') >= ?" in sql
        assert params == [30]

    def test_lt(self):
        sql, params = compile_filter({"age": {"$lt": 30}})
        assert "json_extract(metadata, '$.age') < ?" in sql
        assert params == [30]

    def test_lte(self):
        sql, params = compile_filter({"age": {"$lte": 30}})
        assert "json_extract(metadata, '$.age') <= ?" in sql
        assert params == [30]


class TestSetOperators:
    def test_in(self):
        sql, params = compile_filter({"color": {"$in": ["red", "blue"]}})
        assert "json_extract(metadata, '$.color') IN (?, ?)" in sql
        assert params == ["red", "blue"]

    def test_nin(self):
        sql, params = compile_filter({"color": {"$nin": ["red"]}})
        assert "json_extract(metadata, '$.color') NOT IN (?)" in sql
        assert params == ["red"]


class TestLogicalOperators:
    def test_and(self):
        sql, params = compile_filter({
            "$and": [{"age": {"$gt": 20}}, {"name": {"$eq": "alice"}}]
        })
        assert "AND" in sql
        assert len(params) == 2

    def test_or(self):
        sql, params = compile_filter({
            "$or": [{"age": {"$gt": 20}}, {"name": {"$eq": "alice"}}]
        })
        assert "OR" in sql
        assert len(params) == 2

    def test_nested_and_or(self):
        sql, params = compile_filter({
            "$and": [
                {"$or": [{"x": {"$eq": 1}}, {"x": {"$eq": 2}}]},
                {"y": {"$gt": 10}},
            ]
        })
        assert "AND" in sql
        assert "OR" in sql
        assert len(params) == 3


class TestImplicitAnd:
    def test_multiple_fields_are_implicit_and(self):
        sql, params = compile_filter({"age": {"$gt": 20}, "name": {"$eq": "alice"}})
        assert "AND" in sql
        assert len(params) == 2


class TestErrorCases:
    def test_unknown_operator(self):
        with pytest.raises(InvalidFilterError, match="\\$bad"):
            compile_filter({"x": {"$bad": 1}})

    def test_empty_filter(self):
        sql, params = compile_filter({})
        assert sql == "1=1"
        assert params == []

    def test_none_filter(self):
        sql, params = compile_filter(None)
        assert sql == "1=1"
        assert params == []

    def test_in_with_non_list(self):
        with pytest.raises(InvalidFilterError, match="\\$in"):
            compile_filter({"x": {"$in": "not_a_list"}})

    def test_logical_with_non_list(self):
        with pytest.raises(InvalidFilterError, match="\\$and"):
            compile_filter({"$and": "not_a_list"})

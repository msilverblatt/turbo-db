"""Compile $operator filter dicts into parameterized SQL WHERE clauses."""

from __future__ import annotations

from typing import Any

from turbodb.exceptions import InvalidFilterError

__all__ = ["compile_filter"]

_COMPARISON_OPS = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
}

_SET_OPS = {"$in": "IN", "$nin": "NOT IN"}

_LOGICAL_OPS = {"$and", "$or"}


def compile_filter(where: dict[str, Any] | None) -> tuple[str, list]:
    """Compile a filter dict into a SQL WHERE clause and parameter list."""
    if not where:
        return "1=1", []
    clause, params = _compile_node(where)
    return clause, params


def _compile_node(node: dict[str, Any]) -> tuple[str, list]:
    """Recursively compile a filter node."""
    clauses = []
    params: list = []

    for key, value in node.items():
        if key in _LOGICAL_OPS:
            if not isinstance(value, list):
                raise InvalidFilterError(
                    f"Logical operator {key} requires a list, got {type(value).__name__}"
                )
            sub_clauses = []
            for sub_node in value:
                sub_sql, sub_params = _compile_node(sub_node)
                sub_clauses.append(sub_sql)
                params.extend(sub_params)
            joiner = " AND " if key == "$and" else " OR "
            clauses.append(f"({joiner.join(sub_clauses)})")
        elif isinstance(value, dict):
            field_sql, field_params = _compile_field(key, value)
            clauses.append(field_sql)
            params.extend(field_params)
        else:
            # Bare value treated as $eq
            col = f"json_extract(metadata, '$.{key}')"
            clauses.append(f"{col} = ?")
            params.append(value)

    if len(clauses) == 1:
        return clauses[0], params
    return f"({' AND '.join(clauses)})", params


def _compile_field(field: str, ops: dict[str, Any]) -> tuple[str, list]:
    """Compile operator dict for a single field."""
    col = f"json_extract(metadata, '$.{field}')"
    clauses = []
    params: list = []

    for op, value in ops.items():
        if op in _COMPARISON_OPS:
            clauses.append(f"{col} {_COMPARISON_OPS[op]} ?")
            params.append(value)
        elif op in _SET_OPS:
            if not isinstance(value, list):
                raise InvalidFilterError(
                    f"Operator {op} requires a list, got {type(value).__name__}"
                )
            placeholders = ", ".join("?" for _ in value)
            clauses.append(f"{col} {_SET_OPS[op]} ({placeholders})")
            params.extend(value)
        else:
            raise InvalidFilterError(f"Unknown filter operator: {op}")

    if len(clauses) == 1:
        return clauses[0], params
    return f"({' AND '.join(clauses)})", params

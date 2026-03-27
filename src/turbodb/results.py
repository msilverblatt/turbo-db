"""Query result types and format converters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["QueryResult", "to_chroma_format"]


@dataclass(frozen=True, slots=True)
class QueryResult:
    """A single query result."""

    id: str
    score: float
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        return f"QueryResult(id={self.id!r}, score={self.score:.4f}, metadata={self.metadata!r})"


def to_chroma_format(
    results: list[QueryResult],
) -> dict[str, list[list]]:
    """Convert results to ChromaDB column-oriented format."""
    return {
        "ids": [[r.id for r in results]],
        "distances": [[r.score for r in results]],
        "metadatas": [[r.metadata for r in results]],
    }

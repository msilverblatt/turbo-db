"""turbo-db: Lightweight embedded vector database built on TurboQuant."""

from turbodb.collection import Collection
from turbodb.db import TurboDB
from turbodb.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    InvalidFilterError,
    TurboDBError,
)
from turbodb.results import QueryResult

__all__ = [
    "Collection",
    "CollectionExistsError",
    "CollectionNotFoundError",
    "DimensionMismatchError",
    "InvalidFilterError",
    "QueryResult",
    "TurboDB",
    "TurboDBError",
]

__version__ = "0.1.0"

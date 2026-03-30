"""turbo-db: Lightweight embedded vector database built on TurboQuant."""

from turbodb.bm25 import default_tokenizer
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
    "default_tokenizer",
]

__version__ = "0.2.0"

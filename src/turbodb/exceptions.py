"""Custom exception hierarchy for turbo-db."""

__all__ = [
    "CollectionExistsError",
    "CollectionNotFoundError",
    "DimensionMismatchError",
    "InvalidFilterError",
    "TurboDBError",
]


class TurboDBError(Exception):
    """Base exception for all turbo-db errors."""


class CollectionExistsError(TurboDBError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Collection already exists: '{name}'")


class CollectionNotFoundError(TurboDBError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Collection not found: '{name}'")


class DimensionMismatchError(TurboDBError):
    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"Dimension mismatch: collection expects dim={expected}, got dim={got}")


class InvalidFilterError(TurboDBError):
    """Raised when a metadata filter is malformed."""

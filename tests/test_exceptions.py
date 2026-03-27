from turbodb.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    InvalidFilterError,
    TurboDBError,
)


def test_hierarchy():
    assert issubclass(CollectionExistsError, TurboDBError)
    assert issubclass(CollectionNotFoundError, TurboDBError)
    assert issubclass(DimensionMismatchError, TurboDBError)
    assert issubclass(InvalidFilterError, TurboDBError)


def test_collection_exists_message():
    err = CollectionExistsError("docs")
    assert "docs" in str(err)
    assert err.name == "docs"


def test_collection_not_found_message():
    err = CollectionNotFoundError("docs")
    assert "docs" in str(err)
    assert err.name == "docs"


def test_dimension_mismatch_message():
    err = DimensionMismatchError(384, 768)
    assert "384" in str(err)
    assert "768" in str(err)
    assert err.expected == 384
    assert err.got == 768


def test_invalid_filter_message():
    err = InvalidFilterError("unknown operator $bad")
    assert "$bad" in str(err)

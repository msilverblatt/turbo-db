from turbodb.results import QueryResult, to_chroma_format


def test_query_result_attributes():
    r = QueryResult(id="doc1", score=0.95, metadata={"k": "v"})
    assert r.id == "doc1"
    assert r.score == 0.95
    assert r.metadata == {"k": "v"}


def test_query_result_repr():
    r = QueryResult(id="doc1", score=0.95, metadata={})
    text = repr(r)
    assert "doc1" in text
    assert "0.95" in text


def test_to_chroma_format_single_query():
    results = [
        QueryResult(id="a", score=0.9, metadata={"x": 1}),
        QueryResult(id="b", score=0.8, metadata={"x": 2}),
    ]
    chroma = to_chroma_format(results)
    assert chroma["ids"] == [["a", "b"]]
    assert chroma["distances"] == [[0.9, 0.8]]
    assert chroma["metadatas"] == [[{"x": 1}, {"x": 2}]]


def test_to_chroma_format_empty():
    chroma = to_chroma_format([])
    assert chroma["ids"] == [[]]
    assert chroma["distances"] == [[]]
    assert chroma["metadatas"] == [[]]

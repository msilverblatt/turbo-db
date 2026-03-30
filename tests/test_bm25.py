# tests/test_bm25.py
import json
import math

import pytest

from turbodb.bm25 import BM25Index, default_tokenizer


class TestDefaultTokenizer:
    def test_basic(self):
        assert default_tokenizer("Hello World") == ["hello", "world"]

    def test_drops_short_tokens(self):
        assert default_tokenizer("I am a cat") == ["am", "cat"]

    def test_splits_on_punctuation(self):
        assert default_tokenizer("it's a well-known fact!") == [
            "it", "well", "known", "fact",
        ]

    def test_numbers_preserved(self):
        assert default_tokenizer("version 42 released") == ["version", "42", "released"]

    def test_empty_string(self):
        assert default_tokenizer("") == []


class TestBM25Index:
    @pytest.fixture
    def index(self, tmp_path):
        return BM25Index(tmp_path / "col")

    def test_empty_search(self, index):
        assert index.search("hello") == {}
        assert index.num_docs == 0
        assert index.avg_doc_length == 0.0

    def test_add_and_search(self, index):
        index.add(
            positions=[0, 1, 2],
            documents=[
                "the quick brown fox jumps over the lazy dog",
                "a quick brown cat sits on the mat",
                "the dog chased the cat around the yard",
            ],
        )
        assert index.num_docs == 3

        scores = index.search("quick fox")
        assert len(scores) > 0
        # Doc 0 has both "quick" and "fox", should score highest
        assert scores[0] > scores.get(1, 0.0)

    def test_search_with_valid_positions(self, index):
        index.add(
            positions=[0, 1, 2],
            documents=[
                "the quick brown fox",
                "the quick brown cat",
                "the lazy brown dog",
            ],
        )
        scores = index.search("quick", valid_positions={1, 2})
        assert 0 not in scores
        assert 1 in scores

    def test_search_no_match(self, index):
        index.add(positions=[0], documents=["hello world"])
        scores = index.search("xyzzy")
        assert scores == {}

    def test_remove(self, index):
        index.add(positions=[0, 1], documents=["hello world", "world peace"])
        assert index.num_docs == 2
        index.remove([0])
        assert index.num_docs == 1
        scores = index.search("hello")
        assert 0 not in scores

    def test_remove_nonexistent(self, index):
        index.add(positions=[0], documents=["hello"])
        index.remove([99])  # should not raise
        assert index.num_docs == 1

    def test_rebuild(self, index):
        index.add(positions=[0, 1], documents=["hello world", "world peace"])
        index.rebuild(positions=[0], documents=["brand new document"])
        assert index.num_docs == 1
        scores = index.search("brand")
        assert 0 in scores

    def test_remap_positions(self, index):
        index.add(positions=[5, 10], documents=["hello world", "world peace"])
        index.remap_positions({5: 0, 10: 1})
        assert index.num_docs == 2
        scores = index.search("hello")
        assert 0 in scores
        assert 5 not in scores

    def test_persistence(self, tmp_path):
        path = tmp_path / "col"
        idx1 = BM25Index(path)
        idx1.add(positions=[0, 1], documents=["alpha beta", "gamma delta"])
        assert idx1.num_docs == 2

        # Load from same path — should restore from cache
        idx2 = BM25Index(path)
        assert idx2.num_docs == 2
        scores = idx2.search("alpha")
        assert 0 in scores

    def test_corrupted_cache(self, tmp_path):
        path = tmp_path / "col"
        path.mkdir(parents=True)
        (path / "bm25_index.json").write_text("not valid json{{{")
        idx = BM25Index(path)
        assert idx.num_docs == 0  # graceful degradation

    def test_idf_weighting(self, index):
        """Terms that appear in fewer documents should get higher IDF."""
        index.add(
            positions=[0, 1, 2],
            documents=[
                "common rare",
                "common common",
                "common common common",
            ],
        )
        scores = index.search("rare")
        # Only doc 0 contains "rare" — it should be the only result
        assert set(scores.keys()) == {0}

    def test_custom_tokenizer(self, tmp_path):
        def upper_tokenizer(text):
            return text.upper().split()

        idx = BM25Index(tmp_path / "col", tokenizer=upper_tokenizer)
        idx.add(positions=[0], documents=["hello world"])
        scores = idx.search("HELLO")
        assert 0 in scores

    def test_bm25_params(self, tmp_path):
        idx = BM25Index(tmp_path / "col", k1=2.0, b=0.5)
        idx.add(positions=[0], documents=["test document with some words"])
        scores = idx.search("test")
        assert 0 in scores

"""Tests for the embeddings module: EmbeddingStore + semantic FileMemory."""

import math
import unittest
from unittest.mock import MagicMock, patch, call

from multi_agent_coder.embedding_store import (
    EmbeddingStore,
    _cosine_similarity,
    _chunk_text,
)
from multi_agent_coder.llm.base import LLMClient


# ---------------------------------------------------------------------------
# Helper: deterministic "embeddings" for testing
# ---------------------------------------------------------------------------

def _vec(*components):
    """Shorthand to build a list of floats."""
    return list(components)


# ---------------------------------------------------------------------------
# Cosine similarity unit tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        v = _vec(1, 0, 0)
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        a = _vec(1, 0, 0)
        b = _vec(0, 1, 0)
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)

    def test_opposite_vectors(self):
        a = _vec(1, 0)
        b = _vec(-1, 0)
        self.assertAlmostEqual(_cosine_similarity(a, b), -1.0)

    def test_empty_vectors(self):
        self.assertEqual(_cosine_similarity([], []), 0.0)

    def test_mismatched_lengths(self):
        self.assertEqual(_cosine_similarity([1, 2], [1]), 0.0)

    def test_zero_vector(self):
        self.assertEqual(_cosine_similarity([0, 0], [1, 2]), 0.0)


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestChunking(unittest.TestCase):

    def test_short_text_no_chunking(self):
        text = "short"
        chunks = _chunk_text(text, chunk_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_text_is_chunked(self):
        text = "A" * 300
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be <= chunk_size
        for c in chunks:
            self.assertLessEqual(len(c), 100)

    def test_chunks_cover_full_text(self):
        """Concatenating non-overlapping parts should cover the full text."""
        text = "ABCDEFGHIJ" * 50  # 500 chars
        chunks = _chunk_text(text, chunk_size=100, overlap=0)
        reassembled = "".join(chunks)
        self.assertEqual(reassembled, text)


# ---------------------------------------------------------------------------
# EmbeddingStore tests
# ---------------------------------------------------------------------------

class TestEmbeddingStore(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock(spec=LLMClient)
        self.store = EmbeddingStore(self.mock_llm, embed_model="test-model")

    def test_add_and_search(self):
        """Files with similar embeddings to the query should rank higher."""
        # Each file gets a different embedding direction
        self.mock_llm.generate_embedding.side_effect = [
            _vec(1, 0, 0),   # file_a embedding
            _vec(0, 1, 0),   # file_b embedding
            _vec(0.9, 0.1, 0),  # query embedding (close to file_a)
        ]

        self.store.add("file_a.py", "def foo(): pass")
        self.store.add("file_b.py", "class Bar: ...")

        results = self.store.search("implement foo", top_k=2)

        self.assertEqual(len(results), 2)
        # file_a should rank first (closer to query)
        self.assertEqual(results[0][0], "file_a.py")
        self.assertGreater(results[0][1], results[1][1])

    def test_add_returns_false_on_embed_failure(self):
        self.mock_llm.generate_embedding.return_value = []
        ok = self.store.add("bad.py", "content")
        self.assertFalse(ok)

    def test_search_empty_store(self):
        self.mock_llm.generate_embedding.return_value = _vec(1, 0)
        results = self.store.search("query")
        self.assertEqual(results, [])

    def test_search_query_embed_fails(self):
        self.mock_llm.generate_embedding.side_effect = [
            _vec(1, 0),  # add
            [],          # search query fails
        ]
        self.store.add("a.py", "x")
        results = self.store.search("q")
        self.assertEqual(results, [])

    def test_size_property(self):
        self.mock_llm.generate_embedding.return_value = _vec(1, 0)
        self.assertEqual(self.store.size, 0)
        self.store.add("a.py", "x")
        self.assertEqual(self.store.size, 1)

    def test_has_key(self):
        self.mock_llm.generate_embedding.return_value = _vec(1, 0)
        self.assertFalse(self.store.has_key("a.py"))
        self.store.add("a.py", "x")
        self.assertTrue(self.store.has_key("a.py"))


# ---------------------------------------------------------------------------
# FileMemory integration tests (semantic vs fallback)
# ---------------------------------------------------------------------------

class TestFileMemorySemantic(unittest.TestCase):
    """FileMemory should use the EmbeddingStore when available."""

    def test_semantic_context_used_when_store_present(self):
        mock_llm = MagicMock(spec=LLMClient)
        store = EmbeddingStore(mock_llm, embed_model="test")

        # Return distinct embeddings for files and query
        mock_llm.generate_embedding.side_effect = [
            _vec(1, 0),    # utils.py embed
            _vec(0, 1),    # main.py embed
            _vec(0.9, 0.1),  # query embed
        ]

        from multi_agent_coder.orchestrator import FileMemory
        mem = FileMemory(embedding_store=store, top_k=2)
        mem.update({"src/utils.py": "def helper(): pass", "src/main.py": "import utils"})

        context = mem.related_context("implement a helper function")
        # utils.py should appear first (more relevant)
        self.assertIn("utils.py", context)

    def test_fallback_without_store(self):
        from multi_agent_coder.orchestrator import FileMemory
        mem = FileMemory()  # no store
        mem.update({"src/utils.py": "def helper(): pass", "src/main.py": "# app"})

        # Only utils.py basename appears in step text
        context = mem.related_context("fix utils.py")
        self.assertIn("utils.py", context)
        self.assertNotIn("main.py", context)


if __name__ == "__main__":
    unittest.main()

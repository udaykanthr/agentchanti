"""Concurrency test for SQLiteEmbeddingStore.

The store is written from FileMemory's background embed threads while
pipeline wave threads read — unsynchronized access to the shared sqlite3
connection caused 'bad parameter or other API misuse' errors and silent
native crashes mid-run. This test hammers one store instance from many
threads; it fails (or crashes) without the connection lock.
"""

import os
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

from agentchanti.embedding_store_sqlite import SQLiteEmbeddingStore


class TestSQLiteEmbeddingStoreConcurrency(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="embconc_")
        llm = MagicMock()
        llm.generate_embedding.return_value = [0.1, 0.2, 0.3]
        self.store = SQLiteEmbeddingStore(
            llm, db_path=os.path.join(self.dir, "emb.db"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_parallel_writers_and_readers(self):
        errors: list[BaseException] = []
        n_threads, n_ops = 8, 25

        def worker(tid: int):
            try:
                for i in range(n_ops):
                    # Mix of distinct keys (inserts), shared keys
                    # (replace + stale-delete), and cache-hit reads.
                    self.store.add(f"file_{tid}_{i}.py", f"content {tid} {i}")
                    self.store.add("shared.py", f"shared content {i % 3}")
                    self.store.add(f"file_{tid}_{i}.py", f"content {tid} {i}")
            except BaseException as exc:  # noqa: BLE001 — collect everything
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(errors, [], f"thread errors: {errors}")
        # Every distinct key must be retrievable from the persistent cache
        fresh = SQLiteEmbeddingStore(
            MagicMock(), db_path=os.path.join(self.dir, "emb.db"))
        try:
            import hashlib
            content_hash = hashlib.sha256(
                b"content 0 0").hexdigest()[:16]
            cached = fresh._load_cached("file_0_0.py", content_hash)
            self.assertIsNotNone(cached)
            self.assertEqual(cached[0][1], [0.1, 0.2, 0.3])
        finally:
            fresh.close()


if __name__ == "__main__":
    unittest.main()

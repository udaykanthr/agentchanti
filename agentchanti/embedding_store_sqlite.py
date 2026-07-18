"""
Persistent Embedding Store — SQLite-backed vector cache.

Extends the in-memory EmbeddingStore to persist embeddings to disk,
avoiding recomputation when file content hasn't changed.
"""

import hashlib
import json
import os
import sqlite3
import threading
from typing import List, Optional, Tuple

from .embedding_store import EmbeddingStore, _chunk_text
from .cli_display import log
from .llm.base import LLMClient


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    key         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_idx   INTEGER NOT NULL,
    vector      TEXT NOT NULL,
    PRIMARY KEY (key, content_hash, chunk_idx)
)
"""

_DELETE_STALE = "DELETE FROM embeddings WHERE key = ? AND content_hash != ?"
_INSERT = "INSERT OR REPLACE INTO embeddings (key, content_hash, chunk_idx, vector) VALUES (?, ?, ?, ?)"
_SELECT = "SELECT chunk_idx, vector FROM embeddings WHERE key = ? AND content_hash = ? ORDER BY chunk_idx"


class SQLiteEmbeddingStore(EmbeddingStore):
    """Persistent embedding store backed by SQLite.

    Embeddings are cached by ``(key, content_hash)`` so that unchanged
    files are never re-embedded.  Changed files have their stale entries
    replaced automatically.
    """

    def __init__(self, llm_client: LLMClient, embed_model: Optional[str] = None,
                 db_path: str = ".agentchanti/embeddings.db"):
        super().__init__(llm_client, embed_model)
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # This store is touched from several threads: FileMemory's
        # background _bg_embed workers write while pipeline wave threads
        # read. A single sqlite3 connection shared across threads (even
        # with check_same_thread=False) is officially unsupported and can
        # corrupt the interpreter heap — a Windows fast-fail (0xc0000409)
        # that faulthandler cannot catch, seen as a silent mid-run exit.
        #
        # Instead each thread gets its OWN connection via thread-local
        # storage. WAL mode + busy_timeout let independent connections
        # coordinate safely at the SQLite level, and the default
        # check_same_thread=True turns any accidental cross-thread use
        # into a loud Python exception rather than a native crash.
        # _db_lock still serialises writers to avoid SQLITE_BUSY churn
        # and to keep the in-memory _vectors dict consistent.
        self._db_lock = threading.Lock()
        self._local = threading.local()
        conn = self._get_conn()
        with self._db_lock:
            conn.execute(_CREATE_TABLE)
            conn.commit()
        log.debug(f"[SQLiteEmbeddingStore] Opened {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Return this thread's own SQLite connection (lazily opened)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def add(self, key: str, text: str) -> bool:
        """Embed text and cache persistently. Skips API call if cached."""
        content_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

        # Check SQLite cache first
        cached = self._load_cached(key, content_hash)
        if cached:
            self._vectors[key] = cached
            log.debug(f"[SQLiteEmbeddingStore] Cache hit for '{key}' (hash={content_hash})")
            return True

        # Compute embeddings via API
        chunks = _chunk_text(text)
        stored_any = False
        chunk_vectors: List[Tuple[int, List[float]]] = []

        for idx, chunk in enumerate(chunks):
            vec = self.llm_client.generate_embedding(chunk, model=self.embed_model)
            if vec and all(v is not None for v in vec):
                chunk_vectors.append((idx, vec))
                stored_any = True

        if stored_any:
            self._vectors[key] = chunk_vectors
            self._save_vectors(key, content_hash, chunk_vectors)
            self._failed_keys.discard(key)
            log.debug(f"[SQLiteEmbeddingStore] Stored {len(chunk_vectors)} chunk(s) "
                      f"for '{key}' (hash={content_hash})")
        else:
            if key not in self._failed_keys:
                log.warning(f"[SQLiteEmbeddingStore] Failed to embed '{key}'")
                self._failed_keys.add(key)

        return stored_any

    def _load_cached(self, key: str, content_hash: str) -> List[Tuple[int, List[float]]] | None:
        """Load cached embedding vectors from SQLite."""
        try:
            with self._db_lock:
                cursor = self._get_conn().execute(_SELECT, (key, content_hash))
                rows = cursor.fetchall()
            if not rows:
                return None
            return [(row[0], json.loads(row[1])) for row in rows]
        except (sqlite3.Error, json.JSONDecodeError) as e:
            log.warning(f"[SQLiteEmbeddingStore] Cache read error: {e}")
            return None

    def _save_vectors(self, key: str, content_hash: str,
                      vectors: List[Tuple[int, List[float]]]):
        """Persist embedding vectors to SQLite, replacing stale entries."""
        try:
            with self._db_lock:
                conn = self._get_conn()
                conn.execute(_DELETE_STALE, (key, content_hash))
                for chunk_idx, vec in vectors:
                    vec_json = json.dumps(vec)
                    conn.execute(
                        _INSERT, (key, content_hash, chunk_idx, vec_json))
                conn.commit()
        except sqlite3.Error as e:
            log.warning(f"[SQLiteEmbeddingStore] Cache write error: {e}")

    def close(self):
        """Close this thread's SQLite connection.

        Only the calling thread's connection is closed; connections owned
        by other (daemon) threads are released when the process exits.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._local.conn = None

"""
Error dictionary — SQLite wrapper for the global errors.db.

Provides fast error-type lookups, regex-pattern matching, and tag-based
fuzzy matching for error messages.  Reuses the SQLite connection patterns
established in ``kb.local.manifest``.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS errors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    error_type    TEXT NOT NULL,
    language      TEXT NOT NULL,
    pattern       TEXT NOT NULL,
    cause         TEXT,
    fix_template  TEXT NOT NULL,
    severity      TEXT DEFAULT 'error',
    tags          TEXT,
    source        TEXT DEFAULT 'core'
);

CREATE INDEX IF NOT EXISTS idx_language   ON errors(language);
CREATE INDEX IF NOT EXISTS idx_error_type ON errors(error_type);

CREATE TABLE IF NOT EXISTS content_fixes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    file_glob       TEXT NOT NULL,
    content_pattern TEXT NOT NULL,
    replacement     TEXT NOT NULL DEFAULT '',
    ensure_content  TEXT DEFAULT '',
    flags           TEXT DEFAULT 'MULTILINE',
    collapse_blanks INTEGER DEFAULT 1,
    language        TEXT DEFAULT 'all',
    source          TEXT DEFAULT 'core',
    description     TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_cf_language  ON content_fixes(language);
CREATE INDEX IF NOT EXISTS idx_cf_file_glob ON content_fixes(file_glob);
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ErrorFix:
    """A single error-to-fix mapping."""

    error_type: str
    language: str
    pattern: str = ""
    cause: str = ""
    fix_template: str = ""
    severity: str = "error"
    tags: str = ""
    source: str = "core"

    def tag_list(self) -> list[str]:
        """Return tags split into a list."""
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(",") if t.strip()]


@dataclass
class ContentFix:
    """A deterministic file-content replacement rule.

    Applied after LLM code generation to fix known patterns that
    LLMs persistently emit from stale training data (e.g. Tailwind
    CSS v3 directives in a v4 project).

    Stored in the ``content_fixes`` table of ``errors.db`` and loaded
    once per pipeline run.  Users add new rules by editing the seeder
    and re-running ``seed()``.
    """

    name: str               # unique rule id, e.g. "tailwind-v4-directives"
    file_glob: str           # fnmatch pattern, e.g. "*.css"
    content_pattern: str     # regex to search for in file content
    replacement: str = ""    # replacement string (supports \\1 backrefs)
    ensure_content: str = "" # if set, prepend when not already present
    flags: str = "MULTILINE" # comma-separated re flag names
    collapse_blanks: bool = True
    language: str = "all"
    source: str = "core"
    description: str = ""

    def compiled_flags(self) -> int:
        """Convert the *flags* string to ``re`` module constants."""
        flag_map = {
            "MULTILINE": re.MULTILINE,
            "IGNORECASE": re.IGNORECASE,
            "DOTALL": re.DOTALL,
        }
        result = 0
        for f in self.flags.split(","):
            f = f.strip().upper()
            if f in flag_map:
                result |= flag_map[f]
        return result


# ---------------------------------------------------------------------------
# ErrorDict
# ---------------------------------------------------------------------------

class ErrorDict:
    """
    SQLite-backed error dictionary for the global knowledge base.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if absent.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        """Yield a connected SQLite connection with WAL mode."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create the errors table and indexes."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        error_message: str,
        language: Optional[str] = None,
    ) -> list[ErrorFix]:
        """
        Look up fixes for *error_message*.

        Matching strategy (ranked):
        1. Exact ``error_type`` match
        2. Regex ``pattern`` match against the full error message
        3. Tag-based fuzzy match (any word in the message matches a tag)

        Parameters
        ----------
        error_message:
            The error text to look up.
        language:
            Optional language filter (e.g. ``"python"``).  If None, all
            languages (including ``"all"``) are searched.

        Returns
        -------
        list[ErrorFix]
            Matched results ranked by match quality.
        """
        exact: list[ErrorFix] = []
        regex: list[ErrorFix] = []
        fuzzy: list[ErrorFix] = []

        with self._connect() as conn:
            rows = self._fetch_candidates(conn, language)

        error_lower = error_message.lower()
        error_words = set(re.findall(r"[a-z_]+", error_lower))

        for row in rows:
            ef = self._row_to_errorfix(row)

            # 1. Exact error_type match
            if ef.error_type.lower() in error_lower:
                exact.append(ef)
                continue

            # 2. Regex pattern match
            if ef.pattern:
                try:
                    if re.search(ef.pattern, error_message, re.IGNORECASE):
                        regex.append(ef)
                        continue
                except re.error:
                    pass

            # 3. Tag-based fuzzy match
            if ef.tags:
                tag_set = {t.strip().lower() for t in ef.tags.split(",")}
                if tag_set & error_words:
                    fuzzy.append(ef)

        # Deduplicate while preserving rank order
        seen: set[str] = set()
        results: list[ErrorFix] = []
        for ef in exact + regex + fuzzy:
            key = f"{ef.error_type}:{ef.language}"
            if key not in seen:
                seen.add(key)
                results.append(ef)

        return results

    def add(self, error: ErrorFix) -> None:
        """
        Insert a single error-fix record.

        Parameters
        ----------
        error:
            The error-fix mapping to insert.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO errors
                    (error_type, language, pattern, cause, fix_template,
                     severity, tags, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    error.error_type,
                    error.language,
                    error.pattern,
                    error.cause,
                    error.fix_template,
                    error.severity,
                    error.tags,
                    error.source,
                ),
            )

    def bulk_insert(self, errors: list[ErrorFix]) -> None:
        """
        Insert multiple error-fix records in a single transaction.

        Parameters
        ----------
        errors:
            List of error-fix mappings to insert.
        """
        if not errors:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO errors
                    (error_type, language, pattern, cause, fix_template,
                     severity, tags, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.error_type,
                        e.language,
                        e.pattern,
                        e.cause,
                        e.fix_template,
                        e.severity,
                        e.tags,
                        e.source,
                    )
                    for e in errors
                ],
            )

    def count(self, language: Optional[str] = None) -> int:
        """
        Return the total number of error records.

        Parameters
        ----------
        language:
            Optional filter.  If given, count only that language.
        """
        with self._connect() as conn:
            if language:
                row = conn.execute(
                    "SELECT COUNT(*) FROM errors WHERE language = ?",
                    (language,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM errors").fetchone()
        return row[0] if row else 0

    def count_by_language(self) -> dict[str, int]:
        """Return a dict mapping language to error count."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT language, COUNT(*) AS cnt FROM errors GROUP BY language"
            ).fetchall()
        return {r["language"]: r["cnt"] for r in rows}

    def clear(self) -> None:
        """Delete all error records."""
        with self._connect() as conn:
            conn.execute("DELETE FROM errors")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_candidates(
        self, conn: sqlite3.Connection, language: Optional[str]
    ) -> list[sqlite3.Row]:
        """Fetch error rows filtered by language (includes 'all')."""
        if language:
            return conn.execute(
                "SELECT * FROM errors WHERE language = ? OR language = 'all'",
                (language.lower(),),
            ).fetchall()
        return conn.execute("SELECT * FROM errors").fetchall()

    @staticmethod
    def _row_to_errorfix(row: sqlite3.Row) -> ErrorFix:
        """Convert a SQLite Row to an ErrorFix dataclass."""
        return ErrorFix(
            error_type=row["error_type"],
            language=row["language"],
            pattern=row["pattern"],
            cause=row["cause"] or "",
            fix_template=row["fix_template"],
            severity=row["severity"] or "error",
            tags=row["tags"] or "",
            source=row["source"] or "core",
        )

    # ------------------------------------------------------------------
    # Content fixes
    # ------------------------------------------------------------------

    def bulk_insert_content_fixes(self, fixes: list[ContentFix]) -> None:
        """Insert multiple content-fix rules in a single transaction."""
        if not fixes:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO content_fixes
                    (name, file_glob, content_pattern, replacement,
                     ensure_content, flags, collapse_blanks, language,
                     source, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f.name, f.file_glob, f.content_pattern,
                        f.replacement, f.ensure_content, f.flags,
                        1 if f.collapse_blanks else 0, f.language,
                        f.source, f.description,
                    )
                    for f in fixes
                ],
            )

    def get_content_fixes(
        self, language: Optional[str] = None,
    ) -> list[ContentFix]:
        """Return all content-fix rules, optionally filtered by language."""
        with self._connect() as conn:
            if language:
                rows = conn.execute(
                    "SELECT * FROM content_fixes "
                    "WHERE language = ? OR language = 'all'",
                    (language.lower(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM content_fixes"
                ).fetchall()
        return [self._row_to_content_fix(r) for r in rows]

    def clear_content_fixes(self) -> None:
        """Delete all content-fix records."""
        with self._connect() as conn:
            conn.execute("DELETE FROM content_fixes")

    def count_content_fixes(self) -> int:
        """Return total number of content-fix records."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM content_fixes"
            ).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _row_to_content_fix(row: sqlite3.Row) -> ContentFix:
        """Convert a SQLite Row to a ContentFix dataclass."""
        return ContentFix(
            name=row["name"],
            file_glob=row["file_glob"],
            content_pattern=row["content_pattern"],
            replacement=row["replacement"] or "",
            ensure_content=row["ensure_content"] or "",
            flags=row["flags"] or "MULTILINE",
            collapse_blanks=bool(row["collapse_blanks"]),
            language=row["language"] or "all",
            source=row["source"] or "core",
            description=row["description"] or "",
        )

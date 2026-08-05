"""
Unified global knowledge base query interface.

Provides a single ``search()`` method that queries both the SQLite vector
``global_kb`` collection and the SQLite ``errors.db``, returning
ranked results across all categories.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from .error_dict import ErrorDict, ErrorFix, ContentFix

logger = logging.getLogger(__name__)

_GLOBAL_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_GLOBAL_DIR, "core")

# Minimum cosine similarity score for vector search results.
# Results below this threshold are dropped to avoid injecting
# loosely related docs that waste LLM tokens.
MIN_RELEVANCE_SCORE = 0.25


def _language_matches(doc_lang: str, project_lang: str) -> bool:
    """True when a doc tagged *doc_lang* belongs in a *project_lang* project.

    ``language:`` accepts a comma-separated list so one doc can serve a
    family — React material is equally valid for ``javascript`` and
    ``typescript``, and before this it had to claim ``all`` to reach both.
    ``all`` still means "any project".
    """
    langs = {p.strip().lower() for p in (doc_lang or "all").split(",")}
    langs.discard("")
    return (not langs) or ("all" in langs) or (project_lang.lower() in langs)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GlobalKBResult:
    """A single result from the global knowledge base."""

    title: str
    category: str  # "pattern" | "adr" | "doc" | "behavioral" | "error"
    content: str
    file: str
    score: float
    tags: list[str] = field(default_factory=list)
    language: str = "all"


# ---------------------------------------------------------------------------
# GlobalKBStore
# ---------------------------------------------------------------------------

class GlobalKBStore:
    """
    Unified interface over all global KB content.

    Combines:
    - SQLite vector store for semantic search over documents
    - SQLite ``errors.db`` for deterministic error lookups

    Parameters
    ----------
    errors_db_path:
        Path to the errors.db SQLite database.  Defaults to
        ``kb/global/core/errors.db``.
    """

    def __init__(
        self,
        errors_db_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        self._base_dir = base_dir
        if errors_db_path:
            self._errors_db_path = errors_db_path
        elif base_dir:
            self._errors_db_path = os.path.join(base_dir, "core", "errors.db")
        else:
            self._errors_db_path = os.path.join(_CORE_DIR, "errors.db")
        self._error_dict: Optional[ErrorDict] = None
        self._vector_store = None
        # Frontmatter metadata cache: {filepath -> (mtime, {title, tags, language})}
        # Avoids re-reading .md files on every fallback search call.
        self._fm_cache: dict[str, tuple[float, dict]] = {}
        # Query embedding cache: {(query, model) -> vector}
        # The same task description is searched multiple times per run
        # (planning + per-step context); cache avoids redundant API calls.
        self._query_embed_cache: dict[tuple[str, str], list[float]] = {}

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _get_error_dict(self) -> ErrorDict:
        """Return the ErrorDict, creating it if needed."""
        if self._error_dict is None:
            self._error_dict = ErrorDict(self._errors_db_path)
        return self._error_dict

    def _get_vector_store(self):
        """Return the global vector store (lazy)."""
        if self._vector_store is None:
            self._vector_store = _get_global_vector_store()
        return self._vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        language: Optional[str] = None,
        top_k: int = 5,
        api_client=None,
    ) -> list[GlobalKBResult]:
        """
        Semantic search across all global KB content.

        Tries vector search first, then supplements with file-based
        keyword search so that registry files added via ``kb update``
        (not yet embedded) are still discoverable.

        Parameters
        ----------
        query:
            Natural-language search query.
        categories:
            Filter by category (e.g. ``["pattern", "adr"]``).
        language:
            Filter by language (e.g. ``"python"``).
        top_k:
            Maximum number of results.
        api_client:
            LLM client to use for embedding the query.

        Returns
        -------
        list[GlobalKBResult]
            Ranked results.
        """
        results: list[GlobalKBResult] = []
        try:
            if api_client is None:
                raise ValueError("api_client required for vector search")
            results = self._vector_search(query, categories, language, top_k, api_client)
        except Exception as exc:
            logger.warning("Vector search failed, falling back to file search: %s", exc)

        # Merge file-based search results only when vector search left gaps —
        # i.e. un-embedded docs (added via `kb update` but not yet `kb embed`)
        # need to be discoverable.  Skip the scan entirely when vector results
        # already fill top_k to avoid reading all .md files on every call.
        try:
            file_results = self._fallback_file_search(
                query, categories, language, max(0, top_k - len(results)),
            ) if len(results) < top_k else []
            seen = {r.file for r in results}
            added = 0
            for fr in file_results:
                if fr.file not in seen:
                    results.append(fr)
                    seen.add(fr.file)
                    added += 1
            if added:
                logger.debug(
                    "File search added %d result(s): %s",
                    added,
                    [r.title for r in results[-added:]],
                )
            # Re-rank by score and trim to top_k
            if len(results) > top_k:
                results.sort(key=lambda r: r.score, reverse=True)
                results = results[:top_k]
        except Exception:
            pass

        if results:
            logger.debug(
                "KB search returned %d result(s): %s",
                len(results),
                [r.title for r in results],
            )

        return results

    def search_errors(
        self,
        error_message: str,
        language: Optional[str] = None,
    ) -> list[ErrorFix]:
        """
        Search for error fixes.

        Strategy:
        1. Exact/regex lookup via ErrorDict (fast, deterministic)
        2. Semantic search filtered to behavioral docs
        3. Error dict results ranked above semantic results

        Parameters
        ----------
        error_message:
            The error text to look up.
        language:
            Optional language filter.

        Returns
        -------
        list[ErrorFix]
            Matched error-fix records ranked by match quality.
        """
        edict = self._get_error_dict()
        return edict.lookup(error_message, language)

    def get_behavioral_instructions(
        self,
        context: str,
        api_client=None,
        language: Optional[str] = None,
    ) -> list[GlobalKBResult]:
        """
        Retrieve behavioral instructions relevant to *context*.

        Used by Phase 4 context_builder to inject agent instructions.

        Parameters
        ----------
        context:
            Description of the current task or situation.
        api_client:
            LLM client for embedding the query.
        language:
            Project language (e.g. ``"javascript"``).  Passed to the
            fallback file search so that language-specific docs from
            other stacks (e.g. ``Python-Specific Instructions``) are
            filtered out when the project language is known.

        Returns
        -------
        list[GlobalKBResult]
            Behavioral instruction results.
        """
        return self.search(
            query=context,
            categories=["behavioral"],
            top_k=3,
            api_client=api_client,
            language=language,
        )

    def get_content_fixes(
        self,
        language: Optional[str] = None,
    ) -> list[ContentFix]:
        """
        Return all content-fix rules, optionally filtered by language.

        No embedding needed — deterministic data from ``errors.db``.
        """
        edict = self._get_error_dict()
        return edict.get_content_fixes(language)

    def get_by_titles(
        self,
        titles: list[str],
    ) -> list[GlobalKBResult]:
        """
        Fetch KB docs by exact title match (case-insensitive).

        Used when the planner has already declared which KB docs are
        needed for a step via ``kb_docs:`` in the structured plan.
        Avoids a semantic embedding search — pure title lookup against
        the frontmatter cache, falling back to a directory scan when a
        title isn't cached yet.

        Parameters
        ----------
        titles:
            List of KB doc titles declared by the planner.

        Returns
        -------
        list[GlobalKBResult]
            Matched docs in the same order as *titles*.
            Titles with no matching doc are silently skipped.
        """
        if not titles:
            return []

        from .seeder import _REGISTRY_DIR, _parse_frontmatter

        registry_dir = (
            os.path.join(self._base_dir, "registry")
            if self._base_dir else _REGISTRY_DIR
        )
        global_dir = self._base_dir or _GLOBAL_DIR

        wanted = {t.strip().lower() for t in titles if t.strip()}
        if not wanted:
            return []

        category_dirs = {
            "pattern": "patterns",
            "adr": "adrs",
            "doc": "docs",
            "behavioral": "behavioral",
        }

        matched: dict[str, GlobalKBResult] = {}  # title_lower -> result

        for cat, dirname in category_dirs.items():
            cat_dir = os.path.join(registry_dir, dirname)
            if not os.path.isdir(cat_dir):
                continue

            for fname in os.listdir(cat_dir):
                if not fname.endswith(".md"):
                    continue

                filepath = os.path.join(cat_dir, fname)
                try:
                    mtime = os.path.getmtime(filepath)
                except OSError:
                    continue

                cached = self._fm_cache.get(filepath)
                if cached and cached[0] == mtime:
                    meta = cached[1]
                else:
                    try:
                        with open(filepath, encoding="utf-8") as fh:
                            raw = fh.read()
                    except OSError:
                        continue
                    meta = _parse_frontmatter(raw)
                    self._fm_cache[filepath] = (mtime, meta)

                title = meta.get("title", "")
                title_lower = title.lower().strip()
                if title_lower not in wanted or title_lower in matched:
                    continue

                try:
                    with open(filepath, encoding="utf-8") as fh:
                        content = fh.read()
                except OSError:
                    continue

                body = content
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end != -1:
                        body = content[end + 3:].strip()

                tags_str = meta.get("tags", "")
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                rel_path = os.path.relpath(filepath, global_dir).replace("\\", "/")

                matched[title_lower] = GlobalKBResult(
                    title=title,
                    category=cat,
                    content=body[:4000],
                    file=rel_path,
                    score=1.0,
                    tags=tags,
                    language=meta.get("language", "all"),
                )

        # Preserve the planner-declared order
        results = []
        for t in titles:
            r = matched.get(t.strip().lower())
            if r:
                results.append(r)
        return results

    def batch_search(
        self,
        query: str,
        category_limits: dict[str, int],
        language: Optional[str] = None,
        api_client=None,
    ) -> dict[str, list[GlobalKBResult]]:
        """
        Single-embedding vector search split by category.

        Performs ONE embedding call and ONE vector scan, then partitions
        results by category — saving embedding API calls compared to
        multiple ``search()`` invocations.

        After vector search, supplements underfilled categories with
        file-based keyword search so that registry files added via
        ``kb update`` (which are not yet embedded) are still discoverable.

        Parameters
        ----------
        query:
            Natural-language search query.
        category_limits:
            Mapping of category name to max results for that category.
            E.g. ``{"pattern": 3, "adr": 3, "doc": 2, "behavioral": 3}``.
        language:
            Optional language filter.
        api_client:
            LLM client for embedding the query.

        Returns
        -------
        dict[str, list[GlobalKBResult]]
            Results keyed by category.
        """
        # Start with vector search if possible
        result: dict[str, list[GlobalKBResult]] = {}
        try:
            if api_client is None:
                raise ValueError("api_client required for vector search")
            result = self._vector_batch_search(
                query, category_limits, language, api_client,
            )
        except Exception as exc:
            logger.warning(
                "Batch vector search failed, falling back to per-category "
                "file search: %s", exc,
            )

        # Always merge file-based keyword search results so that registry
        # docs added after the last `kb update` are still discoverable.
        for cat, top_k in category_limits.items():
            current = result.get(cat, [])
            try:
                file_results = self._fallback_file_search(
                    query, categories=[cat], language=language,
                    top_k=top_k,
                )
                # Deduplicate by file path
                seen = {r.file for r in current}
                for fr in file_results:
                    if fr.file not in seen:
                        current.append(fr)
                        seen.add(fr.file)
                # Re-rank by score and trim to top_k
                if len(current) > top_k:
                    current.sort(key=lambda r: r.score, reverse=True)
                    current = current[:top_k]
            except Exception:
                pass
            result[cat] = current

        return result

    def _vector_batch_search(
        self,
        query: str,
        category_limits: dict[str, int],
        language: Optional[str],
        api_client,
    ) -> dict[str, list[GlobalKBResult]]:
        """Single embedding + single scan, split results by category."""
        from ..local.embedder import _embed_batch
        from ...config import Config

        cfg = Config.load()
        embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL

        # One embedding call for the query
        vectors = _embed_batch(api_client, [query], embed_model)
        if not vectors:
            return {cat: [] for cat in category_limits}
        query_vector = vectors[0]

        store = self._get_vector_store()

        # Fetch enough results to fill all category buckets.
        # Use a multiplier (not flat offset) so that chunk deduplication
        # doesn't starve categories — a single doc may produce 2-4
        # chunks, all consuming top_k slots before dedup discards them.
        total_top_k = max(sum(category_limits.values()) * 3, 15)
        filters: Optional[dict] = {}
        if language:
            filters["language"] = language
        if not filters:
            filters = None

        raw_results = store.search(
            query_vector=query_vector,
            top_k=total_top_k,
            filters=filters,
        )

        # Partition results into category buckets (deduplicated by file and title)
        buckets: dict[str, list[GlobalKBResult]] = {
            cat: [] for cat in category_limits
        }
        seen_files: set[str] = set()
        seen_titles: set[str] = set()

        for hit in raw_results:
            # Drop results below minimum relevance threshold
            if hit.get("score", 0.0) < MIN_RELEVANCE_SCORE:
                continue

            payload = hit.get("payload", {})
            cat = payload.get("category", "")

            if cat not in category_limits:
                continue
            if len(buckets[cat]) >= category_limits[cat]:
                continue

            # Deduplicate by normalized file path and by title
            rel_file = payload.get("file", "").replace("\\", "/")
            title = payload.get("title", "")
            if rel_file and rel_file in seen_files:
                continue
            if title and title in seen_titles:
                continue
            if rel_file:
                seen_files.add(rel_file)
            if title:
                seen_titles.add(title)

            # Load content from the markdown file on disk
            content = ""
            if rel_file:
                abs_path = os.path.join(_GLOBAL_DIR, rel_file)
                if os.path.isfile(abs_path):
                    try:
                        with open(abs_path, encoding="utf-8") as fh:
                            raw = fh.read()
                        if raw.startswith("---"):
                            end = raw.find("---", 3)
                            if end != -1:
                                raw = raw[end + 3:].strip()
                        content = raw[:4000]
                    except OSError:
                        pass

            # Skip results whose file no longer exists on disk
            if not content:
                continue

            buckets[cat].append(GlobalKBResult(
                title=payload.get("title", ""),
                category=cat,
                content=content,
                file=rel_file,
                score=hit.get("score", 0.0),
                tags=payload.get("tags", []),
                language=payload.get("language", "all"),
            ))

        return buckets

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query: str,
        categories: Optional[list[str]],
        language: Optional[str],
        top_k: int,
        api_client,
    ) -> list[GlobalKBResult]:
        """Perform semantic search via the vector store."""
        from ..local.embedder import _embed_batch
        from ...config import Config

        cfg = Config.load()
        embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL

        cache_key = (query, embed_model or "")
        if cache_key in self._query_embed_cache:
            query_vector = self._query_embed_cache[cache_key]
        else:
            vectors = _embed_batch(api_client, [query], embed_model)
            if not vectors:
                return []
            query_vector = vectors[0]
            self._query_embed_cache[cache_key] = query_vector

        store = self._get_vector_store()

        # Build filters
        filters: Optional[dict] = {}
        if categories and len(categories) == 1:
            filters["category"] = categories[0]
        if language:
            filters["language"] = language
        if not filters:
            filters = None

        # Overfetch to compensate for chunk dedup and multi-category
        # post-filtering — a single doc may produce 2-4 chunks that
        # all consume raw result slots before dedup discards them.
        raw_results = store.search(
            query_vector=query_vector,
            top_k=max(top_k * 3, 15),
            filters=filters,
        )

        results: list[GlobalKBResult] = []
        seen_files: set[str] = set()
        seen_titles: set[str] = set()
        for hit in raw_results:
            # Drop results below minimum relevance threshold
            if hit.get("score", 0.0) < MIN_RELEVANCE_SCORE:
                continue

            payload = hit.get("payload", {})
            cat = payload.get("category", "")

            # Category filter (multi-category)
            if categories and len(categories) > 1 and cat not in categories:
                continue

            # Deduplicate by normalized file path and by title
            rel_file = payload.get("file", "").replace("\\", "/")
            title = payload.get("title", "")
            if rel_file and rel_file in seen_files:
                continue
            if title and title in seen_titles:
                continue
            if rel_file:
                seen_files.add(rel_file)
            if title:
                seen_titles.add(title)

            # Load content from the markdown file on disk
            content = ""
            if rel_file:
                abs_path = os.path.join(_GLOBAL_DIR, rel_file)
                if os.path.isfile(abs_path):
                    try:
                        with open(abs_path, encoding="utf-8") as fh:
                            raw = fh.read()
                        # Strip YAML frontmatter
                        if raw.startswith("---"):
                            end = raw.find("---", 3)
                            if end != -1:
                                raw = raw[end + 3:].strip()
                        content = raw[:4000]
                    except OSError:
                        pass

            # Skip results whose file no longer exists on disk — don't
            # waste a top_k slot on a stale vector-store entry.
            if not content:
                continue

            results.append(GlobalKBResult(
                title=payload.get("title", ""),
                category=cat,
                content=content,
                file=rel_file,
                score=hit.get("score", 0.0),
                tags=payload.get("tags", []),
                language=payload.get("language", "all"),
            ))

        return results

    # ------------------------------------------------------------------
    # Fallback file search (offline / no vector store)
    # ------------------------------------------------------------------

    def _fallback_file_search(
        self,
        query: str,
        categories: Optional[list[str]],
        language: Optional[str],
        top_k: int,
    ) -> list[GlobalKBResult]:
        """
        Keyword-based file search for un-embedded registry docs.

        Two-pass approach:
        1. Score all docs using cached frontmatter metadata (title + tags) —
           no file I/O for docs already in ``_fm_cache``.
        2. Read full body only for docs that scored above zero.

        The metadata cache (``_fm_cache``) stores ``{filepath: (mtime, meta)}``
        and invalidates individual entries when the file changes on disk.
        """
        if top_k <= 0:
            return []

        from .seeder import _REGISTRY_DIR, _parse_frontmatter

        registry_dir = (
            os.path.join(self._base_dir, "registry")
            if self._base_dir else _REGISTRY_DIR
        )
        global_dir = self._base_dir or _GLOBAL_DIR

        import re as _re
        # Strip punctuation so "React," → "react", "vite:" → "vite"
        query_words = set(
            w for w in (
                _re.sub(r'[^\w]', '', tok)
                for tok in query.lower().split()
            )
            if len(w) >= 2
        )
        scored: list[tuple[float, str, str, dict]] = []  # (score, filepath, cat, meta)

        category_dirs = {
            "pattern": "patterns",
            "adr": "adrs",
            "doc": "docs",
            "behavioral": "behavioral",
        }

        # Pass 1: score using cached frontmatter — zero file reads for warm cache
        for cat, dirname in category_dirs.items():
            if categories and cat not in categories:
                continue

            cat_dir = os.path.join(registry_dir, dirname)
            if not os.path.isdir(cat_dir):
                continue

            for fname in os.listdir(cat_dir):
                if not fname.endswith(".md"):
                    continue

                filepath = os.path.join(cat_dir, fname)
                try:
                    mtime = os.path.getmtime(filepath)
                except OSError:
                    continue

                # Serve from cache when file hasn't changed
                cached = self._fm_cache.get(filepath)
                if cached and cached[0] == mtime:
                    meta = cached[1]
                else:
                    # Read only to populate / refresh cache entry
                    try:
                        with open(filepath, encoding="utf-8") as fh:
                            raw = fh.read()
                    except OSError:
                        continue
                    meta = _parse_frontmatter(raw)
                    self._fm_cache[filepath] = (mtime, meta)

                doc_lang = meta.get("language", "all")
                if language and not _language_matches(doc_lang, language):
                    continue

                title_lower = meta.get("title", "").lower()
                tags_lower = meta.get("tags", "").lower()
                title_compressed = _re.sub(r'[\s\-_]+', '', title_lower)
                title_words = set(_re.findall(r'\w+', title_lower))
                tag_words = set(_re.findall(r'\w+', tags_lower))

                title_hits = sum(
                    3.0 for w in query_words
                    if w in title_words or (len(w) >= 4 and w in title_compressed)
                )
                tag_hits = sum(
                    2.0 for w in query_words if w in tag_words
                )

                # language="all" docs require a title/tag match to avoid
                # generic false-positives on words like "install", "run".
                if doc_lang == "all" and title_hits == 0 and tag_hits == 0:
                    continue

                # Metadata-only score (body hits added in pass 2)
                meta_score = (title_hits + tag_hits) / max(len(query_words), 1)
                if meta_score > 0:
                    scored.append((meta_score, filepath, cat, meta))

        if not scored:
            return []

        # Pass 2: read body only for docs that passed metadata scoring
        scored.sort(key=lambda x: -x[0])
        results: list[tuple[float, GlobalKBResult]] = []

        for meta_score, filepath, cat, meta in scored:
            try:
                with open(filepath, encoding="utf-8") as fh:
                    content = fh.read()
            except OSError:
                continue

            # Strip frontmatter
            body = content
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    body = content[end + 3:].strip()

            content_lower = content.lower()
            body_hits = sum(1.0 for w in query_words if w in content_lower)
            score = meta_score + body_hits / max(len(query_words), 1)

            tags_str = meta.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            rel_path = os.path.relpath(filepath, global_dir).replace("\\", "/")

            results.append((score, GlobalKBResult(
                title=meta.get("title", os.path.basename(filepath)),
                category=cat,
                content=body[:4000],
                file=rel_path,
                score=score,
                tags=tags,
                language=meta.get("language", "all"),
            )))

        results.sort(key=lambda x: -x[0])
        return [r for _, r in results[:top_k]]


# ---------------------------------------------------------------------------
# Global Vector Store Factory
# ---------------------------------------------------------------------------

def _get_global_vector_store():
    from ...config import Config
    cfg = Config.load()
    backend = cfg.KB_VECTOR_BACKEND
    
    if backend == "qdrant":
        # Legacy: Qdrant backend is no longer supported.
        # Fall through to SQLite.
        logger.warning("Qdrant backend is no longer supported, using SQLite.")

    # Fallback / Default: SQLite
    from ..local.sqlite_vector_store import SQLiteVectorStore
    db_path = os.path.join(_CORE_DIR, "global_kb.db")
    return SQLiteVectorStore(project_root="", db_path=db_path)

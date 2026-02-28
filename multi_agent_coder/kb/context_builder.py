"""
Context builder — single entry point for all KB context injection.

Phase 4: Gathers context from the Local Semantic KB (Phase 2), Code
Graph (Phase 1), Global KB store (Phase 3), and Error Dictionary
(Phase 3), and formats it for injection into the LLM prompt.

``api.py`` and ``pipeline.py`` interact only with this module —
they never import searcher, graph, or store directly.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent detection keywords
# ---------------------------------------------------------------------------

_ERROR_KEYWORDS = frozenset({
    "error", "exception", "failed", "traceback", "undefined", "null",
    "crash", "fix", "debug", "not working", "bug", "broken",
})

_REVIEW_KEYWORDS = frozenset({
    "review", "refactor", "improve", "clean", "optimize", "pattern",
    "quality", "lint", "style",
})

# File extension → language mapping (subset for fast detection)
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KBContext:
    """Aggregated KB context ready for prompt injection."""

    local_symbols: list = field(default_factory=list)
    related_symbols: list[dict] = field(default_factory=list)
    error_fixes: list = field(default_factory=list)
    global_patterns: list = field(default_factory=list)
    behavioral_instructions: list = field(default_factory=list)
    token_count: int = 0
    kb_available: bool = False
    sources_used: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Gathers and formats KB context for injection into the LLM prompt.

    Lazily initialises Phase 1/2/3 components on first use so that
    importing this module is always cheap.

    Parameters
    ----------
    project_root:
        Absolute path to the project root.  Defaults to ``os.getcwd()``.
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        vector_backend: str = "local",
    ) -> None:
        self._project_root = os.path.abspath(project_root or os.getcwd())
        self._vector_backend = vector_backend
        self._searcher = None
        self._graph = None
        self._global_store = None
        self._initialised = False

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_local(self) -> bool:
        """
        Try to load the local KB (graph + searcher).

        Returns True if the local index is available.
        """
        if self._graph is not None:
            return True

        try:
            from .local.indexer import Indexer, _manifest_path, read_meta

            meta = read_meta(self._project_root)
            if meta is None:
                return False

            indexer = Indexer(self._project_root)
            if not indexer.is_indexed():
                return False

            self._graph = indexer.load_graph()

            from .local.manifest import Manifest
            from .local.sqlite_vector_store import create_vector_store

            manifest = Manifest(_manifest_path(self._project_root))
            vector_store = create_vector_store(
                self._project_root, backend=self._vector_backend
            )

            from .local.searcher import Searcher
            self._searcher = Searcher(
                graph=self._graph,
                manifest=manifest,
                vector_store=vector_store,
                project_root=self._project_root,
            )
            return True
        except Exception as exc:
            logger.debug("[KB] Failed to initialise local KB: %s", exc)
            return False

    def _ensure_global(self) -> None:
        """Lazily initialise the global KB store."""
        if self._global_store is not None:
            return
        try:
            from .global_kb.store import GlobalKBStore
            self._global_store = GlobalKBStore()
        except Exception as exc:
            logger.debug("[KB] Failed to initialise global KB: %s", exc)

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_error_intent(text: str) -> bool:
        """Return True if *text* indicates an error-fixing task."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in _ERROR_KEYWORDS)

    @staticmethod
    def _detect_review_intent(text: str) -> bool:
        """Return True if *text* indicates a review/refactor task."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in _REVIEW_KEYWORDS)

    @staticmethod
    def _detect_language(file_path: Optional[str]) -> Optional[str]:
        """Detect language from a file extension."""
        if not file_path:
            return None
        ext = os.path.splitext(file_path)[1].lower()
        return _EXT_TO_LANG.get(ext)

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: len(text) // 4."""
        return len(text) // 4

    # ------------------------------------------------------------------
    # Primary method
    # ------------------------------------------------------------------

    def build_context(
        self,
        task_description: str,
        current_file: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> KBContext:
        """
        Build aggregated KB context for a single pipeline step.

        Parameters
        ----------
        task_description:
            Description of the current step or task.
        current_file:
            Path to the file currently being edited, if known.
        max_tokens:
            Maximum token budget for the injected context.

        Returns
        -------
        KBContext
            Aggregated context from all KB layers.
        """
        t0 = time.perf_counter()
        ctx = KBContext()

        # 1. Detect intent
        is_error = self._detect_error_intent(task_description)
        is_review = self._detect_review_intent(task_description)
        language = self._detect_language(current_file)

        # 2. Local semantic search (Phase 2)
        try:
            local_available = self._ensure_local()
        except Exception as exc:
            logger.debug("[KB] _ensure_local failed: %s", exc)
            local_available = False
        if local_available and self._searcher is not None:
            ctx.kb_available = True
            try:
                filters: Optional[dict] = None
                if current_file:
                    # Filter to the directory of the current file
                    dir_path = os.path.dirname(current_file)
                    if dir_path:
                        filters = {"file": dir_path}
                results = self._searcher.search(
                    query=task_description, filters=filters, top_k=8,
                )
                ctx.local_symbols = results
                if results:
                    ctx.sources_used.append("local_semantic")
            except Exception as exc:
                logger.debug("[KB] Local semantic search failed: %s", exc)
        else:
            ctx.kb_available = False

        # 3. Graph expansion (Phase 1) — top 3 local results only
        if self._graph is not None and ctx.local_symbols:
            try:
                seen_names: set[str] = set()
                all_related: list[dict] = []
                for result in ctx.local_symbols[:3]:
                    name = result.symbol_name
                    if name in seen_names:
                        continue
                    seen_names.add(name)
                    related = self._graph.get_related_symbols(name, depth=1)
                    for r in related:
                        r_name = r.get("name", "")
                        if r_name not in seen_names:
                            seen_names.add(r_name)
                            all_related.append(r)
                ctx.related_symbols = all_related
                if all_related:
                    ctx.sources_used.append("graph")
            except Exception as exc:
                logger.debug("[KB] Graph expansion failed: %s", exc)

        # 4. Error lookup (Phase 3) — only if error intent
        if is_error:
            self._ensure_global()
            if self._global_store is not None:
                try:
                    fixes = self._global_store.search_errors(
                        task_description, language=language,
                    )
                    ctx.error_fixes = fixes
                    if fixes:
                        ctx.sources_used.append("error_dict")
                except Exception as exc:
                    logger.debug("[KB] Error lookup failed: %s", exc)

        # 5. Global patterns (Phase 3) — only if review intent
        if is_review:
            self._ensure_global()
            if self._global_store is not None:
                try:
                    patterns = self._global_store.search(
                        task_description,
                        categories=["pattern", "adr"],
                        top_k=3,
                    )
                    ctx.global_patterns = patterns
                    if patterns:
                        ctx.sources_used.append("global_kb")
                except Exception as exc:
                    logger.debug("[KB] Global pattern search failed: %s", exc)

        # 6. Behavioral instructions (Phase 3) — always
        self._ensure_global()
        if self._global_store is not None:
            try:
                behavioral = self._global_store.get_behavioral_instructions(
                    task_description,
                )
                ctx.behavioral_instructions = behavioral
            except Exception as exc:
                logger.debug("[KB] Behavioral instructions failed: %s", exc)

        # 7. Token budget management
        ctx = self._apply_token_budget(ctx, max_tokens)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "[KB] build_context completed in %.1fms — tokens=%d, sources=%s",
            elapsed_ms, ctx.token_count, ctx.sources_used,
        )
        return ctx

    # ------------------------------------------------------------------
    # Token budget trimming
    # ------------------------------------------------------------------

    def _apply_token_budget(self, ctx: KBContext, max_tokens: int) -> KBContext:
        """
        Trim context to fit within *max_tokens*.

        Priority (highest first — last to be trimmed):
        1. behavioral_instructions
        2. error_fixes
        3. local_symbols top 3
        4. global_patterns
        5. related_symbols
        6. local_symbols remaining
        """
        # Helper: estimate tokens for a list of items
        def _list_tokens(items: list) -> int:
            total = 0
            for item in items:
                if hasattr(item, "code_snippet"):
                    total += self._estimate_tokens(getattr(item, "code_snippet", "") or "")
                    total += self._estimate_tokens(getattr(item, "symbol_name", "") or "")
                elif hasattr(item, "fix_template"):
                    total += self._estimate_tokens(getattr(item, "fix_template", "") or "")
                    total += self._estimate_tokens(getattr(item, "cause", "") or "")
                elif hasattr(item, "content"):
                    total += self._estimate_tokens(getattr(item, "content", "") or "")
                    total += self._estimate_tokens(getattr(item, "title", "") or "")
                elif isinstance(item, dict):
                    total += self._estimate_tokens(str(item))
                else:
                    total += self._estimate_tokens(str(item))
            return total

        # Calculate current totals
        behavioral_tokens = _list_tokens(ctx.behavioral_instructions)
        error_tokens = _list_tokens(ctx.error_fixes)
        top3_tokens = _list_tokens(ctx.local_symbols[:3])
        pattern_tokens = _list_tokens(ctx.global_patterns)
        related_tokens = _list_tokens(ctx.related_symbols)
        remaining_tokens = _list_tokens(ctx.local_symbols[3:])

        total = (behavioral_tokens + error_tokens + top3_tokens
                 + pattern_tokens + related_tokens + remaining_tokens)

        # Trim from lowest priority upward
        if total > max_tokens:
            ctx.local_symbols = ctx.local_symbols[:3]
            total -= remaining_tokens
            remaining_tokens = 0

        if total > max_tokens:
            ctx.related_symbols = []
            total -= related_tokens
            related_tokens = 0

        if total > max_tokens:
            ctx.global_patterns = []
            total -= pattern_tokens
            pattern_tokens = 0

        if total > max_tokens and ctx.local_symbols:
            ctx.local_symbols = ctx.local_symbols[:3]
            # Already trimmed above

        # behavioral_instructions and error_fixes are never trimmed

        ctx.token_count = total
        return ctx

    # ------------------------------------------------------------------
    # Formatter
    # ------------------------------------------------------------------

    def format_context_for_prompt(self, context: KBContext) -> str:
        """
        Format a :class:`KBContext` into a clean text block for prompt injection.

        Parameters
        ----------
        context:
            The aggregated KB context.

        Returns
        -------
        str
            Formatted text ready to prepend to the system prompt.
        """
        if not context.kb_available and not context.behavioral_instructions:
            return ""

        parts: list[str] = ["=== KNOWLEDGE BASE CONTEXT ==="]

        # Behavioral instructions (always first)
        if context.behavioral_instructions:
            parts.append("")
            parts.append("[BEHAVIORAL INSTRUCTIONS]")
            for item in context.behavioral_instructions:
                content = getattr(item, "content", "") or getattr(item, "title", "")
                if content:
                    parts.append(content)

        # Relevant code from project
        if context.local_symbols:
            parts.append("")
            parts.append("[RELEVANT CODE FROM THIS PROJECT]")
            for result in context.local_symbols:
                location = f"{result.file} (lines {result.line_start}-{result.line_end})"
                parts.append(f"File: {location}")
                if result.code_snippet:
                    # Limit snippet to 20 lines
                    snippet_lines = result.code_snippet.splitlines()
                    if len(snippet_lines) > 20:
                        snippet_lines = snippet_lines[:20]
                        snippet_lines.append("  ...")
                    parts.append("\n".join(snippet_lines))
                if result.related_symbols:
                    related_names = [
                        f"{r.get('name', '')}" for r in result.related_symbols[:5]
                    ]
                    parts.append(f"Related: {', '.join(related_names)}")
                parts.append("")

        # Error fix patterns
        if context.error_fixes:
            parts.append("[ERROR FIX PATTERNS]")
            for ef in context.error_fixes:
                parts.append(f"Error: {ef.error_type}")
                if ef.cause:
                    parts.append(f"Cause: {ef.cause}")
                parts.append(f"Fix: {ef.fix_template}")
                parts.append("")

        # Coding patterns
        if context.global_patterns:
            parts.append("[CODING PATTERNS]")
            for gp in context.global_patterns:
                title = getattr(gp, "title", "")
                content = getattr(gp, "content", "")
                if title:
                    parts.append(title)
                if content:
                    parts.append(content)
                parts.append("")

        parts.append("=== END KNOWLEDGE BASE CONTEXT ===")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Option A: Relevant file selection
    # ------------------------------------------------------------------

    def get_relevant_files(
        self,
        task_description: str,
        changed_files: list[str] | None = None,
        max_files: int = 15,
    ) -> list[str]:
        """
        Return file paths most relevant to the task.

        Uses KB search results and graph impact analysis to identify a
        minimal set of files the coder should work with.

        Parameters
        ----------
        task_description:
            Natural-language description of the current task/step.
        changed_files:
            Files already modified/created in this session.
        max_files:
            Maximum number of file paths to return.

        Returns
        -------
        list[str]
            Ranked list of relative file paths.
        """
        relevant: dict[str, float] = {}  # path → relevance score

        # 1. Semantic search → extract file paths
        try:
            local_available = self._ensure_local()
        except Exception:
            local_available = False

        if local_available and self._searcher is not None:
            try:
                results = self._searcher.search(
                    query=task_description, top_k=10,
                )
                for result in results:
                    file_path = getattr(result, "file", "")
                    score = getattr(result, "score", 0.5)
                    if file_path:
                        relevant[file_path] = max(
                            relevant.get(file_path, 0), score
                        )
            except Exception as exc:
                logger.debug("[KB] get_relevant_files search failed: %s", exc)

        # 2. Graph impact analysis on changed files
        if self._graph is not None and changed_files:
            try:
                for cf in changed_files[:10]:
                    # Find dependents of this file using impact_analysis
                    impacted = self._graph.impact_analysis(cf, depth=2)
                    for item in impacted:
                        file_path = item.get("file_path", "")
                        if file_path and file_path not in relevant:
                            # Lower score than direct search hits
                            relevant[file_path] = 0.3
            except Exception as exc:
                logger.debug("[KB] get_relevant_files impact failed: %s", exc)

        # 3. Graph expansion — neighbours of top search results
        if self._graph is not None and relevant:
            try:
                top_files = sorted(
                    relevant.items(), key=lambda x: x[1], reverse=True
                )[:5]
                for file_path, _ in top_files:
                    related = self._graph.get_related_symbols(
                        file_path, depth=1
                    )
                    for r in related:
                        r_file = r.get("file_path", "")
                        if r_file and r_file not in relevant:
                            relevant[r_file] = 0.2
            except Exception:
                pass

        # 4. Always include changed files
        if changed_files:
            for cf in changed_files:
                if cf not in relevant:
                    relevant[cf] = 0.9

        # 5. Sort by score, return top max_files
        sorted_files = sorted(
            relevant.items(), key=lambda x: x[1], reverse=True
        )
        result = [f for f, _ in sorted_files[:max_files]]

        logger.info(
            "[KB] Relevant files: %d identified (from %d candidates)",
            len(result), len(relevant),
        )
        return result


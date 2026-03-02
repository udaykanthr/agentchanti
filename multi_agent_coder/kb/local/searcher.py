"""
Semantic search over the Local Knowledge Base — Phase 2.

Combines vector search (SQLite vector store) with Phase 1 graph
look-ups to return ranked :class:`SearchResult` objects enriched
with source code snippets and related-symbol information.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import CodeGraph
    from .manifest import Manifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    A single semantic search result.

    Attributes
    ----------
    symbol_name:
        Name of the matched symbol.
    symbol_type:
        ``"function"`` | ``"method"`` | ``"class"``
    file:
        Relative file path containing the symbol.
    line_start:
        First line of the symbol body (1-indexed).
    line_end:
        Last line of the symbol body (1-indexed).
    code_snippet:
        Actual source lines from *line_start* to *line_end*.
    score:
        Cosine similarity score from the vector store (0–1).  Set to 0.0 for
        graph-only fallback results.
    related_symbols:
        1-hop graph neighbours from Phase 1.  Each entry is a dict
        returned by :meth:`~agentchanti.kb.local.graph.CodeGraph._node_summary`.
    """

    symbol_name: str
    symbol_type: str
    file: str
    line_start: int
    line_end: int
    code_snippet: str
    score: float
    related_symbols: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Embedding helper (reuses OpenAI client logic from embedder)
# ---------------------------------------------------------------------------

def _make_embed_client():
    """Create a dedicated embedding client matching the configured provider.

    This ensures the same provider+model is used for query embeddings
    as was used during ``kb embed``, regardless of what the main
    generation LLM client happens to be.
    """
    from ...config import Config
    cfg = Config.load()
    embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL
    provider = cfg.PROVIDER
    llm_kwargs = dict(
        max_retries=cfg.LLM_MAX_RETRIES,
        retry_delay=cfg.LLM_RETRY_DELAY,
        stream=False,
    )

    if provider == "ollama":
        from ...llm.ollama import OllamaClient
        return OllamaClient(
            base_url=cfg.OLLAMA_BASE_URL, model=embed_model, **llm_kwargs)
    elif provider == "openai":
        from ...llm.openai_client import OpenAIClient
        return OpenAIClient(
            base_url=cfg.OPENAI_BASE_URL, model=embed_model,
            api_key=cfg.OPENAI_API_KEY, **llm_kwargs)
    elif provider == "gemini":
        from ...llm.gemini_client import GeminiClient
        return GeminiClient(
            base_url=cfg.GEMINI_BASE_URL, model=embed_model,
            api_key=cfg.GEMINI_API_KEY, **llm_kwargs)
    elif provider == "anthropic":
        from ...llm.anthropic_client import AnthropicClient
        return AnthropicClient(
            base_url=cfg.ANTHROPIC_BASE_URL, model=embed_model,
            api_key=cfg.ANTHROPIC_API_KEY, **llm_kwargs)
    else:
        from ...llm.lm_studio import LMStudioClient
        return LMStudioClient(
            base_url=cfg.LM_STUDIO_BASE_URL, model=embed_model, **llm_kwargs)


def _embed_query(query: str, api_client: Any = None) -> list[float]:
    """
    Embed *query* using the same provider and model as ``kb embed``.

    A dedicated embedding client is created from the config so that the
    query embedding always matches the stored vectors, even when the
    main generation LLM uses a different provider or model.

    Parameters
    ----------
    query:
        Natural-language search string.
    api_client:
        Ignored (kept for backward compatibility).

    Returns
    -------
    list[float]
        Embedding vector (dimension depends on the model used).
    """
    from ...config import Config
    cfg = Config.load()
    embed_model = cfg.EMBEDDING_MODEL or cfg.DEFAULT_MODEL

    # Build a client that matches the provider used during kb embed
    try:
        client = _make_embed_client()
        vec = client.generate_embedding(query, model=embed_model)
        if vec:
            return vec
    except Exception as exc:
        logger.warning("Embedding client failed: %s", exc)

    # Fallback: try the passed api_client (legacy path)
    if api_client is not None:
        try:
            vec = api_client.generate_embedding(query, model=embed_model)
            if vec:
                return vec
        except Exception as exc:
            logger.warning("api_client embedding fallback failed: %s", exc)

    raise RuntimeError(
        f"Could not generate query embedding with provider={cfg.PROVIDER!r}, "
        f"model={embed_model!r}. Check your .agentchanti.yaml config."
    )


# ---------------------------------------------------------------------------
# Source-line reader
# ---------------------------------------------------------------------------

def _read_snippet(project_root: str, file_path: str, line_start: int, line_end: int) -> str:
    """
    Read source lines *line_start* through *line_end* (1-indexed, inclusive).

    Parameters
    ----------
    project_root:
        Absolute project root path.
    file_path:
        Relative path to the source file.
    line_start:
        First line to read.
    line_end:
        Last line to read.

    Returns
    -------
    str
        Source lines joined with newlines, or an empty string on error.
    """
    abs_path = os.path.join(project_root, file_path)
    if not os.path.exists(abs_path):
        return ""
    try:
        with open(abs_path, encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
        start = max(0, line_start - 1)
        end = line_end if (line_end and line_end > 0) else len(all_lines)
        return "".join(all_lines[start:end]).rstrip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Graph-only keyword fallback
# ---------------------------------------------------------------------------

def _graph_keyword_search(
    query: str,
    graph: "CodeGraph",
    manifest: "Manifest",
    project_root: str,
    top_k: int,
    filters: Optional[dict],
) -> list[SearchResult]:
    """
    Naive keyword search over the graph when the vector store is unavailable.

    Searches symbol names and file paths for tokens in *query*.

    Parameters
    ----------
    query:
        Natural-language search string.
    graph:
        Loaded code graph.
    manifest:
        Manifest instance.
    project_root:
        Absolute project root path.
    top_k:
        Maximum number of results.
    filters:
        Optional ``{"file": ..., "language": ...}`` filters.

    Returns
    -------
    list[SearchResult]
        Results sorted by match score (number of matching tokens).
    """
    tokens = [t.lower() for t in query.split() if t]
    results: list[SearchResult] = []
    seen: set[str] = set()

    from .graph import NodeType
    file_filter: str = (filters or {}).get("file", "")
    lang_filter: str = (filters or {}).get("language", "")

    for nid, attrs in graph._g.nodes(data=True):
        node_type = attrs.get("node_type", "")
        if node_type not in (NodeType.FUNCTION, NodeType.CLASS):
            continue

        name: str = attrs.get("name", "")
        file_path: str = attrs.get("file_path", "")

        if file_filter and file_filter not in file_path:
            continue

        # Language filter: look up the file node's language attribute
        if lang_filter:
            fid = f"FILE:{file_path}"
            lang = ""
            if graph._g.has_node(fid):
                lang = graph._g.nodes[fid].get("language", "")
            if lang_filter.lower() not in lang.lower():
                continue

        unique_key = f"{file_path}:{name}:{attrs.get('line_start', 0)}"
        if unique_key in seen:
            continue

        # Simple token-match score
        target = (name + " " + file_path).lower()
        score = sum(1 for t in tokens if t in target) / max(len(tokens), 1)
        if score == 0:
            continue

        seen.add(unique_key)
        line_start: int = attrs.get("line_start", 0)
        line_end: int = attrs.get("line_end", 0)
        snippet = _read_snippet(project_root, file_path, line_start, line_end)
        related = graph.get_related_symbols(name, depth=1)

        sym_type: str
        if node_type == NodeType.FUNCTION:
            sym_type = "method" if attrs.get("parent_class") else "function"
        else:
            sym_type = "class"

        results.append(
            SearchResult(
                symbol_name=name,
                symbol_type=sym_type,
                file=file_path,
                line_start=line_start,
                line_end=line_end,
                code_snippet=snippet,
                score=score,
                related_symbols=related,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------

class Searcher:
    """
    Combines vector search with Phase 1 graph look-ups.

    Parameters
    ----------
    graph:
        Loaded :class:`~agentchanti.kb.local.graph.CodeGraph`.
    manifest:
        Loaded :class:`~agentchanti.kb.local.manifest.Manifest`.
    vector_store:
        Any vector store with a ``.search()`` method — either
        :class:`SQLiteVectorStore`.
    project_root:
        Absolute path to the project root.
    """

    def __init__(
        self,
        graph: "CodeGraph",
        manifest: "Manifest",
        vector_store: Any = None,
        project_root: str = "",
        api_client: Any = None,
    ) -> None:
        self._graph = graph
        self._manifest = manifest
        self._vector_store = vector_store
        self._project_root = os.path.abspath(project_root)
        self._api_client = api_client

    def search(
        self,
        query: str,
        filters: Optional[dict] = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Perform a semantic (or keyword-fallback) search.

        Parameters
        ----------
        query:
            Natural-language search string.
        filters:
            Optional payload filters.  Supported keys:

            - ``"file"``     — filter by file path substring
            - ``"language"`` — filter by language string (e.g. ``"python"``)
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked list of search results, sorted by score descending.
        """
        t0 = time.perf_counter()

        # No vector store available — use keyword fallback
        if self._vector_store is None:
            logger.debug("No vector store available — using keyword fallback")
            results = _graph_keyword_search(
                query, self._graph, self._manifest,
                self._project_root, top_k, filters
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("Keyword fallback search returned %d results in %.1fms", len(results), elapsed)
            return results

        # Check if vector store has any data
        try:
            info = self._vector_store.collection_info()
            if info is None or info.get("points_count", 0) == 0:
                logger.debug("Vector store is empty — using keyword fallback")
                return _graph_keyword_search(
                    query, self._graph, self._manifest,
                    self._project_root, top_k, filters
                )
        except Exception:
            pass  # proceed to try search anyway

        # ------------------------------------------------------------------
        # Semantic search path
        # ------------------------------------------------------------------
        try:
            query_vector = _embed_query(query, api_client=self._api_client)
        except Exception as exc:
            logger.warning("Failed to embed query: %s — falling back to keyword search", exc)
            return _graph_keyword_search(
                query, self._graph, self._manifest,
                self._project_root, top_k, filters
            )

        # Build store filters (works for SQLite vector store)
        store_filters: Optional[dict] = None
        if filters:
            store_filters = {}
            if "language" in filters:
                store_filters["language"] = filters["language"]
            if "symbol_type" in filters:
                store_filters["symbol_type"] = filters["symbol_type"]
            # "file" partial match is handled post-retrieval

        try:
            raw_results = self._vector_store.search(
                query_vector=query_vector,
                top_k=top_k * 2,  # over-fetch to allow post-filtering
                filters=store_filters,
            )
        except Exception as exc:
            logger.warning("Vector store search failed: %s — falling back to keyword search", exc)
            return _graph_keyword_search(
                query, self._graph, self._manifest,
                self._project_root, top_k, filters
            )

        # Post-filter for file path substring (vector store doesn't do prefix match)
        file_filter = (filters or {}).get("file", "")
        if file_filter:
            raw_results = [
                r for r in raw_results
                if file_filter in r["payload"].get("file", "")
            ]

        # Deduplicate by symbol key
        seen: set[str] = set()
        results: list[SearchResult] = []

        for hit in raw_results:
            if len(results) >= top_k:
                break
            payload = hit["payload"]
            score: float = hit["score"]
            symbol_name: str = payload.get("symbol_name", "")
            file_path: str = payload.get("file", "")
            line_start: int = payload.get("line_start", 0)
            line_end: int = payload.get("line_end", 0)
            sym_type: str = payload.get("symbol_type", "")

            unique_key = f"{file_path}:{symbol_name}:{line_start}"
            if unique_key in seen:
                continue
            seen.add(unique_key)

            snippet = _read_snippet(self._project_root, file_path, line_start, line_end)
            related = self._graph.get_related_symbols(symbol_name, depth=1)

            results.append(
                SearchResult(
                    symbol_name=symbol_name,
                    symbol_type=sym_type,
                    file=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    code_snippet=snippet,
                    score=score,
                    related_symbols=related,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Semantic search returned %d results in %.1fms", len(results), elapsed)
        return results

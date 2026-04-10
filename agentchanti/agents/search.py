"""
Search Agent — searches the web for error documentation when the LLM
encounters an unrecognized error during diagnosis.
"""

import re

from ..cli_display import log
from ..search_provider import web_search, fetch_page_text, SearchResult


class SearchAgent:
    """Optional agent that enriches error diagnosis with web search context.

    Performs web searches and fetches documentation pages.  When an
    ``llm_client`` is provided, raw results are summarized into a compact
    actionable guide before being returned — keeping downstream LLM context
    small.  Without an LLM client the raw results are returned as-is.

    All methods are best-effort: exceptions are caught and logged, never
    propagated.  An empty string return means "no useful context found".
    """

    def __init__(self, provider: str = "duckduckgo",
                 api_key: str = "", api_url: str = "",
                 max_results: int = 3, max_page_chars: int = 3000,
                 llm_client=None):
        self.provider = provider
        self.api_key = api_key
        self.api_url = api_url
        self.max_results = max_results
        self.max_page_chars = max_page_chars
        self.llm_client = llm_client

    # ── Public API ───────────────────────────────────────────

    def search_for_error(self, error_info: str,
                         step_text: str = "",
                         language: str | None = None,
                         kb_context: str = "",
                         query_override: str | None = None) -> str:
        """Search the web for information about an error.

        Args:
            error_info: The error output / traceback from the failed step.
            step_text: The step description (for additional context).
            language: Programming language (adds focused keywords).
            kb_context: Knowledge base context (tech stack, installed
                packages, patterns) so the summarizer can filter outdated
                advice and prefer solutions matching the project.

        Returns:
            Formatted string with search results and excerpts, or empty
            string if nothing useful was found or on any error.
        """
        try:
            query = (query_override.strip()
                     if query_override and query_override.strip()
                     else self._build_search_query(error_info, language,
                                                   kb_context=kb_context))
            if not query:
                return ""

            log.info(f"[SearchAgent] Searching: {query}")

            results = web_search(
                query,
                provider=self.provider,
                api_key=self.api_key,
                api_url=self.api_url,
                max_results=self.max_results,
            )

            if not results:
                log.info("[SearchAgent] No search results found")
                return ""

            raw = self._format_results(results)
            return self._summarize(raw, mode="error", query=query,
                                   kb_context=kb_context)

        except Exception as exc:
            log.warning(f"[SearchAgent] Search failed: {exc}")
            return ""

    def search_for_task(self, task: str,
                        language: str | None = None,
                        kb_context: str = "") -> str:
        """Search the web for latest documentation relevant to a task.

        Called **before** the planner generates steps so that local LLMs
        receive up-to-date, command-level guidance (framework CLI flags,
        install commands, API changes, etc.).

        Args:
            task: The user's task description / prompt.
            language: Programming language (adds focused keywords).
            kb_context: Knowledge base context (tech stack, installed
                packages, patterns) so the summarizer can prefer results
                matching the project's actual stack and versions.

        Returns:
            Formatted string with search results and page excerpts, or
            empty string if nothing useful was found or on any error.
        """
        try:
            query = self._build_task_query(task, language,
                                          kb_context=kb_context)
            if not query:
                return ""

            log.info(f"[SearchAgent] Planning search: {query}")

            results = web_search(
                query,
                provider=self.provider,
                api_key=self.api_key,
                api_url=self.api_url,
                max_results=self.max_results,
            )

            if not results:
                log.info("[SearchAgent] No planning search results found")
                return ""

            header = ("=== Web Search Context (latest documentation) ===\n"
                      "Use the information below for accurate, up-to-date "
                      "commands and flags in your plan.")
            raw = self._format_results(results, header=header)
            return self._summarize(raw, mode="task", query=query,
                                   kb_context=kb_context)

        except Exception as exc:
            log.warning(f"[SearchAgent] Planning search failed: {exc}")
            return ""

    # ── Internals ────────────────────────────────────────────

    @staticmethod
    def _extract_stack_keywords(kb_context: str) -> str:
        """Extract tech stack keywords from KB context for search queries.

        Parses lines like ``Stack: typescript, React 18, tests: vitest``
        and ``Installed: express, cors, jest`` to pull framework names and
        version numbers that make search queries version-specific.

        Returns a short string like ``React 18 vitest typescript`` (capped
        at ~60 chars so it doesn't dominate the query).
        """
        if not kb_context:
            return ""

        keywords: list[str] = []

        # Extract from "Stack: ..." line
        stack_match = re.search(r'Stack:\s*(.+)', kb_context)
        if stack_match:
            raw = stack_match.group(1)
            # Split on commas, strip labels like "tests:", "pkg:"
            for part in raw.split(","):
                part = re.sub(r'\b(tests|pkg|test_framework|package_manager)\s*:\s*', '', part)
                part = part.strip()
                if part and len(part) < 30:
                    keywords.append(part)

        # Extract from "Installed: pkg1, pkg2, ..." line (may include scoped
        # packages like @tailwindcss/postcss — strip scope prefix for readability)
        installed_match = re.search(r'Installed:\s*(.+)', kb_context)
        if installed_match:
            existing_lower = {kw.lower() for kw in keywords}
            for pkg in installed_match.group(1).split(","):
                pkg = pkg.strip()
                # Strip flags (--save-dev etc.) and scope prefix (@org/)
                pkg = re.sub(r'^@[\w-]+/', '', pkg)  # @scope/name → name
                pkg = pkg.split("@")[0].strip()       # name@version → name
                if pkg and len(pkg) < 30 and pkg.lower() not in existing_lower:
                    keywords.append(pkg)
                    existing_lower.add(pkg.lower())

        # Extract framework/tool with version patterns like "React 18.2"
        # from anywhere in the context (only if not already captured above)
        versioned = re.findall(
            r'\b((?:React|Angular|Vue|Next|Nuxt|Express|Django|Flask|'
            r'FastAPI|Spring|Rails|Laravel|Vite|Webpack|Jest|Vitest|'
            r'Mocha|pytest|Node|Deno|Bun)\s+[\d]+[\d.]*)',
            kb_context, re.IGNORECASE,
        )
        existing_lower = {kw.lower() for kw in keywords}
        for v in versioned:
            v = v.strip()
            if v.lower() not in existing_lower:
                keywords.append(v)
                existing_lower.add(v.lower())

        if not keywords:
            return ""

        result = " ".join(keywords)
        # Cap so it doesn't overwhelm the actual query
        if len(result) > 60:
            result = result[:60].rsplit(' ', 1)[0]
        return result

    def _build_search_query(self, error_info: str,
                            language: str | None = None,
                            kb_context: str = "") -> str:
        """Extract a focused search query from the error output.

        Strategy:
        1. Find the first "key error line" — the line that contains the
           actual error message (e.g., ``ModuleNotFoundError: No module
           named 'foo'``).
        2. Strip file paths, line numbers, and timestamp noise.
        3. Append language keyword for relevance.
        4. Append tech stack keywords from KB for version-specific results.
        5. Cap length so the search engine doesn't choke.
        """
        if not error_info or not error_info.strip():
            return ""

        key_line = self._extract_key_error_line(error_info)
        if not key_line:
            # Fallback: use the last non-empty line
            lines = [l.strip() for l in error_info.strip().splitlines() if l.strip()]
            key_line = lines[-1] if lines else ""

        if not key_line:
            return ""

        # Strip file paths and line numbers
        query = re.sub(r'(?:File\s+)?["\']?[\w/\\.:]+\.(py|js|jsx|ts|tsx|go|rb|java|rs)\b["\']?',
                       '', key_line)
        query = re.sub(r',?\s*line\s+\d+', '', query, flags=re.IGNORECASE)
        # Strip ANSI escape codes
        query = re.sub(r'\x1b\[[0-9;]*m', '', query)
        # Collapse whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        # Add language keyword
        lang_keywords = {
            "python": "python",
            "javascript": "javascript node.js",
            "typescript": "typescript",
            "go": "golang",
            "ruby": "ruby",
            "java": "java",
            "rust": "rust",
        }
        if language and language in lang_keywords:
            query = f"{query} {lang_keywords[language]}"

        # Add tech stack keywords from KB for version-specific results
        stack_kw = self._extract_stack_keywords(kb_context)
        if stack_kw:
            query = self._append_stack_keywords(query, stack_kw)

        # Cap length (search engines typically handle ~150 chars well)
        if len(query) > 150:
            query = query[:150].rsplit(' ', 1)[0]

        return query

    def _build_task_query(self, task: str,
                          language: str | None = None,
                          kb_context: str = "") -> str:
        """Build a documentation-focused search query from a task description.

        Unlike ``_build_search_query`` (which targets error messages), this
        method extracts **technology keywords** from the user prompt and
        appends terms like "latest docs guide" so the search engine returns
        setup guides, CLI references, and recent release notes.

        When ``kb_context`` is provided, tech stack keywords (framework
        names with versions) are appended to get version-specific docs.
        """
        if not task or not task.strip():
            return ""

        # Strip ANSI codes (just in case)
        clean = re.sub(r'\x1b\[[0-9;]*m', '', task)
        # Collapse whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Build query: use the task itself (trimmed) + documentation focus
        query = clean

        # Add language keyword for relevance
        lang_keywords = {
            "python": "python",
            "javascript": "javascript node.js",
            "typescript": "typescript",
            "go": "golang",
            "ruby": "ruby",
            "java": "java",
            "rust": "rust",
        }
        if language and language in lang_keywords:
            query = f"{query} {lang_keywords[language]}"

        # Add tech stack keywords from KB for version-specific results
        stack_kw = self._extract_stack_keywords(kb_context)
        if stack_kw:
            query = self._append_stack_keywords(query, stack_kw)

        # Append documentation focus
        query = f"{query} latest docs guide"

        # Cap length
        if len(query) > 150:
            query = query[:150].rsplit(' ', 1)[0]

        return query

    @staticmethod
    def _append_stack_keywords(query: str, stack_keywords: str) -> str:
        """Append stack keywords to query, skipping terms already present.

        Treats multi-word terms like 'React 18.2' as a unit — only adds
        the full term if the primary word (e.g. 'React') isn't in the query.
        """
        query_lower = query.lower()
        # Split on comma-separated terms first, then individual words
        terms = [t.strip() for t in stack_keywords.split(",") if t.strip()]
        if len(terms) <= 1:
            # No commas — the keywords are space-separated from _extract
            # Re-split intelligently: group "Name Version" pairs
            tokens = stack_keywords.split()
            terms = []
            i = 0
            while i < len(tokens):
                # If next token looks like a version number, group with current
                if (i + 1 < len(tokens)
                        and re.match(r'^\d', tokens[i + 1])):
                    terms.append(f"{tokens[i]} {tokens[i+1]}")
                    i += 2
                else:
                    terms.append(tokens[i])
                    i += 1

        new_terms = []
        for term in terms:
            # Check if the primary word of the term is already in the query
            primary = term.split()[0].lower()
            if primary not in query_lower:
                new_terms.append(term)

        if new_terms:
            return f"{query} {' '.join(new_terms)}"
        return query

    @staticmethod
    def _extract_key_error_line(error_info: str) -> str:
        """Find the most informative error line in a traceback/output.

        Looks for common error patterns:
        - Python: ``SomeError: message``
        - Node.js: ``Error: message`` or ``TypeError: message``
        - Jest/Node: ``Cannot find module``, ``Module not found``
        - Generic: lines containing "error", "failed"
        """
        lines = error_info.strip().splitlines()

        # Python-style: "ErrorClassName: message"
        for line in reversed(lines):
            line = line.strip()
            if re.match(r'^[A-Z]\w*(Error|Exception|Warning):\s', line):
                return line

        # Node.js style: "Error: ..." or "TypeError: ..."
        for line in lines:
            line = line.strip()
            if re.match(r'^(Error|TypeError|ReferenceError|SyntaxError|'
                        r'RangeError|URIError|EvalError):', line):
                return line

        # Jest/Node descriptive errors (no "Error:" prefix but highly
        # informative — e.g. "Cannot find module '@testing-library/react'")
        _DESCRIPTIVE_PATTERNS = (
            r'Cannot find module\b',
            r'Module not found\b',
            r'Module build failed\b',
            r'Failed to compile\b',
            r'is not defined\b',
            r'is not a function\b',
            r'Unexpected token\b',
            r'Cannot read propert',       # Cannot read property / properties
            r'command not found\b',
            r'Unknown command\b',          # npm "Unknown command: set-script"
            r'No such file or directory\b',
            r'Permission denied\b',
        )
        for line in lines:
            stripped = line.strip()
            if stripped and any(re.search(p, stripped, re.IGNORECASE)
                               for p in _DESCRIPTIVE_PATTERNS):
                return stripped

        # Generic: first line containing "error" or "failed" (but skip
        # bare Jest markers like "FAIL src/App.test.tsx")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip bare FAIL markers (just "FAIL <path>")
            if re.match(r'^FAIL\s+\S+$', stripped):
                continue
            if re.search(r'\b(error|failed)\b', stripped, re.IGNORECASE):
                return stripped

        # Fallback: first non-empty line that isn't a bare marker
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("Traceback"):
                continue
            # Skip bare FAIL/PASS markers
            if re.match(r'^(FAIL|PASS)\s+\S+$', stripped):
                continue
            # Skip Jest section markers (● ...)
            if stripped.startswith('●'):
                continue
            return stripped

        return ""

    def search_for_package(self, package_name: str,
                           language: str | None = None,
                           existing_kb_doc: str = "",
                           kb_context: str = "") -> str:
        """Search for latest package docs, merging with any existing KB doc.

        Both sources are passed to the summariser together so the LLM can
        reconcile version conflicts and produce a single authoritative
        reference (install command, import syntax, key API, test notes).

        Args:
            package_name: Bare package name (e.g. "animejs", "lodash").
            language: Programming language for query focus.
            existing_kb_doc: Current KB doc content for this package, if any.
                             May be stale — the LLM will prefer live results
                             when they conflict.
            kb_context: Tech-stack KB context for version filtering.

        Returns:
            Compact package reference string, or empty string on failure.
        """
        try:
            lang_suffixes = {
                "javascript": "javascript npm",
                "typescript": "typescript npm",
                "python": "python pip",
                "go": "golang",
                "ruby": "ruby gem",
                "java": "java maven",
                "rust": "rust cargo",
            }
            lang_suffix = lang_suffixes.get(language or "", language or "")
            query = (
                f"{package_name} install usage api guide {lang_suffix}".strip()
            )
            if len(query) > 150:
                query = query[:150].rsplit(" ", 1)[0]

            log.info(f"[SearchAgent] Package doc search: {query}")

            results = web_search(
                query,
                provider=self.provider,
                api_key=self.api_key,
                api_url=self.api_url,
                max_results=self.max_results,
            )

            raw_live = self._format_results(results) if results else ""

            # Combine existing KB doc with live search results
            combined_parts: list[str] = []
            if existing_kb_doc:
                combined_parts.append(
                    f"=== Existing KB documentation (may be stale) ===\n"
                    f"{existing_kb_doc}"
                )
            if raw_live:
                combined_parts.append(
                    f"=== Live web search results ===\n{raw_live}"
                )

            if not combined_parts:
                log.info("[SearchAgent] No package docs found for: %s",
                         package_name)
                return ""

            combined = "\n\n".join(combined_parts)
            return self._summarize(
                combined,
                mode="package",
                query=query,
                package_name=package_name,
                kb_context=kb_context,
            )

        except Exception as exc:
            log.warning(f"[SearchAgent] Package search failed for "
                        f"'{package_name}': {exc}")
            return ""

    def _summarize(self, raw_context: str, *, mode: str = "error",
                   query: str = "", kb_context: str = "",
                   package_name: str = "") -> str:
        """Use LLM to condense raw search results into a compact guide.

        When ``kb_context`` is provided (tech stack, installed packages,
        patterns from the Knowledge Base), the summarizer uses it to:
        - Discard advice for wrong versions or incompatible frameworks
        - Prefer solutions matching the project's actual stack
        - Flag when search results contradict known project setup

        Falls back to the raw text if no LLM client is available or on error.
        """
        if not raw_context:
            return raw_context
        if self.llm_client is None:
            return raw_context

        # Build project grounding block from KB context
        kb_block = ""
        if kb_context:
            kb_block = (
                "\n=== PROJECT CONTEXT (use to filter search results) ===\n"
                f"{kb_context}\n"
                "IMPORTANT: Discard any search results that recommend outdated "
                "APIs, deprecated flags, or solutions for different versions "
                "than what this project uses. Prefer solutions that match the "
                "project's actual tech stack, package versions, and patterns.\n\n"
            )

        if mode == "error":
            instruction = (
                "You are a concise technical assistant. Given the raw web search "
                "results below about a coding error, produce a SHORT actionable "
                "summary (max ~300 words). Format as:\n"
                "- **Root Cause**: one-line explanation\n"
                "- **Fix Steps**: numbered list of concrete steps/commands\n"
                "- **Key Details**: any version-specific notes or gotchas\n\n"
                "Strip all URLs, page boilerplate, and irrelevant content. "
                "Only include information directly useful for fixing the error.\n\n"
                f"{kb_block}"
                f"Search query: {query}\n\n"
                f"Raw search results:\n{raw_context}"
            )
        elif mode == "package":
            pkg = package_name or query
            instruction = (
                "You are a concise technical assistant. Given the documentation "
                f"below about the '{pkg}' package (from an existing KB doc and/or "
                "live web search), produce a SHORT authoritative reference "
                "(max ~200 words). Format as:\n"
                "- **Install**: exact install command with correct flags "
                "(e.g. npm install animejs, pip install animejs)\n"
                "- **Import**: correct import/require statement for the current "
                "version\n"
                "- **Key API**: 1-3 most important usage patterns with brief "
                "examples\n"
                "- **Test setup**: any mocking or vitest/jest setup required\n"
                "- **Version note**: if KB doc and live docs disagree on API or "
                "import path, state which is current and use the live result\n\n"
                "If KB and live docs conflict, the live web results are "
                "authoritative — clearly prefer them.\n\n"
                f"{kb_block}"
                f"Package: {pkg}\n\n"
                f"Documentation sources:\n{raw_context}"
            )
        else:
            instruction = (
                "You are a concise technical assistant. Given the raw web search "
                "results below about a development task, produce a SHORT actionable "
                "guide (max ~300 words). Format as:\n"
                "- **Setup**: key install/init commands with exact flags\n"
                "- **Steps**: numbered list of concrete actions\n"
                "- **Notes**: version-specific caveats or breaking changes\n\n"
                "Strip all URLs, page boilerplate, and irrelevant content. "
                "Only include information directly useful for the task.\n\n"
                f"{kb_block}"
                f"Search query: {query}\n\n"
                f"Raw search results:\n{raw_context}"
            )

        try:
            summary = self.llm_client.generate_response(instruction)
            if summary and len(summary.strip()) > 20:
                log.info(f"[SearchAgent] Summarized {len(raw_context)} chars "
                         f"-> {len(summary)} chars")
                header = ("=== Web Search Context (summarized) ===" if mode == "error"
                          else "=== Web Search Context (latest docs, summarized) ===")
                return f"{header}\n{summary.strip()}"
            # Summary too short / empty — fall back to raw
            return raw_context
        except Exception as exc:
            log.warning(f"[SearchAgent] Summarization failed, using raw: {exc}")
            return raw_context

    def _format_results(self, results: list[SearchResult], *,
                        header: str | None = None) -> str:
        """Format search results with fetched page excerpts."""
        sections: list[str] = []

        for i, result in enumerate(results, 1):
            section = f"[{i}] {result.title}\n    URL: {result.url}"
            if result.snippet:
                section += f"\n    Snippet: {result.snippet}"

            # Fetch page content for richer context
            page_text = fetch_page_text(result.url,
                                        max_chars=self.max_page_chars)
            if page_text:
                # Truncate to a reasonable excerpt
                excerpt = page_text[:1500]
                if len(page_text) > 1500:
                    excerpt = excerpt.rsplit(' ', 1)[0] + "..."
                section += f"\n    Page excerpt: {excerpt}"

            sections.append(section)

        if not sections:
            return ""

        if header is None:
            header = "=== Web Search Results (documentation/solutions found) ==="
        return header + "\n\n" + "\n\n".join(sections)

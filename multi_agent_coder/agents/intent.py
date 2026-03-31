import logging
import os
import re
from .base import Agent

_logger = logging.getLogger(__name__)

# Stop-words filtered out when extracting a search query from the raw task
_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "with", "this", "that", "are", "was",
    "be", "by", "as", "if", "do", "can", "how", "why", "what", "when",
    "only", "just", "also", "very", "all", "any", "from", "into", "its",
    "i", "me", "we", "you", "he", "she", "they", "my", "your", "their",
    "please", "make", "add", "create", "fix", "update", "change", "need",
    "want", "should", "will", "would", "could", "there", "here", "has",
    "have", "been", "does", "did", "then", "than", "so", "about", "which",
}


class IntentAgent(Agent):
    """
    Analyzes the raw user prompt and gathers necessary context before planning.

    The agent acts as a code investigator: it explores the actual project before
    concluding, then produces a REQUIREMENTS_SPEC grounded in what it found.

    Investigation is driven by the LLM's own judgment — not forced by gates or
    counters.  The REQUIREMENTS_SPEC format itself requires evidence fields that
    can only be filled by inspecting real code, which naturally motivates the LLM
    to search before concluding.

    KB_SEARCH hits BOTH indexes:
      - Local semantic index  → actual project source code snippets
      - Global KB store       → framework docs, patterns

    Supported commands the LLM can emit:
      SEARCH: <query>      — delegates to SearchAgent (web search)
      KB_SEARCH: <query>   — queries local source + global KB docs
    """

    def process(self, task: str, context: str = "") -> str:
        """Implementation of base Agent process. Fallback if called directly."""
        return self.analyze_intent(task)

    def analyze_intent(
        self,
        raw_task: str,
        search_agent=None,
        kb_context_builder=None,
        kb_context: str = "",
        max_iterations: int = 5,
    ) -> str:
        """
        Investigate the task and produce a grounded REQUIREMENTS_SPEC.

        The LLM decides when it has enough context to conclude — the prompt
        structure makes thorough investigation the natural path, not a forced one.

        Returns the original task with a short INTENT_CLARIFICATION appended.
        """
        _logger.info("[IntentAnalysis] Starting intent analysis for task.")

        accumulated_context = ""
        if kb_context:
            accumulated_context += f"Project Stack / Knowledge Base Context:\n{kb_context}\n\n"

        # ── Pre-seed: give the LLM a starting point, not a conclusion ─────────
        # Run one automatic KB search before the loop so the LLM can see what
        # exists in the project and decide what it still needs to investigate.
        # This is a hint, not a gate — the LLM is free to search further or
        # pivot to different files based on what it sees here.
        if kb_context_builder is not None:
            pre_query = self._extract_search_query(raw_task)
            _logger.info("[IntentAnalysis] Pre-seeding context with KB_SEARCH '%s'", pre_query)
            pre_result = self._query_kb(kb_context_builder, pre_query)
            if pre_result:
                accumulated_context += (
                    f"Initial Project Context (auto-retrieved, query: '{pre_query}'):\n"
                    f"{pre_result}\n\n"
                )
                _logger.info("[IntentAnalysis] Pre-seed successful.")
            else:
                _logger.info("[IntentAnalysis] Pre-seed returned no results.")

        # ── Iterative investigation loop ──────────────────────────────────────
        last_response = ""
        # Track commands already executed to prevent the LLM from looping on
        # the same tool call repeatedly without making progress.
        executed_commands: set[str] = set()
        # After the first duplicate is detected, force conclude on the very
        # next iteration so the LLM doesn't burn more rounds ignoring the note.
        force_conclude = False

        for iteration in range(1, max_iterations + 1):
            is_last = iteration == max_iterations or force_conclude
            prompt = self._build_prompt(raw_task, accumulated_context, conclude=is_last)

            try:
                response = self.llm_client.generate_response(prompt)
                last_response = response

                # ── KB_SEARCH command ─────────────────────────────────────────
                # Use \b instead of ^ so prefixes like "COMMAND: KB_SEARCH:"
                # or "Tool: KB_SEARCH:" that the LLM may echo don't cause a miss.
                kb_match = re.search(
                    r'\bKB_SEARCH:\s*(.+)$', response,
                    re.MULTILINE | re.IGNORECASE,
                )
                if kb_match and kb_context_builder is not None:
                    query = kb_match.group(1).strip()
                    cmd_key = f"KB_SEARCH:{query.lower()}"
                    if cmd_key in executed_commands:
                        _logger.info(
                            "[IntentAnalysis] Iteration %d: KB_SEARCH '%s' already executed — forcing conclude.",
                            iteration, query,
                        )
                        accumulated_context += (
                            f"[KB_SEARCH '{query}' already ran — results are above. "
                            f"You have enough context. Write REQUIREMENTS_SPEC now.]\n\n"
                        )
                        force_conclude = True
                        continue
                    executed_commands.add(cmd_key)
                    _logger.info(
                        "[IntentAnalysis] Iteration %d: KB_SEARCH '%s'", iteration, query,
                    )
                    # LLMs often pack multiple file names into one query
                    # (e.g. "Food.jsx; SnakeSegment.jsx"). Split on common
                    # separators and run each sub-query so every file is found.
                    sub_queries = [
                        q.strip()
                        for q in re.split(r'[;,\n]+', query)
                        if q.strip()
                    ]
                    all_results: list[str] = []
                    for sq in sub_queries:
                        r = self._query_kb(kb_context_builder, sq)
                        if r:
                            all_results.append(r)
                    kb_result = "\n\n".join(all_results)
                    if kb_result:
                        accumulated_context += (
                            f"KB Search Results for '{query}':\n{kb_result}\n\n"
                        )
                    else:
                        _logger.warning(
                            "[IntentAnalysis] KB search returned no results for: %s", query,
                        )
                        accumulated_context += (
                            f"KB Search for '{query}': no results found.\n\n"
                        )
                    continue

                # ── SEARCH command ────────────────────────────────────────────
                # Negative lookbehind on a single char (_) prevents matching
                # "KB_SEARCH:" as "SEARCH:".  Fixed-length lookbehind required
                # by Python's re module.
                search_match = re.search(
                    r'(?<!_)\bSEARCH:\s*(.+)$', response,
                    re.MULTILINE | re.IGNORECASE,
                )
                if search_match and search_agent is not None:
                    query = search_match.group(1).strip()
                    cmd_key = f"SEARCH:{query.lower()}"
                    if cmd_key in executed_commands:
                        _logger.info(
                            "[IntentAnalysis] Iteration %d: SEARCH '%s' already executed — forcing conclude.",
                            iteration, query,
                        )
                        accumulated_context += (
                            f"[SEARCH '{query}' already ran — results are above. "
                            f"You have enough context. Write REQUIREMENTS_SPEC now.]\n\n"
                        )
                        force_conclude = True
                        continue
                    executed_commands.add(cmd_key)
                    _logger.info(
                        "[IntentAnalysis] Iteration %d: SEARCH '%s'", iteration, query,
                    )
                    search_result = search_agent.search_for_task(
                        query, kb_context=kb_context,
                    )
                    if search_result:
                        accumulated_context += (
                            f"Web Search Results for '{query}':\n{search_result}\n\n"
                        )
                    else:
                        _logger.warning(
                            "[IntentAnalysis] Web search returned no results for: %s", query,
                        )
                        accumulated_context += (
                            f"Web Search for '{query}': no results found.\n\n"
                        )
                    continue

                # ── FIND_USAGES command ───────────────────────────────────────
                # Finds files that import a component/function and shows the
                # actual call site lines — lets the LLM compare props passed
                # at the call site against props declared in the component.
                usages_match = re.search(
                    r'\bFIND_USAGES:\s*(.+)$', response,
                    re.MULTILINE | re.IGNORECASE,
                )
                if usages_match and kb_context_builder is not None:
                    name = usages_match.group(1).strip()
                    cmd_key = f"FIND_USAGES:{name.lower()}"
                    if cmd_key in executed_commands:
                        _logger.info(
                            "[IntentAnalysis] Iteration %d: FIND_USAGES '%s' already executed — forcing conclude.",
                            iteration, name,
                        )
                        accumulated_context += (
                            f"[FIND_USAGES '{name}' already ran — results are above. "
                            f"You have enough context. Write REQUIREMENTS_SPEC now.]\n\n"
                        )
                        force_conclude = True
                        continue
                    executed_commands.add(cmd_key)
                    _logger.info(
                        "[IntentAnalysis] Iteration %d: FIND_USAGES '%s'", iteration, name,
                    )
                    usage_result = self._find_usages(kb_context_builder, name)
                    if usage_result:
                        accumulated_context += (
                            f"Usages of '{name}':\n{usage_result}\n\n"
                        )
                    else:
                        accumulated_context += (
                            f"FIND_USAGES '{name}': no call sites found.\n\n"
                        )
                    continue

                # ── REQUIREMENTS_SPEC ─────────────────────────────────────────
                spec_match = re.search(
                    r'REQUIREMENTS_SPEC:(.+)', response,
                    re.IGNORECASE | re.DOTALL,
                )
                if spec_match:
                    spec = spec_match.group(1).strip()

                    # ── BUG_FIX gate: enforce at least one tool call ──────────
                    # A BUG_FIX Root cause cited from training-data patterns
                    # rather than actual source is unreliable.  If the spec
                    # declares BUG_FIX but no tool has been called yet (no real
                    # code was read this session beyond the pre-seed), push back
                    # once and demand a FIND_USAGES or KB_SEARCH first.
                    is_bug_fix = bool(re.search(
                        r'Task type:\s*BUG_FIX', spec, re.IGNORECASE,
                    ))
                    if is_bug_fix and not executed_commands and not is_last:
                        _logger.info(
                            "[IntentAnalysis] BUG_FIX spec without any tool calls "
                            "— requiring code trace before accepting.",
                        )
                        accumulated_context += (
                            f"[Your REQUIREMENTS_SPEC was rejected: task is BUG_FIX "
                            f"but you have not read any actual component source yet. "
                            f"Root cause must be grounded in real code, not patterns.\n"
                            f"Use FIND_USAGES on the component that fails to render, "
                            f"or KB_SEARCH for its source file, then write the spec.]\n\n"
                        )
                        # Store the draft so we can fall back to it if needed
                        last_response = response
                        continue

                    _logger.info("[IntentAnalysis] Successfully generated REQUIREMENTS_SPEC.")
                    return (
                        f"{raw_task}\n\n"
                        f"=== INTENT CLARIFICATION (from requirements analyst) ===\n"
                        f"{spec}"
                    )

                # ── No recognised command or marker ──────────────────────────
                # LLM produced a response but forgot the marker — treat as spec
                _logger.info(
                    "[IntentAnalysis] No exact marker found — treating response as spec.",
                )
                return (
                    f"{raw_task}\n\n"
                    f"=== INTENT CLARIFICATION (from requirements analyst) ===\n"
                    f"{response.strip()}"
                )

            except Exception as e:
                _logger.warning(
                    "[IntentAnalysis] Failed during iteration %d: %s", iteration, e,
                )
                break

        # Loop exhausted — use the last LLM response rather than discarding the
        # investigation.  This prevents the silent fallback where all KB searches
        # complete but the LLM never got a final call to produce the spec.
        if last_response:
            _logger.warning(
                "[IntentAnalysis] Max iterations (%d) reached — using last response as spec.",
                max_iterations,
            )
            return (
                f"{raw_task}\n\n"
                f"=== INTENT CLARIFICATION (from requirements analyst) ===\n"
                f"{last_response.strip()}"
            )
        _logger.warning("[IntentAnalysis] Returning original raw task as fallback.")
        return raw_task

    # ── Private helpers ───────────────────────────────────────────────────────

    # ── Task-type spec templates ──────────────────────────────────────────────
    _SPEC_BUG = (
        "REQUIREMENTS_SPEC:\n"
        "Task type: BUG_FIX\n"
        "Goal: <what the user wants fixed, one sentence>\n"
        "Symptom: <what the user observes — not rendering, crash, wrong value>\n"
        "Root cause: <specific file + evidence — wrong prop name, bad logic, "
        "missing condition — cite actual code>\n"
        "Fix scope: <ONLY the files/functions that must change to fix the root cause>\n"
        "Do not touch: <files/components that are working correctly — keep the "
        "planner from over-engineering the fix>\n"
        "Interface analysis: <declared props vs what callers pass — mismatches that "
        "cause the bug; 'none' if root cause is not an interface issue>\n"
        "Constraints: <ambiguities resolved from actual code>\n"
        "KB topics: <comma-separated framework/library topics the planner will need "
        "docs for — e.g. 'Tailwind z-index, React refs'; write 'none' if the fix "
        "is pure logic/config with no external library involved>\n"
    )
    _SPEC_FEATURE = (
        "REQUIREMENTS_SPEC:\n"
        "Task type: FEATURE\n"
        "Goal: <what new capability the user wants, one sentence>\n"
        "Create: <new files or components to build>\n"
        "Integrate with: <existing files, functions, import paths to connect to>\n"
        "Reuse: <existing patterns, components, or utilities already in the codebase>\n"
        "Constraints: <API contracts to preserve, tests that cover adjacent code>\n"
        "Packages/versions: <new dependencies needed, or 'none'>\n"
        "KB topics: <comma-separated topics the planner needs docs for — e.g. "
        "'Anime.js install, React hooks'; write 'none' if no external docs needed>\n"
    )
    _SPEC_MODIFY = (
        "REQUIREMENTS_SPEC:\n"
        "Task type: MODIFY\n"
        "Goal: <what should change, one sentence>\n"
        "Current behavior: <what the code does now — cite file, function, logic>\n"
        "Target behavior: <what it should do after the change>\n"
        "Change scope: <minimal list of files/functions to edit>\n"
        "Preserve: <behavior, API surface, or tests that must not break>\n"
        "Constraints: <from actual code>\n"
        "KB topics: <comma-separated topics the planner needs docs for; 'none' if "
        "this is a pure internal refactor>\n"
    )
    _SPEC_QUESTION = (
        "REQUIREMENTS_SPEC:\n"
        "Task type: QUESTION\n"
        "Goal: <what the user wants to understand>\n"
        "Answer: <direct answer grounded in the code evidence>\n"
        "Evidence: <file names, function signatures, prop names that support the answer>\n"
        "KB topics: none\n"
    )

    def _build_prompt(self, raw_task: str, accumulated_context: str, conclude: bool = False) -> str:
        """
        Build the investigation prompt.

        The LLM first classifies the task type (BUG_FIX / FEATURE / MODIFY /
        QUESTION), then applies a type-specific investigation strategy, then
        produces the matching REQUIREMENTS_SPEC format.

        Tools are only called when the LLM can name a specific gap it cannot
        fill from the evidence already shown — not as a mandatory phase.

        When *conclude* is True, tool lines are not accepted.
        """
        prompt = (
            f"Role: {self.role}\n"
            f"Goal: {self.goal}\n\n"
            f"User Prompt:\n{raw_task}\n\n"
        )

        if accumulated_context:
            prompt += f"Evidence gathered so far:\n{accumulated_context}\n\n"

        if conclude:
            prompt += (
                "You have enough evidence. Classify the task and write the matching "
                "REQUIREMENTS_SPEC now — no tool calls.\n\n"
                "Classification reminder: if the user reports something broken,\n"
                "not visible, or not working — even if phrased as a question —\n"
                "that is BUG_FIX, not QUESTION.\n\n"
                + self._spec_all_formats()
            )
            return prompt

        prompt += (
            "── STEP 1: Classify the task ──────────────────────────────────────\n"
            "Decide which type fits the user prompt:\n\n"
            "  BUG_FIX  — something is broken, not rendering, not visible, crashing,\n"
            "             wrong output, or behaving incorrectly.\n"
            "             IMPORTANT: if the user asks 'is this X or Y?' or 'why is\n"
            "             this not working?' about a broken/invisible feature, that is\n"
            "             BUG_FIX — not QUESTION. The question is about the cause of\n"
            "             a bug, not a request for general code explanation.\n\n"
            "  FEATURE  — add new functionality, build something that does not exist.\n\n"
            "  MODIFY   — change existing behaviour, refactor, update, rename.\n\n"
            "  QUESTION — ONLY use this when the user wants to understand how working\n"
            "             code functions (e.g. 'explain how auth works', 'what does\n"
            "             this function do'). NOT for broken/invisible/crashing things.\n\n"

            "── STEP 2: Investigation rules per type ───────────────────────────\n"
            "BUG_FIX (mandatory tracing — you MUST read actual code before concluding):\n"
            "  1. Identify the component that should render the failing element.\n"
            "  2. Use FIND_USAGES to compare what that component DECLARES (props,\n"
            "     params, data shape) vs what its CALLER actually PASSES.\n"
            "     This catches prop-name mismatches, wrong data formats, missing props\n"
            "     — the most common cause of 'renders nothing' bugs.\n"
            "  3. If the interface matches, use KB_SEARCH to read the component body\n"
            "     and trace the rendering logic (conditional returns, null checks,\n"
            "     CSS that hides the element).\n"
            "  4. For visual / layout bugs (invisible, wrong color, wrong position):\n"
            "     also check CSS/SCSS files. Use KB_SEARCH: <filename>.css to read\n"
            "     global or component-scoped stylesheets. FIND_USAGES also searches\n"
            "     CSS files for class-name references (.SnakeBoard, .snake-board).\n"
            "  5. Root cause MUST cite: specific file + specific prop/line/condition\n"
            "     or CSS rule that is wrong. Generic hypotheses are NOT acceptable.\n"
            "  You CANNOT write Root cause from patterns or training knowledge alone.\n"
            "  If you have not read the actual component/CSS code, use a tool first.\n\n"
            "FEATURE:\n"
            "  Find what already exists first — reuse > rebuild.\n"
            "  Identify the exact integration point (file, function, import path).\n"
            "  Check for naming conflicts or overlapping tests.\n\n"
            "MODIFY:\n"
            "  Locate the current implementation — read it before suggesting changes.\n"
            "  Identify what must be preserved (tests, public API, callers).\n"
            "  Define the minimal diff, not a rewrite.\n\n"
            "QUESTION (only for genuine code understanding requests):\n"
            "  Read actual code evidence. Do NOT use training-data patterns or\n"
            "  general best-practices as evidence — cite actual file content.\n\n"

            "── STEP 3: Conclude or investigate ───────────────────────────────\n"
            "Can you fill every required field of the REQUIREMENTS_SPEC with specific\n"
            "code evidence (file names, prop names, function signatures, data shapes)\n"
            "from what is already shown in the evidence above?\n\n"
            "  YES → write REQUIREMENTS_SPEC immediately.\n\n"
            "  NO  → state the ONE specific question you cannot answer from existing\n"
            "        evidence, then on the next line emit exactly one tool call.\n\n"
            "        Available tools:\n\n"
            "        KB_SEARCH: <query>\n"
            "          When: you need source code for a file or symbol not yet shown.\n\n"
            "        FIND_USAGES: <ComponentName or functionName>\n"
            "          When: you need to see both sides of an interface — what a\n"
            "          component DECLARES vs what its CALLER PASSES.\n\n"
            "        SEARCH: <query>\n"
            "          When: you need external docs or package versions.\n\n"
            "        Example:\n"
            "          I need to see what props SnakeGame passes to SnakeBoard\n"
            "          to check for a prop-name mismatch.\n"
            "          FIND_USAGES: SnakeBoard\n\n"
            + self._spec_all_formats()
        )
        return prompt

    def _spec_all_formats(self) -> str:
        """Return all four REQUIREMENTS_SPEC format templates as a single block."""
        return (
            "── REQUIREMENTS_SPEC formats (use the one matching your task type) ──\n\n"
            + self._SPEC_BUG + "\n"
            + self._SPEC_FEATURE + "\n"
            + self._SPEC_MODIFY + "\n"
            + self._SPEC_QUESTION
        )

    @staticmethod
    def _extract_search_query(raw_task: str) -> str:
        """
        Derive a compact KB search query from any raw task text.

        Strips stop-words and punctuation, keeps the first 8 meaningful tokens.
        Works for all task types: bugs, features, questions, refactors.
        """
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', raw_task)
        keywords = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2]
        return " ".join(keywords[:8])

    @staticmethod
    def _query_local_source(kb_context_builder, query: str) -> str:
        """
        Search the LOCAL source index (actual project files) for *query*.

        Strategy:
          1. Filename-direct lookup — if the query looks like a filename
             (contains a dot extension) read the full file from disk so the
             LLM sees the complete source, not just a semantic snippet.
          2. Semantic search — for conceptual queries, returns top-5 matching
             code snippets from the embedded index.

        This means KB_SEARCH: SnakeSegment.jsx reliably returns the full file
        even when the semantic ranker would rank a different file higher.
        """
        try:
            if not kb_context_builder._ensure_local():
                return ""

            project_root = kb_context_builder._project_root or ""

            # ── Strategy 1: filename-direct lookup ─────────────────────────
            # Detect if the query is a bare filename or ends with a known
            # source extension.  Walk the project to find it and return its
            # full content (capped at 200 lines to stay within context limits).
            _FILE_EXTS = {
                ".js", ".jsx", ".ts", ".tsx", ".py", ".vue", ".svelte",
                ".css", ".html", ".json",
            }
            query_stripped = query.strip().split("/")[-1].split("\\")[-1]
            if (
                "." in query_stripped
                and os.path.splitext(query_stripped)[1].lower() in _FILE_EXTS
                and project_root
            ):
                target_name = query_stripped.lower()
                skip_dirs = {"node_modules", ".git", "dist", "build", ".next", "__pycache__"}
                for dirpath, dirnames, filenames in os.walk(project_root):
                    dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                    for fname in filenames:
                        if fname.lower() == target_name:
                            abs_path = os.path.join(dirpath, fname)
                            rel_path = os.path.relpath(abs_path, project_root).replace("\\", "/")
                            try:
                                with open(abs_path, encoding="utf-8", errors="ignore") as fh:
                                    all_lines = fh.readlines()
                                cap = 200
                                truncated = len(all_lines) > cap
                                content = "".join(all_lines[:cap])
                                if truncated:
                                    content += f"\n  ... ({len(all_lines) - cap} more lines)"
                                return (
                                    f"### {fname} (full source) — {rel_path}\n"
                                    f"{content}"
                                )
                            except OSError:
                                pass

            # ── Strategy 2: semantic search ────────────────────────────────
            searcher = kb_context_builder._searcher
            if searcher is None:
                return ""
            results = searcher.search(query=query, top_k=5)
            if not results:
                return ""
            sections = []
            for r in results:
                header = (
                    f"### {r.symbol_name} ({r.symbol_type}) "
                    f"— {r.file} lines {r.line_start}-{r.line_end}"
                )
                lines = (r.code_snippet or "").splitlines()
                if len(lines) > 30:
                    lines = lines[:30] + ["  ..."]
                sections.append(header + "\n" + "\n".join(lines))
            return "\n\n".join(sections)
        except Exception as exc:
            _logger.debug("[IntentAnalysis] Local source search failed: %s", exc)
            return ""

    @staticmethod
    def _query_kb(kb_context_builder, query: str) -> str:
        """
        Query the LOCAL source index for *query*.

        Returns actual project file snippets — the same index the planner and
        coder use.  Global KB docs (framework guides, patterns) are deliberately
        excluded here: they are irrelevant to intent analysis (the intent agent
        needs to know what the code currently *does*, not how to write new code),
        and they are already passed to the planner separately via pre-analysis.
        Excluding them keeps the per-iteration context small and focused.
        """
        return IntentAgent._query_local_source(kb_context_builder, query)

    @staticmethod
    def _find_usages(kb_context_builder, name: str) -> str:
        """
        Find files that import *name* and show the actual call site lines.

        Strategy (three escalating levels):
          1. Graph symbol lookup  — exact name match across all node types
          2. Graph file-basename  — finds the file whose stem matches *name*
             (covers anonymous default exports, const arrow components, etc.)
          3. Filesystem grep      — walks project source files looking for
             ``import … name`` / ``<name`` patterns when the graph returns no
             importers (handles the case where the graph was not built or was
             built before the import was added)

        Returns a formatted string showing the definition side (what props the
        component declares) and the caller side (what props the caller passes),
        so the LLM can spot prop-name mismatches, format mismatches, and
        missing props in a single response.
        """
        try:
            if not kb_context_builder._ensure_local():
                return ""
            graph = kb_context_builder._graph
            if graph is None:
                return ""

            project_root = kb_context_builder._project_root

            # ── Level 1: exact symbol lookup ──────────────────────────────────
            symbols = graph.find_symbol(name)
            definition_files = list({
                s.get("file_path", "") for s in symbols if s.get("file_path")
            })

            # ── Level 2: file-basename fallback ───────────────────────────────
            # Covers components defined as `const X = () => …` (VARIABLE nodes
            # may or may not carry the right name) and anonymous default exports.
            if not definition_files:
                name_lower = name.lower()
                for file_node in graph.get_all_file_nodes():
                    fp = file_node.get("path", "")
                    if not fp:
                        continue
                    stem = os.path.splitext(os.path.basename(fp))[0].lower()
                    if stem == name_lower:
                        definition_files.append(fp)

            # ── Find importers via graph impact_analysis ───────────────────────
            importer_files: set[str] = set()
            for def_file in definition_files:
                # Normalise to forward slashes so _file_id lookup is consistent
                def_file_norm = def_file.replace("\\", "/")
                importers = graph.impact_analysis(def_file_norm)
                if not importers:
                    # Also try native separators (Windows graph may use \)
                    importers = graph.impact_analysis(def_file)
                importer_files.update(importers)

            # ── Level 3: filesystem grep fallback for importers ────────────────
            # Used when graph has no importer edges (e.g. not yet embedded, or
            # edge was added after the last kb build).
            if not importer_files and project_root:
                importer_files = IntentAgent._grep_importers(
                    project_root, name, exclude_files=set(definition_files),
                )

            if not definition_files and not importer_files:
                return f"No definition or importers found for '{name}'."

            sections = []

            # Show definition side: component signature / prop declarations
            for def_file in sorted(definition_files)[:2]:
                abs_path = os.path.join(project_root, def_file)
                snippet = IntentAgent._extract_lines_containing(abs_path, name, context=5)
                if not snippet:
                    # Show the top of the file (signature often near the top)
                    snippet = IntentAgent._read_head(abs_path, lines=40)
                if snippet:
                    sections.append(f"### Definition ({def_file})\n{snippet}")

            # Show caller side: actual props / arguments passed
            for imp_file in sorted(importer_files)[:3]:
                abs_path = os.path.join(project_root, imp_file)
                snippet = IntentAgent._extract_lines_containing(abs_path, name, context=5)
                if snippet:
                    sections.append(f"### Used in ({imp_file})\n{snippet}")

            if not sections:
                return f"Files found but no lines containing '{name}'."
            return "\n\n".join(sections)

        except Exception as exc:
            _logger.debug("[IntentAnalysis] FIND_USAGES failed for '%s': %s", name, exc)
            return ""

    @staticmethod
    def _grep_importers(project_root: str, name: str, exclude_files: set) -> set[str]:
        """
        Walk *project_root* and return relative paths of source files that
        import or use *name* (JSX tag, ES6 import, CommonJS require).

        Skips node_modules, .git, and dist/build directories.
        Returns at most 5 matches to keep context bounded.
        """
        matches: set[str] = set()
        # Kebab-case variant: SnakeBoard → snake-board, Food → food
        _kebab = re.sub(r'(?<!^)(?=[A-Z])', '-', name).lower()
        patterns = [
            # JS/JSX import and usage
            f"import {name}",
            f"import {{ {name}",
            f"import {{{name}",
            f"require('{name}')",
            f'require("{name}")',
            f"<{name}",
            f"<{name} ",
            f"<{name}/",
            # CSS/SCSS class selectors (PascalCase and kebab-case)
            f".{name}",
            f".{name} ",
            f".{name}{{",
            f".{_kebab}",
            f".{_kebab} ",
            f".{_kebab}{{",
        ]
        skip_dirs = {"node_modules", ".git", "dist", "build", ".next", "__pycache__"}
        source_exts = {".js", ".jsx", ".ts", ".tsx", ".py", ".vue", ".svelte",
                       ".css", ".scss", ".sass", ".less"}

        for dirpath, dirnames, filenames in os.walk(project_root):
            # Prune skip directories in-place
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() not in source_exts:
                    continue
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, project_root).replace("\\", "/")
                if rel_path in exclude_files:
                    continue
                try:
                    content = open(abs_path, encoding="utf-8", errors="ignore").read()
                    if any(p in content for p in patterns):
                        matches.add(rel_path)
                        if len(matches) >= 5:
                            return matches
                except OSError:
                    continue
        return matches

    @staticmethod
    def _read_head(file_path: str, lines: int = 40) -> str:
        """Return the first *lines* lines of *file_path* as a numbered string."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as fh:
                head = []
                for i, line in enumerate(fh):
                    if i >= lines:
                        break
                    head.append(f"{i + 1}: {line}")
            return "".join(head)
        except OSError:
            return ""

    @staticmethod
    def _extract_lines_containing(file_path: str, name: str, context: int = 4) -> str:
        """
        Return lines from *file_path* that contain *name*, with *context*
        lines of surrounding code so the LLM sees enough to understand usage.

        Limits output to the first 3 matches to keep context size small.
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as fh:
                all_lines = fh.readlines()
        except OSError:
            return ""

        chunks = []
        seen_ranges: list[tuple[int, int]] = []

        for i, line in enumerate(all_lines):
            if name not in line:
                continue
            start = max(0, i - context)
            end = min(len(all_lines), i + context + 1)
            # Skip if this range overlaps an already-included range
            if any(s <= i <= e for s, e in seen_ranges):
                continue
            seen_ranges.append((start, end))
            chunk = "".join(
                f"{start + j + 1}: {l}" for j, l in enumerate(all_lines[start:end])
            )
            chunks.append(chunk)
            if len(chunks) >= 3:
                break

        return "\n---\n".join(chunks)

import os
import platform
import re
import logging

from .base import Agent, uniquify_context

_logger = logging.getLogger(__name__)


def _shell_example() -> str:
    """Return an OS-appropriate file-listing command example."""
    if os.name == 'nt':
        return "  1. List all project files with `dir /s /b`"
    return "  1. List all project files with `find . -type f`"


def _os_context_for_prompt() -> str:
    """Return OS/shell context string for LLM prompts."""
    if os.name == 'nt':
        return (
            f"HOST OS: Windows ({platform.version()})\n"
            "SHELL: cmd.exe (NOT bash, NOT PowerShell)\n"
            "All shell commands in backticks MUST use Windows cmd.exe syntax.\n"
            "  - Use `mkdir` instead of `mkdir -p`\n"
            "  - Use `call venv\\Scripts\\activate` instead of `source venv/bin/activate`\n"
            "  - Use `set VAR=value` instead of `export VAR=value`\n"
            "  - Use `del` instead of `rm`, `copy` instead of `cp`\n"
            "  - Use `dir /s /b` instead of `find . -type f` or `ls`\n"
            "  - Use `type <file>` instead of `cat <file>`\n"
            "  - Do NOT use PowerShell cmdlets (Get-ChildItem, etc.)\n"
        )
    sysname = platform.system()
    os_label = "macOS" if sysname == "Darwin" else sysname
    shell = os.environ.get('SHELL', '/bin/bash').rsplit('/', 1)[-1]
    return (
        f"HOST OS: {os_label} ({platform.release()})\n"
        f"SHELL: {shell}\n"
        "All shell commands in backticks should use standard Unix shell syntax.\n"
    )


# ── Task intent classification (regex-based, no LLM) ────────────

_INTENT_PATTERNS: dict[str, list[re.Pattern]] = {
    "bug_fix": [
        re.compile(r'\b(fix|bug|error|crash|broken|fail|issue|wrong|incorrect|not working)\b', re.I),
        re.compile(r'\b(debug|traceback|exception|stack\s*trace|segfault)\b', re.I),
    ],
    "refactor": [
        re.compile(r'\b(refactor|restructure|reorganize|clean\s*up|simplify|optimize|improve)\b', re.I),
        re.compile(r'\b(rename|extract|move|split|merge|consolidate)\b', re.I),
    ],
    "test": [
        re.compile(r'\b(test|spec|coverage|unittest|pytest|jest|vitest)\b', re.I),
        re.compile(r'\b(write\s+\w*\s*tests?|add\s+\w*\s*tests?|create\s+\w*\s*tests?|unit\s+tests?|integration\s+tests?)\b', re.I),
    ],
    "feature": [
        re.compile(r'\b(add|create|implement|build|develop|new|introduce)\b', re.I),
        re.compile(r'\b(feature|endpoint|page|component|module|api|route)\b', re.I),
    ],
}


def _classify_task_intent(task: str) -> str:
    """Classify task intent without LLM. Returns one of:
    bug_fix, refactor, test, feature, general.
    """
    scores: dict[str, int] = {k: 0 for k in _INTENT_PATTERNS}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if pat.search(task):
                scores[intent] += 1

    if not any(scores.values()):
        return "general"

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _build_file_skeleton(content: str, max_lines: int = 30) -> str:
    """Extract a compact skeleton from file content: imports + signatures."""
    lines = content.splitlines()
    skeleton: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Keep imports
        if stripped.startswith(("import ", "from ", "require(", "const ", "let ", "var ")):
            if "import" in stripped or "require" in stripped:
                skeleton.append(line)
                continue
        # Keep class/function/method definitions
        if re.match(r'^\s*(def |class |async def |function |export |const \w+ = |async function )', line):
            skeleton.append(line)
            continue
        # Keep decorators
        if stripped.startswith("@"):
            skeleton.append(line)
            continue

    if len(skeleton) > max_lines:
        skeleton = skeleton[:max_lines]

    return "\n".join(skeleton) if skeleton else content[:500]


def _find_relevant_files(task: str, source_files: dict[str, str] | None,
                         kb_context_builder=None, max_files: int = 5
                         ) -> list[tuple[str, str, str]]:
    """Find files relevant to the task.

    Returns list of (path, reason, skeleton) tuples.
    """
    results: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()

    # Strategy 1: KB semantic search (best quality)
    if kb_context_builder is not None:
        try:
            kb_results = kb_context_builder.get_relevant_files(
                task_description=task, changed_files=[], max_files=max_files)
            if kb_results and source_files:
                for fpath in kb_results[:max_files]:
                    if fpath in source_files:
                        skeleton = _build_file_skeleton(source_files[fpath])
                        results.append((fpath, "KB semantic match", skeleton))
                        seen_paths.add(fpath)
        except Exception as e:
            _logger.debug(f"[PreAnalysis] KB search failed: {e}")

    # Strategy 2: Keyword matching — always runs to catch filename-specific
    # matches that semantic search may miss (e.g. "SnakeGame" in task vs
    # App.jsx routing wrapper ranked higher by embeddings).
    # Results are merged with KB results, deduplicated, capped at max_files.
    if source_files:
        stop_words = {
            "the", "a", "an", "to", "in", "for", "of", "and", "or", "is",
            "it", "on", "at", "by", "with", "from", "as", "be", "this",
            "that", "all", "are", "was", "were", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "can", "not", "but", "if", "then", "so",
            "add", "create", "update", "fix", "implement", "make", "use",
            "new", "file", "code", "project",
        }
        words = set(re.findall(r'\b[a-zA-Z_]\w{2,}\b', task.lower())) - stop_words

        if words:
            scored: list[tuple[str, int]] = []
            for fpath, content in source_files.items():
                score = 0
                fpath_lower = fpath.lower()
                content_lower = content.lower()[:2000]  # limit scan

                for word in words:
                    if word in fpath_lower:
                        score += 3  # filename match is strong signal
                    if word in content_lower:
                        score += 1

                if score > 0:
                    scored.append((fpath, score))

            scored.sort(key=lambda x: -x[1])
            for fpath, score in scored[:max_files]:
                if fpath not in seen_paths and len(results) < max_files:
                    skeleton = _build_file_skeleton(source_files[fpath])
                    results.append((fpath, f"keyword match (score={score})", skeleton))
                    seen_paths.add(fpath)

    return results


def _extract_question_answer(enriched_task: str) -> str:
    """
    Pull the Answer + Evidence block out of a QUESTION REQUIREMENTS_SPEC.

    The enriched task looks like:
      <raw task>

      === INTENT CLARIFICATION ===
      REQUIREMENTS_SPEC:
      Task type: QUESTION
      Goal: ...
      Answer: ...
      Evidence: ...
      KB topics: none

    Returns everything from 'Answer:' to the end of the spec, stripped.
    Falls back to the full spec block when the Answer field is not found.
    """
    # Find the spec block
    spec_m = re.search(
        r'REQUIREMENTS_SPEC:(.+)',
        enriched_task,
        re.IGNORECASE | re.DOTALL,
    )
    if not spec_m:
        return enriched_task.strip()
    spec = spec_m.group(1).strip()

    # Extract from Answer: onward (drop Task type / Goal header noise)
    answer_m = re.search(r'(?:^|\n)(Answer:\s*.+)', spec, re.IGNORECASE | re.DOTALL)
    if answer_m:
        # Strip trailing KB topics line — it's internal metadata, not for users
        answer_text = answer_m.group(1).strip()
        answer_text = re.sub(r'\nKB topics:.*$', '', answer_text, flags=re.IGNORECASE | re.DOTALL).strip()
        return answer_text
    return spec


class PlannerAgent(Agent):

    def _build_prompt(self, task: str, context: str, language: str | None = None) -> str:
        """Override base to skip the platform line — the planner's hard-coded
        template already includes a detailed HOST ENVIRONMENT section via
        _os_context_for_prompt(), so the base class one-liner is redundant."""
        prompt = f"Role: {self.role}\nGoal: {self.goal}\n\n"
        if language:
            from ..language import get_language_name
            prompt += f"Language: {get_language_name(language)}\n\n"
        # NOTE: platform line intentionally omitted — provided later
        if self.prompt_suffix:
            prompt += f"Instructions: {self.prompt_suffix}\n\n"
        if context:
            context = uniquify_context(context)
            prompt += f"Context: {context}\n\n"
        prompt += f"Task: {task}\n\nResponse:"
        return prompt

    def pre_analyze(self, task: str, *,
                    source_files: dict[str, str] | None = None,
                    kb_context_builder=None,
                    knowledge_base=None,
                    test_analysis: str | None = None,
                    language: str | None = None,
                    baseline_passing_files: list[str] | None = None,
                    baseline_failing_files: list[str] | None = None,
                    search_agent=None,
                    intent_agent=None,
                    cli_display=None,
                    subproject_cwd: str | None = None) -> str:
        """Analyze the task and project to build enriched planner context.

        Runs BEFORE process(). Returns a context string to prepend to
        the existing planner context. Returns empty string if nothing useful.

        Side-effect: sets ``self._detected_language`` when the LLM disagrees
        with the heuristic language — callers can read this to override the
        language for subsequent agents.
        """
        parts: list[str] = []

        # 0. LLM-based language detection — runs before anything else so the
        # corrected language can be used by the rest of the pipeline.
        # Only triggers when we have file paths to reason about.
        self._detected_language: str | None = None
        if source_files:
            try:
                from ..language import detect_language_llm
                _llm_lang = detect_language_llm(
                    file_paths=list(source_files.keys()),
                    task=task,
                    llm_client=self.llm_client,
                    current_language=language,
                )
                if _llm_lang and _llm_lang != language:
                    _logger.info(
                        "[PreAnalysis] LLM corrected language: %s → %s",
                        language, _llm_lang,
                    )
                    self._detected_language = _llm_lang
                    language = _llm_lang  # use corrected language for this analysis
            except Exception as _ld_exc:
                _logger.debug("[PreAnalysis] LLM language detection skipped: %s", _ld_exc)

        # 1. Task intent classification
        intent = _classify_task_intent(task)
        if intent != "general":
            intent_hints = {
                "bug_fix": "This is a BUG FIX task. Focus on identifying the root cause and fixing it. Avoid unnecessary refactoring.",
                "refactor": "This is a REFACTORING task. Focus on restructuring without changing behavior. Preserve all existing tests.",
                "test": "This is a TESTING task. Focus on writing comprehensive tests for existing code.",
                "feature": "This is a NEW FEATURE task. Plan for implementation, integration with existing code, and proper error handling.",
            }
            parts.append(f"[Task Analysis] {intent_hints.get(intent, '')}")

        # 2. Find and annotate relevant files (KB search first, keyword fallback)
        relevant = _find_relevant_files(
            task, source_files, kb_context_builder, max_files=5)

        if relevant:
            parts.append("\n[Relevant Files Analysis]")
            parts.append("The following existing files are most relevant to this task:")
            for fpath, reason, skeleton in relevant:
                parts.append(f"\n--- {fpath} ({reason}) ---")
                parts.append(skeleton)
                if intent == "bug_fix":
                    parts.append(f"  ^ Check this file for the bug described in the task")
                elif intent == "feature":
                    parts.append(f"  ^ This file may need modification for the new feature")

        # 2b. Intent Analysis — runs BEFORE task briefing so the enriched
        # task is available for the briefing LLM call.
        # For blank folders: IntentAgent runs in "scaffold mode" — skips
        # local KB search (nothing exists) but uses web search to fetch
        # setup guides for the frameworks mentioned in the task, and
        # version detection to find the right docs.
        _raw_task = task  # preserve original for clean KB search queries
        if intent_agent is not None:
            if cli_display:
                cli_display.show_status("Analyzing intent and gathering requirements...")

            _logger.info("[PreAnalysis] Compiling semantic context for IntentAgent...")
            _semantic_kb = []

            # Local knowledge base entries (tech stack, patterns, etc.)
            if knowledge_base and knowledge_base.size > 0:
                _semantic_kb.append(knowledge_base.format_for_planner() or "")

            # Global KB doc titles — pass only titles (no content) so the
            # IntentAgent can pick the relevant ones and emit them as
            # `KB docs:` in its spec.  The planner will then load those docs
            # directly via get_by_titles() — no re-embedding needed.
            _available_kb_titles: list[str] = []
            if kb_context_builder is not None:
                try:
                    kb_context_builder._ensure_global()
                    _gs = kb_context_builder._global_store
                    if _gs is not None:
                        _title_hits = _gs.search(
                            query=_raw_task,
                            categories=["doc", "pattern"],
                            top_k=12,
                            api_client=kb_context_builder._api_client,
                        )
                        _available_kb_titles = [
                            r.title for r in (_title_hits or []) if r.title
                        ]
                        if _available_kb_titles:
                            _logger.info(
                                "[PreAnalysis] %d global KB titles surfaced "
                                "for IntentAgent",
                                len(_available_kb_titles),
                            )
                except Exception:
                    pass

            # Relevant source file skeletons (for existing projects)
            if relevant:
                _semantic_kb.append("Relevant Files Current State:\n" + "\n".join(
                    f"[{fpath}] {skeleton}" for fpath, _, skeleton in relevant
                ))

            _full_semantic_context = "\n\n".join(_semantic_kb)

            task = intent_agent.analyze_intent(
                task, search_agent=search_agent,
                kb_context_builder=kb_context_builder,
                kb_context=_full_semantic_context,
                cli_display=cli_display,
                available_kb_docs=_available_kb_titles or None,
                subproject_cwd=subproject_cwd,
            )
            self._enriched_task = task
            _logger.info("[PreAnalysis] Task intent enriched.")

            # Short-circuit for QUESTION tasks — the IntentAgent already
            # produced a complete answer.  Skip briefing, global KB search,
            # and the planner entirely; callers inspect _is_question_task.
            if re.search(r'Task type:\s*QUESTION', task, re.IGNORECASE):
                self._is_question_task = True
                self._question_answer = _extract_question_answer(task)
                _logger.info(
                    "[PreAnalysis] Task type is QUESTION — skipping briefing and planner.",
                )
                return "\n".join(parts)

        # 2c. LLM-based task briefing using ONLY the pre-filtered relevant files.
        # We deliberately pass only the files from step 2 (not the whole project)
        # so the LLM gets focused context without being flooded.  The result is
        # a concrete TASK BRIEFING block injected as the highest-priority context.
        # Clear the investigation trail now — the user has had time to read the
        # spec detail rows while we prepared contracts below.
        if cli_display:
            cli_display.clear_intent_events()
        if relevant or test_analysis:
            try:
                from ..orchestrator.test_analyzer import analyze_task_for_planner
                import os as _os

                # Build behavioral contracts for each editable file:
                #   source  — full content from source_files (richer than skeleton)
                #   tests   — content of any passing test files whose name matches
                #             the source file basename (e.g. SnakeGame → SnakeGame.test.jsx)
                # When no test file exists, source content alone forms the contract.
                _contracts: dict[str, dict] = {}
                _MAX_SRC_CHARS = 6000   # cap per file to stay within prompt budget

                for _fpath, _reason, _skeleton in relevant:
                    _entry: dict = {}

                    # Full source content (truncated if large)
                    if source_files and _fpath in source_files:
                        _src = source_files[_fpath]
                        _entry["source"] = (
                            _src[:_MAX_SRC_CHARS]
                            + ("\n... (truncated)" if len(_src) > _MAX_SRC_CHARS else "")
                        )
                    else:
                        _entry["source"] = _skeleton  # fallback to skeleton

                    # Matching test files — name-based matching
                    _entry["tests"] = {}
                    _src_base = _os.path.splitext(
                        _os.path.basename(_fpath))[0].lower()
                    if baseline_passing_files and source_files:
                        for _tpath in baseline_passing_files:
                            _tbase = _os.path.basename(_tpath).lower()
                            # Match: SnakeGame.jsx ↔ SnakeGame.test.jsx / snakegame.spec.ts
                            if _src_base in _tbase and _tpath in source_files:
                                _tcontent = source_files[_tpath]
                                _entry["tests"][_tpath] = (
                                    _tcontent[:_MAX_SRC_CHARS]
                                    + ("\n... (truncated)"
                                       if len(_tcontent) > _MAX_SRC_CHARS else "")
                                )

                    _contracts[_fpath] = _entry

                # ── Pre-fetch KB package docs BEFORE generating the briefing ──
                # If the IntentAgent already named specific KB docs, load them
                # directly (no keyword search needed — avoids 5 redundant
                # embedding calls and ensures the briefing uses exactly the
                # same docs the IntentAgent selected).
                # Fallback: keyword-token scan for tasks without KB docs.
                _pre_pkg_docs: list[str] = []
                if kb_context_builder is not None:
                    try:
                        from ..orchestrator.cli import _parse_kb_doc_titles as _pkdt
                        _briefing_doc_titles = _pkdt(task)
                        kb_context_builder._ensure_global()
                        _gs_pre = kb_context_builder._global_store
                        if _gs_pre is not None and _briefing_doc_titles:
                            # Fast path: use IntentAgent's picks directly
                            _briefing_hits = _gs_pre.get_by_titles(_briefing_doc_titles)
                            for _h in (_briefing_hits or []):
                                if _h.content or _h.title:
                                    _pre_pkg_docs.append(
                                        f"### {_h.title}\n{_h.content or ''}"
                                    )
                        elif _gs_pre is not None:
                            # Fallback: keyword-token search
                            import re as _re_pkg
                            _COMMON_WORDS = frozenset({
                                "the", "and", "for", "with", "this", "that",
                                "using", "use", "add", "implement", "enhance",
                                "game", "visuals", "visual", "animation",
                            })
                            _task_tokens = [
                                w.lower() for w in _re_pkg.findall(
                                    r'[a-zA-Z][a-zA-Z0-9._-]{2,}', task)
                                if w.lower() not in _COMMON_WORDS
                            ]
                            for _tok in _task_tokens[:5]:
                                _hits = _gs_pre.search(
                                    query=_tok,
                                    categories=["doc"],
                                    top_k=2,
                                    api_client=kb_context_builder._api_client,
                                )
                                for _h in (_hits or []):
                                    if _tok in (_h.title or "").lower():
                                        _pre_pkg_docs.append(
                                            f"### {_h.title}\n{_h.content or ''}"
                                        )
                                        break
                    except Exception:
                        pass

                _pre_pkg_context = (
                    "\n\n".join(_pre_pkg_docs) if _pre_pkg_docs else None
                )
                if _pre_pkg_context:
                    _logger.info(
                        "[PreAnalysis] Pre-fetched %d KB doc(s) for briefing",
                        len(_pre_pkg_docs),
                    )


                _briefing = analyze_task_for_planner(
                    task=task,
                    relevant_files=relevant,           # (path, reason, skeleton) tuples
                    test_analysis=test_analysis or "",
                    llm_client=self.llm_client,
                    passing_files=baseline_passing_files,
                    failing_files=baseline_failing_files,
                    editable_contracts=_contracts,
                    package_docs=_pre_pkg_context,
                )
                if _briefing:
                    parts.insert(
                        0,
                        "╔══ TASK BRIEFING (independent analysis before planning) ══╗\n"
                        + _briefing
                        + "\n╚════════════════════════════════════════════════════════╝\n"
                        "CRITICAL: The briefing above was produced by an independent analyst\n"
                        "who read your task against the actual project files and live test results.\n"
                        "Your plan MUST follow the 'Agent directive' line exactly.\n"
                        "The 'Preserve:' line lists behaviors the coder must NOT break — treat\n"
                        "each item as a hard constraint that applies to every CODE step you generate.\n"
                        "Do not second-guess the file list or add steps the briefing does not call for.",
                    )
                    _logger.info("[PreAnalysis] Task briefing injected.")
                    # Store on instance so callers can propagate it to memory
                    self._task_briefing = _briefing

                    # Inject full source of files being modified so the planner
                    # can copy them verbatim and make only the minimal change.
                    # Without this, the planner reconstructs files from memory
                    # and inevitably introduces unintended rewrites.
                    _MAX_INJECT_CHARS = 8000  # per-file cap
                    _src_sections: list[str] = []
                    for _fpath, _reason, _skeleton in relevant:
                        if source_files and _fpath in source_files:
                            _full = source_files[_fpath]
                            _truncated = len(_full) > _MAX_INJECT_CHARS
                            _src_sections.append(
                                f"### {_fpath}\n"
                                f"```\n"
                                + (_full[:_MAX_INJECT_CHARS]
                                   + ("\n... (truncated)" if _truncated else ""))
                                + "\n```"
                            )
                    if _src_sections:
                        parts.insert(
                            1,
                            "╔══ FILES TO MODIFY — FULL CURRENT SOURCE ══╗\n"
                            "CRITICAL: When writing content: blocks for CODE steps that MODIFY\n"
                            "these files, you MUST start from this exact source and change ONLY\n"
                            "what the 'Agent directive' above requires. Do NOT restructure,\n"
                            "rename, reorder, or add comments. Copy every line verbatim except\n"
                            "the specific className/value/line the directive points to.\n\n"
                            + "\n\n".join(_src_sections)
                            + "\n╚═══════════════════════════════════════════╝",
                        )
                        _logger.info(
                            "[PreAnalysis] Full source injected for %d file(s)",
                            len(_src_sections),
                        )

                    # 2c. Package documentation: parse "New packages:" from
                    # the briefing, then for each new package combine the
                    # existing KB doc (if any) with a live web search and let
                    # the LLM produce a single authoritative reference.
                    # This ensures the coder uses the correct install command,
                    # import path, and API for the current package version —
                    # even when the KB doc is stale.
                    if search_agent is not None and _briefing:
                        _pkg_line = ""
                        for _bl in _briefing.splitlines():
                            if _bl.startswith("New packages:"):
                                _pkg_line = _bl[len("New packages:"):].strip()
                                break

                        if _pkg_line and _pkg_line.upper() != "NONE":
                            _new_pkgs = [
                                p.strip()
                                for p in _pkg_line.split(",")
                                if p.strip() and p.strip().upper() != "NONE"
                            ][:3]  # cap at 3 to bound latency

                            _pkg_docs: list[str] = []
                            for _pkg in _new_pkgs:
                                # Pull any existing KB doc for this package
                                _existing_doc = ""
                                if kb_context_builder is not None:
                                    try:
                                        kb_context_builder._ensure_global()
                                        _gs = kb_context_builder._global_store
                                        if _gs is not None:
                                            _kb_hits = _gs.search(
                                                query=_pkg,
                                                categories=["doc", "pattern"],
                                                top_k=3,
                                                api_client=kb_context_builder._api_client,
                                            )
                                            if _kb_hits:
                                                # Prefer a hit whose title
                                                # contains the package name
                                                for _kh in _kb_hits:
                                                    if _pkg.lower() in (
                                                        _kh.title or ""
                                                    ).lower():
                                                        _existing_doc = (
                                                            _kh.content or ""
                                                        )
                                                        break
                                                if not _existing_doc:
                                                    _existing_doc = (
                                                        _kb_hits[0].content or ""
                                                    )
                                    except Exception:
                                        pass

                                _logger.info(
                                    "[PreAnalysis] Fetching docs for new "
                                    "package: %s (kb_doc=%d chars)",
                                    _pkg, len(_existing_doc),
                                )
                                _pkg_summary = search_agent.search_for_package(
                                    _pkg,
                                    language=language,
                                    existing_kb_doc=_existing_doc,
                                )
                                if _pkg_summary:
                                    _pkg_docs.append(
                                        f"### {_pkg}\n{_pkg_summary}"
                                    )

                            if _pkg_docs:
                                # Insert right after the briefing (index 0)
                                # so the planner sees package refs immediately
                                parts.insert(
                                    1,
                                    "\n[New Package Documentation]\n"
                                    "CRITICAL: Use EXACTLY these install "
                                    "commands, import paths, and APIs — they "
                                    "are verified against current package "
                                    "docs and override your training data:\n\n"
                                    + "\n\n".join(_pkg_docs),
                                )
                                _logger.info(
                                    "[PreAnalysis] %d package doc(s) injected",
                                    len(_pkg_docs),
                                )

                                # Re-generate the briefing now that we have
                                # verified (search + KB) package docs so the
                                # "Agent directive" uses the correct import
                                # paths / APIs instead of training-data guesses.
                                _full_pkg_context = (
                                    (_pre_pkg_context + "\n\n"
                                     if _pre_pkg_context else "")
                                    + "\n\n".join(_pkg_docs)
                                )
                                _logger.info(
                                    "[PreAnalysis] Re-generating briefing with "
                                    "verified package docs"
                                )
                                _updated_briefing = analyze_task_for_planner(
                                    task=task,
                                    relevant_files=relevant,
                                    test_analysis=test_analysis or "",
                                    llm_client=self.llm_client,
                                    passing_files=baseline_passing_files,
                                    failing_files=baseline_failing_files,
                                    editable_contracts=_contracts,
                                    package_docs=_full_pkg_context,
                                )
                                if _updated_briefing:
                                    _updated_section = (
                                        "╔══ TASK BRIEFING (independent analysis before planning) ══╗\n"
                                        + _updated_briefing
                                        + "\n╚════════════════════════════════════════════════════════╝\n"
                                        "CRITICAL: The briefing above was produced by an independent analyst\n"
                                        "who read your task against the actual project files and live test results.\n"
                                        "Your plan MUST follow the 'Agent directive' line exactly.\n"
                                        "The 'Preserve:' line lists behaviors the coder must NOT break — treat\n"
                                        "each item as a hard constraint that applies to every CODE step you generate.\n"
                                        "Do not second-guess the file list or add steps the briefing does not call for."
                                    )
                                    parts[0] = _updated_section
                                    self._task_briefing = _updated_briefing
                                    _logger.info(
                                        "[PreAnalysis] Briefing updated with "
                                        "verified package docs."
                                    )

            except Exception as _br_exc:
                _logger.warning("[PreAnalysis] Task briefing failed: %s", _br_exc)

        # 3. Knowledge base context — SKIPPED here to avoid duplication.
        # knowledge_base.format_for_planner() is already injected by api.py
        # into the planner context. Adding it again here would duplicate
        # installed packages, tech stack, and patterns.

        # 3b. Task-only intent pass for blank projects.
        # When there are no source files the IntentAgent was skipped (step 2b)
        # because it has nothing to investigate.  Instead of falling back to
        # pure vector search (which matches by embedding similarity and
        # over-fetches), we make ONE lightweight LLM call:
        #   task description + available KB doc titles → LLM picks exact subset
        # The output is appended to `task` as a "KB docs:" line so the
        # fast-path in step 4 loads them via get_by_titles() — no re-embedding.
        if not source_files and intent_agent is not None and kb_context_builder is not None:
            try:
                kb_context_builder._ensure_global()
                _gs = kb_context_builder._global_store
                if _gs is not None:
                    _blank_hits = _gs.search(
                        query=task,
                        categories=["doc", "pattern"],
                        top_k=15,
                        api_client=kb_context_builder._api_client,
                        language=language,
                    ) or []
                    _blank_titles = [r.title for r in _blank_hits if r.title]
                    if _blank_titles:
                        _numbered = "\n".join(
                            f"{i + 1}. {t}" for i, t in enumerate(_blank_titles)
                        )
                        _select_prompt = (
                            "You are selecting documentation for a coding task in a "
                            "brand-new empty project (no existing source files).\n\n"
                            f"Task: {task}\n\n"
                            "Available documentation titles:\n"
                            f"{_numbered}\n\n"
                            "Select ONLY the titles directly needed to complete this task.\n"
                            "Output EXACTLY one line in this format:\n"
                            "KB docs: <comma-separated exact titles from the list above>\n\n"
                            "Rules:\n"
                            "- Use exact title strings as shown above\n"
                            "- Maximum 16 titles\n"
                            "- Prioritize setup/install guides for every framework or tool "
                            "explicitly named in the task — these are essential for a new project\n"
                            "- Exclude titles about unrelated frameworks or languages\n"
                            "- If none are relevant write: KB docs: none"
                        )
                        _kb_select_resp = intent_agent.llm_client.generate_response(
                            _select_prompt
                        )
                        _kbm = re.search(
                            r'KB docs[^:\n]*:\s*(.+)', _kb_select_resp, re.IGNORECASE
                        )
                        if _kbm:
                            _selected = _kbm.group(1).strip()
                            if _selected.lower() not in ('none', 'n/a', ''):
                                # ── Force-include setup guides for task-mentioned tech ──
                                # Smaller LLMs sometimes omit setup/install guides even
                                # when the technology is explicitly named in the task.
                                # For blank projects every named framework needs its
                                # setup doc, so we patch the selection with a keyword
                                # safety net — zero LLM cost.
                                _task_kw = {
                                    w.lower() for w in re.findall(r'[A-Za-z][\w.-]*', task)
                                } - {
                                    "a", "an", "the", "is", "it", "in", "on", "at",
                                    "to", "for", "of", "and", "or", "with", "using",
                                    "have", "has", "be", "by", "as", "all", "app",
                                    "implement", "create", "build", "add", "make",
                                    "use", "integrated", "components", "component",
                                    "web", "page", "responsive", "testcases",
                                }
                                _selected_lower = _selected.lower()
                                _forced: list[str] = []
                                # Build task tech keywords for framework
                                # conflict filtering — reuse the same
                                # normalize_tech_keywords used later in
                                # the doc filter pipeline.
                                try:
                                    from ..orchestrator.plan_optimizer import (
                                        _TECH_KEYWORDS as _TK_f,
                                        has_framework_conflict as _hfc,
                                        normalize_tech_keywords as _ntk,
                                    )
                                    _task_techs_f = _ntk(set(
                                        w.lower() for w in _TK_f.findall(task)
                                    ))
                                except ImportError:
                                    _task_techs_f = set()
                                    _hfc = None  # type: ignore[assignment]
                                    _TK_f = None  # type: ignore[assignment]
                                    _ntk = None  # type: ignore[assignment]
                                for _t in _blank_titles:
                                    _tl = _t.lower()
                                    if ("setup" in _tl or "install" in _tl) and \
                                            _t not in _selected:
                                        if any(kw in _tl for kw in _task_kw):
                                            # Check framework conflict before
                                            # force-including (e.g. don't add
                                            # Django guide for a pyglet project
                                            # or Anime.js for a Python project)
                                            if _hfc and _TK_f and _ntk and _task_techs_f:
                                                _doc_techs_f = _ntk(set(
                                                    w.lower()
                                                    for w in _TK_f.findall(_t)
                                                ))
                                                if _doc_techs_f and _hfc(
                                                    _task_techs_f, _doc_techs_f
                                                ):
                                                    _logger.debug(
                                                        "[PreAnalysis] Skipping "
                                                        "force-include '%s' — "
                                                        "framework conflict", _t,
                                                    )
                                                    continue
                                            # Also check language mismatch:
                                            # skip guides for a different
                                            # language family
                                            if language:
                                                _hit = next(
                                                    (r for r in _blank_hits
                                                     if r.title == _t), None)
                                                if (_hit
                                                        and _hit.language != "all"
                                                        and _hit.language.lower()
                                                        != language.lower()):
                                                    _logger.debug(
                                                        "[PreAnalysis] Skipping "
                                                        "force-include '%s' — "
                                                        "language mismatch "
                                                        "(doc=%s, project=%s)",
                                                        _t, _hit.language,
                                                        language,
                                                    )
                                                    continue
                                            _forced.append(_t)
                                if _forced:
                                    _selected = _selected + ", " + ", ".join(_forced)
                                    _logger.info(
                                        "[PreAnalysis] Force-included setup guides: %s",
                                        ", ".join(_forced),
                                    )
                                task = task + f"\n\nKB docs: {_selected}"
                                _logger.info(
                                    "[PreAnalysis] Task-only KB select (blank project): %s",
                                    _selected,
                                )
                                if cli_display:
                                    cli_display.show_status(
                                        f"KB docs selected: {_selected[:70]}"
                                    )
            except Exception as _toi_exc:
                _logger.debug(
                    "[PreAnalysis] Task-only intent pass failed: %s", _toi_exc
                )

        # 4. Global KB documentation (framework guides, installation docs)
        if kb_context_builder is not None:
            try:
                kb_context_builder._ensure_global()
                if kb_context_builder._global_store is not None:
                    from ..orchestrator.cli import _parse_kb_topics, _parse_kb_doc_titles

                    # Resolve docs via fast-path (title lookup) or fallback
                    # (vector search + topic filters).
                    _kb_doc_titles = _parse_kb_doc_titles(task)
                    _parsed_topics: list[str] = []
                    _use_topics = False
                    docs = None

                    if _kb_doc_titles:
                        # Fast path: IntentAgent named specific docs.
                        # get_by_titles() is a pure filesystem scan — no embedding.
                        _logger.info(
                            "[PreAnalysis] IntentAgent named %d KB doc(s) — "
                            "loading by title (skipping vector search)",
                            len(_kb_doc_titles),
                        )
                        docs = kb_context_builder._global_store.get_by_titles(
                            _kb_doc_titles
                        ) or []
                        if docs:
                            _preloaded_fp = list(docs)
                            # Also parse KB topics so context_builder can use
                            # them to filter per-step batch_search results even
                            # when the fast-path handles planner doc loading.
                            _fp_topics = _parse_kb_topics(task, re)
                            _fp_per_topic_kws: list[set[str]] = [
                                {w.lower() for w in re.findall(r'[a-zA-Z]{3,}', t)}
                                for t in _fp_topics if t
                            ]
                            if kb_context_builder is not None:
                                kb_context_builder._preloaded_docs = _preloaded_fp
                                kb_context_builder._intent_topics = _fp_per_topic_kws or []
                            _fp_hints = [
                                f"### {d.title}\n{d.content or d.title}"
                                for d in _preloaded_fp
                                if (d.content or d.title)
                            ]
                            if _fp_hints:
                                parts.append("\n[Framework/Library Documentation]")
                                parts.append(
                                    "CRITICAL: These KB docs are curated and up-to-date. "
                                    "You MUST follow them exactly:\n"
                                    "- Use the EXACT install commands from these docs "
                                    "(including all peer dependencies like postcss, jsdom, etc.)\n"
                                    "- If a doc says a command is deprecated/removed, "
                                    "do NOT use that command\n"
                                    "- Prefer patterns shown in these docs over your training data"
                                )
                                parts.extend(_fp_hints)
                                _logger.info(
                                    "[PreAnalysis] %d doc(s) injected (KB docs fast-path)",
                                    len(_fp_hints),
                                )
                        docs = None  # fully handled above; skip fallback filter loop
                    else:
                        # Fallback: vector search + Filter 0–4.
                        _parsed_topics = _parse_kb_topics(task, re)
                        _use_topics = bool(_parsed_topics)
                        _kb_topics_raw = ', '.join(_parsed_topics)
                        _search_query = _kb_topics_raw if _use_topics else _raw_task
                        _top_k = 10 if _use_topics else 20
                        _logger.info(
                            "[PreAnalysis] Querying global KB %s: %s",
                            "via KB topics" if _use_topics else "for task",
                            _search_query,
                        )
                        docs = kb_context_builder._global_store.search(
                            query=_search_query,
                            categories=["doc", "pattern"],
                            top_k=_top_k,
                            api_client=kb_context_builder._api_client,
                        )
                        _logger.info(
                            "[PreAnalysis] Global KB search returned %d docs",
                            len(docs) if docs else 0,
                        )

                    if docs:
                        # Fallback-path filter loop.
                        # Pre-compute per-topic keyword sets for Filter 0.
                        _topic_kws: set[str] = set()
                        _per_topic_kws: list[set[str]] = []
                        if _use_topics:
                            for _t in _parsed_topics:
                                kws = {
                                    w.lower()
                                    for w in re.findall(r'[a-zA-Z]{3,}', _t)
                                }
                                if kws:
                                    _per_topic_kws.append(kws)
                                    _topic_kws.update(kws)

                        def _topic_stem_match(a: str, b: str) -> bool:
                            """True when a and b share a common stem prefix."""
                            n = min(len(a), len(b))
                            if n < 3:
                                return False
                            pfx = min(n, 5)
                            return a[:pfx] == b[:pfx]

                        # Filter docs to only include relevant ones.
                        # Five filters applied in order:
                        # 1. Framework conflict: reject docs about a
                        #    conflicting framework (e.g. Angular for React)
                        # 2. Tech mismatch: skip docs whose tech keywords
                        #    have zero overlap with the task's tech keywords
                        # 3. Title relevance: skip docs where fewer than
                        #    _MIN_TITLE_SCORE of their title words appear in
                        #    the task — catches docs that share one secondary
                        #    tech (e.g. Vitest) but are primarily about
                        #    something unrelated (e.g. Three.js, OAuth)
                        # 4. Generic cap: allow a few topic-relevant generic
                        #    docs (no tech keywords) through as a backstop
                        from ..orchestrator.plan_optimizer import (
                            _TECH_KEYWORDS as _TK,
                            has_framework_conflict,
                            normalize_tech_keywords,
                        )
                        task_techs = normalize_tech_keywords(set(
                            w.lower() for w in _TK.findall(task)
                        ))
                        # Precompute task word set (≥4 chars) for title
                        # relevance scoring (Filter 3).
                        # Include the detected project language so that
                        # language-specific KB docs (e.g. "Python Stdlib
                        # Reference") are not filtered out when the task
                        # description doesn't mention the language by name.
                        _task_words_set = set(
                            re.findall(r'[a-zA-Z]{4,}', task.lower())
                        )
                        if language and len(language) >= 4:
                            _task_words_set.add(language.lower())
                        _MIN_TITLE_SCORE = 0.20  # ≥20% of title words in task
                        doc_hints: list[str] = []
                        _preloaded: list = []  # filtered GlobalKBResult objects
                        _MAX_GENERIC_DOCS = 2  # cap for topic-relevant generics
                        _generic_count = 0
                        for doc in docs:
                            doc_text = (
                                (doc.title or "") + " "
                                + " ".join(doc.tags or [])
                                + " " + ((doc.content or "")[:500])
                            )
                            doc_techs = normalize_tech_keywords(set(
                                w.lower() for w in _TK.findall(doc_text)
                            ))

                            # Filter 0: KB-topics allowlist (per-topic)
                            # When the IntentAgent specified explicit topics,
                            # a doc must stem-match ≥ min(2, topic_size)
                            # keywords from at least ONE complete topic.
                            # Single-word matching against the merged pool
                            # was too loose: "react" from topic "React
                            # (functional components)" admitted every React
                            # doc regardless of relevance.
                            if _per_topic_kws:
                                title_kws = set(
                                    re.findall(r'[a-zA-Z]{3,}',
                                               (doc.title or "").lower())
                                )
                                _passed_f0 = False
                                for _tkws in _per_topic_kws:
                                    _threshold = max(1, min(2, len(_tkws)))
                                    _matches = sum(
                                        1 for tw in title_kws
                                        if any(
                                            _topic_stem_match(tw, tk)
                                            for tk in _tkws
                                        )
                                    )
                                    if _matches >= _threshold:
                                        _passed_f0 = True
                                        break
                                if not _passed_f0:
                                    _logger.debug(
                                        "[PreAnalysis] Skipping '%s' — "
                                        "not in KB topics allowlist",
                                        doc.title,
                                    )
                                    continue

                            # Filter 1: framework conflict
                            if task_techs and has_framework_conflict(task_techs, doc_techs):
                                _logger.debug(
                                    "[PreAnalysis] Skipping '%s' — "
                                    "framework conflict",
                                    doc.title,
                                )
                                continue

                            # Filter 2: tech mismatch
                            if task_techs and doc_techs and not (task_techs & doc_techs):
                                _logger.debug(
                                    "[PreAnalysis] Skipping '%s' — "
                                    "tech mismatch (%s vs task %s)",
                                    doc.title, doc_techs, task_techs,
                                )
                                continue

                            # Filter 3: title relevance
                            # Compute what fraction of the doc's title words
                            # (≥4 chars) match the task text.  Uses prefix
                            # matching so "testing"↔"test", "errors"↔"error",
                            # "fixing"↔"fix", etc. are treated as matches.
                            title_words = set(
                                re.findall(r'[a-zA-Z]{4,}',
                                           (doc.title or "").lower())
                            )
                            if title_words:
                                def _title_word_matches(tw: str) -> bool:
                                    if tw in _task_words_set:
                                        return True
                                    for sw in _task_words_set:
                                        if tw.startswith(sw) or sw.startswith(tw):
                                            return True
                                    return False
                                title_score = (
                                    sum(1 for tw in title_words
                                        if _title_word_matches(tw))
                                    / len(title_words)
                                )
                                if title_score < _MIN_TITLE_SCORE:
                                    if doc_techs:
                                        # Tech-matched but off-topic title
                                        _logger.debug(
                                            "[PreAnalysis] Skipping '%s' — "
                                            "low title relevance "
                                            "(%.0f%% overlap, needs ≥%.0f%%)",
                                            doc.title,
                                            title_score * 100,
                                            _MIN_TITLE_SCORE * 100,
                                        )
                                        continue
                                    else:
                                        # Generic doc with irrelevant title
                                        _logger.debug(
                                            "[PreAnalysis] Skipping '%s' — "
                                            "off-topic generic "
                                            "(%.0f%% title overlap)",
                                            doc.title, title_score * 100,
                                        )
                                        continue
                                elif not doc_techs:
                                    # Filter 4: relevant generic doc — cap
                                    _generic_count += 1
                                    if _generic_count > _MAX_GENERIC_DOCS:
                                        _logger.debug(
                                            "[PreAnalysis] Skipping '%s' — "
                                            "generic cap reached (%d/%d)",
                                            doc.title, _generic_count,
                                            _MAX_GENERIC_DOCS,
                                        )
                                        continue
                                    _logger.debug(
                                        "[PreAnalysis] Including generic "
                                        "doc '%s' (%d/%d)",
                                        doc.title, _generic_count,
                                        _MAX_GENERIC_DOCS,
                                    )
                            elif not doc_techs:
                                # Generic doc with no parseable title words
                                # — apply cap as fallback
                                _generic_count += 1
                                if _generic_count > _MAX_GENERIC_DOCS:
                                    _logger.debug(
                                        "[PreAnalysis] Skipping '%s' — "
                                        "generic cap reached (%d/%d)",
                                        doc.title, _generic_count,
                                        _MAX_GENERIC_DOCS,
                                    )
                                    continue
                                _logger.debug(
                                    "[PreAnalysis] Including generic "
                                    "doc '%s' (%d/%d)",
                                    doc.title, _generic_count,
                                    _MAX_GENERIC_DOCS,
                                )

                            content = doc.content or doc.title
                            if content:
                                doc_hints.append(f"### {doc.title}\n{content}")
                                _preloaded.append(doc)
                                _logger.info(
                                    "[PreAnalysis] Loaded doc: '%s'",
                                    doc.title,
                                )
                        # Persist filtered docs so build_context merges
                        # them into every step's KB context automatically.
                        if _preloaded and kb_context_builder is not None:
                            kb_context_builder._preloaded_docs = _preloaded
                        # Also propagate topic filter so batch_search in
                        # context_builder can reuse the same allowlist.
                        if _per_topic_kws and kb_context_builder is not None:
                            kb_context_builder._intent_topics = _per_topic_kws
                        if doc_hints:
                            parts.append("\n[Framework/Library Documentation]")
                            parts.append(
                                "CRITICAL: These KB docs are curated and up-to-date. "
                                "You MUST follow them exactly:\n"
                                "- Use the EXACT install commands from these docs "
                                "(including all peer dependencies like postcss, jsdom, etc.)\n"
                                "- If a doc says a command is deprecated/removed, "
                                "do NOT use that command\n"
                                "- If a doc shows specific packages to install together, "
                                "install ALL of them in one step — do not omit any\n"
                                "- Your training data may be outdated — these docs override it")
                            parts.extend(doc_hints)
                        _logger.info(
                            "[PreAnalysis] %d doc(s) injected into planner context",
                            len(doc_hints),
                        )
            except Exception as e:
                _logger.debug(f"[PreAnalysis] Global KB doc search failed: {e}")

        # 5. Baseline test analysis results (raw evidence — the TASK BRIEFING
        # above is the derived conclusion; this is the supporting data).
        if test_analysis:
            parts.append(f"\n[Baseline Test Analysis]\n{test_analysis}")
            parts.append(
                "\nIMPORTANT: The analysis above was produced by ACTUALLY RUNNING "
                "the test suite. It is authoritative. Your plan MUST respect it:\n"
                "- If files are listed as HEALTHY/PASSING, do NOT include steps that modify them.\n"
                "- If only specific files are BROKEN, plan ONLY code changes to those files.\n"
                "- If the error is a test assertion failure (not a setup/config error), "
                "fix the test code — do NOT recreate config or setup files."
            )

        return "\n".join(parts) if parts else ""

    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """

You are a SENIOR SOFTWARE ARCHITECT creating an execution plan that will be
carried out by an automated pipeline. Each step is executed by one of four
agents: a CODER (writes files), a CMD runner (executes shell commands), a
TESTER (generates and runs unit tests), or a SEARCHER (searches the web for
documentation and latest info). Your plan MUST be precise enough for
these agents to succeed on the first attempt.

═══════ HOST ENVIRONMENT ═══════
""" + _os_context_for_prompt() + """
═══════ OUTPUT FORMAT ═══════
Output your plan using this EXACT line-based format. Each step starts with
a --STEP header line followed by metadata lines and a description.

IMPORTANT: If the TASK BRIEFING says the task is already satisfied (e.g.
"Already satisfied: Yes", "Required changes: NONE", or "Agent directive"
says no changes are needed), output ONLY this — no steps, no explanation:

==DONE==
reason: <one sentence explaining why no action is needed>
==END==

Otherwise, output steps using the format below:

==PLAN==

--STEP 1.1 [CMD] depends:none
Install Express and CORS
> npm install express cors
produces: package.json

--STEP 2.1 [CODE] depends:1.1
Create the Express server with GET /api/health endpoint
target: src/server.js
exports: app, startServer
imports: none
content:
```js
const express = require('express');
const app = express();
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));
function startServer(port = 3000) { return app.listen(port); }
module.exports = { app, startServer };
```
---file-content-end---

--STEP 2.2 [CODE] depends:1.1
Create input validation utility
target: src/utils/validate.js
exports: validateInput, sanitize
imports: none
content:
```js
function validateInput(input) { return input != null && input !== ''; }
function sanitize(input) { return String(input).trim().replace(/[<>]/g, ''); }
module.exports = { validateInput, sanitize };
```
---file-content-end---

--STEP 3.1 [CODE] depends:2.1, 2.2
Add validation middleware to existing server (surgical edit — file already exists)
target: src/server.js
imports: src/utils/validate.js:validateInput, src/utils/validate.js:sanitize
edit:
<<<FIND>>>
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));
<<<REPLACE>>>
app.use(express.json());
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));
app.post('/api/data', (req, res) => {
  const { validateInput, sanitize } = require('./utils/validate');
  if (!validateInput(req.body.value)) return res.status(400).json({ error: 'invalid' });
  res.json({ value: sanitize(req.body.value) });
});
<<<END>>>

--STEP 4.1 [TEST] depends:3.1
Write and run tests for the server and validation
target: src/__tests__/server.test.js
imports: src/server.js:app, src/utils/validate.js:validateInput

==END==

═══════ LINE REFERENCE ═══════
--STEP <id> [<TYPE>] depends:<deps>   ← REQUIRED. id=wave.seq, TYPE=CMD|CODE|TEST, deps=comma-separated step ids or "none"
<description text>                     ← REQUIRED. One-line description of what this step does.
> <shell command>                      ← CMD steps only. The exact command to run.
target: <file1>, <file2>               ← CODE/TEST steps. Files to create or modify.
exports: <Symbol1>, <Symbol2>          ← CODE steps. Symbols this file will export.
imports: <file>:<Symbol>, ...          ← CODE/TEST steps. File:Symbol pairs this step needs. "none" if no imports.
imported_by: <file1>, <file2>          ← CODE steps. Optional. Files that will import this step's target. Use when no explicit wiring step exists.
edit:                                  ← CODE steps that MODIFY existing files. Preferred over content: for modifications.
<<<FIND>>>                             ←   Exact text to find (copied verbatim from current source).
<old text>
<<<REPLACE>>>                          ←   Replacement text.
<new text>
<<<END>>>                              ←   Closes one find/replace pair. Repeat FIND/REPLACE/END for multiple changes.
content:                               ← CODE/TEST steps that CREATE new files. Required for new files.
```<lang>                              ←   Fenced code block immediately after content:
<complete file source>                 ←   Full file — not a snippet. Every line.
```                                    ←   Close the fence.
---file-content-end---                 ←   REQUIRED closing marker after every content: block.
produces: <file1>, <file2>             ← CMD steps. Files created by the command.
kb_docs: <DocTitle1>, <DocTitle2>      ← CODE/TEST steps. Exact titles of KB docs you used when writing the inline code. Omit if none.
content:                               ← CODE/TEST steps. ALWAYS include complete file source here.
```<lang>                              ←   Fenced code block immediately after content:
<complete file source>                 ←   Full file — not a snippet. Every line.
```                                    ←   Close the fence.
---file-content-end---                 ←   REQUIRED closing marker after every content: block.

═══════ STEP ID FORMAT ═══════
Use wave.sequence numbering:
  - Wave 1 (no deps): 1.1, 1.2, 1.3
  - Wave 2 (depends on wave 1): 2.1, 2.2
  - Wave 3 (depends on wave 2): 3.1, 3.2, 3.3
Steps in the same wave can run in parallel. Each wave runs after the previous.

═══════ STEP RULES (CRITICAL) ═══════

1. **Reference EXACT file paths**: Every CODE step must name the specific
   file(s) in the target: line. Use the paths from the project context above.
   Say "target: src/index.js" NOT "target: the main file".

2. **One action per step**: Each step should do ONE thing. Don't combine
   "create file AND install package" in one step. Split them.
   Exception: ALL modifications to the SAME file MUST be combined into a single CODE step.

3. **CMD steps for shell commands**: Installing packages, running scripts,
   creating directories. Put the exact command on a line starting with "> ".

4. **CODE steps for file changes**: Creating or modifying source files.
   Always specify target: with the file path.
   CRITICAL: You MUST combine ALL changes for a single file into exactly ONE CODE step.

5. **SEARCH steps for web lookups**: When you need up-to-date documentation.
   Use [SEARCH] type. No target: line needed.

6. **Existing files = MODIFY, not recreate**: When files already exist in the
   project, plan to UPDATE them. Reference their exact paths.

7. **Declare imports explicitly**: For every CODE step, list all file:symbol
   pairs it will import on the imports: line. This ensures correct dependency
   tracking and context injection.

8. **Declare exports explicitly**: For every CODE step, list the main symbols
   (functions, classes, constants) the file will export on the exports: line.

9. **NO meta-steps**: Do NOT include steps like "Analyze the project",
   "Review the code". Jump straight to actionable steps.

10. **TEST steps only when explicitly requested**: Do NOT include TEST steps
    unless the user's task explicitly asks for tests. Tests consume significant
    tokens and time. When tests ARE requested, place them AFTER all CODE steps.
    CRITICAL: If the [Baseline Test Analysis] shows tests are PASSING,
    do NOT include any steps for "Fixing tests" or "Ensuring test setup".
    NEVER include steps to modify or "fix" test files that are explicitly
    marked as PASSING or Healthy in the analysis.
    Only include global setup/config modification steps if the analysis
    indicates a global failure (e.g., all tests fail or execution fails to start).

11. **Shell commands are non-interactive**: Always include --yes, -y, or
    --defaults flags for tools that prompt for input.

12. **Sub-project paths**: Use the EXACT SAME folder name in ALL steps.
    CMD steps must include `cd <name> &&` before the command.

13. **SKIP already-installed packages**: If the project knowledge lists
    packages as already installed, do NOT add install steps for them.

14. **KB documentation overrides your training data**: Use exact commands
    from KB docs — including ALL packages and peer dependencies.

15. **Configuration BEFORE code**: All setup/config steps MUST come before
    code that relies on them (test framework, CSS config, build tools).

16. **Scaffold blank projects FIRST**: When project is BLANK/EMPTY, scaffold
    first using the appropriate tool for the detected language
    (Python: `python -m venv venv && pip install ...`;
     JavaScript/TypeScript: `npm create vite@latest`, `npm install`;
     Go: `go mod init`; Rust: `cargo init`)
    before any code steps. Use the language-specific examples from the
    PROJECT STATE context above — do NOT default to npm for non-JS projects.

17. **Leaf components BEFORE parents**: Create child components first, then
    parents that import them. Declare imports: to enforce correct ordering.

18. **Explicitly wire entry-point and parent files**: When a new component or
    module must be mounted/imported by an existing file (e.g. main.jsx, App.tsx,
    router.py, index.ts), you MUST include a dedicated CODE step that modifies
    that entry-point file to add the import. Do NOT assume the pipeline will
    auto-wire it. The wiring step must declare:
      imports: <new-component-file>:<ExportedSymbol>
    This also lets the pipeline derive `imported_by` automatically — so the
    dependency checker knows the exact parent without guessing.
    If you cannot add a wiring step, declare `imported_by: <parent-file>` on
    the child component step so the dependency checker knows where to wire it.

19. **Inline code for CODE and TEST steps — two formats**:

    **For MODIFYING an existing file** (the file appears in [FILES TO MODIFY]):
    Use an `edit:` block with find/replace pairs. This is PREFERRED — it is
    smaller, surgical, and cannot accidentally change unrelated code.
    Format:
      edit:
      <<<FIND>>>
      <exact text to find — copy verbatim from the source above>
      <<<REPLACE>>>
      <replacement text>
      <<<END>>>
    You may include multiple FIND/REPLACE/END sequences in one `edit:` block
    for multiple changes to the same file.
    RULES:
    - The <<<FIND>>> text must be an exact substring of the current file.
      Copy it character-for-character from the [FILES TO MODIFY] source.
      Do NOT add, remove, or alter ANY characters — including comments,
      whitespace, blank lines, or code — while writing the FIND block.
      If the source has no comment on a line, the FIND block must have no
      comment on that line. If you cannot reproduce a section verbatim,
      make the FIND block smaller (fewer lines) so it stays exact.
    - Change ONLY what the 'Agent directive' requires. Do NOT alter anything
      else in the file (no comments, no renaming, no restructuring).
    - NEVER invent or "clean up" code in the FIND block. It is a literal
      copy — not a paraphrase or improved version of the original.

    **For CREATING a new file** (file does not exist yet):
    Use a `content:` block with the complete file source. This is MANDATORY
    for new files — zero Coder LLM calls needed.
    Format:
      content:
      ```<lang>
      <full file content>
      ```
      ---file-content-end---
    The code must be COMPLETE — every import, every function, every line.
    Do NOT write a stub or partial implementation.

20. **Declare KB docs used for inline code**: For every CODE or TEST step
    with a content: block, add a `kb_docs:` line listing the exact titles
    of any KB documents from the [KNOWLEDGE BASE] context above that you
    referenced while writing that inline code. Use the titles verbatim —
    no paraphrasing. Place the kb_docs: line immediately before content:.
    Omit this line entirely if no KB docs were relevant to the step.
    Example:
      kb_docs: React TailwindCSS Responsive Header, React Component Patterns
      content:
      ```jsx
      ...
      ```
      ---file-content-end---

═══════ QUALITY CHECKLIST ═══════
- [ ] Every CODE step has a target: line with exact file paths
- [ ] Every CODE step has exports: and imports: lines
- [ ] Every CODE step that modifies an existing file uses an edit: block (NOT content:)
- [ ] Every CODE/TEST step that creates a new file has a content: block with complete source
- [ ] No two steps have the same target: file (consolidate into one step)
- [ ] If step B imports from step A's target file, B depends on A
- [ ] No vague steps — each step is specific and actionable
- [ ] Total steps between 2-15
- [ ] No install steps for already-installed packages
- [ ] Config/tooling steps come BEFORE code that depends on them
- [ ] Leaf components created BEFORE parent components
- [ ] Every new component/module that mounts in an entry-point (main.jsx, App.tsx, etc.) has an explicit CODE step to update that entry-point, OR has imported_by: declared
"""
        return self.llm_client.generate_response(prompt)

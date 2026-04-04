import logging
import os
import re
import shlex
import subprocess
from .base import Agent

_logger = logging.getLogger(__name__)

# ── Safe-command allowlist for RUN_CMD ───────────────────────────────────────
# Only patterns listed here are ever executed.  The list is intentionally
# narrow: read-only git operations, directory listing, file search, and test
# runners.  Nothing that writes, installs, or deletes is permitted.
_SAFE_CMD_PATTERNS: list[re.Pattern] = [
    re.compile(r'^git\s+(diff|log|status|blame|show)\b'),   # read-only git
    re.compile(r'^(ls|dir)\b'),                              # directory listing
    re.compile(r'^(grep|findstr)\b'),                        # text search
    re.compile(r'^npx\s+(vitest|jest)\s+(run|--run)\b'),     # test runner (no install)
    re.compile(r'^python\s+-m\s+pytest\b'),                  # pytest read-only run
]

# Shell tokens that are never allowed anywhere in the command string, regardless
# of the matched allowlist pattern.  These prevent injection via arguments.
_FORBIDDEN_TOKENS: frozenset[str] = frozenset({
    "rm", "del", "rmdir", "rd", "mv", "move", "cp", "copy",
    "chmod", "chown", "sudo", "su", "eval", "exec",
    "npm", "pip", "yarn", "pnpm",           # no installs
    ">", ">>", "2>", "&>", "|",             # no output redirection or pipes
    "`", "$(", "${",                         # no command substitution
    ";", "&&", "||",                         # no command chaining
})

# Output caps for RUN_CMD results injected into LLM context
_CMD_MAX_LINES = 80
_CMD_MAX_CHARS = 4000
_CMD_TIMEOUT_SECS = 15

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


def _extract_reasoning(response: str) -> str:
    """
    Return the reasoning text that precedes a tool command in *response*.

    The LLM typically writes a sentence or two explaining its gap before
    emitting the tool command.  We show this so the user can follow the
    investigation chain without reading raw logs.

    Returns up to 2 sentences / 140 chars, or empty string if the response
    starts immediately with a tool command or REQUIREMENTS_SPEC.
    """
    tool_line = re.compile(
        r'^\s*(KB_SEARCH|SEARCH|RUN_CMD|FIND_USAGES|REQUIREMENTS_SPEC):',
        re.IGNORECASE | re.MULTILINE,
    )
    m = tool_line.search(response)
    reasoning = response[:m.start()].strip() if m else response.strip()
    if not reasoning:
        return ""
    # Collapse to first 2 non-empty lines
    lines = [l.strip() for l in reasoning.splitlines() if l.strip()]
    text = " ".join(lines[:2])
    if len(text) > 140:
        text = text[:137] + "…"
    return text


def _show_spec_details(cli_display, spec: str) -> None:
    """
    Add 2-3 condensed detail rows to the investigation panel after the
    REQUIREMENTS_SPEC row, so the user can see the key fields without
    reading the full spec text.

    Extracts Goal, Change scope / Create, and KB topics.
    Each is shown as a dim indented row using kind='detail'.
    """
    def _first_line(text: str, max_chars: int = 70) -> str:
        line = text.strip().splitlines()[0].strip() if text.strip() else ""
        return line[:max_chars] + ("…" if len(line) > max_chars else "")

    # Goal
    goal_m = re.search(r'Goal:\s*(.+?)(?=\n\S|\Z)', spec, re.IGNORECASE | re.DOTALL)
    if goal_m:
        cli_display.show_intent_event("detail", f"Goal: {_first_line(goal_m.group(1))}")

    # Change scope / Create / Modify (whatever is present)
    for field in ("Change scope", "Create", "Modify", "Fix scope"):
        m = re.search(rf'{field}:\s*(.+?)(?=\n\S|\Z)', spec, re.IGNORECASE | re.DOTALL)
        if m:
            cli_display.show_intent_event("detail",
                                          f"{field}: {_first_line(m.group(1))}")
            break

    # KB topics
    topics_m = re.search(r'KB topics[^:\n]*:\s*(.+)', spec, re.IGNORECASE)
    if topics_m:
        topics = topics_m.group(1).strip()
        if topics.lower() not in ("none", "n/a", ""):
            cli_display.show_intent_event("detail", f"KB topics: {topics[:60]}")


def _is_safe_cmd(cmd: str) -> tuple[bool, str]:
    """
    Validate *cmd* against the allowlist and forbidden-token set.

    Returns (True, "") when safe, or (False, reason) when rejected.
    The check is intentionally strict: the command must match at least one
    allowlist pattern AND must not contain any forbidden token.
    """
    stripped = cmd.strip()
    if not stripped:
        return False, "empty command"

    # Check forbidden tokens first — these override any allowlist match
    try:
        tokens = set(shlex.split(stripped, posix=True))
    except ValueError:
        tokens = set(stripped.split())
    # Also check raw string for shell metacharacters that shlex may not split
    for ft in _FORBIDDEN_TOKENS:
        if ft in tokens or ft in stripped:
            return False, f"forbidden token '{ft}'"

    # Must match at least one allowlist pattern
    for pat in _SAFE_CMD_PATTERNS:
        if pat.match(stripped):
            return True, ""

    return False, "command not in allowlist"


def _translate_cmd_for_os(cmd: str) -> str:
    """
    Translate a Unix-style command to the host OS equivalent.

    Only maps the small set of commands used by IntentAgent's RUN_CMD;
    git commands are universal and need no translation.
    """
    if os.name != "nt":
        return cmd  # Unix: no translation needed

    stripped = cmd.strip()

    # ls [flags] [path...] → dir [path...]
    # Strip all Unix flags (e.g. -la, -a) — they are meaningless to cmd.exe.
    # Keep any non-flag path arguments so "ls src/" → "dir src/".
    if re.match(r'^ls\b', stripped):
        parts = stripped.split()
        paths = [p for p in parts[1:] if not p.startswith('-')]
        return ("dir " + " ".join(paths)).rstrip() if paths else "dir"

    # grep → findstr, preserve all arguments
    if re.match(r'^grep\b', stripped):
        return re.sub(r'^grep\b', 'findstr', stripped, count=1)

    return stripped


def _run_cmd(cmd: str, cwd: str | None = None) -> str:
    """
    Execute *cmd* in a subprocess and return its output (stdout + stderr),
    capped at _CMD_MAX_LINES lines / _CMD_MAX_CHARS characters.

    Caller MUST validate with _is_safe_cmd before calling this.
    """
    translated = _translate_cmd_for_os(cmd)
    try:
        result = subprocess.run(
            translated,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_CMD_TIMEOUT_SECS,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        if not output.strip():
            return "(no output)"
        lines = output.splitlines()
        if len(lines) > _CMD_MAX_LINES:
            output = "\n".join(lines[:_CMD_MAX_LINES]) + f"\n... ({len(lines) - _CMD_MAX_LINES} more lines)"
        if len(output) > _CMD_MAX_CHARS:
            output = output[:_CMD_MAX_CHARS] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        return f"(command timed out after {_CMD_TIMEOUT_SECS}s)"
    except Exception as exc:
        return f"(command failed: {exc})"


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
      RUN_CMD: <command>   — runs a safe read-only shell command and returns output
      FIND_USAGES: <name>  — finds callers of a component/function
    """

    def process(self, task: str, context: str = "") -> str:
        """Implementation of base Agent process. Fallback if called directly."""
        return self.analyze_intent(task)

    # Hard cap — pure runaway guard, never meant to be hit in normal operation.
    # The loop exits naturally when the LLM writes REQUIREMENTS_SPEC, or when
    # the duplicate-command guard sets force_conclude.
    _SAFETY_CAP = 20
    # Max investigation-only iterations for BUG_FIX before auto-concluding.
    # After this many iterations without a tool call, we switch to conclude
    # mode rather than rejecting the spec endlessly.
    _BUG_FIX_INVESTIGATE_CAP = 3
    # If accumulated context grows beyond this, force the LLM to conclude so
    # we don't overflow its context window or burn cloud tokens indefinitely.
    _MAX_CONTEXT_CHARS = 60_000

    def analyze_intent(
        self,
        raw_task: str,
        search_agent=None,
        kb_context_builder=None,
        kb_context: str = "",
        cli_display=None,
        available_kb_docs: list[str] | None = None,
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

        # ── Available global KB docs ──────────────────────────────────────────
        # When the planner pre-fetched global KB titles, surface them so the
        # LLM can pick the most relevant ones and emit them as `KB docs:` in
        # the REQUIREMENTS_SPEC.  The planner will then load those docs directly
        # via get_by_titles() — no re-embedding needed.
        if available_kb_docs:
            _titles_block = "\n".join(f"  - {t}" for t in available_kb_docs)
            accumulated_context += (
                "Available Reference Docs (global KB — list all relevant ones "
                "under 'KB docs:' in your spec, including setup/install guides "
                "when creating a new project):\n"
                f"{_titles_block}\n\n"
            )

        # ── Detect visual/layout bug early ──────────────────────────────────
        _task_lower = raw_task.lower()
        _is_visual_bug = any(kw in _task_lower for kw in (
            "not visible", "nothing happens", "doesn't appear", "doesn't show",
            "does not appear", "does not show", "hidden", "invisible",
            "not rendering", "not displayed", "not showing", "can't see",
            "menu", "modal", "dropdown", "popup", "overlay", "sidebar",
            "z-index", "behind", "underneath",
        ))

        # ── Pre-seed: give the LLM a starting point, not a conclusion ─────────
        # Run one automatic KB search before the loop so the LLM can see what
        # exists in the project and decide what it still needs to investigate.
        # This is a hint, not a gate — the LLM is free to search further or
        # pivot to different files based on what it sees here.
        if kb_context_builder is not None:
            pre_query = self._extract_search_query(raw_task)
            _logger.info("[IntentAnalysis] Pre-seeding context with KB_SEARCH '%s'", pre_query)
            if cli_display:
                cli_display.show_intent_event("preseed", f"Pre-seed: {pre_query}")
            pre_result = self._query_kb(kb_context_builder, pre_query)
            if pre_result:
                accumulated_context += (
                    f"Initial Project Context (auto-retrieved, query: '{pre_query}'):\n"
                    f"{pre_result}\n\n"
                )
                _logger.info("[IntentAnalysis] Pre-seed successful.")
                if cli_display:
                    cli_display.update_last_intent_event(f"{len(pre_result):,} chars")
            else:
                _logger.info("[IntentAnalysis] Pre-seed returned no results.")
                if cli_display:
                    cli_display.update_last_intent_event("no results")

            # ── BUG_FIX enhancement: fetch full file content for target ───────
            # Semantic snippets (30 lines) are insufficient for tracing render
            # logic and CSS layering.  For bug-related tasks, identify the
            # likely target file(s) from the pre-seed and inject their full
            # source so the LLM has the complete picture before it concludes.
            if pre_result:
                _target_files = self._extract_target_files_from_preseed(pre_result)
                for tf in _target_files[:2]:  # cap at 2 files
                    full_src = self._read_full_file(kb_context_builder, tf)
                    if full_src:
                        accumulated_context += (
                            f"Full Source (auto-loaded for investigation): {tf}\n"
                            f"{full_src}\n\n"
                        )
                        _logger.info(
                            "[IntentAnalysis] Auto-loaded full source: %s (%d chars)",
                            tf, len(full_src),
                        )
                        if cli_display:
                            cli_display.show_intent_event(
                                "preseed", f"Full source: {tf}"
                            )

            # ── Visual bug enhancement: auto-fetch CSS/style files ────────────
            # For visual/layout bugs, the root cause is often in CSS (z-index,
            # display, overflow, pointer-events) rather than JS logic.  Auto-
            # fetch CSS files that could affect the target component.
            if _is_visual_bug and pre_result:
                _logger.info("[IntentAnalysis] Visual bug detected — auto-searching CSS/style files.")
                css_results = self._search_related_css(kb_context_builder, raw_task, pre_result)
                if css_results:
                    accumulated_context += (
                        f"CSS/Style Context (auto-retrieved for visual bug investigation):\n"
                        f"{css_results}\n\n"
                    )
                    _logger.info(
                        "[IntentAnalysis] Auto-loaded CSS context: %d chars", len(css_results),
                    )
                    if cli_display:
                        cli_display.show_intent_event("preseed", "CSS/style files loaded")

        # ── Iterative investigation loop ──────────────────────────────────────
        last_response = ""
        # Track commands already executed to prevent the LLM from looping on
        # the same tool call repeatedly without making progress.
        executed_commands: set[str] = set()
        # After the first duplicate is detected, force conclude on the very
        # next iteration so the LLM doesn't burn more rounds ignoring the note.
        force_conclude = False
        # Count consecutive BUG_FIX rejections (spec produced but no tools used).
        # After _BUG_FIX_INVESTIGATE_CAP, switch to investigation-only prompt
        # instead of rejecting forever.
        bug_fix_reject_count = 0

        iteration = 0
        while True:
            iteration += 1
            context_overflow = len(accumulated_context) > self._MAX_CONTEXT_CHARS
            is_last = force_conclude or context_overflow
            if iteration > self._SAFETY_CAP:
                _logger.warning(
                    "[IntentAnalysis] Safety cap (%d iterations) reached — breaking.",
                    self._SAFETY_CAP,
                )
                break
            prompt = self._build_prompt(raw_task, accumulated_context, conclude=is_last)
            if cli_display:
                cli_display.show_intent_event(
                    "think", f"Querying LLM (iteration {iteration})",
                    iteration=iteration, max_iterations=0,
                )

            try:
                response = self.llm_client.generate_response(prompt)
                # ── Duplicate response guard ──────────────────────────────────
                # If the LLM produces the same response twice in a row it is
                # stuck and retrying will not help.  Force conclude immediately
                # so the loop exits rather than burning more iterations/tokens.
                if response.strip() == last_response.strip() and last_response:
                    _logger.warning(
                        "[IntentAnalysis] Iteration %d: identical response to previous "
                        "— forcing conclude to avoid infinite loop.",
                        iteration,
                    )
                    force_conclude = True
                last_response = response
                if cli_display:
                    cli_display.update_last_intent_event(
                        f"~{len(response.split()):,} words"
                    )
                    cli_display.set_intent_response(response)
                    reasoning = _extract_reasoning(response)
                    if reasoning:
                        cli_display.show_intent_event("reason", reasoning)

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
                    if cli_display:
                        cli_display.show_intent_event("kb", f"KB_SEARCH: {query}",
                                                       iteration=iteration,
                                                       max_iterations=0)
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
                        if cli_display:
                            cli_display.update_last_intent_event(f"{len(kb_result):,} chars")
                    else:
                        _logger.warning(
                            "[IntentAnalysis] KB search returned no results for: %s", query,
                        )
                        accumulated_context += (
                            f"KB Search for '{query}': no results found.\n\n"
                        )
                        if cli_display:
                            cli_display.update_last_intent_event("no results")
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
                    if cli_display:
                        cli_display.show_intent_event("web", f"SEARCH: {query}",
                                                       iteration=iteration,
                                                       max_iterations=0)
                    search_result = search_agent.search_for_task(
                        query, kb_context=kb_context,
                    )
                    if search_result:
                        accumulated_context += (
                            f"Web Search Results for '{query}':\n{search_result}\n\n"
                        )
                        if cli_display:
                            cli_display.update_last_intent_event(f"{len(search_result):,} chars")
                    else:
                        _logger.warning(
                            "[IntentAnalysis] Web search returned no results for: %s", query,
                        )
                        accumulated_context += (
                            f"Web Search for '{query}': no results found.\n\n"
                        )
                        if cli_display:
                            cli_display.update_last_intent_event("no results")
                    continue

                # ── RUN_CMD command ───────────────────────────────────────────
                # Executes a shell command and returns its output so the LLM
                # can use live project state (git diff, test results, file list)
                # as investigation evidence.
                # Safety: validated against _SAFE_CMD_PATTERNS + _FORBIDDEN_TOKENS
                # before any subprocess is spawned.  Rejected commands are
                # reported back to the LLM so it can reformulate.
                cmd_match = re.search(
                    r'\bRUN_CMD:\s*(.+)$', response,
                    re.MULTILINE | re.IGNORECASE,
                )
                if cmd_match:
                    raw_cmd = cmd_match.group(1).strip()
                    cmd_key = f"RUN_CMD:{raw_cmd.lower()}"
                    if cmd_key in executed_commands:
                        _logger.info(
                            "[IntentAnalysis] Iteration %d: RUN_CMD '%s' already executed — forcing conclude.",
                            iteration, raw_cmd,
                        )
                        accumulated_context += (
                            f"[RUN_CMD '{raw_cmd}' already ran — output is above. "
                            f"You have enough context. Write REQUIREMENTS_SPEC now.]\n\n"
                        )
                        force_conclude = True
                        continue
                    executed_commands.add(cmd_key)
                    safe, reason = _is_safe_cmd(raw_cmd)
                    if not safe:
                        _logger.warning(
                            "[IntentAnalysis] Iteration %d: RUN_CMD '%s' rejected — %s",
                            iteration, raw_cmd, reason,
                        )
                        if cli_display:
                            cli_display.show_intent_event("reject",
                                                           f"RUN_CMD rejected: {reason}",
                                                           iteration=iteration,
                                                           max_iterations=0)
                        accumulated_context += (
                            f"[RUN_CMD '{raw_cmd}' was rejected: {reason}. "
                            f"Only read-only git/grep/ls/test-runner commands are allowed. "
                            f"Rephrase or use KB_SEARCH instead.]\n\n"
                        )
                        continue
                    _logger.info(
                        "[IntentAnalysis] Iteration %d: RUN_CMD '%s'", iteration, raw_cmd,
                    )
                    if cli_display:
                        cli_display.show_intent_event("cmd", f"RUN_CMD: {raw_cmd}",
                                                       iteration=iteration,
                                                       max_iterations=0)
                    project_root = (
                        getattr(kb_context_builder, "_project_root", None)
                        if kb_context_builder is not None else None
                    )
                    cmd_output = _run_cmd(raw_cmd, cwd=project_root)
                    accumulated_context += (
                        f"RUN_CMD output for `{raw_cmd}`:\n"
                        f"```\n{cmd_output}\n```\n\n"
                    )
                    if cli_display:
                        lines = cmd_output.count("\n") + 1
                        cli_display.update_last_intent_event(f"{lines} lines")
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
                    if cli_display:
                        cli_display.show_intent_event("usage", f"FIND_USAGES: {name}",
                                                       iteration=iteration,
                                                       max_iterations=0)
                    usage_result = self._find_usages(kb_context_builder, name)
                    if usage_result:
                        accumulated_context += (
                            f"Usages of '{name}':\n{usage_result}\n\n"
                        )
                        if cli_display:
                            cli_display.update_last_intent_event(f"{len(usage_result):,} chars")
                    else:
                        accumulated_context += (
                            f"FIND_USAGES '{name}': no call sites found.\n\n"
                        )
                        if cli_display:
                            cli_display.update_last_intent_event("no usages found")
                    continue

                # ── REQUIREMENTS_SPEC ─────────────────────────────────────────
                spec_match = re.search(
                    r'REQUIREMENTS_SPEC:(.+)', response,
                    re.IGNORECASE | re.DOTALL,
                )
                if spec_match:
                    spec = spec_match.group(1).strip()
                    task_type_m = re.search(r'Task type:\s*(\w+)', spec, re.IGNORECASE)
                    task_type_label = task_type_m.group(1) if task_type_m else "UNKNOWN"

                    # ── BUG_FIX guard: reject premature spec ─────────────────
                    # For BUG_FIX tasks, the LLM MUST have used at least one
                    # investigation tool (KB_SEARCH / FIND_USAGES / RUN_CMD)
                    # beyond the auto-preseed.  Pre-seed snippets alone are not
                    # enough to correctly identify root cause — the LLM tends
                    # to guess from partial context and get it wrong (e.g.
                    # "button lacks onClick" when onClick exists but z-index
                    # is missing).  Force it to actually investigate.
                    if (
                        task_type_label.upper() == "BUG_FIX"
                        and not executed_commands
                        and not force_conclude
                    ):
                        bug_fix_reject_count += 1
                        if bug_fix_reject_count >= self._BUG_FIX_INVESTIGATE_CAP:
                            # ── Phase switch: investigation-only prompt ──────
                            # The LLM keeps producing specs instead of tool
                            # calls.  Rather than rejecting forever, run ONE
                            # automatic investigation round: pick the most
                            # likely target file from the spec and KB_SEARCH
                            # it.  This gives the LLM real evidence and
                            # breaks the deadlock.
                            _logger.warning(
                                "[IntentAnalysis] BUG_FIX rejected %d times — "
                                "auto-investigating to break deadlock.",
                                bug_fix_reject_count,
                            )
                            auto_target = self._extract_target_file(spec)
                            if auto_target and kb_context_builder is not None:
                                cmd_key = f"KB_SEARCH:{auto_target.lower()}"
                                if cmd_key not in executed_commands:
                                    executed_commands.add(cmd_key)
                                    _logger.info(
                                        "[IntentAnalysis] Auto-investigating: KB_SEARCH '%s'",
                                        auto_target,
                                    )
                                    if cli_display:
                                        cli_display.show_intent_event(
                                            "kb",
                                            f"Auto KB_SEARCH: {auto_target}",
                                            iteration=iteration,
                                            max_iterations=0,
                                        )
                                    kb_result = self._query_kb(
                                        kb_context_builder, auto_target,
                                    )
                                    if kb_result:
                                        accumulated_context += (
                                            f"[Auto-investigation] KB Search Results "
                                            f"for '{auto_target}':\n{kb_result}\n\n"
                                        )
                                    else:
                                        accumulated_context += (
                                            f"[Auto-investigation] KB Search for "
                                            f"'{auto_target}': no results found.\n\n"
                                        )
                            # Now the LLM has evidence.  Let the next iteration
                            # produce a spec — executed_commands is non-empty
                            # so the guard won't fire again.
                            accumulated_context += (
                                "[Investigation complete. You now have source code "
                                "evidence above. Write your REQUIREMENTS_SPEC using "
                                "the evidence — cite specific files and lines.]\n\n"
                            )
                            if cli_display:
                                cli_display.show_intent_event(
                                    "reject",
                                    f"BUG_FIX: auto-investigated after {bug_fix_reject_count} rejections",
                                    iteration=iteration, max_iterations=0,
                                )
                            continue

                        _logger.info(
                            "[IntentAnalysis] BUG_FIX spec rejected on iteration %d "
                            "— no investigation tools used yet (%d/%d). "
                            "Sending investigation-only prompt.",
                            iteration, bug_fix_reject_count,
                            self._BUG_FIX_INVESTIGATE_CAP,
                        )
                        # Instead of repeating the same full prompt with a
                        # rejection note (which the LLM ignores), inject a
                        # targeted directive that ONLY asks for a tool call —
                        # no spec template to latch onto.
                        accumulated_context += (
                            "[INVESTIGATION REQUIRED — do NOT write REQUIREMENTS_SPEC yet.]\n"
                            "You jumped to a conclusion without reading the code. "
                            "Before you can write a spec, you MUST investigate.\n\n"
                            "Your ONLY allowed response is ONE of these tool calls:\n"
                            "  KB_SEARCH: <filename or symbol>\n"
                            "  FIND_USAGES: <ComponentName or functionName>\n"
                            "  RUN_CMD: <read-only command>\n\n"
                            "Pick the file most likely to contain the bug and search for it. "
                            "Do NOT write any other output.\n\n"
                        )
                        if cli_display:
                            cli_display.show_intent_event(
                                "reject",
                                f"BUG_FIX: requesting investigation ({bug_fix_reject_count}/{self._BUG_FIX_INVESTIGATE_CAP})",
                                iteration=iteration, max_iterations=0,
                            )
                        continue

                    # ── Root-cause cross-validation for BUG_FIX ──────────────
                    # Check if the spec claims something is "missing" or "lacks"
                    # but the evidence already shows it exists.  This catches
                    # the common LLM error of guessing without reading.
                    if task_type_label.upper() == "BUG_FIX" and not force_conclude:
                        contradiction = self._check_root_cause_contradiction(
                            spec, accumulated_context,
                        )
                        if contradiction:
                            _logger.warning(
                                "[IntentAnalysis] Root-cause contradiction detected: %s",
                                contradiction,
                            )
                            accumulated_context += (
                                f"[ROOT CAUSE CONTRADICTION: {contradiction}\n"
                                "Re-examine the code evidence above. Your claimed root "
                                "cause does not match the actual code. Look for OTHER "
                                "possible causes:\n"
                                "  - CSS issues: z-index layering, display:none, "
                                "overflow:hidden, pointer-events:none\n"
                                "  - Conditional rendering: state that never becomes true\n"
                                "  - Event handler that sets state but UI doesn't react "
                                "(wrong CSS, element behind another, opacity:0)\n"
                                "Write a corrected REQUIREMENTS_SPEC.]\n\n"
                            )
                            if cli_display:
                                cli_display.show_intent_event(
                                    "reject",
                                    f"Root-cause contradiction: {contradiction[:60]}",
                                    iteration=iteration, max_iterations=0,
                                )
                            continue

                    # Normalise KB topics: LLMs sometimes output a bullet list
                    # instead of a comma-separated line.  Convert to one line so
                    # downstream parsers (cli.py / api.py) can split on ',' reliably.
                    spec = self._normalize_kb_topics(spec)
                    _logger.info("[IntentAnalysis] Successfully generated REQUIREMENTS_SPEC.")
                    if cli_display:
                        cli_display.show_intent_event("spec",
                                                       f"REQUIREMENTS_SPEC: {task_type_label}",
                                                       iteration=iteration,
                                                       max_iterations=0)
                        _show_spec_details(cli_display, spec)
                        # Events stay visible — cleared by pre_analyze when
                        # the briefing/planning phase begins.
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
                if cli_display:
                    cli_display.show_intent_event("spec", "REQUIREMENTS_SPEC: (inferred)",
                                                   iteration=iteration,
                                                   max_iterations=0)
                    _show_spec_details(cli_display, response)
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
        # investigation, but ONLY if it looks like a final answer.  If the last
        # response still contains a tool command the loop was interrupted while
        # mid-research (e.g. iteration N+1 returned empty after iteration N
        # emitted KB_SEARCH).  Using a mid-research question as the spec causes
        # downstream Task Briefing to generate wrong goals like "state the file
        # name" instead of "fix the bug", so fall through to the raw-task
        # fallback instead.
        _mid_research = re.search(
            r'^\s*(KB_SEARCH|SEARCH|RUN_CMD|FIND_USAGES):', last_response,
            re.IGNORECASE | re.MULTILINE,
        ) if last_response else None
        if last_response and not _mid_research:
            _logger.warning(
                "[IntentAnalysis] Loop exited after %d iterations — using last response as spec.",
                iteration,
            )
            return (
                f"{raw_task}\n\n"
                f"=== INTENT CLARIFICATION (from requirements analyst) ===\n"
                f"{last_response.strip()}"
            )
        if _mid_research:
            _logger.warning(
                "[IntentAnalysis] Last response is mid-research (contains tool call) "
                "— discarding, falling back to raw task.",
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
        "KB topics: <2-5 short keywords the planner needs docs for — library or "
        "framework names only, e.g. 'Tailwind, React refs, Vitest'; "
        "write 'none' if the fix is pure logic with no external library>\n"
        "KB docs: <exact titles from Available Reference Docs that are relevant "
        "to this task; omit this line (or write 'none') if none apply>\n"
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
        "KB topics: <2-5 short keywords — library/framework names only, "
        "e.g. 'Anime.js, React hooks'; write 'none' if no new packages needed>\n"
        "KB docs: <exact titles from Available Reference Docs that are relevant "
        "to this task; omit this line (or write 'none') if none apply>\n"
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
        "KB topics: <2-5 short keywords — library/framework names only; "
        "'none' if this is a pure internal refactor>\n"
        "KB docs: <exact titles from Available Reference Docs that are relevant "
        "to this task; omit this line (or write 'none') if none apply>\n"
    )
    _SPEC_QUESTION = (
        "REQUIREMENTS_SPEC:\n"
        "Task type: QUESTION\n"
        "Goal: <what the user wants to understand — one sentence>\n"
        "Answer: <direct answer grounded in evidence; for git questions cite commit\n"
        "        hashes, changed files, and the actual diff lines as proof>\n"
        "Git evidence: <paste the key lines from git log / git diff that support\n"
        "              your answer; write 'n/a' if the question is not git-related>\n"
        "Evidence: <file names, function names, or source lines that support the answer>\n"
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
                "Classification reminder:\n"
                "  - If the user reports something broken, not visible, or not working\n"
                "    — even if phrased as a question — that is BUG_FIX, not QUESTION.\n"
                "  - If the user says 'understand', 'explain', 'what changed', 'what did\n"
                "    these commits do', 'summarise recent changes' — that is QUESTION,\n"
                "    NOT MODIFY. The user wants an explanation, not code edits.\n\n"
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
            "  MODIFY   — change existing behaviour, refactor, update, rename.\n"
            "             IMPORTANT: MODIFY means the user wants you to MAKE changes.\n"
            "             If the user only wants to UNDERSTAND or EXPLAIN changes that\n"
            "             were already made, that is QUESTION — not MODIFY.\n\n"
            "  QUESTION — the user wants to understand, explain, or summarise something.\n"
            "             Use this when:\n"
            "               • 'explain how X works' / 'what does this function do'\n"
            "               • 'understand the recent changes' / 'explain the last N commits'\n"
            "               • 'what changed in git' / 'summarise the recent commits'\n"
            "               • any 'understand/explain/summarise' phrasing where no code\n"
            "                 edit is being requested — only an explanation.\n"
            "             NOT for broken/invisible/crashing things (those are BUG_FIX).\n\n"

            "── STEP 2: Investigation rules per type ───────────────────────────\n"
            "BUG_FIX (mandatory tracing — you MUST use at least one tool before concluding):\n"
            "  1. Identify the component that should render the failing element.\n"
            "  2. Use KB_SEARCH to read the FULL source of the target component.\n"
            "     DO NOT rely only on pre-seed snippets — they may be incomplete.\n"
            "  3. Use FIND_USAGES to compare what that component DECLARES (props,\n"
            "     params, data shape) vs what its CALLER actually PASSES.\n"
            "     This catches prop-name mismatches, wrong data formats, missing props.\n"
            "  4. For 'nothing happens' / 'not visible' / menu / modal / overlay bugs:\n"
            "     The root cause is OFTEN CSS, not JavaScript. You MUST check:\n"
            "     a) z-index layering: does the element have a z-index high enough\n"
            "        to appear above siblings? (e.g. header z-40 vs menu with no z-index)\n"
            "     b) display/visibility: is the element display:none or visibility:hidden?\n"
            "     c) overflow:hidden on a parent that clips the element\n"
            "     d) pointer-events:none preventing click interaction\n"
            "     e) position: does the element need fixed/absolute positioning to overlay?\n"
            "     f) width/height: is the element 0-sized?\n"
            "     Use KB_SEARCH: <filename>.css to read stylesheets.\n"
            "     Also read Tailwind classes in the JSX — they ARE the CSS.\n"
            "  5. VERIFY before claiming: if you think something is 'missing' or 'lacks',\n"
            "     search the evidence to CONFIRM it is truly absent. Do NOT claim\n"
            "     'button lacks onClick' when onClick is present with a different value.\n"
            "  6. Root cause MUST cite: specific file + specific prop/line/condition\n"
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
            "QUESTION:\n"
            "  Your job is to produce an explanation grounded in real evidence.\n"
            "  If the task mentions 'git commits', 'recent changes', 'last N commits',\n"
            "  'what changed', 'git log' — you MUST use git tools first:\n"
            "    Step A: RUN_CMD: git log --oneline -<N>   (see what was committed)\n"
            "    Step B: RUN_CMD: git diff HEAD~<N> HEAD   (see exact line changes)\n"
            "    Step C: Use the diff output as your primary evidence for the Answer.\n"
            "  Do NOT infer 'what was changed' from the current file state alone —\n"
            "  the current state tells you what the code looks like NOW, not what\n"
            "  was CHANGED. Only git diff/log shows what actually changed.\n"
            "  For non-git questions: use KB_SEARCH to read actual source code.\n\n"

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
            "        RUN_CMD: <command>\n"
            "          When: you need live project state that KB_SEARCH cannot provide.\n"
            "          Allowed: git diff, git log, git status, git blame, git show,\n"
            "                   ls/dir, grep/findstr, npx vitest run, python -m pytest\n"
            "          Examples:\n"
            "            RUN_CMD: git diff HEAD -- src/components/Header.jsx\n"
            "            RUN_CMD: git log --oneline -10 src/components/Header.jsx\n"
            "            RUN_CMD: grep -r 'useHeader' src/\n"
            "          NOT allowed: npm install, rm, pipes (|), redirects (>)\n\n"
            "        SEARCH: <query>\n"
            "          When: you need external docs or package versions.\n\n"
            "        Example:\n"
            "          I need to see what props SnakeGame passes to SnakeBoard\n"
            "          to check for a prop-name mismatch.\n"
            "          FIND_USAGES: SnakeBoard\n\n"
            + self._spec_all_formats()
        )
        return prompt

    @staticmethod
    def _normalize_kb_topics(spec: str) -> str:
        """
        Ensure 'KB topics:' is a single comma-separated line.

        LLMs sometimes output a markdown bullet list:
          KB topics:
          - Tailwind CSS
          - React hooks

        This converts it to:
          KB topics: Tailwind CSS, React hooks

        Also strips trailing punctuation and collapses whitespace.
        """
        def _replace_block(m: re.Match) -> str:
            block = m.group(0)
            # Collect lines after "KB topics:" that start with "- "
            lines = block.splitlines()
            header = lines[0]  # "KB topics:" or "KB topics: something"
            after_colon = header.split(":", 1)[1].strip().rstrip(".")
            bullet_items: list[str] = []
            for line in lines[1:]:
                stripped = line.strip()
                if stripped.startswith("- "):
                    bullet_items.append(stripped[2:].strip().rstrip("."))
                elif stripped:
                    # Non-bullet continuation — treat as additional item
                    bullet_items.append(stripped.rstrip("."))
                else:
                    break  # blank line ends the block
            if bullet_items:
                items = ", ".join(bullet_items)
                return f"KB topics: {items}"
            # No bullets — just clean up the header value
            return f"KB topics: {after_colon}" if after_colon else header

        # Match "KB topics" (with any optional text before the colon, e.g.
        # "KB topics (for developer/tester reference):") followed by optional
        # inline text and optional bullet-list lines on subsequent lines.
        return re.sub(
            r'KB topics[^:\n]*:[^\n]*(?:\n[ \t]*-[^\n]+)*',
            _replace_block,
            spec,
            flags=re.IGNORECASE,
        )

    def _spec_all_formats(self) -> str:
        """Return all four REQUIREMENTS_SPEC format templates as a single block."""
        return (
            "── REQUIREMENTS_SPEC formats (use the one matching your task type) ──\n\n"
            + self._SPEC_BUG + "\n"
            + self._SPEC_FEATURE + "\n"
            + self._SPEC_MODIFY + "\n"
            + self._SPEC_QUESTION
        )

    # ── New helpers for enhanced BUG_FIX investigation ─────────────────────

    @staticmethod
    def _extract_target_files_from_preseed(pre_result: str) -> list[str]:
        """
        Extract file paths mentioned in pre-seed results.

        Looks for common patterns:
          ### Header.jsx (full source) — src/components/Header.jsx
          ### Header (function) — src/components/Header.jsx lines 1-112
        Returns a list of unique file basenames suitable for full-file lookup.
        """
        # Match filenames with extensions in ### headers
        header_files = re.findall(
            r'###\s+\S+.*?—\s*(\S+\.(?:jsx?|tsx?|vue|svelte|py|css|scss))',
            pre_result,
        )
        # Also match bare filenames in the content (e.g. "Header.jsx")
        inline_files = re.findall(
            r'\b(\w+\.(?:jsx?|tsx?|vue|svelte|py|css|scss))\b',
            pre_result,
        )
        seen = set()
        result = []
        for f in header_files + inline_files:
            basename = f.split("/")[-1].split("\\")[-1]
            if basename.lower() not in seen:
                seen.add(basename.lower())
                result.append(basename)
        return result

    @staticmethod
    def _read_full_file(kb_context_builder, filename: str) -> str:
        """
        Read the full source of *filename* from the project.

        Walks the project tree to find the file, returns its content capped at
        300 lines.  Returns empty string if not found.
        """
        project_root = getattr(kb_context_builder, '_project_root', None)
        if not project_root:
            return ""
        skip_dirs = {"node_modules", ".git", "dist", "build", ".next", "__pycache__"}
        target_lower = filename.lower()
        for dirpath, dirnames, filenames in os.walk(project_root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                if fname.lower() == target_lower:
                    abs_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(abs_path, project_root).replace("\\", "/")
                    try:
                        with open(abs_path, encoding="utf-8", errors="ignore") as fh:
                            all_lines = fh.readlines()
                        cap = 300
                        content = "".join(all_lines[:cap])
                        if len(all_lines) > cap:
                            content += f"\n  ... ({len(all_lines) - cap} more lines)"
                        return f"### {fname} (full source) — {rel_path}\n{content}"
                    except OSError:
                        pass
        return ""

    @staticmethod
    def _search_related_css(kb_context_builder, raw_task: str, pre_result: str) -> str:
        """
        Find and return CSS/style files that could affect the target component.

        Strategy:
          1. Extract component names from pre-seed results
          2. Search for matching .css/.scss files by component name
          3. Also search for global stylesheets (index.css, App.css, globals.css)
          4. Return the content of z-index, display, overflow, position, and
             pointer-events rules found in those files
        """
        project_root = getattr(kb_context_builder, '_project_root', None)
        if not project_root:
            return ""

        # Extract component names from pre-seed
        components = re.findall(r'###\s+(\w+)', pre_result)
        # Add common global CSS file stems
        css_targets = {"index", "app", "global", "globals", "main", "style", "styles"}
        for comp in components:
            css_targets.add(comp.lower())

        skip_dirs = {"node_modules", ".git", "dist", "build", ".next", "__pycache__"}
        css_exts = {".css", ".scss", ".sass", ".less"}
        sections: list[str] = []

        for dirpath, dirnames, filenames in os.walk(project_root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in css_exts:
                    continue
                stem = os.path.splitext(fname)[0].lower()
                if stem not in css_targets:
                    continue
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, project_root).replace("\\", "/")
                try:
                    with open(abs_path, encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    if not content.strip():
                        continue
                    # Cap at 200 lines
                    lines = content.splitlines()
                    if len(lines) > 200:
                        content = "\n".join(lines[:200]) + f"\n  ... ({len(lines) - 200} more lines)"
                    sections.append(f"### {fname} — {rel_path}\n{content}")
                except OSError:
                    continue

        return "\n\n".join(sections[:4])  # cap at 4 CSS files

    @staticmethod
    def _extract_target_file(spec: str) -> str:
        """Extract the most likely target filename from a REQUIREMENTS_SPEC.

        Looks for 'target:', 'Modify:', or file-path patterns in the spec
        and returns the first match.  Used by the auto-investigation fallback
        when the LLM keeps producing specs without using tools.
        """
        # Try explicit target/modify lines first
        target_m = re.search(
            r'(?:target|modify|file|component)\s*:\s*(\S+)',
            spec, re.IGNORECASE,
        )
        if target_m:
            return target_m.group(1).strip().rstrip(",;")
        # Fall back to any file-path-like pattern
        path_m = re.search(r'[\w/\\]+\.\w{1,5}', spec)
        if path_m:
            return path_m.group(0)
        return ""

    @staticmethod
    def _check_root_cause_contradiction(spec: str, evidence: str) -> str:
        """
        Cross-check BUG_FIX root cause against accumulated evidence.

        Returns a contradiction description if found, empty string if consistent.

        Checks for the common LLM error pattern: spec claims something is
        "missing" or "lacks" but the evidence shows it actually exists.
        """
        root_m = re.search(
            r'Root cause:\s*(.+?)(?=\n\w|\Z)',
            spec,
            re.IGNORECASE | re.DOTALL,
        )
        if not root_m:
            return ""
        root_cause = root_m.group(1).strip()

        # Pattern: "lacks an onClick" / "missing onClick" / "no onClick handler"
        # Check if the evidence actually shows this element exists
        missing_patterns = re.findall(
            r'(?:lacks?|missing|no|absent|without|does not have)\s+(?:an?\s+)?'
            r'[`\'"]?(\w+(?:\s*\([^)]*\))?)[`\'"]?(?:\s+handler|\s+prop|\s+attribute)?',
            root_cause,
            re.IGNORECASE,
        )
        for claimed_missing in missing_patterns:
            # Clean up: "onClick" from "onClick handler"
            token = claimed_missing.strip().split("(")[0].strip()
            if not token or len(token) < 3:
                continue
            # Search evidence for the token actually existing in code context
            # Look for it in code blocks (``` ... ```) within evidence
            code_blocks = re.findall(r'```[\s\S]*?```', evidence)
            full_source_blocks = re.findall(
                r'### .+?\(full source\).*?\n([\s\S]*?)(?=\n###|\Z)', evidence,
            )
            searchable = " ".join(code_blocks + full_source_blocks)
            if token in searchable:
                return (
                    f"Spec claims '{token}' is missing/lacking, but the evidence "
                    f"shows '{token}' already exists in the code. The root cause "
                    f"is likely something else (CSS z-index, display, conditional "
                    f"rendering, or event propagation)."
                )
        return ""

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


# ── IntentSpec ─────────────────────────────────────────────────────────────────
# Structured representation of the REQUIREMENTS_SPEC the IntentAgent produces.
# Parsed zero-cost from the enriched task string — no extra LLM call.

from dataclasses import dataclass, field as _dc_field


@dataclass
class IntentSpec:
    """Structured fields extracted from the IntentAgent's REQUIREMENTS_SPEC."""
    task_type: str = ""
    do_not_touch: list[str] = _dc_field(default_factory=list)   # BUG_FIX
    fix_scope: list[str] = _dc_field(default_factory=list)      # BUG_FIX
    preserve: list[str] = _dc_field(default_factory=list)       # MODIFY
    change_scope: list[str] = _dc_field(default_factory=list)   # MODIFY
    constraints: list[str] = _dc_field(default_factory=list)    # all types
    packages: list[str] = _dc_field(default_factory=list)       # FEATURE
    integrate_with: list[str] = _dc_field(default_factory=list) # FEATURE


def _extract_spec_field(spec_text: str, field_name: str) -> str:
    """Return the value for *field_name* from REQUIREMENTS_SPEC text (single line)."""
    m = re.search(
        rf'^{re.escape(field_name)}:\s*(.+)',
        spec_text,
        re.IGNORECASE | re.MULTILINE,
    )
    return m.group(1).strip() if m else ""


def _split_spec_values(text: str) -> list[str]:
    """Split a comma/semicolon-separated spec field into individual items."""
    if not text or text.lower().strip() in ('none', 'n/a', '-'):
        return []
    items = re.split(r'[,;]', text)
    return [
        item.strip().strip('"\'`<>')
        for item in items
        if item.strip() and item.strip().lower() not in ('none', 'n/a')
    ]


def parse_intent_spec(enriched_task: str) -> "IntentSpec | None":
    """Parse IntentSpec from the enriched task string produced by IntentAgent.

    Returns ``None`` when no REQUIREMENTS_SPEC block is present (blank projects,
    fallback paths, QUESTION tasks) so callers can skip intent-aware logic.
    """
    if not enriched_task:
        return None

    # Locate the REQUIREMENTS_SPEC block (ends at next '===' header or EOF)
    m = re.search(
        r'REQUIREMENTS_SPEC:\s*\n(.*?)(?=\n===|\Z)',
        enriched_task,
        re.DOTALL,
    )
    if not m:
        return None

    spec = m.group(1)

    task_type_m = re.search(r'^Task type:\s*(\S+)', spec, re.IGNORECASE | re.MULTILINE)
    task_type = task_type_m.group(1).strip() if task_type_m else ""

    return IntentSpec(
        task_type=task_type,
        do_not_touch=_split_spec_values(_extract_spec_field(spec, "Do not touch")),
        fix_scope=_split_spec_values(_extract_spec_field(spec, "Fix scope")),
        preserve=_split_spec_values(_extract_spec_field(spec, "Preserve")),
        change_scope=_split_spec_values(_extract_spec_field(spec, "Change scope")),
        constraints=_split_spec_values(_extract_spec_field(spec, "Constraints")),
        packages=_split_spec_values(_extract_spec_field(spec, "Packages/versions")),
        integrate_with=_split_spec_values(_extract_spec_field(spec, "Integrate with")),
    )

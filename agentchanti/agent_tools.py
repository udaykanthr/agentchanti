"""
Agent tool registry — the agent-computer interface for tool-calling loops.

Wraps existing subsystems (Executor, KB Searcher, FileMemory) as a small
set of :class:`~agentchanti.llm.chat_types.ToolDef` tools that a model can
invoke through ``LLMClient.chat()``. Execution never raises: every outcome
(including errors) is returned as a string so it can be fed straight back
to the model as a tool-result message.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Optional

from .cli_display import log
from .llm.chat_types import Message, ToolCall, ToolDef

# Directories never listed/searched — build artifacts and VCS internals.
_IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".agentchanti", ".pytest_cache", ".mypy_cache",
    ".next", ".nuxt", "coverage", "target",
})

# Caps keep tool results within a predictable token budget.
_MAX_READ_CHARS = 40_000
_MAX_CMD_OUTPUT_CHARS = 8_000
_MAX_LIST_ENTRIES = 300

# POSIX heredoc (`python - << 'PY' ... PY`). cmd.exe parses `<<` as two
# redirects, so on Windows the command exits 1 with no useful output and
# the model retries variations of the same broken syntax.
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?\w+['\"]?")


# Both `python -m unittest` and `python -m pytest` exit 5 when the runner
# COLLECTED NOTHING — a discovery problem, not a failing assertion. The
# tool result only ever said "exit: FAILED", so the model could not tell
# the two apart and debugged the wrong thing: observed a loop spending
# four consecutive run_command turns re-running a suite that had no tests
# to run, then editing source that was never the problem. 19 occurrences
# across 7 of 8 measured runs.
_NO_TESTS_EXIT = 5
# Substring match, not a regex: the command only has to LOOK like a test
# runner for exit 5 to be meaningful.
_TEST_RUNNER_TOKENS = ("pytest", "unittest", "nose2", "tox",
                       "manage.py test", "go test", "npm test")
_NO_TESTS_OUTPUT_MARKERS = ("no tests ran", "ran 0 tests",
                            "collected 0 items")


def _no_tests_collected(command: str, exit_code, output: str) -> bool:
    """True when a test runner found nothing to run.

    Exit 5 alone is not enough — it is an ordinary failure code for other
    programs — so the command must look like a test runner, or the output
    must say so outright.
    """
    low = (command or "").lower()
    if not any(tok in low for tok in _TEST_RUNNER_TOKENS):
        return False
    if (isinstance(exit_code, int) and not isinstance(exit_code, bool)
            and exit_code == _NO_TESTS_EXIT):
        return True
    low_out = (output or "").lower()
    return any(m in low_out for m in _NO_TESTS_OUTPUT_MARKERS)


_NO_TESTS_HINT = (
    "\n\nNOTE: the runner exited having COLLECTED NO TESTS. This is a "
    "discovery problem, not a failing assertion — nothing was executed, so "
    "there is no bug in the code under test to chase here. Check that the "
    "test file exists, is named test_*.py, sits in a directory with an "
    "__init__.py if you are importing it as a package, and that you are "
    "running from the project root."
)


def _truncate(text: str, limit: int, what: str = "output") -> str:
    if len(text) <= limit:
        return text
    return (text[:limit]
            + f"\n... [{what} truncated at {limit} chars"
              f" of {len(text)} total]")


class AgentTools:
    """Executable tool set scoped to one project root.

    Parameters
    ----------
    project_root:
        Directory all file paths resolve against; access outside it is
        rejected.
    executor:
        :class:`~agentchanti.executor.Executor` for ``run_command``.
        Created lazily when omitted.
    searcher:
        Optional KB :class:`~agentchanti.kb.local.searcher.Searcher` backing
        ``search_code``. Without it the tool degrades to a hint message.
    memory:
        Optional :class:`~agentchanti.orchestrator.memory.FileMemory`;
        writes/edits are recorded so the rest of the pipeline sees them.
    command_timeout:
        Seconds before ``run_command`` gives up.
    """

    def __init__(self, project_root: str = ".", executor=None,
                 searcher=None, memory=None, command_timeout: int = 120):
        self.project_root = os.path.abspath(project_root)
        self._executor = executor
        self._searcher = searcher
        self._memory = memory
        self._command_timeout = command_timeout

    # ── Definitions ──

    def definitions(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="list_files",
                description=(
                    "List files under a directory (recursive), relative to "
                    "the project root. Build artifacts and VCS directories "
                    "are skipped."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Directory to list, relative "
                                                "to project root. Default: "
                                                "project root."},
                    },
                },
            ),
            ToolDef(
                name="read_file",
                description=(
                    "Read a file's content. Returns numbered lines. "
                    "Optionally restrict to a line range."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "File path relative to "
                                                "project root."},
                        "start_line": {"type": "integer",
                                       "description": "First line (1-based)."},
                        "end_line": {"type": "integer",
                                     "description": "Last line (inclusive)."},
                    },
                    "required": ["path"],
                },
            ),
            ToolDef(
                name="write_file",
                description=(
                    "Create or fully overwrite a file with the given "
                    "content. Use edit_file for partial changes to an "
                    "existing file."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "File path relative to "
                                                "project root."},
                        "content": {"type": "string",
                                    "description": "Complete file content."},
                    },
                    "required": ["path", "content"],
                },
            ),
            ToolDef(
                name="edit_file",
                description=(
                    "Replace one exact occurrence of old_text with new_text "
                    "in a file. old_text must match exactly (including "
                    "whitespace) and be unique in the file — include enough "
                    "surrounding lines to make it unique."),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "File path relative to "
                                                "project root."},
                        "old_text": {"type": "string",
                                     "description": "Exact text to replace."},
                        "new_text": {"type": "string",
                                     "description": "Replacement text."},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            ),
            ToolDef(
                name="run_command",
                description=(
                    "Run a shell command in the project root and return its "
                    "combined output. Non-interactive; commands that prompt "
                    "for input will fail."),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string",
                                    "description": "Shell command to run."},
                    },
                    "required": ["command"],
                },
            ),
            ToolDef(
                name="search_code",
                description=(
                    "Semantic search over the project's code (knowledge "
                    "base). Returns matching symbols with file, line range "
                    "and snippet."),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Natural-language query, "
                                                 "e.g. 'where is user auth "
                                                 "validated'."},
                    },
                    "required": ["query"],
                },
            ),
        ]

    # ── Execution ──

    def execute(self, call: ToolCall) -> str:
        """Execute one tool call; always returns a string result."""
        handler = getattr(self, f"_tool_{call.name}", None)
        if handler is None:
            names = ", ".join(t.name for t in self.definitions())
            return f"ERROR: unknown tool '{call.name}'. Available: {names}"
        try:
            return handler(**call.arguments)
        except TypeError as e:
            return f"ERROR: bad arguments for {call.name}: {e}"
        except Exception as e:
            log.warning(f"[AgentTools] {call.name} failed: {e}")
            return f"ERROR: {call.name} failed: {e}"

    def execute_all(self, calls: list[ToolCall]) -> list[Message]:
        """Execute tool calls and wrap results as ``role="tool"`` messages."""
        return [
            Message(role="tool", content=self.execute(c),
                    tool_call_id=c.id, tool_name=c.name)
            for c in calls
        ]

    # ── Helpers ──

    def _resolve(self, path: str) -> str:
        """Resolve *path* inside the project root; reject escapes."""
        full = os.path.abspath(os.path.join(self.project_root, path))
        if os.path.commonpath([full, self.project_root]) != self.project_root:
            raise ValueError(f"path '{path}' is outside the project root")
        return full

    def _record(self, rel_path: str, content: str) -> None:
        if self._memory is not None:
            try:
                self._memory.update({rel_path: content})
            except Exception as e:
                log.debug(f"[AgentTools] FileMemory update failed: {e}")

    # ── Tool implementations ──

    def _tool_list_files(self, path: str = ".") -> str:
        root = self._resolve(path)
        if not os.path.isdir(root):
            return f"ERROR: '{path}' is not a directory"
        entries: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in _IGNORED_DIRS
                                 and not d.startswith("."))
            for fname in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, fname),
                                      self.project_root)
                entries.append(rel.replace("\\", "/"))
                if len(entries) >= _MAX_LIST_ENTRIES:
                    entries.append(f"... [listing truncated at "
                                   f"{_MAX_LIST_ENTRIES} entries]")
                    return "\n".join(entries)
        return "\n".join(entries) if entries else "(empty directory)"

    def _tool_read_file(self, path: str, start_line: Optional[int] = None,
                        end_line: Optional[int] = None) -> str:
        full = self._resolve(path)
        if not os.path.isfile(full):
            return f"ERROR: file not found: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        lo = max(1, start_line or 1)
        hi = min(len(lines), end_line or len(lines))
        if lo > len(lines):
            return f"ERROR: start_line {lo} beyond end of file ({len(lines)} lines)"
        numbered = "\n".join(
            f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1))
        header = f"{path} (lines {lo}-{hi} of {len(lines)})\n"
        return header + _truncate(numbered, _MAX_READ_CHARS, "file content")

    def _tool_write_file(self, path: str, content: str) -> str:
        full = self._resolve(path)
        os.makedirs(os.path.dirname(full) or self.project_root, exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        self._record(os.path.relpath(full, self.project_root), content)
        return f"OK: wrote {len(content)} chars to {path}"

    def _tool_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        full = self._resolve(path)
        if not os.path.isfile(full):
            return f"ERROR: file not found: {path}"
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        count = content.count(old_text)
        if count == 0:
            return ("ERROR: old_text not found in file. It must match "
                    "exactly, including whitespace and indentation. "
                    "Re-read the file and try again.")
        if count > 1:
            return (f"ERROR: old_text matches {count} locations. Include "
                    "more surrounding lines to make it unique.")

        updated = content.replace(old_text, new_text, 1)

        # Syntax-validate Python edits before committing them to disk.
        if full.endswith(".py"):
            try:
                ast.parse(updated)
            except SyntaxError as e:
                return (f"ERROR: edit rejected — resulting Python has a "
                        f"syntax error at line {e.lineno}: {e.msg}")
        # Same for JSON: a structurally broken package.json breaks every
        # subsequent npm/node invocation with a confusing downstream error.
        # tsconfig*.json is JSONC (comments/trailing commas allowed) — skip.
        if (full.endswith(".json")
                and not os.path.basename(full).startswith("tsconfig")):
            import json as _json
            try:
                _json.loads(updated)
            except ValueError as e:
                return (f"ERROR: edit rejected — resulting JSON is invalid: "
                        f"{e}. Re-read the file and fix commas/braces.")

        with open(full, "w", encoding="utf-8", newline="\n") as f:
            f.write(updated)
        self._record(os.path.relpath(full, self.project_root), updated)
        return f"OK: replaced 1 occurrence in {path}"

    def _tool_run_command(self, command: str) -> str:
        if os.name == "nt" and _HEREDOC_RE.search(command):
            return ("ERROR: POSIX heredoc syntax (<<) does not work on "
                    "Windows cmd — the command would fail without a useful "
                    "error. Write the script to a file with write_file and "
                    "run that file, or use python -c \"...\" for one-liners.")
        if self._executor is None:
            from .executor import Executor
            self._executor = Executor()
        success, output = self._executor.run_command(
            command, timeout=self._command_timeout, cwd=self.project_root)
        status = "exit: success" if success else "exit: FAILED"
        body = _truncate(output or "(no output)", _MAX_CMD_OUTPUT_CHARS)
        # "exit: FAILED" reads identically whether an assertion failed or
        # the runner never found a test to run. Say which.
        hint = ""
        if not success and _no_tests_collected(
                command, getattr(self._executor, "last_exit_code", None),
                output):
            hint = _NO_TESTS_HINT
        return f"{status}\n{body}{hint}"

    def _tool_search_code(self, query: str) -> str:
        if self._searcher is None:
            return ("search_code unavailable (no knowledge base index). "
                    "Use list_files and read_file to explore instead.")
        results = self._searcher.search(query, top_k=5)
        if not results:
            return f"No results for: {query}"
        parts = []
        for r in results:
            snippet = _truncate(r.code_snippet or "", 1_200, "snippet")
            parts.append(
                f"{r.file}:{r.line_start}-{r.line_end} "
                f"[{r.symbol_type}] {r.symbol_name} (score {r.score:.2f})\n"
                f"{snippet}")
        return "\n\n".join(parts)

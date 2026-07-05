"""Runtime smoke verification — actually launch the app after the pipeline.

Tests can pass while the application crashes or renders nothing at runtime
(GUI apps especially: tests mock the graphics library and never draw a
frame).  This stage launches the project's entry point briefly through the
Executor (which resolves the project venv):

- process crashes within the launch window → the traceback is fed to an LLM
  fix loop (bounded attempts), then the launch is retried;
- process is still running after the window → healthy launch, kill it;
- process exits 0 quickly → CLI-style script, success.

The whole stage is best-effort: no entry point, non-Python projects, or a
headless environment (no display) all skip silently rather than fail.
"""

import logging
import os
import re

_logger = logging.getLogger(__name__)

# A module with a __main__ guard is a runnable entry point
_MAIN_GUARD_RE = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']')

# Preferred entry-point basenames, most likely first
_ENTRY_PRIORITY = ("__main__.py", "main.py", "app.py", "run.py", "cli.py", "game.py")

# Crash signatures caused by the environment (headless CI), not the code
_HEADLESS_SIGNATURES = (
    "NoSuchDisplayException",
    "ScreenNotFoundException",
    "no display",
    "cannot connect to display",
    "couldn't connect to display",
    "DISPLAY environment variable",
)

# Traceback file references: File "path", line N
_TB_FILE_RE = re.compile(r'File "([^"]+)"')

_MAX_FIX_ATTEMPTS = 2


def find_python_entrypoint(memory_files: dict[str, str]) -> str | None:
    """Pick the most likely runnable Python entry point from session files."""
    from .pipeline import _is_test_file

    candidates = []
    for path, content in memory_files.items():
        norm = path.replace("\\", "/")
        if not norm.endswith(".py") or norm.startswith("_"):
            continue
        base = norm.rsplit("/", 1)[-1]
        if base == "conftest.py" or _is_test_file(norm):
            continue
        if not isinstance(content, str) or not _MAIN_GUARD_RE.search(content):
            continue
        candidates.append(norm)

    if not candidates:
        return None

    def rank(p: str):
        base = p.rsplit("/", 1)[-1]
        try:
            prio = _ENTRY_PRIORITY.index(base)
        except ValueError:
            prio = len(_ENTRY_PRIORITY)
        return (prio, p.count("/"), len(p))

    return min(candidates, key=rank)


def build_run_command(entry_path: str) -> str:
    """Build the launch command.

    ``python -m pkg.mod`` when the file's own directory is a package
    (has ``__init__.py``) — running such files as scripts breaks their
    relative imports.  Parent levels may be namespace packages (no
    ``__init__.py``); ``-m`` handles those fine.  Otherwise ``python path``.
    """
    norm = entry_path.replace("\\", "/").lstrip("./")
    parts = norm.split("/")
    if len(parts) > 1 and os.path.isfile(
        os.path.join(*parts[:-1], "__init__.py")
    ):
        if parts[-1] == "__main__.py":
            module = ".".join(parts[:-1])
        else:
            module = ".".join(parts)[: -len(".py")]
        return f"python -m {module}"
    return f'python "{norm}"'


def _is_headless_failure(output: str) -> bool:
    low = output.lower()
    return any(sig.lower() in low for sig in _HEADLESS_SIGNATURES)


def _launch(executor, cmd: str) -> tuple[bool, str]:
    """Launch *cmd* in the background and classify the result.

    The Executor's background mode waits ~3s: a crash inside that window
    returns ``(False, traceback)``; still-running means a healthy launch
    (we kill the process tree afterwards).
    """
    baseline = len(executor._background_processes)
    try:
        ok, out = executor.run_command(cmd, background=True)
    finally:
        executor.stop_background_processes_from(baseline)
    return ok, out or ""


def _files_from_traceback(output: str, memory_files: dict[str, str]) -> list[str]:
    """Map traceback file paths back to session files (max 4)."""
    matched: list[str] = []
    for tb_path in _TB_FILE_RE.findall(output):
        tb_norm = tb_path.replace("\\", "/")
        for mem_path in memory_files:
            mem_norm = mem_path.replace("\\", "/").lstrip("./")
            if tb_norm.endswith(mem_norm) and mem_path not in matched:
                matched.append(mem_path)
                break
        if len(matched) >= 4:
            break
    return matched


# Tokens that commonly appear with call syntax in generic error prose and
# would over-match nearly every source file.
_ERROR_SYMBOL_STOPWORDS = frozenset({
    "print", "warn", "warning", "warnings", "exit", "main", "run", "init",
    "str", "int", "type", "format", "len", "open", "input", "call",
    "called", "self", "super",
})


def _files_mentioning_error_symbols(
    crash_output: str, memory_files: dict[str, str]
) -> list[str]:
    """Session files whose content references a symbol the crash text
    names with call syntax (e.g. ``start_render() can only be called
    once``).

    Rescues fix targeting when an app-level try/except swallowed the
    traceback: with no ``File "..."`` lines the loop previously fell back
    to the entry point alone and rewrote the wrong file every attempt
    while the failing call sat in a file the error message named.
    """
    tokens = {
        t for t in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(", crash_output or "")
        if t.lower() not in _ERROR_SYMBOL_STOPWORDS
    }
    if not tokens:
        return []
    from .pipeline import _is_test_file
    matched: list[str] = []
    for path, content in memory_files.items():
        norm = path.replace("\\", "/")
        if not norm.endswith(".py") or _is_test_file(norm):
            continue
        if not isinstance(content, str):
            continue
        if any(f"{t}(" in content for t in tokens):
            matched.append(path)
    return matched


def _same_crash(a: str, b: str) -> bool:
    """True when two crash outputs describe the same failure."""
    if not a or not b:
        return False
    return a.strip()[-400:] == b.strip()[-400:]


def _installed_versions_line(executor) -> str:
    """Installed-packages prompt block so fixes target the real versions."""
    try:
        from .api_grounding import get_installed_package_versions
        versions = get_installed_package_versions(executor=executor)
    except Exception:
        return ""
    pkgs = [f"{n}=={v}" for n, v in sorted(versions.items())
            if n not in ("pip", "setuptools", "wheel")]
    if not pkgs:
        return ""
    return (
        "=== INSTALLED PACKAGES (write code against these EXACT versions) "
        "===\n" + ", ".join(pkgs[:40]) + "\n\n"
    )


def _probe_fix_files(fix_files: dict[str, str], executor, memory_files) -> list[str]:
    """API-probe a proposed fix before it is written. Best-effort."""
    py_fixes = {p: c for p, c in fix_files.items() if p.endswith(".py")}
    if not py_fixes:
        return []
    try:
        from .api_grounding import (local_top_levels_from_files,
                                    probe_api_usage)
        return probe_api_usage(
            py_fixes, executor,
            local_top_levels=local_top_levels_from_files(memory_files.keys()),
        )
    except Exception as exc:
        _logger.debug("[SmokeTest] API probe failed (non-fatal): %s", exc)
        return []


def _attempt_fix(
    crash_output: str,
    cmd: str,
    memory,
    executor,
    coder,
    entry_path: str,
    stuck_note: str = "",
) -> list[str]:
    """One LLM fix attempt from the crash output. Returns the list of
    files written (empty when no fix was applied)."""
    memory_files = memory.all_files()
    fix_targets = _files_from_traceback(crash_output, memory_files)
    if not fix_targets and entry_path in memory_files:
        fix_targets = [entry_path]
    # Widen with files the error text implicates by symbol name — the
    # traceback may have been swallowed by an app-level try/except.
    for extra in _files_mentioning_error_symbols(crash_output, memory_files):
        if extra not in fix_targets and len(fix_targets) < 4:
            fix_targets.append(extra)
    if not fix_targets:
        return []

    sources = []
    for p in fix_targets:
        sources.append(f"--- {p} ---\n```\n{memory_files[p]}\n```")

    prompt = (
        f"The application crashed when launched with `{cmd}`.\n\n"
        f"=== CRASH OUTPUT ===\n{crash_output[-3000:]}\n\n"
        + _installed_versions_line(executor)
        + "=== SOURCE FILES ===\n" + "\n\n".join(sources) + "\n\n"
        "Fix the crash. Pay close attention to the exact error message — it "
        "often states the correct API to use. Reply with the COMPLETE "
        "corrected content of each file that needs changes, using exactly "
        "this format for each file:\n"
        "#### [FILE]: path/to/file.py\n"
        "```python\n"
        "# complete file content — never abbreviate\n"
        "```\n"
        "Only modify the files shown above."
        + stuck_note
    )

    allowed = {p.replace("\\", "/") for p in fix_targets}
    feedback = ""
    fix_files: dict[str, str] = {}
    for ground_attempt in range(2):
        try:
            response = coder.llm_client.generate_response(prompt + feedback)
        except Exception as exc:
            _logger.warning("[SmokeTest] Fix LLM call failed: %s", exc)
            return []

        fix_files = executor.parse_code_blocks(response)
        if not fix_files:
            fix_files = executor.parse_code_blocks_fuzzy(response)

        # Only accept fixes for files the crash implicated
        fix_files = {
            p: c for p, c in (fix_files or {}).items()
            if p.replace("\\", "/").lstrip("./") in allowed
        }
        if not fix_files:
            _logger.warning(
                "[SmokeTest] No applicable fix files in LLM response")
            return []

        # A fix that swaps one missing API for another just burns a launch
        # attempt — probe it first and re-ask with the probe's suggestions.
        api_errs = _probe_fix_files(fix_files, executor, memory_files)
        if not api_errs:
            break
        _logger.warning(
            "[SmokeTest] Proposed fix uses missing APIs (attempt %d): %s",
            ground_attempt + 1,
            "; ".join(e.split(" — ")[0] for e in api_errs))
        feedback = (
            "\n\n=== PREVIOUS ATTEMPT REJECTED ===\n"
            "Your previous fix used APIs that do NOT exist in the installed "
            "package versions:\n"
            + "\n".join(f"- {e}" for e in api_errs)
            + "\nUse the suggested replacements above."
        )
    else:
        _logger.warning(
            "[SmokeTest] Fix still uses missing APIs after re-ask — rejecting")
        return []

    executor.write_files(fix_files)
    memory.update(fix_files)
    _logger.info("[SmokeTest] Applied fix to: %s", list(fix_files.keys()))
    return list(fix_files.keys())


def run_smoke_verification(
    memory,
    executor,
    coder,
    display,
    task: str,
    language: str | None,
    cfg=None,
    max_fix_attempts: int = _MAX_FIX_ATTEMPTS,
) -> tuple[bool, str]:
    """Launch the app entry point and fix launch crashes. Returns (ok, error)."""
    if cfg is not None and not getattr(cfg, "SMOKE_TEST_ENABLED", True):
        return True, ""
    if language not in (None, "python"):
        return True, ""  # Python entry points only, for now

    memory_files = memory.all_files()
    entry = find_python_entrypoint(memory_files)
    if not entry:
        _logger.info("[SmokeTest] No runnable Python entry point — skipping")
        return True, ""

    cmd = build_run_command(entry)
    _logger.info("[SmokeTest] Launching app: %s", cmd)

    out = ""
    prev_out = ""
    prev_fixed: list[str] = []
    for attempt in range(max_fix_attempts + 1):
        ok, out = _launch(executor, cmd)
        if ok:
            _logger.info("[SmokeTest] App launched successfully (%s)", entry)
            return True, ""
        if _is_headless_failure(out):
            _logger.info(
                "[SmokeTest] No display available — skipping runtime check")
            return True, ""

        _logger.warning(
            "[SmokeTest] App crashed on launch (attempt %d/%d):\n%s",
            attempt + 1, max_fix_attempts + 1, out[-1500:])
        if attempt >= max_fix_attempts:
            break
        _show_status = getattr(display, "show_status", None)
        if callable(_show_status):
            _show_status("[SmokeTest] App crashed on launch — attempting fix")
        # Tell the LLM when its previous fix changed nothing — otherwise
        # it re-polishes the same wrong file every attempt (observed:
        # three identical crashes, three rewrites of the entry point,
        # while the failing call lived in another file).
        stuck_note = ""
        if prev_fixed and _same_crash(prev_out, out):
            _logger.warning(
                "[SmokeTest] Crash unchanged after fixing %s — redirecting",
                prev_fixed)
            stuck_note = (
                "\n\nIMPORTANT: A previous fix modified "
                + ", ".join(f"`{p}`" for p in prev_fixed)
                + " but the crash output is UNCHANGED. The bug is almost "
                "certainly in a DIFFERENT file — fix the file that actually "
                "contains the failing call named in the error message."
            )
        prev_out = out
        fixed = _attempt_fix(out, cmd, memory, executor, coder, entry,
                             stuck_note=stuck_note)
        if not fixed:
            break
        prev_fixed = fixed

    return False, (
        f"[SmokeTest] Application crashes when launched with `{cmd}`:\n"
        f"{out[-1500:]}"
    )

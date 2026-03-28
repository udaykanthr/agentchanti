import logging
import re
import os
from .memory import FileMemory
from ..executor import Executor
from ..cli_display import CLIDisplay, log
from ..language import get_test_framework

_logger = logging.getLogger(__name__)

# Regex for common error types
_ERROR_TYPE_RE = re.compile(
    r'(ModuleNotFoundError|ImportError|SyntaxError|NameError|'
    r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
    r'AssertionError|AssertError|KeyError|ValueError|ReferenceError|'
    r'RangeError|RuntimeError|OSError|IOError|PermissionError|'
    r'expect\(received\))'
)

# ANSI escape sequence regex
_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def _count_test_failures(output: str) -> int:
    """Count the number of individual test failures in test runner output."""
    if not output:
        return 0

    clean = _ANSI_RE.sub('', output)
    file_failures = 0
    test_failures = 0

    m_files = re.search(r'Test Files?\s+(\d+)\s+failed', clean)
    if m_files:
        file_failures = int(m_files.group(1))
    m_tests = re.search(r'Tests:\s*(\d+)\s+failed', clean)
    if m_tests:
        test_failures = int(m_tests.group(1))

    if file_failures or test_failures:
        return max(file_failures, test_failures)

    m = re.search(r'(\d+)\s+failed', clean)
    if m:
        return int(m.group(1))

    count = len(re.findall(r'---\s+FAIL:', clean))
    if count:
        return count

    count = 0
    for line in clean.splitlines():
        stripped = line.strip()
        if re.match(r'(FAILED\s+|×\s+|✕\s+|✗\s+)', stripped):
            count += 1

    return max(count, 1) if 'FAIL' in clean.upper() else 0

def _identify_test_files(
    output: str,
    all_files: dict[str, str],
    language: str | None = None,
) -> tuple[list[str], list[str]]:
    """Identify which test files had failures and which passed.

    Returns (failing_paths, passing_paths).
    """
    if not output:
        return [], []

    clean = _ANSI_RE.sub('', output)
    failed_basenames: set[str] = set()

    # Vitest: "❯ path/file.test.jsx (N tests | M failed)" or "❯ path/file.test.jsx:line:col"
    for m in re.finditer(r'[❯]\s+([^\s\(\:]+)', clean):
        fname = m.group(1).rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        failed_basenames.add(fname)

    # Jest: "FAIL path/file.test.jsx" or "FAIL  path/file.spec.js"
    for m in re.finditer(r'FAIL\s+(\S+\.(?:test|spec)\.\w+)', clean):
        fname = m.group(1).rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        failed_basenames.add(fname)

    # pytest: "FAILED path/test_file.py::test_name"
    for m in re.finditer(r'FAILED\s+(\S+\.py)::', clean):
        fname = m.group(1).rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        failed_basenames.add(fname)

    # Identify candidate test files from all files
    from ..language import TEST_FRAMEWORKS
    fw = TEST_FRAMEWORKS.get(language or "python", TEST_FRAMEWORKS["python"])
    test_dir = fw.get("dir", "tests")
    test_suffix = fw.get("suffix", ".test")
    test_prefix = fw.get("prefix", "test_")

    failing: list[str] = []
    passing: list[str] = []

    for fpath in all_files:
        is_test = False
        if test_dir and (test_dir in fpath or "__tests__" in fpath):
            is_test = True
        elif test_suffix and any(fpath.endswith(test_suffix + ext) for ext in ['.js', '.jsx', '.ts', '.tsx', fw.get("ext", "")]):
            is_test = True
        elif test_prefix and os.path.basename(fpath).startswith(test_prefix):
            is_test = True

        if not is_test:
            continue

        basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        if basename in failed_basenames:
            failing.append(fpath)
        else:
            passing.append(fpath)

    return failing, passing

def _extract_per_file_errors(output: str, failed_basenames: set[str],
                              max_chars_per_file: int = 800) -> dict[str, str]:
    """Extract condensed error details per failing test file from runner output.

    Returns a dict mapping basename → error summary string.
    """
    if not output:
        return {}

    clean = _ANSI_RE.sub('', output)
    lines = clean.splitlines()

    # Patterns that mark the start of a file's error block
    _FILE_HEADER = re.compile(
        r'(?:FAIL\s+(\S+)|[❯]\s+([^\s\(\:]+)|---\s+FAIL:\s+(\S+))')

    # Patterns for lines we want to keep (error messages, assertions, source pointers)
    _KEEP = re.compile(
        r'(Error[:\s]|error[:\s]|TypeError|ReferenceError|SyntaxError'
        r'|NameError|ModuleNotFoundError|ImportError|AttributeError'
        r'|AssertionError|AssertError|KeyError|ValueError'
        r'|expect\(|Expected\b|Received\b|Difference:'
        r'|×\s|✕\s|✗\s|FAIL\b|FAILED\b'
        r'|Unable to find|TestingLibraryElementError'
        r'|Transform failed|PARSE_ERROR|Unterminated'
        r'|╭─\[|─{3,}'  # Vite/OXC error box: "╭─[ file.jsx:132:27 ]"
        r'|\d+\s*\|)',  # source pointer lines like "  29 | expect(...)"
        re.IGNORECASE)

    # Collect per-file error blocks
    per_file: dict[str, list[str]] = {b: [] for b in failed_basenames}
    current_file: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check if this line starts a new file block
        m = _FILE_HEADER.search(stripped)
        if m:
            fname_raw = m.group(1) or m.group(2) or m.group(3) or ''
            fname = fname_raw.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
            if fname in per_file:
                current_file = fname
            continue

        # If we're inside a known failing file block, collect relevant lines
        if current_file and _KEEP.search(stripped):
            per_file[current_file].append(stripped)

    # Detect transform errors — the actual broken file is the SOURCE file
    # referenced in the Vite/OXC error box "╭─[ path/file.jsx:line:col ]",
    # NOT the test file that imports it.  Append a clear directive so the
    # planner doesn't rewrite the test file instead of fixing the source.
    _transform_source_re = re.compile(
        r'(?:╭─\[\s*|File:\s*)([^\s:\]]+\.[jt]sx?):(\d+)', re.IGNORECASE)
    transform_source_hint: str = ""
    if re.search(r'Transform failed|PARSE_ERROR', clean, re.IGNORECASE):
        m = _transform_source_re.search(clean)
        if m:
            src_file, src_line = m.group(1), m.group(2)
            transform_source_hint = (
                f"\nNOTE: This is a TRANSFORM error — the syntax problem is in the SOURCE FILE "
                f"'{src_file}' at line {src_line}, NOT in the test file. "
                f"Fix the source file '{src_file}' to resolve the parse error."
            )

    # Also do a fallback scan: if we couldn't attribute errors to files,
    # collect all error-like lines as a generic block
    result: dict[str, str] = {}
    for basename, err_lines in per_file.items():
        if err_lines:
            text = '\n'.join(err_lines[:15])  # cap at 15 lines
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + '\n... [truncated]'
            if transform_source_hint:
                text += transform_source_hint
            result[basename] = text

    # Fallback: if no file-specific errors found, extract generic error lines
    if not result and failed_basenames:
        generic_errors: list[str] = []
        for line in lines:
            stripped = line.strip()
            if _KEEP.search(stripped):
                generic_errors.append(stripped)
                if len(generic_errors) >= 20:
                    break
        if generic_errors:
            text = '\n'.join(generic_errors)
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + '\n... [truncated]'
            if transform_source_hint:
                text += transform_source_hint
            for basename in failed_basenames:
                result[basename] = text

    return result


def perform_baseline_test_analysis(
    memory: FileMemory,
    executor: Executor,
    language: str | None,
    project_profile=None,
    display: CLIDisplay | None = None,
    step_idx: int = 0,
    task_intent: str | None = None,
) -> str:
    """Perform baseline test analysis for planning or execution.

    *task_intent* is one of ``"test"``, ``"bug_fix"``, ``"feature"``,
    ``"refactor"``, or ``"general"``.  When the intent is ``"test"`` or
    ``"bug_fix"`` the directives are strict (don't touch passing files).
    For ``"feature"``/``"refactor"``/``"general"`` the directives allow
    updating passing test files when the new code requires it.

    Returns a summary string of the analysis.
    """
    if getattr(memory, '_tester_pre_analysis_done', False):
        return getattr(memory, '_tester_pre_analysis_summary', "")

    if display:
        display.step_info(step_idx, "Performing pre-execution analysis...")

    # Detect subproject root
    from .step_handlers import _detect_subproject_root, _read_js_project_env
    subproject_cwd = _detect_subproject_root(memory)

    # Detect test runner
    test_runner = None
    if language in ("javascript", "typescript"):
        js_env = _read_js_project_env(subproject_cwd)
        test_runner = js_env.get("test_runner")

    fw = get_test_framework(language or "python", test_runner=test_runner)
    test_cmd = fw["command"]

    _logger.info("Performing pre-execution baseline test analysis via %s", test_cmd)

    # 1. Run baseline tests
    success, output = executor.run_tests(test_cmd, cwd=subproject_cwd)
    _logger.info("Baseline test run success=%s", success)

    analysis_lines = ["PRE-EXECUTION ANALYSIS:"]

    # Determine if this is a test-fix task (strict mode) or a feature/refactor
    # task (allow updating passing tests to cover new code).
    _is_test_fix = task_intent in ("test", "bug_fix")

    # Track file lists for callers (e.g. task interpretation LLM)
    _baseline_failing_files: list[str] = []
    _baseline_passing_files: list[str] = []

    # Pattern covering common test file conventions across languages
    _TEST_FILE_PAT = re.compile(
        r'(?:'
        r'\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs)$'   # JS/TS
        r'|__tests__/.*\.[jt]sx?$'                         # Jest __tests__ dir
        r'|_test\.py$|test_[^/]+\.py$'                     # Python
        r'|_test\.go$'                                      # Go
        r'|_spec\.rb$'                                      # Ruby
        r')',
        re.I,
    )

    if success:
        # Scan memory to identify every test file that is currently passing.
        # This list is included in the analysis so the task interpreter and
        # planner know EXACTLY which files must not be touched.
        _all_mem_files = memory.all_files()
        _baseline_passing_files = sorted(
            fp for fp in _all_mem_files.keys()
            if _TEST_FILE_PAT.search(fp)
        )

        analysis_lines.append("- All existing tests are currently PASSING.")
        if _baseline_passing_files:
            analysis_lines.append("- PASSING TEST FILES (must NOT be modified):")
            for fp in _baseline_passing_files:
                analysis_lines.append(f"  - {fp}")

        if _is_test_fix:
            analysis_lines.append("- DIRECTIVE: Test environment is HEALTHY. Do NOT add setup or fix steps for tests.")
            analysis_lines.append("  Do NOT modify vitest.config, jest.config, setup files, or any test file.")
        else:
            analysis_lines.append("- DIRECTIVE: Test environment is HEALTHY. Do NOT add setup or fix steps.")
            analysis_lines.append("  Do NOT modify vitest.config, jest.config, or setup files.")
            analysis_lines.append("  You MAY update existing test files ONLY if the new code changes require it")
            analysis_lines.append("  (e.g. adding tests for a new feature, updating assertions after a refactor).")
            analysis_lines.append("  But do NOT rewrite or restructure tests that are unrelated to the task.")
    else:
        # 2. Identify failed/passing files
        all_files = memory.all_files()
        failed_paths, passing_paths = _identify_test_files(output, all_files, language=language)
        _baseline_failing_files = list(failed_paths)
        _baseline_passing_files = list(passing_paths)
        total_fails = _count_test_failures(output)

        # Collect failed basenames for error extraction
        failed_basenames: set[str] = set()
        for p in failed_paths:
            failed_basenames.add(p.rsplit('/', 1)[-1].rsplit('\\', 1)[-1])

        if total_fails == 0:
            analysis_lines.append("- Baseline: Test execution failed (likely setup error or no tests).")
            analysis_lines.append("- DIRECTIVE: Initialize or repair the test environment if needed.")
        else:
            analysis_lines.append(f"- Baseline: {total_fails} test(s) currently FAILING across {len(failed_paths)} file(s).")
            analysis_lines.append(f"- Total passing: {len(passing_paths)} file(s) with all tests green.")

            if passing_paths:
                if _is_test_fix:
                    analysis_lines.append("- HEALTHY (PASSING) files — DO NOT MODIFY:")
                else:
                    analysis_lines.append("- HEALTHY (PASSING) files — only update if new code changes require it:")
                for p in passing_paths:
                    analysis_lines.append(f"  - {p}")

            if failed_paths:
                analysis_lines.append("- BROKEN (FAILING) files — THESE ARE THE ONLY FILES TO FIX:")
                for p in failed_paths:
                    analysis_lines.append(f"  - {p}")

                # 3. Extract and include actual error details per file
                per_file_errors = _extract_per_file_errors(output, failed_basenames)
                if per_file_errors:
                    analysis_lines.append("")
                    analysis_lines.append("ACTUAL ERROR OUTPUT per failing file:")
                    for basename, err_text in per_file_errors.items():
                        # Find the full path for this basename
                        full_path = basename
                        for p in failed_paths:
                            if p.endswith(basename):
                                full_path = p
                                break
                        analysis_lines.append(f"\n  --- {full_path} ---")
                        for err_line in err_text.splitlines():
                            analysis_lines.append(f"  {err_line}")

            analysis_lines.append("")
            if _is_test_fix:
                analysis_lines.append("DIRECTIVES (CRITICAL — violating these wastes tokens and breaks passing tests):")
                analysis_lines.append("1. ONLY modify the BROKEN files listed above.")
                analysis_lines.append("2. Do NOT modify, recreate, or touch HEALTHY files — they already pass.")
                analysis_lines.append("3. Do NOT modify test config files (vitest.config, jest.config, setup files)")
                analysis_lines.append("   UNLESS the error output above shows a config/setup problem (e.g. all tests fail to start).")
                analysis_lines.append("4. Read the actual error messages above to determine the root cause.")
                analysis_lines.append("   A failing assertion (e.g. 'Unable to find role') means the TEST CODE needs fixing,")
                analysis_lines.append("   NOT the test setup or config.")
                analysis_lines.append("5. If only 1 file fails, generate exactly 1 CODE step targeting that file. Do not plan extra steps.")
            else:
                analysis_lines.append("DIRECTIVES:")
                analysis_lines.append("1. PRIORITIZE fixing the BROKEN files listed above.")
                analysis_lines.append("2. HEALTHY files may be updated ONLY if the task's new code changes require it")
                analysis_lines.append("   (e.g. a new export needs a new test, a renamed function needs updated assertions).")
                analysis_lines.append("   Do NOT rewrite or restructure tests unrelated to the task.")
                analysis_lines.append("3. Do NOT modify test config files (vitest.config, jest.config, setup files)")
                analysis_lines.append("   UNLESS the error output above shows a config/setup problem.")
                analysis_lines.append("4. Read the actual error messages above to determine the root cause of failures.")

    summary = "\n".join(analysis_lines)
    memory._tester_pre_analysis_done = True
    memory._tester_pre_analysis_summary = summary
    memory._tester_baseline_success = success
    # Store file lists so callers can build richer task interpretations
    memory._tester_baseline_failing_files = _baseline_failing_files
    memory._tester_baseline_passing_files = _baseline_passing_files
    return summary


def analyze_task_for_planner(
    task: str,
    relevant_files: list[tuple[str, str, str]],
    test_analysis: str,
    llm_client,
    passing_files: list[str] | None = None,
    failing_files: list[str] | None = None,
) -> str:
    """LLM-based pre-planning analysis grounded in actual project files and test state.

    *relevant_files* is a list of ``(path, reason, skeleton)`` tuples produced
    by the keyword/KB pre-filter — only the files already deemed relevant to
    the task, not the entire project.  This keeps the prompt focused.

    *passing_files* / *failing_files* are the complete per-file test results
    from ``perform_baseline_test_analysis``.  Passing them explicitly (rather
    than extracting from the text blob) ensures the full list always reaches
    the interpreter regardless of how long the analysis text is.

    Returns a ``TASK BRIEFING`` block string, or empty string on failure.
    The caller (``PlannerAgent.pre_analyze``) injects it as the first thing
    the planner sees.
    """
    # Build the file context from pre-filtered relevant files only
    file_section = ""
    if relevant_files:
        file_lines = []
        for fpath, reason, skeleton in relevant_files:
            file_lines.append(f"### {fpath}  ({reason})")
            if skeleton:
                file_lines.append(f"```\n{skeleton}\n```")
        file_section = (
            "RELEVANT PROJECT FILES (pre-filtered to match the task):\n"
            + "\n".join(file_lines)
        )

    # Build the test state section.
    # Use explicit file lists when provided (complete, no truncation risk).
    # Fall back to extracting key lines from the text blob.
    test_section_lines: list[str] = ["CURRENT TEST STATE (live test run seconds ago):"]

    if passing_files is not None or failing_files is not None:
        # Explicit structured data — always complete
        if failing_files:
            test_section_lines.append(
                f"FAILING ({len(failing_files)} file(s)) — these need fixing:")
            for fp in failing_files:
                test_section_lines.append(f"  FAIL: {fp}")
        else:
            test_section_lines.append("All tests PASSING — 0 failures.")

        if passing_files:
            test_section_lines.append(
                f"PASSING ({len(passing_files)} file(s)) — must NOT be modified:")
            for fp in passing_files:
                test_section_lines.append(f"  PASS: {fp}")
    elif test_analysis:
        # Fallback: extract key lines from the raw text, but do NOT cap the
        # count — we need every file name to reach the interpreter.
        for line in test_analysis.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "  - ", "DIRECTIVE", "BROKEN",
                                    "HEALTHY", "PRE-EXECUTION", "Baseline:",
                                    "PASSING TEST FILES", "ACTUAL ERROR")):
                test_section_lines.append(stripped)

    test_section = "\n".join(test_section_lines) if len(test_section_lines) > 1 else ""

    sections = "\n\n".join(s for s in [file_section, test_section] if s)

    prompt = f"""\
You are a software project analyst preparing a briefing for an AI coding agent.
The agent will plan and execute code changes to accomplish the user's task.
Your job is to determine — based on the ACTUAL current project state below —
exactly what needs to change and what must not be touched.

USER TASK:
{task}

{sections}

Answer these questions concisely and precisely:
1. What is the real goal of this task (one sentence)?
2. Which existing files need to be modified?  Name them specifically.
3. Which new files (if any) need to be created?
4. Which files must NOT be touched and why?
5. What does a successful result look like?  (observable outcome, not process steps)
6. What is the single most important constraint the coding agent must respect?

Respond in this EXACT format — no extra text, no markdown outside the block:
TASK BRIEFING:
Goal: <one sentence>
Modify: <specific file paths, or NONE>
Create: <specific new file paths, or NONE>
Do not touch: <file paths or patterns that must not change>
Expected output: <observable result when done>
Key constraint: <the one rule the agent must not break>
Agent directive: <one concrete instruction for the planner>
"""
    try:
        response = llm_client.generate_response(prompt)
        if "TASK BRIEFING:" in response and "Agent directive:" in response:
            _logger.info("[TaskBriefing] LLM briefing:\n%s", response)
            return response.strip()
        _logger.warning(
            "[TaskBriefing] LLM response missing expected structure, skipping.")
        return ""
    except Exception as exc:
        _logger.warning("[TaskBriefing] LLM call failed: %s", exc)
        return ""

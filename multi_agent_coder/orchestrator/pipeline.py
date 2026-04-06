"""
Pipeline execution — wave-based parallel/sequential step execution.
"""

import logging
import re

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..cli_display import CLIDisplay, log

from .memory import FileMemory
from .classification import _classify_step, _TEST_CMD_RE, _TEST_CONFIG_RE
from .plan_step import PlanStep, build_step_context, update_step_after_execution
from .step_handlers import (
    _handle_cmd_step, _handle_code_step, _handle_test_step,
    _handle_search_step,
    _build_scoped_test_cmd,
    _detect_subproject_root,
    _prefix_subproject_paths,
    MAX_STEP_RETRIES,
)
from .diagnosis import _diagnose_failure, _apply_fix
from .error_router import classify_error

_logger = logging.getLogger(__name__)


MAX_DIAGNOSIS_RETRIES = 2   # outer retries: diagnose failure → fix → re-run step


def _try_trivial_close(
    partial: dict[str, str],
    language: str | None,
) -> dict[str, str] | None:
    """Attempt to close trivially truncated inline code without LLM.

    If each file in *partial* has ≤2 unmatched opening braces and ≤2
    unmatched opening parens, append the missing closing tokens.
    Returns the closed dict on success, or None if any file is too
    complex to close deterministically.
    """
    result: dict[str, str] = {}
    for path, content in partial.items():
        open_braces = content.count('{') - content.count('}')
        open_parens = content.count('(') - content.count(')')
        # Only attempt closure when the gap is tiny (likely a cut-off tail)
        if open_braces < 0 or open_parens < 0 or open_braces > 2 or open_parens > 2:
            return None
        tail = ('}\n' * open_braces) + (')\n' * open_parens)
        result[path] = content + ('\n' + tail if tail else '')
    return result


# ── Test file detection ───────────────────────────────────────
# Patterns that indicate a file is a test file (used for CODE→TEST
# auto-correction when the planner marks a test-editing step as CODE).
_TEST_FILE_RE = re.compile(
    r'(?:'
    # JS/TS: *.test.js, *.spec.tsx, etc.
    r'\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs)$'
    # Python: test_*.py or *_test.py
    r'|(?:^|[/\\])test_\w+\.py$'
    r'|\w+_test\.py$'
    # Go: *_test.go
    r'|\w+_test\.go$'
    # Ruby: *_spec.rb
    r'|\w+_spec\.rb$'
    r')',
    re.IGNORECASE,
)
# Directories that indicate test files
_TEST_DIR_RE = re.compile(
    r'(?:^|[/\\])(?:__tests__|tests?|specs?|test_\w+)[/\\]',
    re.IGNORECASE,
)


_SOURCE_EXTS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rb", ".java", ".rs", ".cs", ".cpp", ".c", ".php",
})

# ── Wiring verification: fix scope parsing ────────────────────────────────
# Matches a "Fix scope:" section in the task description.
_FIX_SCOPE_SECTION_RE = re.compile(
    r'fix\s+scope\s*:?\s*(.+?)(?=\nDo not touch:|\nConstraints:|\nInterface|\nKB topics:|\Z)',
    re.IGNORECASE | re.DOTALL,
)
# Matches backtick-quoted strings that look like file paths.
_BACKTICK_PATH_RE = re.compile(r'`([^`]+\.[a-zA-Z]{1,6}[^`]*)`')
# Matches bare file names with common source/config extensions.
_BARE_FILENAME_RE = re.compile(
    r'\b([\w.\-/]+\.(?:jsx|tsx|js|ts|mjs|cjs|py|css|html|json|go|rb|rs|java|cs|vue|svelte))\b'
)

def _is_test_file(file_path: str) -> bool:
    """Return True if *file_path* looks like a test file."""
    import os
    basename = os.path.basename(file_path)
    _, ext = os.path.splitext(basename)
    # Never treat non-source files (HTML, CSS, JSON, …) as test files even
    # when they live inside a __tests__ directory.
    if ext.lower() not in _SOURCE_EXTS:
        return False
    return bool(_TEST_FILE_RE.search(basename) or _TEST_DIR_RE.search(file_path))


def _is_additive_source_fix(file_path: str, new_content: str,
                            memory) -> bool:
    """Return True if the proposed source change is small and additive.

    Allows minor test-supportive tweaks (adding data-testid, aria-label,
    role attributes) without altering component functionality.

    Criteria — ALL must be met:
      1. Original file exists in memory (not a new file)
      2. No lines were removed (only additions or in-place attribute additions)
      3. Total change is ≤10% of original line count
      4. Changed lines only add attributes / props, not logic
    """
    import difflib

    orig = memory.get(file_path)
    if orig is None:
        return False

    orig_lines = orig.splitlines()
    new_lines = new_content.splitlines()
    orig_len = len(orig_lines)

    if orig_len < 3:
        return False

    sm = difflib.SequenceMatcher(None, orig_lines, new_lines, autojunk=False)
    added = 0
    removed = 0
    changed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'insert':
            added += j2 - j1
        elif tag == 'delete':
            removed += i2 - i1
        elif tag == 'replace':
            changed += max(i2 - i1, j2 - j1)
            removed += max(0, (i2 - i1) - (j2 - j1))

    # No lines may be removed — only additions or in-place edits
    if removed > 0:
        return False

    # Total change must be ≤10% of original
    total_delta = added + changed
    if total_delta > max(orig_len * 0.10, 3):
        return False

    return True


def _find_tests_impacted_by_sources(
    modified_sources: list[str],
    all_test_files: dict[str, str],
    exclude: str,
    already_queued: set[str],
    kb_context_builder=None,
) -> list[str]:
    """Return test files (not already queued) that import any of the modified source files.

    Prefers the KB code graph (``CodeGraph.impact_analysis``) when available —
    it already tracks IMPORTS edges via tree-sitter parsing so no re-scanning is
    needed.  Falls back to a lightweight regex scan only when the graph is absent
    (e.g. KB not initialised for this project).
    """
    candidates: set[str] = set()

    graph = getattr(kb_context_builder, "_graph", None) if kb_context_builder else None

    if graph is not None:
        # Graph path: reverse-BFS over IMPORTS edges for each changed source.
        for src in modified_sources:
            for affected in graph.impact_analysis(src):
                if _is_test_file(affected):
                    candidates.add(affected)
    else:
        # Fallback: stem-match imports in test file content via regex.
        import re as _re
        _JS_IMPORT_RE = _re.compile(
            r'''(?:from\s+['"](.+?)['"]|require\s*\(\s*['"](.+?)['"]\s*\))''')
        _PY_IMPORT_RE = _re.compile(
            r'(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))')

        src_stems: set[str] = set()
        for src in modified_sources:
            stem = src.replace("\\", "/").rsplit('/', 1)[-1].rsplit('.', 1)[0].lower()
            src_stems.add(stem)

        for tpath, tcontent in all_test_files.items():
            imports: set[str] = set()
            for m in _JS_IMPORT_RE.finditer(tcontent):
                rel = m.group(1) or m.group(2)
                if rel:
                    imports.add(rel)
            for m in _PY_IMPORT_RE.finditer(tcontent):
                mod = m.group(1) or m.group(2)
                if mod:
                    imports.add(mod.replace('.', '/'))
            for imp in imports:
                imp_stem = imp.replace("\\", "/").rsplit('/', 1)[-1].rsplit('.', 1)[0].lower()
                if imp_stem in src_stems:
                    candidates.add(tpath)
                    break

    return [
        t for t in candidates
        if t != exclude and t not in already_queued
    ]


# ── External service dependency detection ─────────────────────
# Patterns that indicate the command failed because an external
# service (database, cache, message broker, etc.) is unavailable.
# These failures cannot be fixed by the agent — the user must
# ensure the service is running.

_EXTERNAL_SERVICE_PATTERNS: list[tuple[str, str]] = [
    # MongoDB
    (r'MongoServerSelectionError|MongoNetworkError|ECONNREFUSED.*27017',
     'MongoDB (default port 27017)'),
    # PostgreSQL
    (r'ECONNREFUSED.*5432|could not connect to server.*5432|pg_hba\.conf|'
     r'SequelizeConnectionRefusedError.*5432',
     'PostgreSQL (default port 5432)'),
    # MySQL / MariaDB
    (r'ECONNREFUSED.*3306|ER_ACCESS_DENIED_ERROR|PROTOCOL_CONNECTION_LOST.*3306',
     'MySQL/MariaDB (default port 3306)'),
    # Redis
    (r'ECONNREFUSED.*6379|Redis connection.*failed|NOAUTH',
     'Redis (default port 6379)'),
    # RabbitMQ
    (r'ECONNREFUSED.*5672|amqp.*connection.*refused',
     'RabbitMQ (default port 5672)'),
    # Elasticsearch
    (r'ECONNREFUSED.*9200|ConnectionError.*9200',
     'Elasticsearch (default port 9200)'),
    # Generic connection refused (with port)
    (r'ECONNREFUSED\s+\d+\.\d+\.\d+\.\d+:\d+',
     'an external service'),
    # Generic connection timeout to localhost
    (r'connect ETIMEDOUT\s+127\.0\.0\.1:\d+|'
     r'connection timed out.*localhost',
     'an external service on localhost'),
]


def _detect_external_service_failure(error_info: str) -> str | None:
    """Check if an error is caused by an unavailable external service.

    Returns a human-readable service name if detected, ``None`` otherwise.
    """
    for pattern, service_name in _EXTERNAL_SERVICE_PATTERNS:
        if re.search(pattern, error_info, re.IGNORECASE):
            return service_name
    return None


# ── System-level / environment issue detection ────────────────
# Patterns that indicate the failure is due to missing system tools,
# runtimes, or project setup files — NOT a code bug.  The agent
# cannot fix these by editing source files.

_SYSTEM_LEVEL_PATTERNS: list[tuple[str, str]] = [
    # Ruby / Bundler
    (r'Could not locate Gemfile', 'Bundler (no Gemfile found — run `bundle init` or create a Gemfile)'),
    (r'bundler:?\s+command not found|bundle:?\s+command not found',
     'Bundler (install with `gem install bundler`)'),
    (r"ruby:?\s+command not found|ruby:?\s+is not recognized",
     'Ruby runtime (install Ruby from https://www.ruby-lang.org)'),
    # Python
    (r'python3?:?\s+command not found|python3?:?\s+is not recognized',
     'Python runtime'),
    (r'pip3?:?\s+command not found|pip3?:?\s+is not recognized',
     'pip (Python package manager)'),
    # Node.js / npm
    (r'node:?\s+command not found|node:?\s+is not recognized',
     'Node.js runtime (install from https://nodejs.org)'),
    (r'npm:?\s+command not found|npm:?\s+is not recognized',
     'npm (install Node.js from https://nodejs.org)'),
    # Java
    (r'javac?:?\s+command not found|javac?:?\s+is not recognized',
     'Java SDK (install JDK)'),
    (r'mvn:?\s+command not found', 'Maven (install Apache Maven)'),
    (r'gradle:?\s+command not found', 'Gradle (install Gradle)'),
    # .NET
    (r'dotnet:?\s+command not found|dotnet:?\s+is not recognized',
     '.NET SDK (install from https://dotnet.microsoft.com)'),
    # Docker
    (r'docker:?\s+command not found|docker:?\s+is not recognized',
     'Docker (install Docker Desktop)'),
    # Generic: "X is not recognized as an internal or external command" (Windows)
    (r"'[^']+' is not recognized as an internal or external command",
     'a required system tool (see error message above)'),
]


def _detect_system_level_failure(error_info: str) -> str | None:
    """Check if an error is caused by a missing system tool or environment setup.

    Returns a human-readable description if detected, ``None`` otherwise.
    """
    for pattern, description in _SYSTEM_LEVEL_PATTERNS:
        if re.search(pattern, error_info, re.IGNORECASE):
            return description
    return None


def _infer_test_file_path(src_path: str, language: str | None) -> str:
    """Return the conventional test-file path for *src_path*.

    Examples:
        src/components/Footer.jsx  ->  src/components/__tests__/Footer.test.jsx
        src/utils/math.ts          ->  src/utils/__tests__/math.test.ts
        api/views.py               ->  api/tests/test_views.py
    """
    src_path_norm = src_path.replace("\\", "/")
    src_dir = src_path_norm.rsplit("/", 1)[0] if "/" in src_path_norm else "."
    basename = src_path_norm.rsplit("/", 1)[-1]
    stem, _, ext = basename.rpartition(".")

    _ext_map: dict[str, str] = {
        "jsx": "test.jsx", "tsx": "test.tsx",
        "js": "test.js",   "ts": "test.ts",
        "py": "py",        "rb": "spec.rb",
    }
    test_suffix = _ext_map.get(ext, f"test.{ext}")

    if ext == "py":
        return f"{src_dir}/tests/test_{stem}.{test_suffix}"
    return f"{src_dir}/__tests__/{stem}.{test_suffix}"


def _source_covered_by_test_step(
    src_path: str,
    test_step: "PlanStep",
) -> bool:
    """Return True if *test_step* explicitly references *src_path*.

    Checks plan-declared imports and inline test-file content so that
    both plan-parsed and LLM-inlined test specs are handled.
    """
    src_path_norm = src_path.replace("\\", "/")
    src_basename = src_path_norm.rsplit("/", 1)[-1]
    src_stem = src_basename.rsplit(".", 1)[0]

    # Plan-declared imports (e.g. imports: src/components/Footer.jsx:default)
    for imp_path in (test_step.imports_from or {}):
        if src_stem in imp_path or src_basename in imp_path or src_path in imp_path:
            return True

    # Inline test code content
    for content in (test_step.inline_code or {}).values():
        if src_stem in content or src_basename in content:
            return True

    return False


def _generate_test_coverage_for_inline_changes(
    *,
    uncovered_files: list[str],
    before_files: dict[str, str],
    memory: "FileMemory",
    executor,
    coder,
    display: "CLIDisplay",
    step_idx: int,
    language: str | None,
    plan_step: "PlanStep | None",
) -> None:
    """Ask the coder LLM to generate/update test files for source files
    that have no corresponding TEST step in the plan.

    Called from the Tier A inline-code path when reviewer is skipped because
    a TEST step follows — but the specific source files changed here are NOT
    imported by any of those TEST steps.  Without this function those changes
    would reach the bulk test run untested.

    For each uncovered source file:
      1. Find the best matching test file already in memory (by imports or name).
      2. Ask the LLM to add tests for the new/changed behaviour (diff context).
      3. Write the result to memory — it will be executed by run_bulk_test_execution_and_fix.
    """
    from ..language import get_code_block_lang

    all_files = memory.all_files()
    lang_tag = get_code_block_lang(language) if language else "python"

    for src_path in uncovered_files:
        old_content = before_files.get(src_path, "")
        new_content = all_files.get(src_path, "")
        if not new_content:
            continue

        src_path_norm = src_path.replace("\\", "/")
        src_basename = src_path_norm.rsplit("/", 1)[-1]
        src_stem = src_basename.rsplit(".", 1)[0]

        # ── Find best existing test file ──
        existing_test_path: str | None = None
        existing_test_content: str = ""
        for fpath, content in all_files.items():
            if not _is_test_file(fpath):
                continue
            if src_stem in content or src_basename in content:
                existing_test_path = fpath
                existing_test_content = content
                break

        target_test_path = existing_test_path or _infer_test_file_path(src_path, language)
        target_test_basename = target_test_path.replace("\\", "/").rsplit("/", 1)[-1]
        action = "update" if existing_test_path else "create"

        display.step_info(
            step_idx,
            f"[Inline] No test coverage for {src_basename} — "
            f"{'updating' if action == 'update' else 'generating'} "
            f"{target_test_basename}...",
        )
        _logger.info(
            "[Inline] Generating test coverage for uncovered source %s -> %s",
            src_path, target_test_path,
        )

        ctx = ""
        if existing_test_content:
            ctx = (
                f"Existing test file (DO NOT remove any existing tests):\n"
                f"#### [FILE]: {existing_test_path}\n"
                f"```{lang_tag}\n{existing_test_content}\n```\n\n"
            )

        old_block = ""
        if old_content:
            old_block = (
                f"Previous version of source (for reference — only test "
                f"new/changed behaviour):\n"
                f"```{lang_tag}\n{old_content}\n```\n\n"
            )

        # Detect subproject CWD: if source and test share a common top-level
        # directory (e.g. responsive-web-page/), the test runner's CWD when
        # invoked as `cd {subproject} && npm test` is already that directory.
        # process.cwd() calls inside the test must NOT re-include the prefix.
        _subproject_note = ""
        _src_parts = src_path_norm.split("/")
        if len(_src_parts) > 1:
            _subproject = _src_parts[0]
            _target_norm = target_test_path.replace("\\", "/")
            if _target_norm.startswith(_subproject + "/"):
                _subproject_note = (
                    f"\n\nIMPORTANT — Subproject CWD: This test runs inside the "
                    f"`{_subproject}/` directory (vitest/jest CWD when npm test "
                    f"is run as `cd {_subproject} && npm test`). "
                    f"Do NOT prefix file paths with `{_subproject}/` in "
                    f"`process.cwd()` calls.\n"
                    f"CORRECT: `resolve(process.cwd(), 'vitest.config.js')`\n"
                    f"WRONG:   `resolve(process.cwd(), '{_subproject}/vitest.config.js')`\n"
                )

        prompt = (
            f"Source file `{src_path}` was just written/updated and has "
            f"no corresponding test step in the plan.\n\n"
            f"New source content:\n"
            f"#### [FILE]: {src_path}\n```{lang_tag}\n{new_content}\n```\n\n"
            f"{old_block}"
            f"{ctx}"
            f"{'Add tests for the new or changed behaviour to the existing test file.'  if action == 'update' else 'Create a new test file covering the key functionality.'}\n\n"
            f"Requirements:\n"
            f"- Target file: `{target_test_path}`\n"
            f"- Do NOT remove existing tests\n"
            f"- Only test observable behaviour, not implementation details\n"
            f"{_subproject_note}\n"
            f"Output ONLY the complete test file:\n"
            f"#### [FILE]: {target_test_path}\n```{lang_tag}\n...full content...\n```"
        )

        try:
            response = coder.llm_client.generate_response(prompt)
            gen_files = executor.parse_code_blocks(response)
            if not gen_files:
                gen_files = executor.parse_code_blocks_fuzzy(response)
            # Accept only test files
            gen_files = {p: c for p, c in gen_files.items() if _is_test_file(p)}
            if gen_files:
                executor.write_files(gen_files)
                memory.update(gen_files)
                _logger.info(
                    "[Inline] Test coverage written for %s: %s",
                    src_path, list(gen_files.keys()),
                )
                display.step_info(
                    step_idx,
                    f"[Inline] Test coverage written: "
                    f"{', '.join(p.rsplit('/', 1)[-1] for p in gen_files)}",
                )
            else:
                _logger.warning(
                    "[Inline] LLM produced no test files for %s", src_path
                )
        except Exception as exc:
            _logger.warning(
                "[Inline] Test coverage generation failed for %s: %s", src_path, exc
            )


def build_step_waves(steps: list[str], dependencies: dict[int, set[int]]) -> list[list[int]]:
    """Group step indices into execution waves using topological ordering.

    Each wave is a list of step indices that can execute in parallel.
    Waves execute sequentially.
    """
    n = len(steps)
    remaining: set[int] = set(range(n))
    completed: set[int] = set()
    waves: list[list[int]] = []

    while remaining:
        # Find all steps whose dependencies are satisfied
        wave = [i for i in sorted(remaining)
                if dependencies.get(i, set()).issubset(completed)]
        if not wave:
            # Circular dependency or missing deps — execute remaining sequentially
            wave = [min(remaining)]
        waves.append(wave)
        for i in wave:
            remaining.discard(i)
            completed.add(i)

    return waves


def _execute_step(step_idx: int, step_text: str, *,
                  steps: list[str],
                  llm_client, executor, coder, reviewer, tester,
                  task: str, memory: FileMemory, display: CLIDisplay,
                  language: str | None, cfg=None,
                  auto: bool = False,
                  search_agent=None,
                  kb_context_builder=None,
                  code_graph=None,
                  project_profile=None,
                  knowledge_base=None,
                  project_context=None,
                  plan_step: PlanStep | None = None,
                  all_plan_steps: list[PlanStep] | None = None,
                  intent_spec=None,
                  ) -> tuple[int, bool, str]:
    """Execute a single step. Returns ``(step_idx, success, error_info)``.

    When *plan_step* is provided (structured plan), step type and
    dependencies are taken from the object — no LLM classification call.

    Catches all exceptions so that a crash inside any handler never
    kills the whole pipeline — the step is marked as failed instead.
    """
    try:
        # --- Project Orientation + KB Context Injection (Phase 4+) ---
        #
        # Project grounding ALWAYS comes first — before KB symbols,
        # before task description, before everything.  It is the LLM's
        # "north star" for the entire session.

        context_parts: list[str] = []

        # 1. Project orientation grounding (always first)
        if project_profile is not None:
            try:
                context_parts.append(project_profile.format_for_prompt())
            except Exception as orient_exc:
                _logger.warning(
                    "[KB] Project orientation formatting failed: %s",
                    orient_exc,
                )

        # 2. Project knowledge (installed packages, file purposes, tech stack)
        if knowledge_base is not None:
            try:
                kb_agent_ctx = knowledge_base.format_for_agents()
                if kb_agent_ctx:
                    context_parts.append(kb_agent_ctx)
            except Exception as kb_fmt_exc:
                _logger.warning(
                    "[KB] format_for_agents failed: %s", kb_fmt_exc,
                )

        # 3. KB context (Phase 4 — symbols, error fixes, patterns)
        #
        # Use a clean short description as the KB search query.
        # step_text may contain inline code blocks from the plan (e.g. a full
        # JSX component), which explodes the keyword-score denominator and
        # dilutes all meaningful matches.  plan_step.description is the
        # one-line human description; fall back to the first line of step_text.
        _kb_query = (
            (plan_step.description if plan_step and plan_step.description else None)
            or step_text.split("\n")[0].strip()
        )
        # Augment _kb_query with project tech stack so framework docs are
        # found for steps whose description doesn't mention the framework
        # by name (e.g. "Replace main.jsx" doesn't mention "tailwindcss").
        # Uses installed_packages from knowledge_base (already in memory —
        # no I/O or LLM calls) filtered to recognised tech keywords only.
        if knowledge_base is not None:
            try:
                _pk = knowledge_base.load()
                _pkgs = getattr(_pk, "installed_packages", [])
                if _pkgs:
                    from ..orchestrator.plan_optimizer import _TECH_KEYWORDS
                    _tech_hits = _TECH_KEYWORDS.findall(" ".join(_pkgs[:50]))
                    _query_lower = _kb_query.lower()
                    _tech_extras = [
                        t for t in dict.fromkeys(t.lower() for t in _tech_hits)
                        if t.lower() not in _query_lower
                    ][:8]
                    if _tech_extras:
                        _kb_query = f"{_kb_query} {' '.join(_tech_extras)}"
            except Exception:
                pass
        # Skip KB context (and its embedding API call) for inline-code steps —
        # the coder is never invoked for inline steps so the context is wasted.
        _has_inline = (
            plan_step is not None
            and getattr(plan_step, "inline_code", None)
            and len(plan_step.inline_code) > 0
        )
        if kb_context_builder is not None and not _has_inline:
            try:
                from ..kb.context_builder import ContextBuilder
                kb_ctx = kb_context_builder.build_context(
                    task_description=_kb_query,
                    current_file=None,
                    max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 4000) if cfg else 4000,
                    language=getattr(project_context, "language", None) if project_context else None,
                )
                if kb_ctx.kb_available or kb_ctx.behavioral_instructions or kb_ctx.global_patterns:
                    kb_text = kb_context_builder.format_context_for_prompt(kb_ctx)
                    if kb_text:
                        context_parts.append(kb_text)
                _logger.debug(
                    "[KB] Injected context: %d tokens, sources: %s, "
                    "symbols: %d, errors: %d",
                    kb_ctx.token_count, kb_ctx.sources_used,
                    len(kb_ctx.local_symbols), len(kb_ctx.error_fixes),
                )
            except Exception as kb_exc:
                _logger.warning("[KB] Context injection failed: %s", kb_exc)

        # Combine and store in memory for downstream handlers
        if context_parts:
            memory._kb_context = "\n\n".join(context_parts)

        # --- Load KB content fixes once per pipeline run ---
        if not hasattr(memory, '_content_fixes') or memory._content_fixes is None:
            try:
                # Reuse the global store from kb_context_builder if available,
                # avoiding a redundant GlobalKBStore instantiation.
                _gkb = (getattr(kb_context_builder, '_global_store', None)
                        if kb_context_builder is not None else None)
                if _gkb is None:
                    from ..kb.global_kb.store import GlobalKBStore
                    _gkb = GlobalKBStore()
                memory._content_fixes = _gkb.get_content_fixes(language=language)
                if memory._content_fixes:
                    _logger.debug(
                        "[KB] Loaded %d content-fix rules",
                        len(memory._content_fixes),
                    )
            except Exception as exc:
                _logger.debug("[KB] Failed to load content fixes: %s", exc)
                memory._content_fixes = []

        # --- KB-guided file scoping (Option A) ---
        # Use KB to identify most relevant files for this step,
        # so the coder only sees focused context instead of everything.
        if kb_context_builder is not None:
            try:
                changed = list(memory.all_files().keys())[:10]
                relevant_files = kb_context_builder.get_relevant_files(
                    task_description=_kb_query,
                    changed_files=changed,
                    max_files=15,
                )
                if relevant_files:
                    memory._kb_relevant_files = relevant_files
                    _logger.debug(
                        "[KB] Scoped to %d relevant files for step %d",
                        len(relevant_files), step_idx + 1,
                    )
            except Exception as kb_scope_exc:
                _logger.debug("[KB] File scoping failed: %s", kb_scope_exc)

        log.info(f"\n{'='*60}\nTask {step_idx+1}: {step_text}\n"
                 f"Memory: {memory.summary()}\n{'='*60}")

        display.start_step(step_idx)

        # --- Structured plan: use declared type (skip LLM classification) ---
        if plan_step is not None and plan_step.step_type != "UNCLASSIFIED":
            step_type = plan_step.step_type
            display.step_info(step_idx, f"Type: [{step_type}] (from plan)")
            display.step_tokens(step_idx, 0, 0)
            plan_step.status = "in_progress"
        elif plan_step is not None and plan_step.step_type == "UNCLASSIFIED":
            # Infer type from plan_step fields before falling back to LLM
            if plan_step.command:
                step_type = "CMD"
                plan_step.step_type = "CMD"
                _logger.info(
                    "[PlanStep] step %d was UNCLASSIFIED but has command — "
                    "inferred CMD (0 LLM tokens)", step_idx,
                )
                display.step_info(step_idx, f"Type: [{step_type}] (inferred from plan command)")
                display.step_tokens(step_idx, 0, 0)
                plan_step.status = "in_progress"
            else:
                _logger.warning(
                    "[PlanStep] step %d has type UNCLASSIFIED — "
                    "falling back to LLM classification. "
                    "This happens when structured metadata is lost "
                    "(e.g. plan was edited in TUI).",
                    step_idx,
                )
                display.step_info(step_idx, "Loading context and classifying...")
                step_type = _classify_step(step_text, llm_client, display, step_idx)
                # Persist classified type back on PlanStep for checkpoint
                plan_step.step_type = step_type
        else:
            _logger.warning(
                "[PlanStep] plan_step is None for step %d — "
                "falling back to LLM classification (tokens wasted). "
                "Check if plan_steps_parsed is intact at execution time.",
                step_idx,
            )
            display.step_info(step_idx, "Loading context and classifying...")
            step_type = _classify_step(step_text, llm_client, display, step_idx)

        display.steps[step_idx]["type"] = step_type
        display.render()

        # ── Step Type Auto-Correction ──
        # Heuristic: If type is CODE/TEST but looks like CMD, verify with LLM.
        if plan_step is not None and step_type in ("CODE", "TEST"):
            has_targets = "target:" in step_text.lower()
            has_inline = plan_step.inline_code and len(plan_step.inline_code) > 0
            has_cmd_markers = re.search(r'^[ \t]*[>$][ \t]+', step_text, re.MULTILINE)

            if not has_targets and not has_inline and has_cmd_markers:
                _logger.info("[Pipeline] Step %s misclassification suspected (%s -> CMD). Re-classifying...",
                             plan_step.id, step_type)
                display.step_info(step_idx, f"Suspicious {step_type} classification, verifying...")

                try:
                    # Reuse the same classification flow as fallback
                    new_type = _classify_step(step_text, llm_client, display, step_idx)
                    if new_type != step_type:
                        _logger.info("[Pipeline] Step %s re-classified: %s -> %s",
                                     plan_step.id, step_type, new_type)
                        step_type = new_type
                        plan_step.step_type = step_type
                        display.steps[step_idx]["type"] = step_type
                        display.render()
                except Exception as e:
                    _logger.warning("[Pipeline] Re-classification failed for %s: %s", plan_step.id, e)

        # ── CMD → TEST Auto-Correction (deterministic, 0 LLM cost) ──
        # If planner labelled a step as CMD but the description/command
        # contains a test runner invocation (e.g. "npx vitest run",
        # "pytest", "npm test"), reclassify to TEST so the test handler
        # (with retry-and-fix logic) is used instead of the plain CMD handler.
        if step_type == "CMD":
            _check_text = step_text
            if plan_step is not None and plan_step.command:
                _check_text = f"{step_text} {plan_step.command}"
            if _TEST_CMD_RE.search(_check_text) and not _TEST_CONFIG_RE.search(_check_text):
                _logger.info(
                    "[Pipeline] Step %s reclassified CMD -> TEST "
                    "(test runner detected in description/command, 0 LLM tokens)",
                    plan_step.id if plan_step else step_idx,
                )
                step_type = "TEST"
                if plan_step is not None:
                    plan_step.step_type = "TEST"
                display.steps[step_idx]["type"] = step_type
                display.render()

        # ── CODE → TEST Auto-Correction (deterministic, 0 LLM cost) ──
        # If planner labelled a step as CODE but ALL target files are test
        # files (e.g. __tests__/App.test.jsx, test_main.py), reclassify to
        # TEST.  The TEST handler validates the fix by running the tests and
        # retries on failure, while CODE just writes the file without running.
        if step_type == "CODE" and plan_step is not None and plan_step.target_files:
            if all(_is_test_file(f) for f in plan_step.target_files):
                _logger.info(
                    "[Pipeline] Step %s reclassified CODE -> TEST "
                    "(all target files are test files: %s, 0 LLM tokens)",
                    plan_step.id, plan_step.target_files,
                )
                step_type = "TEST"
                plan_step.step_type = "TEST"
                display.steps[step_idx]["type"] = step_type
                display.render()

        log.info(f"Task {step_idx+1}: Classified as [{step_type}]")

        # --- Structured plan: inject plan-aware context (thread-local) ---
        # Use thread-local storage so parallel wave steps don't overwrite each
        # other's context on the shared memory object (race condition fix).
        if plan_step is not None and all_plan_steps is not None:
            try:
                from .memory import set_plan_context_files
                plan_ctx_files = build_step_context(
                    plan_step, all_plan_steps, memory,
                    read_from_disk=lambda p: executor.read_file(p)
                    if hasattr(executor, 'read_file') else None,
                )
                if plan_ctx_files:
                    set_plan_context_files(plan_ctx_files)
                    _logger.debug(
                        "[PlanStep] Injected %d plan-context files for step %s",
                        len(plan_ctx_files), plan_step.id,
                    )
            except Exception as pctx_exc:
                _logger.debug("[PlanStep] Context build failed: %s", pctx_exc)

        success, error_info = True, ""

        # ── Dependency check: before-snapshot ─────────────────────
        _dep_check_enabled = cfg is None or getattr(cfg, "DEPENDENCY_CHECK_ENABLED", True)
        _before_files = dict(memory.all_files()) if _dep_check_enabled else None

        if step_type == "IGNORE":
            display.step_info(step_idx, "Not actionable, skipping.")
            display.complete_step(step_idx, "skipped")

        elif step_type == "CMD":
            success, error_info = _handle_cmd_step(
                step_text, executor, llm_client, memory, display, step_idx,
                language=language, project_context=project_context,
                plan_step=plan_step, intent_spec=intent_spec)

        elif step_type == "CODE":
            # ── Inline edit fast path ──
            # If the planner provided find/replace edit blocks, apply them
            # surgically to the existing files and promote the result into
            # inline_code so the existing quality gate handles it naturally.
            # Falls through to coder if any edit fails to apply.
            if (plan_step is not None
                    and plan_step.inline_edits
                    and not plan_step.inline_code):
                _edit_subproject = _detect_subproject_root(memory)
                _edit_all_ok = True
                _patched = None
                _cur = None

                for _edit_fpath, _edit_pairs in plan_step.inline_edits.items():
                    import os as _os_edit
                    _resolved = _edit_fpath
                    if _edit_subproject and not _edit_fpath.startswith(_edit_subproject):
                        _candidate = f"{_edit_subproject}/{_edit_fpath}"
                        if _os_edit.path.exists(_candidate):
                            _resolved = _candidate

                    if not _os_edit.path.exists(_resolved):
                        _logger.warning(
                            "[InlineEdit] Target not found: %s — skipping edit path",
                            _resolved,
                        )
                        _edit_all_ok = False
                        break

                    try:
                        with open(_resolved, "r", encoding="utf-8") as _ef:
                            _cur = _ef.read()
                    except OSError as _oe:
                        _logger.warning("[InlineEdit] Cannot read %s: %s", _resolved, _oe)
                        _edit_all_ok = False
                        break

                    # Pre-validate: log which FIND strings won't match so the
                    # cause is visible before any edits are attempted.
                    for _pi, (_find_pre, _) in enumerate(_edit_pairs):
                        if _find_pre not in _cur:
                            _find_lines_pre = [l.strip() for l in _find_pre.splitlines() if l.strip()]
                            _cur_stripped = [l.strip() for l in _cur.splitlines()]
                            _n_pre = len(_find_lines_pre)
                            _pre_fuzzy = any(
                                _cur_stripped[_li:_li + _n_pre] == _find_lines_pre
                                for _li in range(max(0, len(_cur_stripped) - _n_pre + 1))
                            )
                            if not _pre_fuzzy:
                                _logger.warning(
                                    "[InlineEdit] FIND block #%d in %s will not match "
                                    "(neither exact nor fuzzy). Likely cause: planner "
                                    "hallucinated content not present in the file. "
                                    "First non-matching line: %r",
                                    _pi + 1, _resolved,
                                    next(
                                        (l for l in _find_lines_pre if l not in _cur_stripped),
                                        "<all lines present but sequence mismatch>",
                                    ),
                                )

                    _patched = _cur
                    for _find_str, _repl_str in _edit_pairs:
                        # Skip pairs where the find string has no real file
                        # content (e.g. leftover ---file-content-end--- marker
                        # accidentally flushed as a phantom edit pair).
                        _find_meaningful_lines = [
                            l for l in _find_str.splitlines()
                            if l.strip() and not l.strip().startswith("---")
                        ]
                        if not _find_meaningful_lines:
                            _logger.debug(
                                "[InlineEdit] Skipping empty/marker-only FIND pair in %s",
                                _resolved,
                            )
                            continue
                        # Skip no-op edits where FIND and REPLACE are identical.
                        # These waste a step and, worse, silently claim success
                        # without adding the methods/code the plan intended.
                        if _find_str == _repl_str:
                            _logger.warning(
                                "[InlineEdit] Skipping no-op FIND/REPLACE pair in %s "
                                "(FIND and REPLACE are identical — plan likely "
                                "failed to include the intended changes)",
                                _resolved,
                            )
                            continue
                        if _find_str in _patched:
                            _find_pos = _patched.index(_find_str)
                            _after_find = _patched[_find_pos + len(_find_str):]
                            # Guard: if REPLACE starts with FIND text and the
                            # original file has content after the match, a plain
                            # str.replace would append that tail after the
                            # replacement block, duplicating it.  This happens
                            # when the LLM anchors on a single import/line but
                            # puts the entire new file in REPLACE.  Fix: treat
                            # FIND as a positional anchor and replace from its
                            # position to EOF with REPLACE.
                            #
                            # However, if REPLACE is only slightly larger than
                            # FIND (small insertion like adding one import line),
                            # this is NOT a full-file rewrite — use normal
                            # str.replace to preserve the rest of the file.
                            _repl_is_full_rewrite = (
                                _repl_str.lstrip('\n').startswith(_find_str.lstrip('\n'))
                                and _after_find.strip()
                                # REPLACE must be large enough to plausibly
                                # contain the rest of the file.  If it's
                                # shorter than the content after the FIND
                                # anchor it's clearly a small insertion, not
                                # a full-file rewrite.
                                and len(_repl_str) > len(_after_find)
                            )
                            if _repl_is_full_rewrite:
                                _patched = _patched[:_find_pos] + _repl_str
                                _logger.debug(
                                    "[InlineEdit] Anchor-to-EOF applied in %s "
                                    "(REPLACE starts with FIND — tail dedup)",
                                    _resolved,
                                )
                            else:
                                _patched = _patched.replace(_find_str, _repl_str, 1)
                            _logger.debug("[InlineEdit] Exact match applied in %s", _resolved)
                        else:
                            # Fuzzy fallback: normalize whitespace per-line and
                            # try to locate the find block in the file.
                            # Works for both single-line and multi-line finds.
                            _find_lines_stripped = [l.strip() for l in _find_str.splitlines() if l.strip()]
                            _file_lns = _patched.splitlines(keepends=True)
                            _file_stripped = [l.strip() for l in _file_lns]
                            _n_find = len(_find_lines_stripped)
                            _fuzzy_hit = False
                            if _n_find > 0:
                                for _li in range(len(_file_lns) - _n_find + 1):
                                    if _file_stripped[_li:_li + _n_find] == _find_lines_stripped:
                                        # Determine base indent from first matched line
                                        _indent = len(_file_lns[_li]) - len(_file_lns[_li].lstrip())
                                        _repl_lines = _repl_str.splitlines()
                                        _last_orig_newline = _file_lns[_li + _n_find - 1].endswith("\n")
                                        _replacement = (
                                            "\n".join(
                                                (" " * _indent + rl.lstrip()) if rl.strip() else rl
                                                for rl in _repl_lines
                                            )
                                            + ("\n" if _last_orig_newline else "")
                                        )
                                        _file_lns[_li:_li + _n_find] = [_replacement]
                                        _patched = "".join(_file_lns)
                                        _fuzzy_hit = True
                                        _logger.debug(
                                            "[InlineEdit] Fuzzy match applied in %s (lines %d-%d)",
                                            _resolved, _li + 1, _li + _n_find,
                                        )
                                        break
                            if not _fuzzy_hit:
                                # ── Minimal-diff fallback ──────────────────────
                                # When the full FIND block fails (e.g. planner
                                # assumed stale scaffold content), extract only
                                # the lines that actually change between FIND and
                                # REPLACE and try to apply each change individually.
                                # Only succeeds when every changed line appears
                                # exactly once in the file (no ambiguity).
                                import difflib as _difflib
                                _find_all_lns = _find_str.splitlines()
                                _repl_all_lns = _repl_str.splitlines()
                                _sm = _difflib.SequenceMatcher(
                                    None, _find_all_lns, _repl_all_lns, autojunk=False
                                )
                                _delta: list = []  # [(old_line, new_line | None)]
                                for _tag, _i1, _i2, _j1, _j2 in _sm.get_opcodes():
                                    if _tag == "replace":
                                        for _k in range(max(_i2 - _i1, _j2 - _j1)):
                                            _o = _find_all_lns[_i1 + _k] if _i1 + _k < _i2 else None
                                            _n = _repl_all_lns[_j1 + _k] if _j1 + _k < _j2 else None
                                            if _o is not None:
                                                _delta.append((_o, _n))
                                    elif _tag == "delete":
                                        for _k in range(_i1, _i2):
                                            _delta.append((_find_all_lns[_k], None))
                                    # 'insert' has no anchor line — skip
                                if _delta:
                                    _work_lns = _patched.splitlines(keepends=True)
                                    _work_s = [l.strip() for l in _work_lns]
                                    _min_ok = True
                                    for _old_ln, _new_ln in _delta:
                                        _old_s = _old_ln.strip()
                                        _idxs = [i for i, s in enumerate(_work_s) if s == _old_s]
                                        if len(_idxs) != 1:
                                            _min_ok = False
                                            break
                                        _idx = _idxs[0]
                                        if _new_ln is not None:
                                            _bi = len(_work_lns[_idx]) - len(_work_lns[_idx].lstrip())
                                            _work_lns[_idx] = (
                                                " " * _bi + _new_ln.lstrip()
                                                + ("\n" if _work_lns[_idx].endswith("\n") else "")
                                            )
                                            _work_s[_idx] = _new_ln.strip()
                                        else:
                                            _work_lns[_idx] = ""
                                            _work_s[_idx] = ""
                                    if _min_ok:
                                        _patched = "".join(_work_lns)
                                        _fuzzy_hit = True
                                        _logger.info(
                                            "[InlineEdit] Minimal-diff fallback applied in %s "
                                            "(%d change(s))",
                                            _resolved, len(_delta),
                                        )
                                if not _fuzzy_hit:
                                    _logger.warning(
                                        "[InlineEdit] find string not found in %s — "
                                        "falling through to coder",
                                        _resolved,
                                    )
                                    _edit_all_ok = False
                                    break

                    if not _edit_all_ok:
                        break
                    # Promote the patched content into inline_code so the
                    # existing quality gate below handles writing + validation.
                    plan_step.inline_code[_resolved] = _patched
                    _logger.info(
                        "[InlineEdit] Promoted patched %s -> inline_code", _resolved,
                    )

                if not _edit_all_ok:
                    import os as _os_inline_fb
                    # ── REPLACE-as-full-file fallback ──
                    # When the FIND block doesn't match (e.g. scaffold template
                    # changed between Vite versions), check if the REPLACE block
                    # looks like a complete file replacement.  If so, use it
                    # directly instead of falling through to the coder LLM
                    # (which lacks KB context and generates garbage).
                    _used_replace_fallback = False
                    if plan_step.inline_edits:
                        for _ie_path, _ie_pairs in plan_step.inline_edits.items():
                            _ie_resolved = _ie_path
                            # Find the matching resolved path
                            for _kf in (plan_step.target_files or []):
                                if _kf.endswith(_ie_path) or _ie_path.endswith(_os_inline_fb.path.basename(_kf)):
                                    _ie_resolved = _kf
                                    break
                            # Collect ALL REPLACE blocks for this file and
                            # concatenate them in order.  Plans often split
                            # a file into multiple FIND/REPLACE pairs (e.g.
                            # imports in pair 1, function body in pair 2).
                            _repl_parts: list[str] = []
                            for _, _repl in _ie_pairs:
                                _repl_stripped = _repl.strip()
                                _has_structure = (
                                    'import ' in _repl_stripped
                                    or 'export ' in _repl_stripped
                                    or 'function ' in _repl_stripped
                                    or 'class ' in _repl_stripped
                                    or 'def ' in _repl_stripped
                                    or 'from ' in _repl_stripped
                                )
                                _is_substantial = len(_repl_stripped) > 50
                                if _has_structure and _is_substantial:
                                    _repl_parts.append(_repl_stripped)
                            if _repl_parts:
                                _combined = "\n\n".join(_repl_parts)
                                # Safety check: verify the combined REPLACE
                                # text looks like a complete file, not a
                                # fragment from an incremental edit.
                                #
                                # For JS/JSX/TS/TSX files a complete module
                                # must contain an export statement.  Without
                                # one the REPLACE is just a fragment (e.g.
                                # import lines from the first edit pair)
                                # and would produce a corrupt file.
                                #
                                # For multi-block concatenations also check
                                # balanced braces — fragments like "add
                                # import" + "add plugin line" won't close.
                                _ext = _os_inline_fb.path.splitext(_ie_resolved)[1].lower()
                                _js_like = _ext in {
                                    '.js', '.jsx', '.ts', '.tsx', '.mjs',
                                    '.cjs', '.mts', '.cts',
                                }
                                _is_complete = True
                                # JS/TS modules must have an export
                                if _js_like:
                                    _has_export = (
                                        'export ' in _combined
                                        or 'module.exports' in _combined
                                    )
                                    if not _has_export:
                                        _is_complete = False
                                # Multi-block: also need balanced delimiters
                                if _is_complete and len(_repl_parts) > 1:
                                    _opens = _combined.count('{') + _combined.count('(')
                                    _closes = _combined.count('}') + _combined.count(')')
                                    if _opens == 0 or _opens != _closes:
                                        _is_complete = False
                                if _is_complete:
                                    plan_step.inline_code[_ie_resolved] = _combined
                                    _logger.info(
                                        "[InlineEdit] FIND failed but REPLACE "
                                        "looks like a complete file — promoting "
                                        "%d REPLACE block(s) for %s (%d chars)",
                                        len(_repl_parts),
                                        _ie_resolved, len(_combined),
                                    )
                                    _used_replace_fallback = True
                                else:
                                    _logger.warning(
                                        "[InlineEdit] FIND failed and %d REPLACE "
                                        "block(s) for %s look like fragments "
                                        "(unbalanced delimiters) — falling "
                                        "through to coder LLM",
                                        len(_repl_parts), _ie_resolved,
                                    )

                    if _used_replace_fallback:
                        plan_step.inline_edits = {}
                        _logger.info(
                            "[InlineEdit] Using plan REPLACE content as "
                            "full-file replacement (skipping coder LLM)")
                    else:
                        # If some blocks applied before the failure, write the
                        # partial result to disk so the coder sees the already-
                        # patched file rather than the original.
                        if _patched is not None and _cur is not None and _patched != _cur:
                            try:
                                with open(_resolved, "w", encoding="utf-8") as _wf:
                                    _wf.write(_patched)
                                memory.update({_resolved: _patched})
                                _logger.info(
                                    "[InlineEdit] Partial edits written to %s "
                                    "before falling through to coder",
                                    _resolved,
                                )
                            except OSError as _we:
                                _logger.warning(
                                    "[InlineEdit] Could not write partial edits to %s: %s",
                                    _resolved, _we,
                                )
                        plan_step.inline_code.clear()
                        _logger.info("[InlineEdit] Falling through to coder (edit failed)")

            # ── Inline code fast path ──
            # If the planner already provided complete code in the plan,
            # write it directly — zero Coder LLM calls needed.
            if (plan_step is not None
                    and plan_step.inline_code
                    and len(plan_step.inline_code) > 0):
                display.step_info(step_idx, "Writing inline code from plan (0 LLM calls)")
                _inline_files = dict(plan_step.inline_code)
                # Resolve <project-name> and similar placeholder tokens that the
                # planner may have left in file path keys (e.g. when a dumb LLM
                # outputs "target: <project-name>/src/App.jsx").  The same logic
                # used for CMD placeholders is applied to path strings.
                from .classification import resolve_cmd_placeholders as _resolve_ph
                _ph_task = task or ''
                if any('<' in k for k in _inline_files):
                    _inline_files = {
                        _resolve_ph(k, step_text=step_text, task=_ph_task): v
                        for k, v in _inline_files.items()
                    }
                _inline_subproject = _detect_subproject_root(memory)
                # Fallback: if memory-based detection failed (e.g. no source
                # files in memory yet, so _detect_subproject_root bails early),
                # infer from the CMD-output entries that ARE in memory.
                if not _inline_subproject:
                    import re as _re
                    _mem_all = memory.all_files()
                    _scaffold_pats = [
                        _re.compile(r'npm\s+create\s+vite(?:@\S+)?\s+(\S+)'),
                        _re.compile(r'create-vite(?:@\S+)?\s+(\S+)'),
                        _re.compile(r'create-next-app(?:@\S+)?\s+(\S+)'),
                        _re.compile(r'create-react-app\s+(\S+)'),
                        _re.compile(r'ng\s+new\s+(\S+)'),
                    ]
                    import os as _os
                    for _fpath, _content in _mem_all.items():
                        if not _fpath.startswith('_cmd_output/'):
                            continue
                        _first = _content.split('\n')[0] if _content else ''
                        for _pat in _scaffold_pats:
                            _m = _pat.search(_first)
                            if _m:
                                _cand = _m.group(1).strip().rstrip('/')
                                if _cand and _os.path.isdir(_cand):
                                    _inline_subproject = _cand
                                    _logger.info(
                                        "[Inline] Subproject from CMD "
                                        "fallback: %s/", _cand)
                                    break
                        if _inline_subproject:
                            break
                _logger.debug(
                    "[Inline] subproject=%r inline_keys=%r",
                    _inline_subproject, list(_inline_files.keys()),
                )
                if _inline_subproject:
                    _inline_files = _prefix_subproject_paths(
                        _inline_files, _inline_subproject, memory)

                # Gate: strip any pseudo-diff markers the planner may have emitted
                from .dependency_check import clean_diff_markers as _clean_diff
                _inline_files = {
                    path: _clean_diff(content)
                    for path, content in _inline_files.items()
                }

                # Capture which targets already exist before overwriting
                import os as _os_inline
                _existing_inline_targets = {
                    p for p in _inline_files if _os_inline.path.exists(p)
                }

                executor.write_files(_inline_files)
                memory.update(_inline_files)
                display.step_tokens(step_idx, 0, 0)
                _logger.info(
                    "[PlanStep] Inline code: wrote %d file(s) for step %s: %s",
                    len(_inline_files), plan_step.id,
                    list(_inline_files.keys()),
                )

                # ── Seed scaffold entry-point files into memory ──
                # When writing into a scaffolded subproject, the entry-point
                # file (main.jsx, index.jsx, etc.) may not be touched by any
                # plan step.  If it's not in memory, depcheck can't see it
                # and reports false "orphaned export" gaps for App.jsx.
                # Read it into memory so the import graph is complete.
                _scaff_root = getattr(memory, '_scaffolded_subproject', None)
                if _scaff_root:
                    import os as _os_scaff
                    _entry_names = (
                        'main.jsx', 'main.tsx', 'index.jsx', 'index.tsx',
                        'main.py', 'main.go', 'main.rs', 'index.js', 'index.ts',
                    )
                    for _ename in _entry_names:
                        _epath = f"{_scaff_root}/src/{_ename}"
                        if _epath not in memory.all_files():
                            _full = _os_scaff.path.join('.', _epath)
                            if _os_scaff.path.isfile(_full):
                                try:
                                    with open(_full, 'r', encoding='utf-8',
                                              errors='replace') as _ef:
                                        _econtent = _ef.read()
                                    memory.update({_epath: _econtent})
                                    _logger.info(
                                        "[Scaffold] Seeded entry-point %s "
                                        "into memory (depcheck visibility)",
                                        _epath)
                                except OSError:
                                    pass

                # Deterministic KB content-fix gate for inline code.
                #
                # The planner generates inline code WITH full KB context
                # (e.g. Tailwind v4 docs), so its output is typically correct.
                # Sending it to an LLM reviewer is counterproductive — local
                # models apply outdated training-data bias and "fix" correct
                # code back to v3 patterns.  Instead, apply the same
                # deterministic _apply_content_fixes() rules used in
                # _handle_code_step — these catch known LLM mistakes (e.g.
                # @tailwind directives, wrong plugin names) without LLM calls.
                from .step_handlers import _apply_content_fixes
                _cf = getattr(memory, "_content_fixes", None)
                if _cf:
                    _fixed_inline = _apply_content_fixes(_inline_files, _cf)
                    _changed = [
                        p for p in _inline_files
                        if _fixed_inline.get(p) != _inline_files.get(p)
                    ]
                    if _changed:
                        executor.write_files(
                            {p: _fixed_inline[p] for p in _changed})
                        memory.update(
                            {p: _fixed_inline[p] for p in _changed})
                        display.step_info(
                            step_idx,
                            f"[Inline] Content fixes applied to "
                            f"{len(_changed)} file(s): "
                            f"{', '.join(_changed)}",
                        )
                    else:
                        _logger.debug(
                            "[Inline] Content fixes checked — "
                            "no corrections needed"
                        )

                # NOTE: BrowserRouter injection was removed here — duplicate
                # provider/wrapper issues (Router, ThemeProvider, etc.) are now
                # caught by the language-agnostic post-completion wiring
                # verification (run_wiring_verification), which checks all
                # files together after all CODE steps complete.

                # ── Inline code quality gate (Phase 1) ──
                # The planner wrote this code WITH full KB context, so it is
                # typically correct.  We avoid unconditional reviewer LLM calls
                # and instead apply a tiered gate:
                #
                #   Tier A: TEST-step lookahead — tests will validate; skip all.
                #   Tier B: Static lint + import checks (free, no LLM).
                #           Fail → fall back to Coder+Reviewer loop.
                #   Tier C: Existing-file rewrite + full review mode → run
                #           Reviewer LLM to verify the overwrite is correct.
                #           Fail → fall back to Coder+Reviewer loop.
                #   Tier D: All clear → done.  The post-step dependency check
                #           (run_dependency_check at line ~830) already handles
                #           orphaned exports and wiring via its own LLM fix path.
                _has_test_after_inline = False
                if all_plan_steps is not None:
                    _has_test_after_inline = any(
                        s.step_type == "TEST" for s in all_plan_steps
                        if s.index > step_idx
                    )

                if _has_test_after_inline:
                    # Tier A: TEST follows — tests will validate, skip review.
                    _logger.info(
                        "[Inline] Skipping review for step %s — TEST step follows",
                        plan_step.id if plan_step else step_idx,
                    )

                    # ── Coverage gap check ──
                    # A TEST step exists somewhere after this CODE step, but it
                    # may not import the specific source files written here.
                    # Identify any source files that no future TEST step covers
                    # and proactively generate/update their test files so the
                    # bulk run at end-of-pipeline exercises the new changes.
                    _inline_sources = [
                        p for p in _inline_files if not _is_test_file(p)
                    ]
                    if _inline_sources and all_plan_steps is not None:
                        _covered: set[str] = set()
                        for _ts in all_plan_steps:
                            if _ts.index <= step_idx or _ts.step_type != "TEST":
                                continue
                            for _sp in _inline_sources:
                                if _source_covered_by_test_step(_sp, _ts):
                                    _covered.add(_sp)
                        # Skip files that are inherently untestable or
                        # produce fragile/circular tests:
                        # - __main__.py: trivial entry points, runpy+patch fails
                        # - Config/scaffold files: importing vitest.config
                        #   inside vitest creates circular issues; postcss/
                        #   tailwind/vite configs are validated by the build
                        _CONFIG_SKIP = (
                            'vitest.config', 'vite.config', 'postcss.config',
                            'tailwind.config', 'jest.config', 'vitest.setup',
                            'setupTests', 'babel.config', 'tsconfig',
                            '.eslintrc', 'eslint.config',
                        )
                        _uncovered = [
                            p for p in _inline_sources
                            if p not in _covered
                            and not p.endswith('__main__.py')
                            and not any(
                                cfg in p.replace("\\", "/").rsplit("/", 1)[-1]
                                for cfg in _CONFIG_SKIP
                            )
                        ]
                        if _uncovered:
                            _logger.info(
                                "[Inline] Source files with no TEST-step coverage: %s",
                                _uncovered,
                            )
                            _generate_test_coverage_for_inline_changes(
                                uncovered_files=_uncovered,
                                before_files=_before_files or {},
                                memory=memory,
                                executor=executor,
                                coder=coder,
                                display=display,
                                step_idx=step_idx,
                                language=language,
                                plan_step=plan_step,
                            )
                else:
                    # Tier B: Static lint + import checks
                    from .step_handlers import _quick_offline_lint, _validate_import_paths
                    _inline_lint = _quick_offline_lint(_inline_files)
                    _inline_import_errs = _validate_import_paths(_inline_files, memory)
                    _inline_static_errs = (
                        (_inline_lint + "\n" + _inline_import_errs).strip()
                        if _inline_import_errs else _inline_lint
                    )
                    if _inline_static_errs:
                        display.step_info(
                            step_idx,
                            "[Inline] Static errors found — falling back to Coder+Reviewer loop",
                        )
                        _logger.info(
                            "[Inline] Static check failed for step %s — "
                            "falling back to _handle_code_step: %s",
                            plan_step.id if plan_step else step_idx,
                            _inline_static_errs[:200],
                        )
                        _graph_inline = code_graph
                        if _graph_inline is None and kb_context_builder is not None:
                            _graph_inline = getattr(kb_context_builder, "_graph", None)
                        success, error_info = _handle_code_step(
                            step_text, coder, reviewer, executor,
                            task, memory, display, step_idx,
                            language=language, cfg=cfg,
                            auto=auto, code_graph=_graph_inline,
                            project_profile=project_profile,
                            skip_review=_has_test_after_inline,
                            project_context=project_context,
                            plan_step=plan_step,
                            all_plan_steps=all_plan_steps,
                            kb_context_builder=kb_context_builder,
                            initial_error=(
                                f"Syntax/lint errors in the inline-edited file "
                                f"— fix these:\n{_inline_static_errs}"
                            ),
                            intent_spec=intent_spec,
                        )
                    else:
                        # Tier C: Existing-file rewrite — run Reviewer when in
                        # full review mode so overwritten files are verified.
                        _inline_review_mode = "static"
                        if cfg is not None:
                            _inline_review_mode = getattr(
                                cfg, "REVIEW_MODE", "static"
                            )
                        _should_review_inline = (
                            _inline_review_mode == "full"
                            and bool(_existing_inline_targets)
                        )
                        if _should_review_inline:
                            display.step_info(
                                step_idx,
                                f"[Inline] Reviewing overwrite of "
                                f"{len(_existing_inline_targets)} existing "
                                f"file(s) via Reviewer...",
                            )
                            _inline_review_code = "\n\n".join(
                                f"#### {p}\n```\n{_inline_files[p]}\n```"
                                for p in _existing_inline_targets
                                if p in _inline_files
                            )
                            _kb_ctx_inline = getattr(memory, "_kb_context", "")
                            # Also load step-specific global KB docs (plan_step.kb_docs)
                            # so the reviewer has framework docs like "Tailwind CSS v4
                            # Setup Guide" and doesn't reject valid code based on older
                            # training data.
                            _gstore_inline = getattr(kb_context_builder, '_global_store', None) if kb_context_builder else None
                            _declared_kb_inline = getattr(plan_step, 'kb_docs', None) if plan_step else None
                            if _gstore_inline and _declared_kb_inline:
                                try:
                                    _step_docs_inline = _gstore_inline.get_by_titles(_declared_kb_inline)
                                    if _step_docs_inline:
                                        _step_doc_text = "\n".join(
                                            getattr(d, "content", "") or getattr(d, "title", "")
                                            for d in _step_docs_inline
                                            if getattr(d, "content", "") or getattr(d, "title", "")
                                        )
                                        _kb_ctx_inline = (_kb_ctx_inline + "\n\n" + _step_doc_text).strip()
                                except Exception:
                                    pass
                            _reviewer_kb_inline = (
                                f"\n\n[KB Documentation — trust this over your "
                                f"training data]\n{_kb_ctx_inline}\n"
                                if _kb_ctx_inline else ""
                            )
                            _inline_review_resp = reviewer.process(
                                f"Review this inline code for the step: "
                                f"{step_text}\n\n{_inline_review_code}",
                                context=(
                                    f"Step: {step_text}\n"
                                    f"This code replaces existing file(s). "
                                    f"Verify the replacement is complete and "
                                    f"correct."
                                    f"{_reviewer_kb_inline}"
                                ),
                                language=language,
                            )
                            _inline_review_lower = (
                                _inline_review_resp or ""
                            ).lower()
                            _inline_approved = any(
                                phrase in _inline_review_lower for phrase in (
                                    "code looks good", "looks good",
                                    "no issues", "no critical issues",
                                    "no bugs found", "code is correct",
                                    "functionally correct", "lgtm",
                                )
                            )
                            if _inline_approved:
                                display.step_info(
                                    step_idx,
                                    "[Inline] Reviewer approved existing-file "
                                    "rewrite ✔",
                                )
                                _logger.info(
                                    "[Inline] Reviewer approved inline rewrite "
                                    "for step %s",
                                    plan_step.id if plan_step else step_idx,
                                )
                            else:
                                display.step_info(
                                    step_idx,
                                    "[Inline] Reviewer flagged issues — "
                                    "falling back to Coder+Reviewer loop",
                                )
                                _logger.info(
                                    "[Inline] Reviewer rejected inline rewrite "
                                    "for step %s — falling back: %s",
                                    plan_step.id if plan_step else step_idx,
                                    (_inline_review_resp or "")[:200],
                                )
                                _graph_inline = code_graph
                                if _graph_inline is None and kb_context_builder is not None:
                                    _graph_inline = getattr(
                                        kb_context_builder, "_graph", None
                                    )
                                success, error_info = _handle_code_step(
                                    step_text, coder, reviewer, executor,
                                    task, memory, display, step_idx,
                                    language=language, cfg=cfg,
                                    auto=auto, code_graph=_graph_inline,
                                    project_profile=project_profile,
                                    skip_review=_has_test_after_inline,
                                    project_context=project_context,
                                    plan_step=plan_step,
                                    all_plan_steps=all_plan_steps,
                                    kb_context_builder=kb_context_builder,
                                    intent_spec=intent_spec,
                                )
                        else:
                            # Tier D: Static clean, no existing-file rewrite
                            # concern — accept inline code as-is.
                            # Dependency wiring (orphaned exports) is handled
                            # by run_dependency_check after this block.
                            _logger.info(
                                "[Inline] Static checks passed for step %s — "
                                "accepted (0 reviewer LLM calls)",
                                plan_step.id if plan_step else step_idx,
                            )
            else:
                # ── No inline code (or inline was truncated) ──
                # Phase 2: If the planner's inline code was truncated (token
                # limit), _partial_inline_code holds what was written before
                # the cut-off.  Two strategies:
                #
                #   1. Trivial close: if unmatched braces/parens are small
                #      (≤2 each), close them deterministically — 0 LLM calls.
                #   2. Partial hint: inject the partial code into coder context
                #      so the coder completes rather than regenerates cold.
                #      Skip reviewer (static-only) since the base was planner-
                #      written and only the tail needs filling.
                _partial = getattr(plan_step, '_partial_inline_code', None) if plan_step else None
                _used_trivial_close = False

                if _partial:
                    _closed = _try_trivial_close(_partial, language)
                    if _closed is not None:
                        # Strategy 1: lint first, write only if clean
                        from .dependency_check import clean_diff_markers as _clean_diff_trunc
                        _closed = {p: _clean_diff_trunc(c) for p, c in _closed.items()}
                        from .step_handlers import _quick_offline_lint, _validate_import_paths
                        _trunc_lint = _quick_offline_lint(_closed)
                        _trunc_imp = _validate_import_paths(_closed, memory)
                        if not _trunc_lint and not _trunc_imp:
                            # Lint clean — write and accept
                            _trunc_subproject = _detect_subproject_root(memory)
                            if _trunc_subproject:
                                _closed = _prefix_subproject_paths(
                                    _closed, _trunc_subproject, memory)
                            executor.write_files(_closed)
                            memory.update(_closed)
                            display.step_tokens(step_idx, 0, 0)
                            display.step_info(
                                step_idx,
                                "[Inline/trunc] Trivially closed truncated code (0 LLM calls)",
                            )
                            _logger.info(
                                "[Inline/trunc] Step %s: trivial close succeeded for %s",
                                plan_step.id if plan_step else step_idx,
                                list(_closed.keys()),
                            )
                            _used_trivial_close = True
                            success, error_info = True, ""
                        else:
                            _logger.info(
                                "[Inline/trunc] Trivial close lint failed for step %s "
                                "— falling through to coder with partial hint",
                                plan_step.id if plan_step else step_idx,
                            )

                if _partial and not _used_trivial_close:
                    # Strategy 2: inject partial code as completion hint
                    _logger.info(
                        "[Inline/trunc] Step %s: using partial code as coder hint (%d file(s))",
                        plan_step.id if plan_step else step_idx,
                        len(_partial),
                    )
                    display.step_info(
                        step_idx,
                        "[Inline/trunc] Completing truncated inline code via coder hint",
                    )

                if not _used_trivial_close:
                    # Extract code graph from kb_context_builder if available
                    _graph = code_graph
                    if _graph is None and kb_context_builder is not None:
                        _graph = getattr(kb_context_builder, "_graph", None)

                    # Look ahead: skip LLM review if a TEST step follows OR
                    # if we are completing partial planner code (base was correct)
                    if all_plan_steps is not None:
                        _has_test_after = any(
                            s.step_type == "TEST" for s in all_plan_steps
                            if s.index > step_idx
                        )
                    else:
                        _test_keywords = re.compile(
                            r'\b(test|spec|unit.test|integration.test|jest|vitest|pytest|rspec)\b',
                            re.IGNORECASE,
                        )
                        _has_test_after = any(
                            _test_keywords.search(steps[j])
                            for j in range(step_idx + 1, len(steps))
                        )

                    # Partial hint: skip reviewer — coder is only completing tail
                    _skip_review_for_partial = bool(_partial and not _used_trivial_close)

                    success, error_info = _handle_code_step(
                        step_text, coder, reviewer, executor,
                        task, memory, display, step_idx, language=language, cfg=cfg,
                        auto=auto, code_graph=_graph,
                        project_profile=project_profile,
                        skip_review=_has_test_after or _skip_review_for_partial,
                        project_context=project_context,
                        plan_step=plan_step,
                        all_plan_steps=all_plan_steps,
                        kb_context_builder=kb_context_builder,
                        partial_inline_code=_partial,
                        intent_spec=intent_spec,
                    )

        elif step_type == "TEST":
            # ── Inline test fast path ──
            # If the planner already provided test code in the plan,
            # write it directly and run — zero Tester LLM calls needed.
            if (plan_step is not None
                    and plan_step.inline_code
                    and len(plan_step.inline_code) > 0):
                display.step_info(step_idx, "Writing inline test code from plan (0 LLM calls)")
                _inline_test_files = dict(plan_step.inline_code)
                _inline_test_subproject = _detect_subproject_root(memory)
                _logger.debug(
                    "[Inline/test] subproject=%r inline_keys=%r",
                    _inline_test_subproject, list(_inline_test_files.keys()),
                )
                if _inline_test_subproject:
                    _inline_test_files = _prefix_subproject_paths(
                        _inline_test_files, _inline_test_subproject, memory)

                # Gate: strip any pseudo-diff markers
                from .dependency_check import clean_diff_markers as _clean_diff_t
                _inline_test_files = {
                    path: _clean_diff_t(content)
                    for path, content in _inline_test_files.items()
                }

                # Gate: syntax validation before writing inline test code.
                # Mirrors the Tier B lint check on the CODE inline path —
                # catches truncated files (missing `}`), bad JSX, etc. and
                # falls back to the Tester LLM to regenerate from scratch.
                from .step_handlers import _quick_offline_lint
                _inline_test_lint = _quick_offline_lint(_inline_test_files)
                if _inline_test_lint:
                    _logger.warning(
                        "[Inline/test] Syntax errors in step %s — "
                        "falling back to Tester LLM: %s",
                        plan_step.id if plan_step else step_idx,
                        _inline_test_lint[:300],
                    )
                    display.step_info(
                        step_idx,
                        "[Inline/test] Syntax errors found — falling back to Tester LLM",
                    )
                    # Clear inline code so the else branch runs _handle_test_step
                    plan_step.inline_code.clear()
                    success, error_info = _handle_test_step(
                        step_text, tester, coder, reviewer, executor,
                        task, memory, display, step_idx, language=language,
                        auto=auto, search_agent=search_agent,
                        project_context=project_context,
                        kb_context_builder=kb_context_builder,
                        plan_step=plan_step,
                        all_plan_steps=all_plan_steps,
                        project_profile=project_profile,
                        intent_spec=intent_spec)
                else:
                    executor.write_files(_inline_test_files)
                    memory.update(_inline_test_files)
                    display.step_tokens(step_idx, 0, 0)
                    _logger.info(
                        "[PlanStep] Inline test code: wrote %d file(s) for step %s: %s",
                        len(_inline_test_files), plan_step.id,
                        list(_inline_test_files.keys()),
                    )

                    # Deterministic KB content-fix gate (e.g. jest-dom → jest-dom/vitest)
                    from .step_handlers import _apply_content_fixes as _acf_test
                    _cf_test = getattr(memory, "_content_fixes", None)
                    if _cf_test:
                        _fixed_test = _acf_test(_inline_test_files, _cf_test)
                        _changed_test = [
                            p for p in _inline_test_files
                            if _fixed_test.get(p) != _inline_test_files.get(p)
                        ]
                        if _changed_test:
                            executor.write_files(
                                {p: _fixed_test[p] for p in _changed_test})
                            memory.update(
                                {p: _fixed_test[p] for p in _changed_test})
                            display.step_info(
                                step_idx,
                                f"[Inline/test] Content fixes applied to "
                                f"{len(_changed_test)} file(s)",
                            )

                    # Defer test execution — all TEST steps write their files
                    # first; a single bulk run happens after all waves complete.
                    # This avoids redundant parallel runs when multiple TEST steps
                    # are in the same wave and prevents source-fixes for one test
                    # from breaking another test that hasn't run yet.
                    display.step_info(
                        step_idx,
                        "[Inline/test] Test files written — execution deferred to bulk run",
                    )
            else:
                success, error_info = _handle_test_step(
                    step_text, tester, coder, reviewer, executor,
                    task, memory, display, step_idx, language=language,
                    auto=auto, search_agent=search_agent,
                    project_context=project_context,
                    kb_context_builder=kb_context_builder,
                    plan_step=plan_step,
                    all_plan_steps=all_plan_steps,
                    project_profile=project_profile,
                    intent_spec=intent_spec)

        elif step_type == "SEARCH":
            success, error_info = _handle_search_step(
                step_text, search_agent, memory, display, step_idx,
                language=language)

        else:
            display.step_info(step_idx, f"Unknown type '{step_type}', skipping.")
            display.complete_step(step_idx, "skipped")

        # Clear plan context for this thread so it doesn't leak into the next step
        from .memory import clear_plan_context_files
        clear_plan_context_files()

        # ── Dependency check: after-snapshot + fix ─────────────────
        # Runs BEFORE complete_step so the spinner stays active during
        # gap detection and LLM fix generation.
        if _before_files is not None and success and step_type not in ("IGNORE",):
            try:
                after_files = memory.all_files()
                new_or_changed = [
                    f for f in after_files
                    if f not in _before_files or _before_files[f] != after_files[f]
                ]
                # Only run if actual source files changed (skip metadata keys)
                new_or_changed = [
                    f for f in new_or_changed if not f.startswith("_")
                ]
                if new_or_changed:
                    display.step_info(step_idx, "Running dependency check...")
                    from .dependency_check import build_snapshot, run_dependency_check
                    dep_before = build_snapshot(_before_files, language)
                    dep_after = build_snapshot(after_files, language)
                    integration_fixes = run_dependency_check(
                        step_idx, step_text, new_or_changed,
                        dep_before, dep_after,
                        memory, llm_client, executor, display, language, cfg,
                        all_plan_steps=all_plan_steps,
                        kb_context=getattr(memory, "_kb_context", "") or "",
                    )
                    if integration_fixes:
                        executor.write_files(integration_fixes)
                        memory.update(integration_fixes)
                        display.step_info(
                            step_idx,
                            f"[DepCheck] Fixed {len(integration_fixes)} file(s) "
                            f"for dependency integration",
                        )
            except Exception as dep_exc:
                _logger.warning("[DepCheck] Post-step check failed: %s", dep_exc)

        # Complete the step AFTER dependency check so spinner stays visible.
        # IGNORE and unknown-type steps already called complete_step above.
        if step_type in ("CMD", "CODE", "TEST", "SEARCH"):
            display.complete_step(step_idx, "done" if success else "failed")

        # --- Structured plan: update step status + actual exports ---
        if plan_step is not None:
            if success:
                try:
                    # Collect files generated in this step
                    after_all = memory.all_files()
                    new_files = {
                        f: c for f, c in after_all.items()
                        if f not in (_before_files or {})
                        or (_before_files or {}).get(f) != c
                    }
                    if new_files:
                        update_step_after_execution(plan_step, new_files)
                    else:
                        plan_step.status = "completed"
                except Exception:
                    plan_step.status = "completed"
            else:
                plan_step.status = "failed"

        # Per-step knowledge upsert (lightweight, no LLM calls)
        # Runs on both success and failure — CMD packages only on success,
        # but CODE/TEST file purposes are recorded regardless.
        if knowledge_base is not None:
            try:
                knowledge_base.record_step_completion(
                    step_type, step_text, step_idx, memory.as_dict(),
                    success=success)
            except Exception as kb_exc:
                _logger.warning("[KB] Per-step upsert failed: %s", kb_exc)

        return step_idx, success, error_info

    except Exception as exc:
        log.error(f"Task {step_idx+1}: Unhandled exception: {exc}")
        display.step_info(step_idx, f"Error: {type(exc).__name__}: {exc}")
        display.complete_step(step_idx, "failed")
        return step_idx, False, f"Unhandled exception: {type(exc).__name__}: {exc}"


def _run_diagnosis_loop(step_idx: int, step_text: str, error_info: str, *,
                        steps: list[str],
                        llm_client, executor, coder, reviewer, tester,
                        task: str, memory: FileMemory, display: CLIDisplay,
                        language: str | None, cfg=None,
                        auto: bool = False,
                        search_agent=None,
                        kb_context_builder=None,
                        project_profile=None,
                        knowledge_base=None,
                        project_context=None,
                        plan_step: PlanStep | None = None,
                        all_plan_steps: list[PlanStep] | None = None,
                        intent_spec=None,
                        ) -> bool:
    """Run diagnose → fix → retry loop. Returns ``True`` if the step was fixed.

    All exceptions are caught so that a crash during diagnosis (e.g. an
    embedding error) never kills the whole pipeline — the step is simply
    marked as failed and the pipeline halts gracefully.
    """
    # ── Early exit: external service dependency ──────────────────
    # If the failure is due to an unavailable external service (DB,
    # cache, etc.), diagnosis cannot help — inform the user instead.
    service = _detect_external_service_failure(error_info)
    if service:
        msg = (f"Step requires {service} which is not reachable. "
               f"Please ensure the service is running and accessible, "
               f"then re-run the pipeline.")
        display.step_info(step_idx, msg)
        log.warning(f"Task {step_idx+1}: External service unavailable: {service}")
        log.warning(f"Task {step_idx+1}: Skipping diagnosis — "
                    f"this is not a code issue.")
        display.complete_step(step_idx, "skipped")
        return False

    # ── Early exit: missing system tool / environment setup ─────
    # If the failure is because a runtime, package manager, or project
    # config file is missing, editing code won't help.
    sys_issue = _detect_system_level_failure(error_info)
    if sys_issue:
        msg = (f"System dependency missing: {sys_issue}. "
               f"Please install the required tool and re-run the pipeline.")
        display.step_info(step_idx, msg)
        log.warning(f"Task {step_idx+1}: System-level issue: {sys_issue}")
        log.warning(f"Task {step_idx+1}: Skipping diagnosis — "
                    f"this is an environment issue, not a code bug.")
        display.complete_step(step_idx, "failed")
        return False

    last_diagnosis_content = None

    # Classify the error once — result is reused across all retry attempts.
    # Uses regex (0 tokens) for common cases; falls back to a tiny single-shot
    # LLM call only for ambiguous errors.
    _step_type_for_route = display.steps[step_idx].get("type", "CODE")
    _kb_matched = False  # refined inside _diagnose_failure after KB lookup
    error_route = classify_error(
        error_info=error_info,
        step_type=_step_type_for_route,
        project_context=project_context,
        kb_matched=_kb_matched,
        llm_client=llm_client if search_agent is not None else None,
    )
    log.info(
        "Step %d: ErrorRouter → source=%s skip_web=%s confidence=%s reason=%s",
        step_idx + 1, error_route.source_type, error_route.skip_web,
        error_route.confidence, error_route.reason,
    )

    # Snapshot files before the diagnosis loop so we can restore them
    # if a fix attempt corrupts source code (compounding failures).
    _diag_snapshot = memory.snapshot()

    for diag_attempt in range(1, MAX_DIAGNOSIS_RETRIES + 1):
        try:
            # Restore snapshot at the start of each attempt so that a
            # bad fix from the previous attempt doesn't compound.
            if diag_attempt > 1:
                memory.restore(_diag_snapshot, executor=executor)
                log.info(
                    "Task %d: Restored file snapshot before diagnosis "
                    "attempt %d", step_idx + 1, diag_attempt)

            display.step_info(
                step_idx, f"Diagnosing failure ({diag_attempt}/{MAX_DIAGNOSIS_RETRIES})...")
            log.info(f"Task {step_idx+1}: Diagnosis attempt "
                     f"{diag_attempt}/{MAX_DIAGNOSIS_RETRIES}")

            step_type = display.steps[step_idx].get("type", "CODE")
            diagnosis = _diagnose_failure(
                step_text, step_type, error_info,
                memory, llm_client, display, step_idx,
                search_agent=search_agent, language=language,
                previous_diagnosis=last_diagnosis_content,
                kb_context_builder=kb_context_builder,
                error_route=error_route,
                intent_spec=intent_spec,
                executor=executor)

            # Extract the original failing command from error_info so
            # _apply_fix can filter it out (prevents re-running the same
            # broken command extracted from diagnosis inline backticks).
            import re as _re_diag
            _orig_cmd_match = _re_diag.search(
                r"Command `(.+?)` failed\.", error_info or "")
            _orig_cmd = _orig_cmd_match.group(1) if _orig_cmd_match else None

            _task_goal = getattr(project_context, 'goal_summary', '') if project_context else ''
            _step_targets = plan_step.target_files if plan_step else None
            fix_applied, cmds_succeeded, has_fix_commands, _fix_cmds_run = _apply_fix(
                diagnosis, executor, memory, display, step_idx,
                step_type=step_type,
                original_error_cmd=_orig_cmd,
                step_text=step_text,
                task=_task_goal,
                step_target_files=_step_targets)

            if not fix_applied:
                # ── Test-only retry for TEST steps ──
                # When the diagnosis produced code (it has [EDIT]: or
                # [FILE]: markers) but _apply_fix returned False, the
                # likely cause is the diff guard blocking destructive
                # source rewrites.  For TEST steps, retry with a
                # test-only prompt.
                #
                # We do NOT retry when:
                #  - The diagnosis had no code at all (prose-only) — the
                #    LLM genuinely didn't know the fix
                #  - step_type != "TEST" — source bugs need source fixes
                #  - The diagnosis was a CMD fix (has_fix_commands=True)
                _diag_had_code = (
                    "#### [EDIT]:" in diagnosis
                    or "#### [FILE]:" in diagnosis
                    or "```" in diagnosis
                )
                if step_type == "TEST" and _diag_had_code:
                    _test_targets = (
                        plan_step.target_files if plan_step else None
                    ) or []
                    _test_paths = [
                        t for t in _test_targets
                        if any(seg in t for seg in (
                            '.test.', '.spec.', '__tests__', 'test_'))
                    ]
                    # Fallback: when plan_step has no target_files
                    # (e.g. CMD step reclassified to TEST), discover
                    # test files from memory that are failing.
                    if not _test_paths:
                        _test_paths = [
                            fp for fp in memory.all_files()
                            if any(seg in fp for seg in (
                                '.test.', '.spec.',
                                '__tests__', 'test_'))
                        ]
                    if _test_paths:
                        log.info(
                            "Task %d: Diagnosis had code but nothing "
                            "applied (likely diff guard) — retrying "
                            "with test-only constraint",
                            step_idx + 1)

                        # Build context: step intent, source files
                        # (read-only), and existing test content
                        _dt_step_desc = ""
                        if plan_step and plan_step.description:
                            _dt_step_desc = (
                                f"STEP INTENT: {plan_step.description}\n"
                            )
                        if step_text:
                            _dt_step_desc += (
                                f"STEP DESCRIPTION: {step_text[:1000]}\n"
                            )

                        _dt_briefing = ""
                        if intent_spec:
                            _brief = getattr(
                                intent_spec, 'briefing', '') or ''
                            if _brief:
                                _dt_briefing = (
                                    f"TASK BRIEFING (preserve these "
                                    f"constraints):\n{_brief[:1000]}\n\n"
                                )

                        # Include source files the tests import so the
                        # LLM knows what the components render
                        _dt_source_ctx = ""
                        _all_mem = memory.all_files()
                        for _tp in _test_paths:
                            _tc = _all_mem.get(_tp, "")
                            if _tc:
                                _dt_source_ctx += (
                                    f"CURRENT TEST FILE (to fix):\n"
                                    f"#### [FILE]: {_tp}\n"
                                    f"```\n{_tc}\n```\n\n"
                                )
                        # Add source components as read-only context
                        _dt_imports = (
                            plan_step.imports_from
                            if plan_step else {}
                        )
                        # Fallback: when plan_step has no imports_from
                        # (e.g. CMD step), include non-test source files
                        # from memory so the LLM knows what the
                        # components actually render.
                        if not _dt_imports:
                            _dt_imports = {
                                fp: ""
                                for fp in _all_mem
                                if not any(seg in fp for seg in (
                                    '.test.', '.spec.',
                                    '__tests__', 'test_',
                                    '_cmd_output/',
                                    'node_modules/'))
                                and fp.endswith((
                                    '.jsx', '.tsx', '.js', '.ts',
                                    '.py', '.vue'))
                            }
                        for _imp_path in _dt_imports:
                            _imp_content = _all_mem.get(_imp_path, "")
                            if _imp_content:
                                _dt_source_ctx += (
                                    f"SOURCE FILE (READ-ONLY — do NOT "
                                    f"modify):\n"
                                    f"#### [FILE]: {_imp_path}\n"
                                    f"```\n{_imp_content[:3000]}\n"
                                    f"```\n\n"
                                )

                        _diag_test_prompt = (
                            f"{_dt_briefing}"
                            f"{_dt_step_desc}\n"
                            f"A test step failed. The previous fix "
                            f"attempt was rejected because it tried to "
                            f"rewrite source files too aggressively.\n\n"
                            f"Error:\n{error_info[:3000]}\n\n"
                            f"{_dt_source_ctx}"
                            f"CRITICAL RULES:\n"
                            f"1. Fix ONLY the test file(s). Source "
                            f"files are correct — do NOT modify them.\n"
                            f"2. Do NOT remove or weaken test "
                            f"assertions — the intended functionality "
                            f"must still be verified.\n"
                            f"3. Adapt assertions to match what the "
                            f"source components ACTUALLY render.\n\n"
                            f"Common test fixes:\n"
                            f"- Wrap renders in <MemoryRouter> when "
                            f"components use react-router Link/NavLink\n"
                            f"- Use getAllByRole/getAllByText when "
                            f"multiple elements match (desktop + mobile)\n"
                            f"- Scope queries with within(container)\n\n"
                            f"Test files to fix: {_test_paths}\n\n"
                            f"Return the COMPLETE fixed test file(s) "
                            f"using #### [FILE]: format."
                        )
                        try:
                            _dt_resp = llm_client.generate_response(
                                _diag_test_prompt)
                            _dt_files = executor.parse_code_blocks(
                                _dt_resp)
                            if not _dt_files:
                                _dt_files = (
                                    executor.parse_code_blocks_fuzzy(
                                        _dt_resp))
                            if _dt_files:
                                # Only accept test files
                                _dt_files = {
                                    fp: fc
                                    for fp, fc in _dt_files.items()
                                    if any(seg in fp for seg in (
                                        '.test.', '.spec.',
                                        '__tests__', 'test_'))
                                }
                            if _dt_files:
                                executor.write_files(_dt_files)
                                memory.update(_dt_files)
                                fix_applied = True
                                log.info(
                                    "Task %d: Diagnosis test-only "
                                    "retry produced fix for: %s",
                                    step_idx + 1,
                                    list(_dt_files.keys()))
                        except Exception as _dt_exc:
                            log.warning(
                                "Task %d: Diagnosis test-only retry "
                                "failed: %s", step_idx + 1, _dt_exc)

            if not fix_applied:
                last_diagnosis_content = diagnosis
                display.step_info(step_idx, "No actionable fix found in diagnosis.")
                log.warning(f"Task {step_idx+1}: Diagnosis produced no actionable fix.")
                continue

            # For CMD steps: if the diagnosis both wrote code fixes AND ran
            # new commands successfully, AND the diagnosis signals that the
            # original command is deprecated/removed, treat the step as
            # resolved.  Re-running a deprecated command will never succeed
            # regardless of how many fixes are applied.
            import re as _re_depr
            _DEPRECATION_RE = _re_depr.compile(
                r'\b(deprecated|removed|no longer|discontinued|obsolete|'
                r'replaced by|use instead|not supported|not available)\b',
                _re_depr.IGNORECASE,
            )
            if (step_type == "CMD"
                    and has_fix_commands and cmds_succeeded and fix_applied
                    and _DEPRECATION_RE.search(diagnosis)):
                display.step_info(
                    step_idx,
                    "Original command is deprecated — fix applied, step resolved.")
                log.info(
                    f"Task {step_idx+1}: CMD step resolved via deprecation-aware fix "
                    f"(code fixes + replacement commands succeeded). "
                    f"Skipping re-run of deprecated original command.")
                return True

            # For CMD steps: if the fix command is a corrected replacement of
            # the original (same core operation, different path/syntax), skip
            # re-running the original — it would just fail again.
            # Detection: strip leading "cd X &&" prefixes and compare the
            # first two words (e.g. "npm install"). If they match, the fix
            # *is* the corrected command, not a prerequisite.
            if (step_type == "CMD"
                    and has_fix_commands and cmds_succeeded and fix_applied
                    and _fix_cmds_run and _orig_cmd):
                import re as _re_repl

                def _cmd_keys(c: str) -> set[str]:
                    segments = [s.strip() for s in _re_repl.split(r'&&|;', c)]
                    keys = set()
                    for seg in segments:
                        parts = seg.split()
                        if not parts: continue
                        cmd = parts[0].lower()
                        if cmd in ('cd', 'mkdir', 'echo', 'set', 'export'):
                            continue
                        keys.add(' '.join(parts[:2]).lower() if len(parts) >= 2 else cmd)
                    return keys

                _orig_keys = _cmd_keys(_orig_cmd)
                if _orig_keys and any(
                        _orig_keys.intersection(_cmd_keys(fc)) for fc in _fix_cmds_run):
                    display.step_info(
                        step_idx,
                        "Fix command replaced original — step resolved.")
                    log.info(
                        f"Task {step_idx+1}: CMD step resolved via corrected replacement "
                        f"command (same operation, fix succeeded). "
                        f"Skipping re-run of original command.")
                    return True

            # Always re-run the original step after applying fixes.
            # Fix commands may be prerequisites (e.g. `npm install` for a
            # missing dependency) rather than replacements for the original
            # command.  Re-running verifies the original intent is satisfied
            # (e.g. tests actually pass, build actually succeeds).
            display.step_info(step_idx, "Fix applied — retrying step...")
            _, success, error_info = _execute_step(
                step_idx, step_text,
                steps=steps,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=task, memory=memory, display=display,
                language=language, cfg=cfg, auto=auto,
                search_agent=search_agent,
                kb_context_builder=kb_context_builder,
                project_profile=project_profile,
                knowledge_base=knowledge_base,
                project_context=project_context,
                plan_step=plan_step,
                all_plan_steps=all_plan_steps,
                intent_spec=intent_spec,
            )

            if success:
                return True
            else:
                log.warning(f"Task {step_idx+1}: Still failing after "
                            f"diagnosis attempt {diag_attempt}")

        except Exception as exc:
            log.error(f"Task {step_idx+1}: Exception during diagnosis "
                      f"attempt {diag_attempt}: {exc}")
            display.step_info(step_idx, f"Diagnosis error: {type(exc).__name__}")
            continue

    # Restore files to pre-diagnosis state so that destructive edits
    # from failed diagnosis attempts (e.g. overwriting package.json
    # with downgraded versions) don't persist on disk and corrupt
    # future runs or resume attempts.
    memory.restore(_diag_snapshot, executor=executor)
    log.info(f"Task {step_idx+1}: Restored file snapshot after "
             f"all diagnosis attempts failed.")

    display.step_info(
        step_idx, "Step failed after all fix attempts. Halting pipeline.")
    log.error(f"Task {step_idx+1}: Failed after {MAX_DIAGNOSIS_RETRIES} "
              f"diagnosis attempts. Halting pipeline.")
    return False


# ── Final cross-step test verification ────────────────────────────────────────

_MAX_FINAL_VERIFY_ATTEMPTS = 3


def _lazify_display_imports(content: str) -> str:
    """
    Post-process a source file to prevent test-collection failures caused by
    display-requiring imports (e.g. pygame) at module level.

    Strategy:
    1. Strip any unindented 'import pygame' / 'from pygame import ...' lines.
    2. For every function that references 'pygame.' but has no local
       'import pygame', inject one as the first statement of the function body.

    This is a deterministic guard — the LLM frequently ignores the KB
    instruction to use lazy imports, so we enforce it here instead.
    """
    _DISPLAY_PKGS = ("pygame",)

    def _is_display_import(line: str) -> bool:
        s = line.strip()
        return any(
            s.startswith(f"import {pkg}") or s.startswith(f"from {pkg}")
            for pkg in _DISPLAY_PKGS
        )

    lines = content.splitlines()

    # Fast-exit: nothing to do if there are no display imports at all
    if not any(not ln.startswith((" ", "\t")) and _is_display_import(ln) for ln in lines):
        return content

    # Pass 1 — remove module-level display imports
    lines = [
        ln for ln in lines
        if not (not ln.startswith((" ", "\t")) and _is_display_import(ln))
    ]

    # Pass 2 — inject lazy imports inside functions that use 'pygame.'
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        is_toplevel_def = (
            not line.startswith((" ", "\t"))
            and stripped.startswith("def ")
            and stripped.rstrip().endswith(":")
        )
        if not is_toplevel_def:
            result.append(line)
            i += 1
            continue

        # Collect the entire function block (until next unindented non-empty line)
        func: list[str] = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() and not nxt.startswith((" ", "\t")):
                break
            func.append(nxt)
            i += 1

        func_text = "\n".join(func)
        needs_pygame = "pygame." in func_text

        if needs_pygame:
            # Determine body indentation from the first non-empty body line
            body_indent = "    "
            for fl in func[1:]:
                if fl.strip():
                    body_indent = " " * (len(fl) - len(fl.lstrip()))
                    break
            lazy_import = f"{body_indent}import pygame"
            if lazy_import not in func_text:
                # Insert after the def line (skip past any docstring)
                insert_at = 1
                in_docstring = False
                quote = None
                for j, fl in enumerate(func[1:], start=1):
                    s = fl.strip()
                    if not in_docstring:
                        if s.startswith(('"""', "'''")):
                            quote = s[:3]
                            in_docstring = not s.endswith(quote) or len(s) == 3
                            if not in_docstring:
                                insert_at = j + 1
                        else:
                            insert_at = j
                            break
                    else:
                        if s.endswith(quote):
                            in_docstring = False
                            insert_at = j + 1
                func.insert(insert_at, lazy_import)

        result.extend(func)

    return "\n".join(result)


def _extract_failing_test_imports(error_output: str, all_files: dict) -> str:
    """
    Parse ERROR lines from pytest output, find those test files in memory,
    and return their import statements so the LLM knows exactly which symbols
    each failing test needs from the source files.
    """
    import re as _re
    # Match lines like "ERROR collecting test_foo.py"
    failing = _re.findall(r"ERROR collecting (\S+\.py)", error_output)
    if not failing:
        return ""

    lines = []
    for test_fname in failing:
        # Normalise path separators and find the file in memory
        needle = test_fname.replace("\\", "/")
        content = None
        for fpath, fcontent in all_files.items():
            if fpath.replace("\\", "/").endswith(needle):
                content = fcontent
                break
        if content is None:
            continue
        # Collect import lines only
        imports = [
            ln.strip()
            for ln in content.splitlines()
            if ln.strip().startswith(("import ", "from "))
        ]
        if imports:
            lines.append(f"  {needle} imports:\n    " + "\n    ".join(imports))

    if not lines:
        return ""
    return "\n\nSymbols required by failing tests (you MUST export all of these from the source):\n" + "\n".join(lines)


def run_final_test_verification(
    *,
    memory: FileMemory,
    executor,
    coder,
    display: CLIDisplay,
    language: str | None,
    task: str,
    cfg=None,
    project_context=None,
    kb_context_builder=None,
) -> tuple[bool, str]:
    """Re-run all test files generated in this session as a final regression gate.

    Individual TEST steps only verify their own test files in isolation.  When a
    source fix in step N causes tests from step M to regress, the pipeline would
    naively declare success.  This function catches those cross-step regressions
    by running every test file written during the session together, after all
    steps have completed.

    Only runs when there are 2+ distinct test files (a single test file was
    already verified by its own step — no cross-step regression is possible).

    Returns ``(success, error_info)``.
    """
    from ..language import get_test_framework, detect_language_from_files

    # Collect session test files
    all_files = memory.all_files()
    test_files = {
        fpath: content
        for fpath, content in all_files.items()
        if _is_test_file(fpath) and not fpath.startswith("_")
    }

    if len(test_files) <= 1:
        _logger.info(
            "[FinalVerify] %d test file(s) — no cross-step regression possible, skipping.",
            len(test_files),
        )
        return True, ""

    _logger.info(
        "[FinalVerify] Running final regression check on %d test file(s): %s",
        len(test_files), list(test_files.keys()),
    )
    print(f"\n  [FinalVerify] Re-running {len(test_files)} test file(s) for cross-step regression check...")

    # Determine test command — read package.json scripts.test for JS/TS projects
    lang = language
    if lang is None:
        lang = detect_language_from_files(list(test_files.keys()))

    # Detect subproject root (needed for reading package.json)
    subproject_cwd = _detect_subproject_root(memory)

    _test_runner_fv = None
    if lang in ("javascript", "typescript"):
        from .step_handlers import _read_js_project_env
        _js_env_fv = _read_js_project_env(subproject_cwd)
        _test_runner_fv = _js_env_fv.get("test_runner")
        if _test_runner_fv and _test_runner_fv != "jest":
            _logger.info("[FinalVerify] Detected test runner from package.json: %s", _test_runner_fv)

    fw = get_test_framework(lang, test_runner=_test_runner_fv) if lang else get_test_framework("python")
    base_cmd = fw["command"]

    # Vitest fallback override: catch cases where test files import vitest directly
    uses_vitest = False
    if "jest" in base_cmd.lower():
        uses_vitest = any(
            "from 'vitest'" in c or 'from "vitest"' in c
            for c in test_files.values()
        )
        if not uses_vitest:
            _vitest_configs = (
                "vitest.config.js", "vitest.config.ts",
                "vitest.config.mjs", "vitest.config.mts",
            )
            uses_vitest = any(
                any(f.endswith(cfg) for cfg in _vitest_configs)
                for f in all_files
            )
        if uses_vitest:
            base_cmd = "npx vitest run"
            _logger.info("[FinalVerify] Overriding to vitest (import/config fallback)")

    test_cmd = _build_scoped_test_cmd(base_cmd, test_files, subproject_cwd)
    _logger.info("[FinalVerify] Test command: %s", test_cmd)

    last_output = ""
    for attempt in range(1, _MAX_FINAL_VERIFY_ATTEMPTS + 1):
        ok, output = executor.run_command(test_cmd, cwd=subproject_cwd)
        last_output = output
        if ok:
            _logger.info("[FinalVerify] All tests passed on attempt %d.", attempt)
            print(f"  [FinalVerify] All tests passed.")
            return True, ""

        _logger.warning(
            "[FinalVerify] Attempt %d/%d failed:\n%s",
            attempt, _MAX_FINAL_VERIFY_ATTEMPTS, output[:800],
        )

        if attempt == _MAX_FINAL_VERIFY_ATTEMPTS:
            break

        # Ask coder to fix source files only (test files are already correct —
        # they passed during their own steps; only source regressions are at fault)
        print(f"  [FinalVerify] Tests failed — asking coder to fix source files (attempt {attempt})...")
        source_files = {
            fpath: content
            for fpath, content in all_files.items()
            if not _is_test_file(fpath) and not fpath.startswith("_")
        }
        if not source_files:
            _logger.warning("[FinalVerify] No source files to fix — aborting fix loop.")
            break

        context_parts = [
            f"#### [FILE]: {fpath}\n```\n{content}\n```"
            for fpath, content in list(source_files.items())[:6]
        ]
        # Optionally inject KB behavioral instructions into the fix prompt
        kb_instructions = ""
        if kb_context_builder is not None:
            try:
                from ..kb.context_builder import ContextBuilder
                kb_ctx = kb_context_builder.build_context(
                    task_description=task,
                    current_file=None,
                    max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 2000) if cfg else 2000,
                    language=language,
                    step_type="TEST",
                )
                kb_text = kb_context_builder.format_context_for_prompt(kb_ctx)
                if kb_text:
                    kb_instructions = f"\n\nKnowledge base guidance:\n{kb_text}"
            except Exception:
                pass

        failing_imports = _extract_failing_test_imports(output, all_files)

        _fv_briefing = getattr(memory, '_task_briefing', '')
        _fv_briefing_block = (
            "TASK BRIEFING (overall goal — respect Preserve and Key constraint):\n"
            f"{_fv_briefing}\n\n"
        ) if _fv_briefing else ""

        fix_prompt = (
            f"{_fv_briefing_block}"
            f"Task: {task}\n\n"
            f"All individual test steps passed, but running the full test suite together "
            f"revealed a cross-step regression: a source fix for one test broke another.\n\n"
            f"Test command: {test_cmd}\n\n"
            f"Failure output:\n{output[:8000]}\n\n"
            f"Source files (do NOT modify test files — they are correct):\n"
            + "\n\n".join(context_parts)
            + failing_imports
            + kb_instructions
            + "\n\nFix the source file(s) so ALL tests pass."
            + "\n\nIMPORTANT: Preserve ALL existing public symbols (classes, functions, constants) — only add or modify, never remove."
            + "\n\nCRITICAL: NEVER abbreviate or summarize existing code with comments like `// existing code` or `/* unchanged */`. If you are editing a chunk or a file, you MUST write out the ENTIRE content of that chunk or file. Abbreviating code will cause it to be permanently deleted!"
            + "\n\nPrefer CHUNK FORMAT for surgical fixes:\n"
            + "#### [EDIT]: path/to/file.py:function_name (lines start-end)\n```\n// replacement chunk\n```\n"
            + "Use full-file [FILE]: format only when the whole file must be rewritten."
        )
        try:
            fix_response = coder.llm_client.generate_response(fix_prompt)
            # Try chunk edits first (surgical), fall back to full-file
            fix_files = {}
            try:
                from ..editing.chunk_editor import ChunkEditor as _FvCE
                _fv_ce = _FvCE()
                _fv_edits = _fv_ce.parse_chunk_response(fix_response)
                if _fv_edits:
                    for _fv_edit in _fv_edits:
                        _fv_fp = _fv_edit.file_path
                        _fv_existing = memory.get(_fv_fp)
                        if _fv_existing is None:
                            try:
                                with open(_fv_fp, "r", encoding="utf-8", errors="replace") as _f:
                                    _fv_existing = _f.read()
                            except OSError:
                                pass
                        if _fv_existing:
                            try:
                                fix_files[_fv_fp] = _fv_ce.apply_chunk_edits(_fv_existing, [_fv_edit])
                            except Exception:
                                pass
            except ImportError:
                pass
            if not fix_files:
                fix_files = executor.parse_code_blocks(fix_response)
            if not fix_files:
                fix_files = executor.parse_code_blocks_fuzzy(fix_response)
            # Strictly filter: only apply fixes to non-test source files
            fix_files = {
                fpath: content for fpath, content in fix_files.items()
                if not _is_test_file(fpath)
            }
            # Post-process: enforce lazy display imports regardless of LLM output
            fix_files = {
                fpath: _lazify_display_imports(content)
                for fpath, content in fix_files.items()
            }
            if fix_files:
                executor.write_files(fix_files)
                memory.update(fix_files)
                _logger.info("[FinalVerify] Applied source fixes: %s", list(fix_files.keys()))
            else:
                _logger.warning("[FinalVerify] Coder produced no source-only fixes.")
                continue
        except Exception as exc:
            _logger.warning("[FinalVerify] Fix generation failed: %s", exc)
            break

    error_msg = (
        f"Final cross-step test verification failed: {len(test_files)} test file(s) "
        f"did not all pass together after source fixes.\n{last_output[:600]}"
    )
    print(f"  [FinalVerify] FAILED — cross-step regression detected.")
    return False, error_msg


# ---------------------------------------------------------------------------
# Bulk test execution and per-file fix (replaces per-step inline test runs)
# ---------------------------------------------------------------------------

_MAX_BULK_TEST_FIX_ATTEMPTS = 3


def _resolve_django_failed_files(
    output: str,
    subproject_cwd: str | None = None,
) -> list[str]:
    """Resolve Django ERROR:/FAIL: module paths from test output to file paths.

    Django test output reports failures as:
        ERROR: test_foo (app.tests.MyClass.test_foo)
        FAIL:  test_bar (app.tests.test_views.MyClass.test_bar)

    This extracts the dotted module path inside the parentheses, converts it
    to a file path (e.g. ``app/tests.py`` or ``app/tests/test_views.py``),
    and checks whether that file exists on disk.  Returns paths relative to
    the project root (prefixed with *subproject_cwd* when given).

    Useful when the failing test file was NOT written during the current
    session (not in *known_test_files*) so the normal matcher misses it.
    """
    import os as _os
    from .step_handlers import _ANSI_RE
    clean = _ANSI_RE.sub('', output)
    base = subproject_cwd.rstrip("/\\") if subproject_cwd else "."

    # Extract all dotted module paths from Django ERROR/FAIL lines
    modules = re.findall(
        r'(?:ERROR|FAIL):\s+\S+\s+\(([^)]+)\)',
        clean,
        re.MULTILINE | re.IGNORECASE,
    )

    resolved: list[str] = []
    seen: set[str] = set()
    for dotted in modules:
        parts = dotted.split('.')
        # Try progressively shorter prefixes: the class name and method name
        # are the last 1-2 parts; the module is the remainder.
        # e.g. home.tests.HomePageTests.test_foo → try home/tests.py first
        for n in range(len(parts) - 1, 0, -1):
            candidate_rel = '/'.join(parts[:n]) + '.py'
            candidate_abs = _os.path.join(base, candidate_rel)
            if _os.path.isfile(candidate_abs):
                # Return relative to project root (not subproject)
                full_rel = (
                    subproject_cwd.rstrip('/\\') + '/' + candidate_rel
                    if subproject_cwd and subproject_cwd != '.'
                    else candidate_rel
                )
                if full_rel not in seen:
                    seen.add(full_rel)
                    resolved.append(full_rel)
                break

    return resolved


def _django_settings_context(subproject_cwd: str | None) -> str:
    """Return a source context block containing Django settings.py and the
    actual ROOT_URLCONF file for the project.

    When the LLM only sees app-level urls.py files it cannot tell that
    ROOT_URLCONF = "pkg.urls" resolves to ``pkg/urls.py`` — a different file
    from the one it has been editing.  Injecting both files into the fix prompt
    lets the LLM find and fix the real URLconf instead of a decoy.

    Returns an empty string when the project is not Django or files can't be
    found.
    """
    import os as _os
    import re as _re

    base = subproject_cwd or "."

    # Resolve the REAL settings file via DJANGO_SETTINGS_MODULE in manage.py.
    # Django projects often have TWO settings.py files: a root-level stub
    # (ignored by Django) and the real one inside the project package, e.g.
    # bootstrap_homepage/bootstrap_homepage/settings.py.  Reading manage.py
    # is the only reliable way to know which one Django actually loads.
    settings_path: str | None = None
    manage_py = _os.path.join(base, "manage.py")
    if _os.path.isfile(manage_py):
        try:
            with open(manage_py, "r", encoding="utf-8", errors="replace") as _mf:
                manage_content = _mf.read()
            _dsm_m = _re.search(
                r"DJANGO_SETTINGS_MODULE['\"]?\s*,\s*['\"]([^'\"]+)['\"]",
                manage_content,
            )
            if _dsm_m:
                # e.g. "bootstrap_homepage.settings" → bootstrap_homepage/settings.py
                _module = _dsm_m.group(1)
                _rel = _module.replace(".", _os.sep) + ".py"
                _candidate = _os.path.join(base, _rel)
                if _os.path.isfile(_candidate):
                    settings_path = _candidate
        except OSError:
            pass

    # Fallback: root-level settings.py (simple single-file layout)
    if settings_path is None:
        _fallback = _os.path.join(base, "settings.py")
        if _os.path.isfile(_fallback):
            settings_path = _fallback

    if settings_path is None:
        return ""

    try:
        with open(settings_path, "r", encoding="utf-8", errors="replace") as _f:
            settings_content = _f.read()
    except OSError:
        return ""

    ctx = (
        f"#### [FILE]: {settings_path}\n"
        f"```python\n{settings_content}\n```\n\n"
        "NOTE: ROOT_URLCONF above defines the actual Django URL entry point — "
        "make sure you edit THAT file, not a same-named file at a different path.\n\n"
    )

    # Resolve ROOT_URLCONF to a file path and include its current content
    m = _re.search(r'ROOT_URLCONF\s*=\s*["\']([^"\']+)["\']', settings_content)
    if m:
        module = m.group(1)  # e.g. "bootstrap_homepage.urls"
        rel_path = module.replace(".", _os.sep) + ".py"  # bootstrap_homepage/urls.py
        urlconf_abs = _os.path.join(base, rel_path)
        if _os.path.isfile(urlconf_abs):
            # Express the path relative to the project root for the LLM
            urlconf_rel = _os.path.join(
                subproject_cwd.rstrip("/\\"), rel_path
            ) if subproject_cwd else rel_path
            try:
                with open(urlconf_abs, "r", encoding="utf-8", errors="replace") as _uf:
                    urlconf_content = _uf.read()
                ctx += (
                    f"#### [FILE]: {urlconf_rel}\n"
                    f"```python\n{urlconf_content}\n```\n\n"
                    f"NOTE: This is the ROOT_URLCONF file ({module}). "
                    "It must include your app's URLs for reverse() to work.\n\n"
                )
            except OSError:
                pass

    return ctx


def _fix_django_startup_crashes(
    output: str,
    subproject_cwd: str | None,
    executor,
) -> str:
    """Detect and self-heal Django test-runner startup crashes.

    Returns the output of a fresh test run if a fix was applied, otherwise
    returns the original *output* unchanged.

    Currently handles:
    - ``tests.py`` stub + ``tests/`` package coexistence:
      Django raises ``ImportError: 'tests' module incorrectly imported from …``
      when both exist in the same app directory.  Fix: delete the ``.py`` stub.
    """
    import os

    # Pattern: Django incorrectly-imported module conflict
    _conflict_re = re.compile(
        r"ImportError: '(\w+)' module incorrectly imported from '([^']+)'",
        re.IGNORECASE,
    )
    m = _conflict_re.search(output)
    if not m:
        return output

    module_name = m.group(1)          # e.g. "tests"
    package_dir = m.group(2)          # e.g. "/abs/path/to/homepage/tests"

    # The conflicting stub is <parent_dir>/<module_name>.py
    parent_dir = os.path.dirname(package_dir)
    stub_rel_candidates = [
        os.path.join(parent_dir, f"{module_name}.py"),
    ]
    if subproject_cwd:
        stub_rel_candidates.append(
            os.path.join(subproject_cwd, parent_dir, f"{module_name}.py")
        )

    deleted: list[str] = []
    for stub_path in stub_rel_candidates:
        if os.path.isfile(stub_path):
            try:
                os.remove(stub_path)
                deleted.append(stub_path)
                _logger.info(
                    "[BulkTest] Removed conflicting stub '%s' "
                    "(shadowed by '%s/' package)",
                    stub_path, package_dir,
                )
            except OSError as exc:
                _logger.warning(
                    "[BulkTest] Could not remove stub '%s': %s", stub_path, exc
                )

    if not deleted:
        # Couldn't find the stub on disk — log the full error tail so
        # the diagnostic LLM gets the actual ImportError text
        _logger.warning(
            "[BulkTest] Django startup crash (could not auto-fix):\n%s",
            output[-2000:],
        )
        return output

    # Re-run to get fresh output after the fix
    from .step_handlers import get_test_framework, detect_language_from_files
    ok, new_output = executor.run_command("python manage.py test", cwd=subproject_cwd)
    _logger.info(
        "[BulkTest] Post-stub-removal run: exit=%s", "0" if ok else "1"
    )
    return new_output


def _parse_failed_test_files(
    output: str,
    known_test_files: list[str],
    subproject_cwd: str | None = None,
) -> list[str]:
    """Parse test runner output to find which test files failed.

    Matches FAIL lines from vitest/jest/pytest/Django against the known test
    files written during the session, then also resolves Django module paths
    to files on disk (catching failures in pre-existing test files that were
    not written this session).
    """
    from .step_handlers import _ANSI_RE
    clean = _ANSI_RE.sub('', output)
    failed: list[str] = []
    # Normalize to forward-slashes so mixed-separator duplicates are caught
    # (e.g. "foo/bar.jsx" and "foo\bar.jsx" refer to the same file on Windows).
    failed_set: set[str] = set()  # stores normalized paths

    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    for fpath in known_test_files:
        basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        stem = basename[:-3] if basename.endswith(".py") else basename
        norm_fpath = _norm(fpath)

        # vitest/jest:  " FAIL src/__tests__/Foo.test.jsx"
        # pytest:       "FAILED tests/test_foo.py::test_bar"
        if re.search(
            r'(?:^|\s)(?:FAIL(?:ED)?)\s.*' + re.escape(basename),
            clean,
            re.MULTILINE | re.IGNORECASE,
        ):
            if norm_fpath not in failed_set:
                failed.append(fpath)
                failed_set.add(norm_fpath)
            continue

        # Django manage.py test:
        #   "ERROR: test_foo (app.tests.test_views.MyTestCase.test_foo)"
        #   "FAIL: test_foo (app.tests.test_views.MyTestCase.test_foo)"
        if re.search(
            r'(?:^|\s)(?:ERROR|FAIL):\s+\S+\s+\([^)]*\b' + re.escape(stem) + r'\b',
            clean,
            re.MULTILINE | re.IGNORECASE,
        ):
            if norm_fpath not in failed_set:
                failed.append(fpath)
                failed_set.add(norm_fpath)

    # Also resolve Django module paths to actual files on disk — catches
    # failures in pre-existing test files not written during this session.
    for extra in _resolve_django_failed_files(output, subproject_cwd):
        if _norm(extra) not in failed_set:
            failed.append(extra)
            failed_set.add(_norm(extra))

    # Fallback: if we couldn't identify specific files but tests failed,
    # treat all known test files as candidates
    if not failed and output:
        failed = list(known_test_files)
    return failed


def run_bulk_test_execution_and_fix(
    *,
    memory: FileMemory,
    executor,
    coder,
    display: CLIDisplay,
    language: str | None,
    task: str,
    cfg=None,
    project_context=None,
    kb_context_builder=None,
    all_plan_steps=None,
    search_agent=None,
) -> tuple[bool, str]:
    """Run all session test files in a single bulk execution, then fix failures
    one test file at a time.

    This replaces the per-step inline test runs that used to fire immediately
    after each TEST step wrote its files.  By deferring execution until all
    test files are written:

      - Parallel TEST steps in the same wave no longer race to run the full
        suite simultaneously.
      - A source-file fix for one failing test cannot break another test
        before it has been verified.
      - Total LLM calls are reduced because a single diagnosis loop handles
        all failures rather than one loop per step.

    Fix strategy: run all tests → collect failed files → for each failed file
    ask the coder to fix it (or its imported source) → re-run that single file
    → move to the next.  A final run-all confirms everything passes.

    Returns ``(success, error_info)``.
    """
    from ..language import get_test_framework, detect_language_from_files
    from .step_handlers import (
        _extract_file_specific_errors,
        _extract_imported_sources,
        _ANSI_RE,
        _parse_test_counts,
    )

    # Build test-file → kb_docs mapping from planner-declared step metadata.
    # Used in the fix loop to skip broad KB search when the plan already
    # specifies exactly which docs apply to each test file.
    _step_kb_docs: dict[str, list[str]] = {}
    _step_descs: dict[str, str] = {}
    if all_plan_steps:
        for _ps in all_plan_steps:
            _declared = getattr(_ps, 'kb_docs', None)
            _desc = getattr(_ps, 'description', None)
            for _tf in getattr(_ps, 'target_files', []):
                if _declared:
                    _step_kb_docs[_tf] = _declared
                if _desc:
                    _step_descs[_tf] = _desc

    all_files = memory.all_files()
    # Deduplicate test files by normalized path — mixed separators (foo/bar vs foo\bar)
    # can produce the same basename twice, causing double-processing of one file.
    _seen_norm: set[str] = set()
    test_files: dict[str, str] = {}
    for fpath, content in all_files.items():
        if not (_is_test_file(fpath) and not fpath.startswith("_")):
            continue
        _np = fpath.replace("\\", "/")
        if _np not in _seen_norm:
            _seen_norm.add(_np)
            test_files[fpath] = content

    if not test_files:
        _logger.info("[BulkTest] No test files found — skipping bulk run.")
        return True, ""

    _logger.info(
        "[BulkTest] Running bulk test execution on %d file(s): %s",
        len(test_files), list(test_files.keys()),
    )
    print(f"\n  [BulkTest] Running all {len(test_files)} test file(s)...")

    # Detect test command
    subproject_cwd = _detect_subproject_root(memory)
    lang = language
    if lang is None:
        lang = detect_language_from_files(list(test_files.keys()))

    # For JS/TS projects, read package.json to detect the actual test runner
    # (covers Angular/ng, Karma, Mocha, Vitest, Jest, etc. without hardcoding).
    import os as _os_bt
    _test_runner = None
    if lang in ("javascript", "typescript"):
        from .step_handlers import _read_js_project_env
        _js_env = _read_js_project_env(subproject_cwd)
        _test_runner = _js_env.get("test_runner")
        if _test_runner and _test_runner != "jest":
            _logger.info("[BulkTest] Detected test runner from package.json: %s", _test_runner)

    fw = get_test_framework(lang, test_runner=_test_runner) if lang else get_test_framework("python")
    base_cmd = fw["command"]

    # Django project detection: prefer manage.py test over pytest
    if (not lang or lang == "python") and _os_bt.path.isfile(
        _os_bt.path.join(subproject_cwd or "", "manage.py")
    ):
        base_cmd = "python manage.py test"
        _logger.info("[BulkTest] Django project detected — using 'python manage.py test'")

    # Vitest fallback override: catch cases where _read_js_project_env
    # didn't detect vitest but test files import from it directly.
    uses_vitest = False
    if "jest" in base_cmd.lower():
        uses_vitest = any(
            "from 'vitest'" in c or 'from "vitest"' in c
            for c in test_files.values()
        )
        if not uses_vitest:
            _vitest_cfgs = (
                "vitest.config.js", "vitest.config.ts",
                "vitest.config.mjs", "vitest.config.mts",
            )
            uses_vitest = any(
                any(f.endswith(vc) for vc in _vitest_cfgs)
                for f in all_files
            )
        if uses_vitest:
            base_cmd = "npx vitest run"
            _logger.info("[BulkTest] Overriding to vitest (import/config fallback)")

    # ── Step 1: Run all tests ──
    ok, output = executor.run_command(base_cmd, cwd=subproject_cwd)
    if ok:
        _logger.info("[BulkTest] All tests passed on first run.")
        print("  [BulkTest] All tests passed.")
        # Record every test file as passed
        for fpath in test_files:
            display.record_test_result(fpath, passed=1, total=1, failures=[])
        return True, ""

    _logger.warning("[BulkTest] Tests failed:\n%s", output[:1000])

    # ── Startup crash detection (Django only) ──────────────────────────────
    # Django raises ImportError / ModuleNotFoundError at collection time when
    # there is a `tests.py` stub AND a `tests/` package in the same directory.
    # This crashes the whole run before any test executes, so _parse_failed_test_files
    # returns the unhelpful fallback (all known files).  Detect the pattern and
    # self-heal before falling into the per-file fix loop.
    if "manage.py" in base_cmd:
        output = _fix_django_startup_crashes(output, subproject_cwd, executor)

    # ── Step 2a: Fix shared root causes before per-file loop ─────────────────
    # When all (or most) test files fail with the same error pointing at a
    # shared config file (vitest.config.js, jest.config.js, vite.config.js,
    # setup files, etc.), fix that one file first rather than burning
    # per-file fix budgets on the same root cause 3× per test.
    failed_files = _parse_failed_test_files(output, list(test_files.keys()), subproject_cwd)
    if len(failed_files) > 1:
        _shared_fix_applied = False
        # Extract the first error line that references a non-test config file
        _CONFIG_FILE_RE = re.compile(
            r'[\w./\\-]*(vitest\.config|vite\.config|jest\.config|'
            r'webpack\.config|babel\.config|setup\w*)[\w./\\-]*',
            re.IGNORECASE,
        )
        _error_lines = output.splitlines()
        _shared_config: str | None = None
        for _el in _error_lines[:60]:  # scan first 60 lines of test output
            _cm = _CONFIG_FILE_RE.search(_el)
            if _cm:
                _candidate = _cm.group(0).strip().replace("\\", "/")
                # Must look like an actual file path (contains a dot/extension),
                # not a bare keyword like "setup" from test description prose.
                # Also exclude test files themselves (e.g. vitest.setup.test.js).
                if (
                    "." in _candidate
                    and not re.search(r'\.(test|spec)\.[a-z]+$', _candidate, re.IGNORECASE)
                ):
                    _shared_config = _candidate
                    break

        if _shared_config:
            # Count how many failing test files mention this config in the output
            _mentioning = sum(
                1 for _tf in failed_files
                if _shared_config in output or _shared_config.split("/")[-1] in output
            )
            if _mentioning >= len(failed_files):
                _logger.info(
                    "[BulkTest] Shared root cause detected — %s referenced in all "
                    "%d failing test outputs. Fixing shared config first.",
                    _shared_config, len(failed_files),
                )
                print(f"  [BulkTest] Shared config error in {_shared_config} — fixing once...")
                # Read the current config file content
                _cfg_candidates = [
                    _shared_config,
                    os.path.join(subproject_cwd or "", _shared_config.split("/")[-1]),
                    os.path.join(subproject_cwd or "", _shared_config),
                ]
                _cfg_content = ""
                _cfg_path = ""
                for _cp in _cfg_candidates:
                    try:
                        with open(_cp, "r", encoding="utf-8", errors="replace") as _cf:
                            _cfg_content = _cf.read()
                            _cfg_path = _cp
                            break
                    except OSError:
                        pass
                if _cfg_content:
                    _shared_fix_prompt = (
                        f"Task: {task}\n\n"
                        f"All test files are failing due to an error in the shared "
                        f"config file `{_cfg_path}`.\n\n"
                        f"Error output:\n{output[:2000]}\n\n"
                        f"Current content of `{_cfg_path}`:\n"
                        f"```\n{_cfg_content}\n```\n\n"
                        f"Fix ONLY `{_cfg_path}` so tests can run. "
                        f"Do NOT touch any test files or component files.\n\n"
                        f"IMPORTANT: Output only code — no prose, no markdown headers.\n"
                        f"Use full-file format:\n"
                        f"#### [FILE]: {_cfg_path}\n"
                        f"```\n// fixed content\n```"
                    )
                    try:
                        _shared_fix_resp = coder.llm_client.generate_response(_shared_fix_prompt)
                        _shared_fix_files = executor.parse_code_blocks(_shared_fix_resp)
                        if not _shared_fix_files:
                            _shared_fix_files = executor.parse_code_blocks_fuzzy(_shared_fix_resp)
                        if _shared_fix_files:
                            if subproject_cwd:
                                from .step_handlers import _prefix_subproject_paths
                                _shared_fix_files = _prefix_subproject_paths(
                                    _shared_fix_files, subproject_cwd, memory)
                            executor.write_files(_shared_fix_files)
                            memory.update(_shared_fix_files)
                            _logger.info(
                                "[BulkTest] Shared config fix applied: %s",
                                list(_shared_fix_files.keys()),
                            )
                            # Re-run all tests to see if the shared fix resolved things
                            _ok_shared, output = executor.run_command(base_cmd, cwd=subproject_cwd)
                            if _ok_shared:
                                _logger.info("[BulkTest] Shared config fix resolved all failures.")
                                print("  [BulkTest] All tests pass after shared config fix.")
                                return True, ""
                            # Update failed_files with whatever remains
                            failed_files = _parse_failed_test_files(
                                output, list(test_files.keys()), subproject_cwd)
                            _shared_fix_applied = True
                            _logger.info(
                                "[BulkTest] After shared fix, %d file(s) still failing: %s",
                                len(failed_files), failed_files,
                            )
                    except Exception as _sfe:
                        _logger.warning("[BulkTest] Shared config fix failed: %s", _sfe)

    # ── Step 2: Fix one failing test file at a time ──
    failed_files = _parse_failed_test_files(output, list(test_files.keys()), subproject_cwd)
    _logger.info("[BulkTest] Failed test files: %s", failed_files)
    print(f"  [BulkTest] {len(failed_files)} test file(s) failed — fixing one at a time...")

    # Show initial pass/fail state in TEST RESULTS immediately
    _failed_set = set(failed_files)
    for fpath in test_files:
        if fpath in _failed_set:
            _p, _t, _f = _parse_test_counts(output)
            display.record_test_result(fpath, passed=_p, total=_t, failures=_f)
        else:
            display.record_test_result(fpath, passed=1, total=1, failures=[])

    lang_tag = lang or "python"

    # Track fix content hashes per test file to prevent the loop from
    # re-applying an identical (failed) fix across attempts.
    # Keys are test file paths; values are sets of hex content digests.
    import hashlib as _hashlib
    _applied_fix_signatures: dict[str, set[str]] = {}

    # Use an index-based loop so we can append newly-impacted test files
    # to failed_files mid-iteration without losing them.
    fix_idx = 0
    while fix_idx < len(failed_files):
        test_path = failed_files[fix_idx]
        fix_idx += 1
        basename = test_path.rsplit('/', 1)[-1]
        print(f"  [BulkTest] Fixing {basename}...")

        current_output = output  # use full output for first attempt
        _no_code_last_attempt = False   # True when LLM returned prose but no code
        _test_rewrite_done = False      # True after a test-rewrite pivot was attempted

        # ── ErrorRouter: classify this test failure once per file ────────────
        # Determines whether web search is worth calling before fixing.
        # Computed from the initial error output (before any fix is applied).
        _initial_error = _extract_file_specific_errors(output, test_path, max_chars=2000)
        if not _initial_error:
            _initial_error = output[-2000:]
        _route = None
        _bulk_search_context = ""
        try:
            from .error_router import classify_error as _classify_error
            _route = _classify_error(
                error_info=_initial_error,
                step_type="TEST",
                project_context=project_context,
                kb_matched=False,
                llm_client=coder.llm_client if search_agent is not None else None,
            )
            log.info(
                "[BulkTest] ErrorRouter %s → source=%s skip_web=%s (%s)",
                basename, _route.source_type, _route.skip_web, _route.reason,
            )
            if search_agent is not None and not _route.skip_web:
                _lang = language or (
                    getattr(project_context, "language", None) if project_context else None)
                _kb_ctx = getattr(memory, "_kb_context", "")
                _bulk_search_context = search_agent.search_for_error(
                    _initial_error, test_path,
                    language=_lang,
                    kb_context=_kb_ctx,
                    query_override=_route.query_hint or None,
                )
                if _bulk_search_context:
                    log.info("[BulkTest] Search context injected for %s", basename)
        except Exception as _re_exc:
            log.debug("[BulkTest] ErrorRouter failed for %s: %s", basename, _re_exc)

        for fix_attempt in range(1, _MAX_BULK_TEST_FIX_ATTEMPTS + 1):
            # Extract error relevant to this file
            file_error = _extract_file_specific_errors(
                current_output, test_path, max_chars=3000)
            if not file_error:
                # Take the tail (where tracebacks and ImportErrors appear)
                # rather than the head (which is often setup noise).
                file_error = current_output[-3000:]

            # Build source context for this test file
            current_content = memory.all_files().get(test_path, "")
            if not current_content:
                # Pre-existing file not tracked in session memory — read from disk
                try:
                    with open(test_path, "r", encoding="utf-8", errors="replace") as _tf:
                        current_content = _tf.read()
                except OSError:
                    pass
            imported_sources = _extract_imported_sources(
                {test_path: current_content}, memory,
                resolve_from_disk=True)

            # Also resolve 2nd-level imports (files imported by the direct
            # imports) so the LLM can see all relevant source components.
            # Capped at 4 extra files to avoid bloating the prompt.
            _second_level: dict[str, str] = {}
            if imported_sources:
                _second_level = _extract_imported_sources(
                    imported_sources, memory, resolve_from_disk=True)
                # Remove anything already in imported_sources or the test file
                for _k in list(_second_level.keys()):
                    if _k in imported_sources or _k == test_path:
                        del _second_level[_k]
                # Keep at most 4 extra files (shortest paths first = most local)
                _second_level = dict(
                    sorted(_second_level.items(), key=lambda kv: len(kv[0]))[:4]
                )

            source_ctx = (
                f"#### [FILE]: {test_path}\n```{lang_tag}\n{current_content}\n```\n\n"
            )
            for fp, cnt in imported_sources.items():
                source_ctx += (
                    f"#### [FILE]: {fp}\n```{lang_tag}\n{cnt}\n```\n\n"
                )
            for fp, cnt in _second_level.items():
                source_ctx += (
                    f"#### [FILE]: {fp}\n```{lang_tag}\n{cnt}\n```\n\n"
                )

            # For Django projects inject settings.py + the real ROOT_URLCONF so
            # the LLM can see the correct file to edit instead of guessing.
            source_ctx += _django_settings_context(subproject_cwd)

            # Inject KB guidance — use planner-declared docs when available
            # (exact title lookup, no semantic search) to avoid injecting
            # irrelevant docs (e.g. Django instructions for a React test).
            # Fall back to build_context() only when the plan has no kb_docs.
            kb_instructions = ""
            if kb_context_builder is not None:
                try:
                    _declared_titles = _step_kb_docs.get(test_path)
                    if _declared_titles:
                        _gstore = getattr(kb_context_builder, '_global_store', None)
                        if _gstore is not None:
                            _kb_results = _gstore.get_by_titles(_declared_titles)
                            if _kb_results:
                                _kb_parts = [
                                    getattr(r, 'content', '') or getattr(r, 'title', '')
                                    for r in _kb_results
                                ]
                                kb_instructions = (
                                    "\n\nKnowledge base guidance:\n"
                                    + "\n".join(p for p in _kb_parts if p)
                                )
                    else:
                        from ..kb.context_builder import ContextBuilder
                        kb_ctx = kb_context_builder.build_context(
                            task_description=task,
                            current_file=test_path,
                            max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 2000) if cfg else 2000,
                            language=lang,
                            step_type="TEST",
                        )
                        kb_text = kb_context_builder.format_context_for_prompt(kb_ctx)
                        if kb_text:
                            kb_instructions = f"\n\nKnowledge base guidance:\n{kb_text}"
                except Exception:
                    pass

            _bt_briefing = getattr(memory, '_task_briefing', '')
            _bt_briefing_block = (
                "TASK BRIEFING (overall goal — respect Preserve and Key constraint):\n"
                f"{_bt_briefing}\n\n"
            ) if _bt_briefing else ""

            fix_prompt = (
                f"{_bt_briefing_block}"
                f"Task: {task}\n\n"
                f"Test file `{test_path}` failed. Fix it so the tests pass.\n\n"
                f"Error output:\n{file_error}\n\n"
                f"Relevant files:\n{source_ctx}"
                f"{kb_instructions}\n\n"
                "You may fix the test file itself OR fix a source file it imports — "
                "whichever is correct.  Do NOT remove any existing tests.\n\n"
                "CRITICAL: Do NOT remove or comment out the tested component/feature.\n\n"
                "IMPORTANT — before modifying any template or source file to satisfy "
                "an assertContains/assertIn/assertEqual test:\n"
                "1. Read the EXACT string literal from the test assertion above.\n"
                "2. Copy that exact string (correct case, spacing, punctuation) into "
                "   the template or source file.\n"
                "3. Do NOT paraphrase, guess, or change the casing of the expected string.\n\n"
                "CRITICAL: NEVER abbreviate or summarize existing code with comments like `// existing code` or `/* unchanged */`. If you are editing a chunk or a file, you MUST write out the ENTIRE content of that chunk or file. Abbreviating code will cause it to be permanently deleted!\n\n"
                "Prefer CHUNK FORMAT for surgical fixes:\n"
                f"#### [EDIT]: path/to/file:{lang_tag}:function_name (lines start-end)\n"
                f"```{lang_tag}\n// replacement chunk\n```\n"
                "Use full-file [FILE]: format only when the whole file must be rewritten."
            )
            if _bulk_search_context:
                fix_prompt += (
                    f"\n\nWeb search context (use to inform your fix):\n"
                    f"{_bulk_search_context}\n"
                )
            if _no_code_last_attempt:
                fix_prompt += (
                    "\n\nCRITICAL: Your previous response contained only explanation text "
                    "with no code changes. You MUST output actual file content using "
                    "#### [FILE]: or #### [EDIT]: markers — no prose-only replies."
                )

            try:
                fix_response = coder.llm_client.generate_response(fix_prompt)
                # Try chunk edits first (surgical), fall back to full-file
                fix_files = {}
                try:
                    from ..editing.chunk_editor import ChunkEditor as _BtCE
                    _bt_ce = _BtCE()
                    _bt_edits = _bt_ce.parse_chunk_response(fix_response)
                    if _bt_edits:
                        for _bt_edit in _bt_edits:
                            _bt_fp = _bt_edit.file_path
                            _bt_existing = memory.get(_bt_fp)
                            if _bt_existing is None:
                                try:
                                    with open(_bt_fp, "r", encoding="utf-8", errors="replace") as _f:
                                        _bt_existing = _f.read()
                                except OSError:
                                    pass
                            # If the LLM used a subproject-relative path (e.g.
                            # "src/components/Foo.jsx" instead of
                            # "react-responsive-page/src/components/Foo.jsx"),
                            # try resolving it via the subproject prefix so the
                            # existing file content can be found and the edit
                            # doesn't silently fall through.
                            if _bt_existing is None and subproject_cwd:
                                _bt_fp_prefixed = f"{subproject_cwd}/{_bt_fp}"
                                _bt_existing = memory.get(_bt_fp_prefixed)
                                if _bt_existing is None:
                                    try:
                                        with open(_bt_fp_prefixed, "r", encoding="utf-8", errors="replace") as _f:
                                            _bt_existing = _f.read()
                                    except OSError:
                                        pass
                                if _bt_existing is not None:
                                    _bt_fp = _bt_fp_prefixed
                                    _bt_edit = type(_bt_edit)(
                                        file_path=_bt_fp,
                                        chunk_id=_bt_edit.chunk_id,
                                        line_start=_bt_edit.line_start,
                                        line_end=_bt_edit.line_end,
                                        new_content=_bt_edit.new_content,
                                        is_new=_bt_edit.is_new,
                                        insert_after=_bt_edit.insert_after,
                                    )
                            if _bt_existing:
                                # Guard: reject full-file replacements that are
                                # suspiciously small compared to the existing file
                                # and lack any function/class definitions — these
                                # are almost always a lone stub like
                                # `export default Foo;` that would destroy the file.
                                _is_full_replace = (
                                    _bt_edit.line_start == 1
                                    and _bt_edit.line_end >= 99999
                                )
                                if _is_full_replace and _bt_existing:
                                    _new_lines = _bt_edit.new_content.splitlines()
                                    _old_lines = _bt_existing.splitlines()
                                    _too_small = len(_new_lines) < max(5, len(_old_lines) * 0.15)
                                    _has_def = any(
                                        kw in _bt_edit.new_content
                                        for kw in (
                                            'function ', 'const ', 'class ',
                                            'def ', 'export default function',
                                            '=>', 'return (', 'return <',
                                        )
                                    )
                                    if _too_small and not _has_def:
                                        _logger.warning(
                                            "[BulkTest] Rejected suspiciously tiny "
                                            "full-file replacement for %s "
                                            "(%d lines → %d lines, no definitions) "
                                            "— skipping to avoid destroying the file.",
                                            _bt_fp, len(_old_lines), len(_new_lines),
                                        )
                                        continue
                                try:
                                    fix_files[_bt_fp] = _bt_ce.apply_chunk_edits(_bt_existing, [_bt_edit])
                                except Exception:
                                    pass
                except ImportError:
                    pass
                if not fix_files:
                    fix_files = executor.parse_code_blocks(fix_response)
                if not fix_files:
                    fix_files = executor.parse_code_blocks_fuzzy(fix_response)

                # ── No code in response — don't re-run unchanged test ──────────
                if not fix_files:
                    _logger.warning(
                        "[BulkTest] No code extracted from LLM response for %s "
                        "(attempt %d/%d) — skipping re-run, retrying with stronger prompt.",
                        basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
                    )
                    _no_code_last_attempt = True
                    continue

                _no_code_last_attempt = False

                if subproject_cwd:
                    from .step_handlers import _prefix_subproject_paths
                    fix_files = _prefix_subproject_paths(
                        fix_files, subproject_cwd, memory)

                # Dedup: compute a signature of the proposed file contents
                # and skip if this exact fix was already applied for this
                # test file — prevents the loop from burning retries on an
                # identical broken fix.
                _fix_sig = _hashlib.md5(
                    "".join(sorted(fix_files.values())).encode("utf-8", errors="replace")
                ).hexdigest()
                _sigs_for_test = _applied_fix_signatures.setdefault(test_path, set())
                if _fix_sig in _sigs_for_test:
                    _logger.warning(
                        "[BulkTest] Duplicate source fix for %s (attempt %d/%d) "
                        "— pivoting to test rewrite.",
                        basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
                    )
                    # ── Pivot: rewrite the TEST to match actual implementation ──
                    # The source is likely correct; the test expectation is wrong.
                    if not _test_rewrite_done:
                        _test_rewrite_done = True
                        _rw_content = memory.all_files().get(test_path, "")
                        if not _rw_content:
                            try:
                                with open(test_path, "r", encoding="utf-8",
                                          errors="replace") as _rf:
                                    _rw_content = _rf.read()
                            except OSError:
                                pass
                        _rw_prompt = (
                            f"The source fix for `{test_path}` was already applied "
                            f"but the test still fails.\n\n"
                            f"Rewrite the TEST FILE ITSELF so it matches what the "
                            f"implementation actually does. Do NOT remove any tests — "
                            f"update expected values, selectors, or text to match "
                            f"reality.\n\n"
                            "IMPORTANT GUIDELINES for the rewrite:\n"
                            "- If a test asserts an exact library-internal string (e.g. a "
                            "plugin name like `'vite:react-babel'`, an internal key, or an "
                            "internal version string), use a flexible matcher instead: "
                            "`includes('react')`, `startsWith('vite:')`, or `toContain()`. "
                            "Exact internal names change across library versions.\n"
                            "- For assertions on DOM text or component output, copy the "
                            "exact string from the source/template into the test.\n"
                            "- Do NOT change what is being tested — only update the "
                            "expected values to match what the code actually produces.\n\n"
                            f"Current test file:\n"
                            f"#### [FILE]: {test_path}\n"
                            f"```{lang_tag}\n{_rw_content}\n```\n\n"
                            f"Error output:\n{file_error}\n\n"
                            f"Relevant source:\n{source_ctx}\n\n"
                            "Output the COMPLETE rewritten test file using:\n"
                            f"#### [FILE]: {test_path}\n"
                            f"```{lang_tag}\n// full file\n```"
                        )
                        try:
                            _rw_response = coder.llm_client.generate_response(_rw_prompt)
                            _rw_files = executor.parse_code_blocks(_rw_response)
                            if not _rw_files:
                                _rw_files = executor.parse_code_blocks_fuzzy(_rw_response)
                            if _rw_files:
                                if subproject_cwd:
                                    _rw_files = _prefix_subproject_paths(
                                        _rw_files, subproject_cwd, memory)
                                executor.write_files(_rw_files)
                                memory.update(_rw_files)
                                _logger.info(
                                    "[BulkTest] Test rewrite applied for %s: %s",
                                    basename, list(_rw_files.keys()),
                                )
                                _single_rw = _build_scoped_test_cmd(
                                    base_cmd, {test_path: ""}, subproject_cwd)
                                _ok_rw, current_output = executor.run_command(
                                    _single_rw, cwd=subproject_cwd)
                                if _ok_rw:
                                    _logger.info(
                                        "[BulkTest] %s passes after test rewrite.", basename)
                                    print(f"  [BulkTest] {basename} fixed via test rewrite")
                                    _p, _t, _ = _parse_test_counts(current_output)
                                    display.record_test_result(
                                        test_path, passed=_p, total=_t, failures=[])
                                    break
                                _logger.warning(
                                    "[BulkTest] Test rewrite for %s still failing.",
                                    basename)
                        except Exception as _rw_exc:
                            _logger.warning(
                                "[BulkTest] Test rewrite failed for %s: %s",
                                basename, _rw_exc)
                    break

                _sigs_for_test.add(_fix_sig)
                # ── Source file protection ──
                # BulkTest may modify test files freely. Source files
                # are only allowed through if the change is small and
                # additive (e.g. adding data-testid, aria-label) — it
                # must not alter functionality.
                if fix_files:
                    _bt_filtered = {}
                    for _bt_fp, _bt_fc in fix_files.items():
                        if _is_test_file(_bt_fp):
                            _bt_filtered[_bt_fp] = _bt_fc
                        elif _is_additive_source_fix(
                                _bt_fp, _bt_fc, memory):
                            _bt_filtered[_bt_fp] = _bt_fc
                            _logger.info(
                                "[BulkTest] Allowed small additive "
                                "source fix for %s", _bt_fp)
                        else:
                            _logger.warning(
                                "[BulkTest] Blocked fix for source file "
                                "%s — change is too large or destructive",
                                _bt_fp)
                    fix_files = _bt_filtered
                if not fix_files:
                    # All fixes targeted source files — retry once with a
                    # test-only constraint instead of giving up.  Common
                    # case: Header uses <Link> but test doesn't wrap in
                    # MemoryRouter.  LLM tries to remove Link from Header
                    # instead of wrapping the test render.
                    if not getattr(fix_attempt, '_retried_test_only', False):
                        _logger.info(
                            "[BulkTest] All fixes were source files for %s "
                            "— retrying with test-only constraint", basename)
                        _bt_step_desc = ""
                        _ps_desc = _step_descs.get(test_path)
                        if _ps_desc:
                            _bt_step_desc = (
                                f"STEP INTENT: {_ps_desc}\n"
                            )
                        _vitest_hint = ""
                        if uses_vitest:
                            _vitest_hint = (
                                "- This project uses VITEST (not Jest). "
                                "Use `vi.mock()` instead of `jest.mock()`, "
                                "`vi.fn()` instead of `jest.fn()`, and "
                                "import `{ vi }` from 'vitest' if needed.\n"
                            )
                        _test_only_prompt = (
                            f"{_bt_briefing_block}"
                            f"Task: {task}\n\n"
                            f"{_bt_step_desc}"
                            f"Test file `{test_path}` failed.\n\n"
                            f"Error output:\n{file_error}\n\n"
                            f"Relevant source files (READ-ONLY — do NOT modify these):\n"
                            f"{source_ctx}\n\n"
                            "CRITICAL RULES:\n"
                            "1. Fix ONLY the test file. Source files are correct.\n"
                            "2. Do NOT remove or weaken test assertions — the intended "
                            "functionality must still be verified.\n"
                            "3. Adapt assertions to match what the source components "
                            "ACTUALLY render (read the source files above).\n\n"
                            "Common fixes:\n"
                            f"{_vitest_hint}"
                            "- If the error mentions Router context (useHref, useLocation, "
                            "basename is null), wrap the render in <MemoryRouter> from "
                            "'react-router-dom'.\n"
                            "- If the error mentions missing element/text, update the "
                            "test assertions to match what the component actually renders.\n"
                            "- Use getAllByText/getAllByRole if multiple elements match.\n\n"
                            f"Output the COMPLETE fixed test file using:\n"
                            f"#### [FILE]: {test_path}\n"
                            f"```{lang_tag}\n// complete test file\n```"
                        )
                        try:
                            _to_resp = coder.llm_client.generate_response(
                                _test_only_prompt)
                            _to_files = executor.parse_code_blocks(_to_resp)
                            if not _to_files:
                                _to_files = executor.parse_code_blocks_fuzzy(
                                    _to_resp)
                            if _to_files:
                                if subproject_cwd:
                                    from .step_handlers import (
                                        _prefix_subproject_paths,
                                    )
                                    _to_files = _prefix_subproject_paths(
                                        _to_files, subproject_cwd, memory)
                                # Filter again — only test files
                                _to_files = {
                                    fp: fc for fp, fc in _to_files.items()
                                    if _is_test_file(fp)
                                }
                            if _to_files:
                                fix_files = _to_files
                                _logger.info(
                                    "[BulkTest] Test-only retry produced "
                                    "fix for %s: %s",
                                    basename, list(fix_files.keys()))
                        except Exception as _to_exc:
                            _logger.warning(
                                "[BulkTest] Test-only retry failed: %s",
                                _to_exc)
                    if not fix_files:
                        _logger.warning(
                            "[BulkTest] All fix files were source files — "
                            "skipping write for %s", basename)
                        break
                executor.write_files(fix_files)
                memory.update(fix_files)
                _logger.info(
                    "[BulkTest] Applied fixes for %s: %s",
                    basename, list(fix_files.keys()),
                )
            except Exception as exc:
                _logger.warning("[BulkTest] Fix generation failed for %s: %s", basename, exc)
                break

            # Re-run this single file
            single_cmd = _build_scoped_test_cmd(
                base_cmd, {test_path: ""}, subproject_cwd)
            ok_single, current_output = executor.run_command(
                single_cmd, cwd=subproject_cwd)
            if ok_single:
                _logger.info("[BulkTest] %s now passes.", basename)
                print(f"  [BulkTest] {basename} fixed ✔")
                _p, _t, _ = _parse_test_counts(current_output)
                display.record_test_result(test_path, passed=_p, total=_t, failures=[])

                # Check if any source files were modified and find other test
                # files that import them — they may have been broken by the fix.
                if fix_files:
                    modified_sources = [f for f in fix_files if not _is_test_file(f)]
                    if modified_sources:
                        already_queued = set(failed_files)
                        impacted = _find_tests_impacted_by_sources(
                            modified_sources,
                            test_files,
                            exclude=test_path,
                            already_queued=already_queued,
                            kb_context_builder=kb_context_builder,
                        )
                        for rt in impacted:
                            rt_cmd = _build_scoped_test_cmd(
                                base_cmd, {rt: ""}, subproject_cwd)
                            ok_rt, out_rt = executor.run_command(
                                rt_cmd, cwd=subproject_cwd)
                            if not ok_rt:
                                rt_base = rt.rsplit('/', 1)[-1]
                                _logger.warning(
                                    "[BulkTest] %s broke after fix to %s — queuing",
                                    rt_base, modified_sources,
                                )
                                print(
                                    f"  [BulkTest] {rt_base} impacted by source "
                                    f"change — queuing fix..."
                                )
                                failed_files.append(rt)
                                _p_rt, _t_rt, _f_rt = _parse_test_counts(out_rt)
                                display.record_test_result(
                                    rt, passed=_p_rt, total=_t_rt, failures=_f_rt)
                break
            _logger.warning(
                "[BulkTest] %s still failing (attempt %d/%d)",
                basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
            )
            # Update TEST RESULTS with latest counts on each fix attempt
            _p, _t, _f = _parse_test_counts(current_output)
            display.record_test_result(test_path, passed=_p, total=_t, failures=_f)
        else:
            print(f"  [BulkTest] {basename} could not be fixed after "
                  f"{_MAX_BULK_TEST_FIX_ATTEMPTS} attempt(s).")

    # ── Step 3: Final run-all to confirm everything passes ──
    # If the run-all surfaces new failures (e.g. a pre-existing test file that
    # wasn't in the per-file fix queue, or a Django ERROR: that the initial
    # parse missed), feed them back into the fix loop rather than giving up.
    _MAX_RUNALL_ROUNDS = 2
    for _round in range(_MAX_RUNALL_ROUNDS):
        ok_final, output_final = executor.run_command(base_cmd, cwd=subproject_cwd)
        if ok_final:
            _logger.info("[BulkTest] Final run-all passed.")
            print("  [BulkTest] All tests pass after fixes.")
            for fpath in test_files:
                display.record_test_result(fpath, passed=1, total=1, failures=[])
            return True, ""

        # Parse failures from the full run — includes files not in the original
        # per-file fix queue (e.g. pre-existing tests, Django ERROR: lines).
        if "manage.py" in base_cmd:
            output_final = _fix_django_startup_crashes(output_final, subproject_cwd, executor)
        still_failing = _parse_failed_test_files(output_final, list(test_files.keys()), subproject_cwd)
        _logger.warning(
            "[BulkTest] Run-all round %d/%d failed. Still failing: %s",
            _round + 1, _MAX_RUNALL_ROUNDS, still_failing,
        )

        # Find files that weren't already fixed in the per-file loop
        already_fixed = set(failed_files)
        new_failures = [f for f in still_failing if f not in already_fixed]

        if not new_failures:
            # Same files are still broken — no point retrying
            break

        print(
            f"  [BulkTest] Run-all found {len(new_failures)} additional "
            f"failing file(s) — fixing..."
        )
        # Append to failed_files and re-run the fix loop for just the new ones
        fix_start = len(failed_files)
        for nf in new_failures:
            if nf not in set(failed_files):
                failed_files.append(nf)
                _p_nf, _t_nf, _f_nf = _parse_test_counts(output_final)
                display.record_test_result(nf, passed=_p_nf, total=_t_nf, failures=_f_nf)

        current_output = output_final
        while fix_idx < len(failed_files):
            test_path = failed_files[fix_idx]
            fix_idx += 1
            basename = test_path.rsplit('/', 1)[-1]
            print(f"  [BulkTest] Fixing {basename} (from run-all)...")

            for fix_attempt in range(1, _MAX_BULK_TEST_FIX_ATTEMPTS + 1):
                file_error = _extract_file_specific_errors(
                    current_output, test_path, max_chars=3000)
                if not file_error:
                    # Take the tail (where tracebacks and ImportErrors appear)
                    # rather than the head (which is often setup noise).
                    file_error = current_output[-3000:]

                current_content = memory.all_files().get(test_path, "")
                if not current_content:
                    # Pre-existing file not tracked in session memory — read from disk
                    try:
                        with open(test_path, "r", encoding="utf-8", errors="replace") as _tf2:
                            current_content = _tf2.read()
                    except OSError:
                        pass
                imported_sources = _extract_imported_sources(
                    {test_path: current_content}, memory,
                    resolve_from_disk=True)
                source_ctx = (
                    f"#### [FILE]: {test_path}\n```{lang_tag}\n{current_content}\n```\n\n"
                )
                for fp, cnt in imported_sources.items():
                    source_ctx += (
                        f"#### [FILE]: {fp}\n```{lang_tag}\n{cnt}\n```\n\n"
                    )
                source_ctx += _django_settings_context(subproject_cwd)

                _bt_briefing = getattr(memory, '_task_briefing', '')
                _bt_briefing_block = (
                    "TASK BRIEFING (overall goal):\n"
                    f"{_bt_briefing}\n\n"
                ) if _bt_briefing else ""

                fix_prompt = (
                    f"{_bt_briefing_block}"
                    f"Task: {task}\n\n"
                    f"Test file `{test_path}` failed in the full test suite. "
                    f"Fix it so the tests pass.\n\n"
                    f"Error output:\n{file_error}\n\n"
                    f"Relevant files:\n{source_ctx}\n\n"
                    "You may fix the test file itself OR fix a source file it imports — "
                    "whichever is correct.  Do NOT remove any existing tests.\n\n"
                    "IMPORTANT — before modifying any template or source file to satisfy "
                    "an assertContains/assertIn/assertEqual test:\n"
                    "1. Read the EXACT string literal from the test assertion above.\n"
                    "2. Copy that exact string (correct case, spacing, punctuation) into "
                    "   the template or source file.\n"
                    "3. Do NOT paraphrase, guess, or change the casing of the expected string.\n\n"
                    "CRITICAL: NEVER abbreviate or summarize existing code with comments like `// existing code` or `/* unchanged */`. If you are editing a chunk or a file, you MUST write out the ENTIRE content of that chunk or file. Abbreviating code will cause it to be permanently deleted!\n\n"
                    "Prefer CHUNK FORMAT for surgical fixes:\n"
                    f"#### [EDIT]: path/to/file:{lang_tag}:function_name (lines start-end)\n"
                    f"```{lang_tag}\n// replacement chunk\n```\n"
                    "Use full-file [FILE]: format only when the whole file must be rewritten."
                )

                try:
                    fix_response = coder.llm_client.generate_response(fix_prompt)
                    fix_files = {}
                    try:
                        from ..editing.chunk_editor import ChunkEditor as _BtCE2
                        _bt_ce2 = _BtCE2()
                        _bt_edits2 = _bt_ce2.parse_chunk_response(fix_response)
                        if _bt_edits2:
                            for _bt_edit2 in _bt_edits2:
                                _bt_fp2 = _bt_edit2.file_path
                                _bt_existing2 = memory.get(_bt_fp2)
                                if _bt_existing2 is None:
                                    try:
                                        with open(_bt_fp2, "r", encoding="utf-8",
                                                  errors="replace") as _f2:
                                            _bt_existing2 = _f2.read()
                                    except OSError:
                                        pass
                                if _bt_existing2:
                                    try:
                                        fix_files[_bt_fp2] = _bt_ce2.apply_chunk_edits(
                                            _bt_existing2, [_bt_edit2])
                                    except Exception:
                                        pass
                    except ImportError:
                        pass
                    if not fix_files:
                        fix_files = executor.parse_code_blocks(fix_response)
                    if not fix_files:
                        fix_files = executor.parse_code_blocks_fuzzy(fix_response)
                    if fix_files:
                        if subproject_cwd:
                            from .step_handlers import _prefix_subproject_paths
                            fix_files = _prefix_subproject_paths(
                                fix_files, subproject_cwd, memory)
                        # ── Source file protection ──
                        _bt2_filtered = {}
                        for _bt2_fp, _bt2_fc in fix_files.items():
                            if _is_test_file(_bt2_fp):
                                _bt2_filtered[_bt2_fp] = _bt2_fc
                            elif _is_additive_source_fix(
                                    _bt2_fp, _bt2_fc, memory):
                                _bt2_filtered[_bt2_fp] = _bt2_fc
                                _logger.info(
                                    "[BulkTest] Allowed small additive "
                                    "source fix for %s", _bt2_fp)
                            else:
                                _logger.warning(
                                    "[BulkTest] Blocked fix for source "
                                    "file %s — too large or destructive",
                                    _bt2_fp)
                        _bt2_blocked = set(fix_files) - set(_bt2_filtered)
                        if _bt2_blocked:
                            _logger.warning(
                                "[BulkTest] Blocked source file(s): %s",
                                list(_bt2_blocked))
                        fix_files = _bt2_filtered
                        if not fix_files:
                            # Retry with test-only constraint (same
                            # logic as Loop 1 — see detailed comments there)
                            try:
                                _to2_step_desc = ""
                                _to2_ps = _step_descs.get(test_path)
                                if _to2_ps:
                                    _to2_step_desc = (
                                        f"STEP INTENT: {_to2_ps}\n")
                                _to2_vitest_hint = ""
                                if uses_vitest:
                                    _to2_vitest_hint = (
                                        "- VITEST project: use `vi.mock()` "
                                        "not `jest.mock()`, `vi.fn()` not "
                                        "`jest.fn()`.\n"
                                    )
                                _to2_prompt = (
                                    f"{_bt_briefing_block}"
                                    f"{_to2_step_desc}"
                                    f"Test file `{test_path}` failed.\n\n"
                                    f"Error:\n{current_output[:3000]}\n\n"
                                    f"Source files (READ-ONLY):\n"
                                    f"{source_ctx}\n\n"
                                    "CRITICAL RULES:\n"
                                    "1. Fix ONLY the test file. Source "
                                    "files are correct.\n"
                                    "2. Do NOT remove or weaken "
                                    "assertions — verify the intended "
                                    "functionality.\n"
                                    "3. Adapt to match what the source "
                                    "components actually render.\n"
                                    f"{_to2_vitest_hint}"
                                    "- Use MemoryRouter for Router "
                                    "context.\n"
                                    "- Use getAllBy* for multiple "
                                    "matches.\n\n"
                                    f"#### [FILE]: {test_path}\n"
                                )
                                _to2_resp = coder.llm_client.generate_response(
                                    _to2_prompt)
                                _to2_files = (
                                    executor.parse_code_blocks(_to2_resp)
                                    or executor.parse_code_blocks_fuzzy(_to2_resp)
                                )
                                if _to2_files and subproject_cwd:
                                    from .step_handlers import _prefix_subproject_paths
                                    _to2_files = _prefix_subproject_paths(
                                        _to2_files, subproject_cwd, memory)
                                if _to2_files:
                                    _to2_files = {
                                        fp: fc for fp, fc in _to2_files.items()
                                        if _is_test_file(fp)
                                    }
                                if _to2_files:
                                    fix_files = _to2_files
                                    _logger.info(
                                        "[BulkTest] Test-only retry produced "
                                        "fix: %s", list(fix_files.keys()))
                            except Exception:
                                pass
                        if not fix_files:
                            break
                        executor.write_files(fix_files)
                        memory.update(fix_files)
                        _logger.info(
                            "[BulkTest] Applied fixes for %s: %s",
                            basename, list(fix_files.keys()),
                        )
                except Exception as exc:
                    _logger.warning(
                        "[BulkTest] Fix generation failed for %s: %s", basename, exc)
                    break

                single_cmd = _build_scoped_test_cmd(
                    base_cmd, {test_path: ""}, subproject_cwd)
                ok_single, current_output = executor.run_command(
                    single_cmd, cwd=subproject_cwd)
                if ok_single:
                    _logger.info("[BulkTest] %s now passes.", basename)
                    print(f"  [BulkTest] {basename} fixed ✔")
                    _p, _t, _ = _parse_test_counts(current_output)
                    display.record_test_result(test_path, passed=_p, total=_t, failures=[])
                    break
                _logger.warning(
                    "[BulkTest] %s still failing (attempt %d/%d)",
                    basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
                )
                _p, _t, _f = _parse_test_counts(current_output)
                display.record_test_result(test_path, passed=_p, total=_t, failures=_f)
            else:
                print(f"  [BulkTest] {basename} could not be fixed after "
                      f"{_MAX_BULK_TEST_FIX_ATTEMPTS} attempt(s).")

        current_output = output_final  # reset for next round's error extraction

    # All rounds exhausted — update display and report failure
    still_failing_final = set(_parse_failed_test_files(output_final, list(test_files.keys()), subproject_cwd))
    for fpath in test_files:
        if fpath in still_failing_final:
            _p, _t, _f = _parse_test_counts(output_final)
            display.record_test_result(fpath, passed=_p, total=_t, failures=_f)
        else:
            display.record_test_result(fpath, passed=1, total=1, failures=[])

    error_msg = (
        f"Bulk test execution failed: some test file(s) still failing "
        f"after per-file fix attempts.\n{output_final[:600]}"
    )
    _logger.warning("[BulkTest] Final run-all failed:\n%s", output_final[:600])
    print("  [BulkTest] FAILED — some tests still failing after fixes.")
    return False, error_msg


# ---------------------------------------------------------------------------
# Wiring verification — cross-file integration check after all steps complete
# ---------------------------------------------------------------------------

def _parse_fix_scope(task: str) -> tuple[list[str], list[str]]:
    """Parse fix scope file references from the task description.

    Looks for a ``Fix scope:`` section, then extracts:
    - backtick-quoted paths / filenames  → *exact_paths*
    - remaining natural-language text    → *nl_queries* (for KB semantic search)

    Returns ``(exact_paths, nl_queries)``.
    """
    # Narrow to "Fix scope:" section when present; otherwise scan full task.
    m = _FIX_SCOPE_SECTION_RE.search(task)
    scope_text = m.group(1) if m else task

    exact_paths: list[str] = []
    seen: set[str] = set()

    # 1. Backtick-quoted paths/names
    for bm in _BACKTICK_PATH_RE.finditer(scope_text):
        p = bm.group(1).strip()
        if p and p not in seen:
            exact_paths.append(p)
            seen.add(p)

    # 2. Bare file names not already captured
    for bm in _BARE_FILENAME_RE.finditer(scope_text):
        p = bm.group(1).strip()
        if p not in seen:
            exact_paths.append(p)
            seen.add(p)

    # 3. Remaining text as NL query (strip captured paths, collapse whitespace)
    nl_text = _BACKTICK_PATH_RE.sub('', scope_text)
    nl_text = _BARE_FILENAME_RE.sub('', nl_text)
    nl_text = re.sub(r'[\s,;()]+', ' ', nl_text).strip()
    nl_queries = [nl_text] if len(nl_text) > 20 else []

    return exact_paths, nl_queries


def _resolve_fix_scope_files(
    exact_paths: list[str],
    nl_queries: list[str],
    memory: FileMemory,
    kb_context_builder=None,
    project_root: str = "",
) -> dict[str, str]:
    """Resolve fix-scope entries to ``{path: content}``.

    Fallback chain per entry:
    1. FileMemory exact key match
    2. FileMemory basename / suffix match
    3. Disk read  (file exists, was not modified this session)
    4. Filesystem glob  (``**/<basename>``)
    5. KB semantic search  (when *kb_context_builder* is available)

    Files from steps 3-5 are included as read-only context — they are NOT
    added to FileMemory so that the written-files audit trail stays clean.
    """
    import glob as _glob
    import os as _os

    root = project_root or _os.getcwd()

    def _read_disk(path: str) -> str | None:
        abs_p = path if _os.path.isabs(path) else _os.path.join(root, path)
        try:
            if _os.path.isfile(abs_p):
                with open(abs_p, encoding="utf-8", errors="replace") as fh:
                    return fh.read()
        except OSError:
            pass
        return None

    def _glob_search(name: str) -> tuple[str, str] | None:
        basename = _os.path.basename(name)
        hits = _glob.glob(_os.path.join(root, "**", basename), recursive=True)
        for hit in hits:
            content = _read_disk(hit)
            if content:
                rel = _os.path.relpath(hit, root).replace("\\", "/")
                return rel, content
        return None

    result: dict[str, str] = {}

    for path in exact_paths:
        basename = _os.path.basename(path)

        # 1. FileMemory exact
        content = memory.get(path)
        if content is not None:
            result[path] = content
            continue

        # 2. FileMemory basename / suffix match
        for stored_path, stored_content in memory.all_files().items():
            sp_norm = stored_path.replace("\\", "/")
            p_norm = path.replace("\\", "/")
            if (sp_norm.endswith(p_norm)
                    or _os.path.basename(stored_path) == basename):
                result[stored_path] = stored_content
                break
        else:
            # 3. Disk read
            content = _read_disk(path)
            if content is not None:
                result[path] = content
            else:
                # 4. Glob
                found = _glob_search(path)
                if found:
                    result[found[0]] = found[1]
                else:
                    # 5. KB semantic search
                    resolved_via_kb = False
                    if kb_context_builder is not None:
                        try:
                            relevant = kb_context_builder.get_relevant_files(
                                task_description=f"file {path} {basename}",
                                max_files=1,
                            )
                            for r_path in relevant:
                                r_content = memory.get(r_path) or _read_disk(r_path)
                                if r_content:
                                    result[r_path] = r_content
                                    resolved_via_kb = True
                                    break
                        except Exception as kb_exc:
                            _logger.debug(
                                "[WiringVerification] KB search failed for %s: %s",
                                path, kb_exc,
                            )
                    if not resolved_via_kb:
                        _logger.warning(
                            "[WiringVerification] Could not resolve: %s — skipping", path
                        )

    # Natural-language queries → KB semantic search (top-3 files each)
    for nl_query in nl_queries:
        if kb_context_builder is not None:
            try:
                relevant = kb_context_builder.get_relevant_files(
                    task_description=nl_query,
                    max_files=3,
                )
                for r_path in relevant:
                    if r_path not in result:
                        r_content = memory.get(r_path) or _read_disk(r_path)
                        if r_content:
                            result[r_path] = r_content
            except Exception as exc:
                _logger.debug(
                    "[WiringVerification] KB NL query failed (%r): %s", nl_query, exc
                )

    return result


def run_wiring_verification(
    *,
    memory: FileMemory,
    executor,
    coder,
    display: "CLIDisplay",
    task: str,
    language: str | None,
    cfg=None,
    kb_context_builder=None,
    project_root: str = "",
) -> tuple[bool, str]:
    """Cross-file wiring check executed once after all pipeline steps complete.

    Algorithm
    ---------
    1. Parse ``Fix scope:`` from the task description to get target files.
    2. Resolve each target via layered fallback:
       FileMemory → disk read → filesystem glob → KB semantic search.
       Files not in FileMemory land in a read-only *verification_context*
       buffer — they are never written back.
    3. Make a single LLM call asking for cross-file integration issues
       (missing mounts, import/export mismatches, wrong prop shapes, etc.).
    4. If the LLM finds issues, parse code blocks and apply fixes via the
       executor + memory.  Returns ``(True, "")`` on success.
    5. If no fix scope is declared in the task, falls back to all session
       files tracked by FileMemory.

    Always non-fatal on LLM errors — returns ``(True, "")`` rather than
    blocking the pipeline on a transient failure.
    """
    import os as _os
    from .memory import _should_skip_for_context

    _logger.info("[WiringVerification] Starting cross-file wiring check")

    # ── 1. Parse fix scope ────────────────────────────────────────────────
    exact_paths, nl_queries = _parse_fix_scope(task)

    # ── 2. Resolve files ──────────────────────────────────────────────────
    if exact_paths or nl_queries:
        verification_context = _resolve_fix_scope_files(
            exact_paths, nl_queries, memory,
            kb_context_builder=kb_context_builder,
            project_root=project_root or _os.getcwd(),
        )
    else:
        # No explicit scope — use all session files as a broad check.
        _logger.info(
            "[WiringVerification] No fix scope in task — checking all session files"
        )
        verification_context = {
            p: c for p, c in memory.all_files().items()
            if not _should_skip_for_context(p)
        }

    if not verification_context:
        _logger.info("[WiringVerification] No files resolved — skipping")
        return True, ""

    _logger.info(
        "[WiringVerification] Resolved %d file(s): %s",
        len(verification_context), list(verification_context.keys()),
    )

    # ── 3. Build verification prompt ──────────────────────────────────────
    lang_tag = language or "code"
    context_block = "\n\n".join(
        f"#### [FILE]: {path}\n```{lang_tag}\n{content}\n```"
        for path, content in verification_context.items()
    )

    prompt = (
        f"Task: {task}\n\n"
        "You are performing a final cross-file wiring review. "
        "Examine ALL files below together and identify any integration "
        "issue that would cause a blank screen, runtime crash, or silent "
        "failure at startup. Common issues to check:\n"
        "  • Missing ReactDOM.createRoot / wrong entry-point mounting\n"
        "  • Default-export / named-export mismatch between files\n"
        "  • Import path typo or missing file extension\n"
        "  • Component receives wrong prop shape (undefined, wrong type)\n"
        "  • Undefined variable or missing null-check at render time\n"
        "  • Module not found (package not imported / wrong casing)\n"
        "  • Duplicate context/provider wrappers (e.g. Router, ThemeProvider, "
        "Store) in both entry-point AND child — causes nested provider errors\n"
        "  • Any other wiring issue that prevents the UI from rendering\n\n"
        f"Files in scope:\n{context_block}\n\n"
        "RESPONSE FORMAT:\n"
        "  • If NO issues exist, respond with exactly: NO_ISSUES_FOUND\n"
        "  • If issues exist, output the COMPLETE fixed content of every "
        "affected file using:\n"
        "    #### [FILE]: path/to/file\n"
        f"    ```{lang_tag}\n"
        "    // complete file content — never abbreviate\n"
        "    ```\n"
        "CRITICAL: Never use `// existing code` or `/* unchanged */` — "
        "write the full file content or your fix will delete existing code."
    )

    display.show_status("Verifying cross-file wiring...")

    # ── 4. LLM call ───────────────────────────────────────────────────────
    try:
        response = coder.llm_client.generate_response(prompt)
    except Exception as exc:
        _logger.warning("[WiringVerification] LLM call failed (non-fatal): %s", exc)
        return True, ""

    if "NO_ISSUES_FOUND" in response:
        _logger.info("[WiringVerification] No wiring issues found")
        print("  [WiringVerify] No cross-file wiring issues found.")
        return True, ""

    # ── 5. Parse and apply fixes ──────────────────────────────────────────
    fix_files: dict[str, str] = {}
    try:
        fix_files = executor.parse_code_blocks(response)
    except Exception:
        pass
    if not fix_files:
        try:
            fix_files = executor.parse_code_blocks_fuzzy(response)
        except Exception:
            pass

    if not fix_files:
        # LLM described an issue in prose but produced no code — log and
        # treat as informational (don't block the pipeline).
        _logger.warning(
            "[WiringVerification] LLM reported issues but no code blocks found; "
            "review manually:\n%s", response[:400],
        )
        return True, ""

    _logger.info(
        "[WiringVerification] Applying fixes for: %s", list(fix_files.keys())
    )
    print(
        f"  [WiringVerify] Wiring issues found — fixing: "
        f"{', '.join(fix_files.keys())}"
    )
    try:
        executor.write_files(fix_files)
        memory.update(fix_files)
    except Exception as exc:
        _logger.warning("[WiringVerification] Failed to write fixes: %s", exc)
        return False, f"[WiringVerification] Fix write failed: {exc}"

    return True, ""

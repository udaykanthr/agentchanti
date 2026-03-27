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


def _is_test_file(file_path: str) -> bool:
    """Return True if *file_path* looks like a test file."""
    import os
    basename = os.path.basename(file_path)
    return bool(_TEST_FILE_RE.search(basename) or _TEST_DIR_RE.search(file_path))


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
        if kb_context_builder is not None:
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
                plan_step=plan_step)

        elif step_type == "CODE":
            # ── Inline code fast path ──
            # If the planner already provided complete code in the plan,
            # write it directly — zero Coder LLM calls needed.
            if (plan_step is not None
                    and plan_step.inline_code
                    and len(plan_step.inline_code) > 0):
                display.step_info(step_idx, "Writing inline code from plan (0 LLM calls)")
                _inline_files = dict(plan_step.inline_code)
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
                    project_profile=project_profile)

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

    for diag_attempt in range(1, MAX_DIAGNOSIS_RETRIES + 1):
        try:
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
                kb_context_builder=kb_context_builder)

            # Extract the original failing command from error_info so
            # _apply_fix can filter it out (prevents re-running the same
            # broken command extracted from diagnosis inline backticks).
            import re as _re_diag
            _orig_cmd_match = _re_diag.search(
                r"Command `(.+?)` failed\.", error_info or "")
            _orig_cmd = _orig_cmd_match.group(1) if _orig_cmd_match else None

            fix_applied, cmds_succeeded, has_fix_commands = _apply_fix(
                diagnosis, executor, memory, display, step_idx,
                step_type=step_type,
                original_error_cmd=_orig_cmd)

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

    # Determine test command (mirror _handle_test_step detection logic)
    lang = language
    if lang is None:
        lang = detect_language_from_files(list(test_files.keys()))

    fw = get_test_framework(lang) if lang else get_test_framework("python")
    base_cmd = fw["command"]

    # Vitest override: check imports, config files, and package.json
    if "jest" in base_cmd.lower():
        # 1. Explicit vitest imports (works when globals:false)
        uses_vitest = any(
            "from 'vitest'" in c or 'from "vitest"' in c
            for c in test_files.values()
        )
        # 2. vitest.config.* present in session memory (covers globals:true setups)
        if not uses_vitest:
            _vitest_configs = (
                "vitest.config.js", "vitest.config.ts",
                "vitest.config.mjs", "vitest.config.mts",
            )
            uses_vitest = any(
                any(f.endswith(cfg) for cfg in _vitest_configs)
                for f in all_files
            )
        # 3. vitest listed in package.json (covers installed-but-config-not-in-memory)
        if not uses_vitest:
            pkg_content = next(
                (c for f, c in all_files.items() if f.endswith("package.json")),
                "",
            )
            uses_vitest = '"vitest"' in pkg_content or "'vitest'" in pkg_content
        if uses_vitest:
            base_cmd = "npx vitest run"
            _logger.info("[FinalVerify] Overriding to vitest (detected vitest config/package)")

    # Detect subproject root
    subproject_cwd = _detect_subproject_root(memory)

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

        fix_prompt = (
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
            + "\n\nOutput ONLY the complete fixed file(s) using this exact format — no prose, no explanation:\n"
            + "#### [FILE]: path/to/file.py\n```python\n...full file content...\n```"
        )
        try:
            fix_response = coder.llm_client.generate_response(fix_prompt)
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


def _parse_failed_test_files(output: str, known_test_files: list[str]) -> list[str]:
    """Parse test runner output to find which test files failed.

    Matches FAIL lines from vitest/jest/pytest against the known test files
    written during the session.  Returns a list of matching file paths.
    """
    from .step_handlers import _ANSI_RE
    clean = _ANSI_RE.sub('', output)
    failed: list[str] = []
    for fpath in known_test_files:
        basename = fpath.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        # vitest/jest: " FAIL src/__tests__/Foo.test.jsx"
        # pytest:      "FAILED tests/test_foo.py::test_bar"
        if re.search(
            r'(?:^|\s)(?:FAIL(?:ED)?)\s.*' + re.escape(basename),
            clean,
            re.MULTILINE | re.IGNORECASE,
        ):
            failed.append(fpath)
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
    )

    all_files = memory.all_files()
    test_files = {
        fpath: content
        for fpath, content in all_files.items()
        if _is_test_file(fpath) and not fpath.startswith("_")
    }

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

    fw = get_test_framework(lang) if lang else get_test_framework("python")
    base_cmd = fw["command"]

    # Vitest override (mirrors run_final_test_verification detection logic)
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
        if not uses_vitest:
            pkg_content = next(
                (c for f, c in all_files.items() if f.endswith("package.json")),
                "",
            )
            uses_vitest = '"vitest"' in pkg_content or "'vitest'" in pkg_content
        if uses_vitest:
            base_cmd = "npx vitest run"
            _logger.info("[BulkTest] Overriding to vitest")

    # ── Step 1: Run all tests ──
    ok, output = executor.run_command(base_cmd, cwd=subproject_cwd)
    if ok:
        _logger.info("[BulkTest] All tests passed on first run.")
        print("  [BulkTest] All tests passed.")
        return True, ""

    _logger.warning("[BulkTest] Tests failed:\n%s", output[:1000])

    # ── Step 2: Fix one failing test file at a time ──
    failed_files = _parse_failed_test_files(output, list(test_files.keys()))
    _logger.info("[BulkTest] Failed test files: %s", failed_files)
    print(f"  [BulkTest] {len(failed_files)} test file(s) failed — fixing one at a time...")

    lang_tag = lang or "python"

    for test_path in failed_files:
        basename = test_path.rsplit('/', 1)[-1]
        print(f"  [BulkTest] Fixing {basename}...")

        current_output = output  # use full output for first attempt

        for fix_attempt in range(1, _MAX_BULK_TEST_FIX_ATTEMPTS + 1):
            # Extract error relevant to this file
            file_error = _extract_file_specific_errors(
                current_output, test_path, max_chars=3000)
            if not file_error:
                file_error = current_output[:3000]

            # Build source context for this test file
            current_content = memory.all_files().get(test_path, "")
            imported_sources = _extract_imported_sources(
                {test_path: current_content}, memory)

            source_ctx = (
                f"#### [FILE]: {test_path}\n```{lang_tag}\n{current_content}\n```\n\n"
            )
            for fp, cnt in imported_sources.items():
                source_ctx += (
                    f"#### [FILE]: {fp}\n```{lang_tag}\n{cnt}\n```\n\n"
                )

            # Optionally inject KB guidance
            kb_instructions = ""
            if kb_context_builder is not None:
                try:
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

            fix_prompt = (
                f"Task: {task}\n\n"
                f"Test file `{test_path}` failed. Fix it so the tests pass.\n\n"
                f"Error output:\n{file_error}\n\n"
                f"Relevant files:\n{source_ctx}"
                f"{kb_instructions}\n\n"
                "You may fix the test file itself OR fix a source file it imports — "
                "whichever is correct.  Do NOT remove any existing tests.\n\n"
                "Output ONLY the complete fixed file(s) using this exact format:\n"
                f"#### [FILE]: path/to/file\n```{lang_tag}\n...full content...\n```"
            )

            try:
                fix_response = coder.llm_client.generate_response(fix_prompt)
                fix_files = executor.parse_code_blocks(fix_response)
                if not fix_files:
                    fix_files = executor.parse_code_blocks_fuzzy(fix_response)
                if fix_files:
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
                break
            _logger.warning(
                "[BulkTest] %s still failing (attempt %d/%d)",
                basename, fix_attempt, _MAX_BULK_TEST_FIX_ATTEMPTS,
            )
        else:
            print(f"  [BulkTest] {basename} could not be fixed after "
                  f"{_MAX_BULK_TEST_FIX_ATTEMPTS} attempt(s).")

    # ── Step 3: Final run-all to confirm everything passes ──
    ok_final, output_final = executor.run_command(base_cmd, cwd=subproject_cwd)
    if ok_final:
        _logger.info("[BulkTest] Final run-all passed.")
        print("  [BulkTest] All tests pass after fixes.")
        return True, ""

    error_msg = (
        f"Bulk test execution failed: some test file(s) still failing "
        f"after per-file fix attempts.\n{output_final[:600]}"
    )
    _logger.warning("[BulkTest] Final run-all failed:\n%s", output_final[:600])
    print("  [BulkTest] FAILED — some tests still failing after fixes.")
    return False, error_msg

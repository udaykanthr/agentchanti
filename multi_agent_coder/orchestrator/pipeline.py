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
    MAX_STEP_RETRIES,
)
from .diagnosis import _diagnose_failure, _apply_fix

_logger = logging.getLogger(__name__)


MAX_DIAGNOSIS_RETRIES = 2   # outer retries: diagnose failure → fix → re-run step

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
        if kb_context_builder is not None:
            try:
                from ..kb.context_builder import ContextBuilder
                kb_ctx = kb_context_builder.build_context(
                    task_description=step_text,
                    current_file=None,
                    max_tokens=getattr(cfg, "KB_MAX_CONTEXT_TOKENS", 4000) if cfg else 4000,
                    language=getattr(project_context, "language", None) if project_context else None,
                )
                if kb_ctx.kb_available or kb_ctx.behavioral_instructions:
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
                    task_description=step_text,
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
                _inline_files = plan_step.inline_code
                executor.write_files(_inline_files)
                memory.update(_inline_files)
                display.step_tokens(step_idx, 0, 0)
                _logger.info(
                    "[PlanStep] Inline code: wrote %d file(s) for step %s: %s",
                    len(_inline_files), plan_step.id,
                    list(_inline_files.keys()),
                )
            else:
                # Extract code graph from kb_context_builder if available
                _graph = code_graph
                if _graph is None and kb_context_builder is not None:
                    _graph = getattr(kb_context_builder, "_graph", None)

                # Look ahead: skip LLM review if a TEST step follows
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

                success, error_info = _handle_code_step(
                    step_text, coder, reviewer, executor,
                    task, memory, display, step_idx, language=language, cfg=cfg,
                    auto=auto, code_graph=_graph,
                    project_profile=project_profile,
                    skip_review=_has_test_after,
                    project_context=project_context,
                    plan_step=plan_step,
                    all_plan_steps=all_plan_steps,
                    kb_context_builder=kb_context_builder)

        elif step_type == "TEST":
            # ── Inline test fast path ──
            # If the planner already provided test code in the plan,
            # write it directly and run — zero Tester LLM calls needed.
            if (plan_step is not None
                    and plan_step.inline_code
                    and len(plan_step.inline_code) > 0):
                display.step_info(step_idx, "Writing inline test code from plan (0 LLM calls)")
                _inline_test_files = plan_step.inline_code
                executor.write_files(_inline_test_files)
                memory.update(_inline_test_files)
                display.step_tokens(step_idx, 0, 0)
                _logger.info(
                    "[PlanStep] Inline test code: wrote %d file(s) for step %s: %s",
                    len(_inline_test_files), plan_step.id,
                    list(_inline_test_files.keys()),
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

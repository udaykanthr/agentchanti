# Changelog

All notable user-facing changes to `agentchanti` land here. This
project follows [Semantic Versioning](https://semver.org): breaking
changes bump the minor (until 1.0), bugfixes bump the patch.

## 0.4.0 — 2026-07-18

A verification-first release: steps now prove themselves against the
real project environment, regressions roll back automatically, and
Django/web tasks are checked by actually rendering pages rather than
trusting the model's claim. Also hardens the process against silent
native crashes.

### Added

- **Per-step verification gates.** A CODE/TEST step passes only when
  its plan-declared `verify:` command exits 0 — not at end-of-run.
  Failures feed back into the fix loop with the real command output.
- **Wave snapshots + monotonic rollback.** Each green wave is
  committed to a machine-managed git snapshot repo, and a later fix
  round that breaks a previously-passing gate is rolled back to the
  last green snapshot instead of being shipped.
- **Django ground-truth verification.** Template reachability and real
  page-render checks; task-pinned URLs and `Acceptance:` lines become
  executable acceptance probes.
- **Django lint gates** for the cross-file-defaults bug family:
  URL-name vs `app_name` namespace, `{% static %}` without
  `{% load static %}`, `@login_required` without `LOGIN_URL`, the
  `tests.py`/`tests` package shadow, and `{% url %}` names that
  resolve nowhere.
- **Web-page task grounding.** Rendered pages are the source of truth;
  task-quoted lines and `Acceptance:` lines become executable probes.
- **Intent-first planning** (`plan_mode: intent`): plans carry goals
  and gates while the agent loop authors the code.
- **Model escalation.** Failed agent-loop steps and recovery loops
  retry on a stronger model, with per-agent provider routing so
  escalation can cross providers.
- **Crash diagnostics for silent process deaths.** A heartbeat file
  plus a startup scavenger that correlates a stale heartbeat with the
  Windows Application event log and writes a post-mortem to
  `.agentchanti/crash.log`; `faulthandler` is armed to capture native
  stacks for the crashes it *can* catch.

### Changed

- SQLite embedding and vector stores now use **per-thread
  connections** (thread-local) instead of a single shared connection.
  A connection shared across threads is officially unsupported and
  could corrupt the interpreter heap — a Windows fast-fail
  (`0xc0000409`) seen as a silent mid-run exit.
- The planner now **recovers from reasoning-burn** (empty/truncated
  output at the token cap → reduced-effort retry) and **detects
  truncated plans**, re-planning for a complete plan instead of
  silently running a stub.

### Fixed

- Planner-emitted path normalization across every layer — doubled
  backslashes no longer poison verification.
- All multi-file content blocks are captured; phantom-path guard for
  inline edits.
- Package self-imports (`from . import x`) are no longer flagged as
  broken imports.
- Subproject venv resolution and cwd-aware verify; Django app-context
  imports are treated as inconclusive, not load failures.
- The config `models:` section no longer silently drops
  `escalation` / `intent` / `analyser` overrides.
- A false monotonic-gate regression that could roll back working code
  (a subproject gate recorded in a form that was never executed).

## 0.1.1 — 2026-04-19

### Fixed

- PyPI project metadata now points at the correct repository. The
  `Homepage`, `Repository`, and `Issues` URLs were shipped with a
  typo (`udaykanth` missing the trailing `r`) in 0.1.0 and all
  404'd. (#19, #20)
- `SECURITY.md` advisory link is reachable again. Same typo as
  above — private vulnerability reporting was silently broken on
  0.1.0. (#19)

### Removed

- Stale "package will be on PyPI once the first tagged release is
  cut" note in the README installation section.
- Reference to the deleted `setup.py` file in `SECURITY.md`.

## 0.1.0 — 2026-04-19

Initial public release on PyPI.

### Added

- `agentchanti` CLI and Python library for multi-agent AI coding
  tasks (Planner → Coder → Reviewer → Tester pipeline).
- Built-in RAG: tree-sitter code graph across 11 languages, local
  SQLite vector store, global knowledge base, error dictionary.
- Support for local LLMs (Ollama, LM Studio) and cloud providers
  (OpenAI, Gemini, Anthropic).
- Structured `PlanStep` format, KB-first command execution, plan-
  aware context injection, step caching, checkpoint/resume.
- Plugin system (`StepPlugin`) for custom pipeline steps.
- GitHub Actions CI (test matrix on ubuntu + windows × py3.10-3.12,
  ruff lint) and release workflow (PyPI trusted publishing via
  OIDC on tag push).

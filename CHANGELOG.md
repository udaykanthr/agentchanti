# Changelog

All notable user-facing changes to `agentchanti` land here. This
project follows [Semantic Versioning](https://semver.org): breaking
changes bump the minor (until 1.0), bugfixes bump the patch.

## 0.5.1 — 2026-07-21

Fixes for a shared failure theme found by running 0.5.0 on a fresh
Python/pygame task: acceptance gates that no amount of correct work
could turn green.

### Fixed

- **Subproject prefix no longer breaks root-relative verify gates.**
  A gate like `python -m unittest discover -s game` was prefixed with
  `cd game && `, making its path argument resolve to `game/game/` —
  unpassable by correct code (the escalation model "satisfied" it by
  creating a duplicate nested test package). Commands that reference
  the subproject itself by path or module (`-s {sub}`, `{sub}/tests`,
  `python -m {sub}.main`) now run from the repo root as written.
- **`pip install --yes` is sanitized.** Planners hallucinate the
  apt/conda flag onto pip, which rejects it (exit 2). The flag is
  stripped from pip-install segments before the first run; apt/conda
  segments keep their legitimate `-y`.
- **Recovery gates accept a flag-variant success.** A failed CMD step's
  recovery gate re-runs the original command verbatim; when that
  command is malformed, the gate stayed red even after the loop ran the
  corrected command successfully (observed: pygame installed twice,
  run still failed). Recovery loops now accept a success of the same
  command differing only in flag tokens. CODE/TEST verify semantics
  are unchanged.

## 0.5.0 — 2026-07-20

A token-efficiency and reliability release, hardened through ten
consecutive benchmark runs on a real scaffold-and-build task. Best-case
cost dropped ~5× (13.9k tokens sent, zero recovery loops, tests passing
first try — parity with single-shot editors while keeping the full
test + build verification pipeline). Every failure class encountered is
now covered by a deterministic guard plus a regression test.

### Added

- **Prompt-cache accounting.** OpenAI cache hits are read from
  `usage.prompt_tokens_details.cached_tokens` and reported everywhere:
  the final log line shows gross vs cached vs full-price input
  (`sent=X [cached=Y (Z%), full-price=W]`), the live panel shows the
  cached share, cost estimates bill cached input at the discounted rate
  (override per model via `pricing.cached_input`), and the library's
  `TaskResult.token_usage` gains a `"cached"` field.
- **`agentchanti --version`** — reports the actually-installed
  distribution version via `importlib.metadata`.
- **Agent-loop preload + earlier nudge.** The loop's opening message
  pre-loads the step's existing target files (capped) so models stop
  burning turns on `read_file` round-trips; the read-only "act now"
  nudge fires after 2 idle turns instead of 3.
- **Adaptive inline-code budget.** Content-mode plans cap total inline
  code (~150 lines); beyond it, steps carry descriptions + `verify:`
  and are implemented against the real project state. Truncated plans
  whose parsed steps are structurally complete are salvaged instead of
  triggering a full re-plan.
- **Deterministic vitest bootstrap.** DOM test suites about to run
  without a vitest config get a jsdom-enabled `vitest.config.js` +
  setup file written from a fixed template (no LLM), with missing
  testing deps installed. Test-infra files (vitest/jest config + setup)
  are exempt from the fix loop's source-protection guard.
- **Unsolicited-test gating.** When the raw task doesn't ask for tests,
  the pipeline no longer auto-generates per-file coverage tests
  (plan-declared TEST steps still run).
- **Silent no-op CMD guard.** A command that reduces to
  cd/mkdir/parens while the plan declares concrete `produces:` files is
  failed at that step, so recovery runs where the problem is.
- **Repeated-text test-query rules.** Planner and tester prompts now
  require landmark-scoped Testing Library queries
  (`within(getByRole('banner'/'contentinfo'))` / `getAllBy*`) — brand
  text legitimately appears in header and footer, and singular queries
  were the most frequent self-inflicted test failure.

### Fixed

- File-creation `echo`-chain CMD steps (all observed Windows variants:
  spaced, compact `>`/`>>`, caret-escaped, parenthesized) are
  reclassified into CODE steps that write the file directly — cmd.exe
  chokes on `[`/`{` in echoed content.
- Multi-target `edit:` steps: the i-th bare `edit:` block now maps to
  the i-th target file (previously all blocks landed on the first
  target, fusing two files' content); full-file promotion accepts
  exactly one REPLACE block, never a merge.
- Edits targeting files that don't exist at plan time are converted to
  full-file writes (single complete REPLACE) or routed to the grounded
  path — no more FIND-matching against content the planner never saw.
- JSON syntax gates on inline patches, promotions, and the agent-tool
  `edit_file` (tsconfig JSONC exempt) — a trailing comma minimal-diffed
  into `package.json` previously broke every later npm invocation.
- Multi-line CMD steps joined with `&&` no longer fail on repeated
  `cd <dir>` segments (each plan line assumed the project root).
- `validate_plan` matches imports against glob `produces:` entries
  (`app/src/*`), so inline code importing scaffold-created files is no
  longer cleared as dangling.

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

# Changelog

All notable user-facing changes to `agentchanti` land here. This
project follows [Semantic Versioning](https://semver.org): breaking
changes bump the minor (until 1.0), bugfixes bump the patch.

## 0.6.5 — 2026-08-07

Closes the gap left open by 0.6.4's FileMemory work: memory was made to
agree with the filesystem, but nothing stopped the write reaching the
filesystem in the first place.

### Fixed

- **The agent loop cannot replace a manifest it did not write.** The
  classic writer has always refused to overwrite a dependency manifest it
  did not create (`Executor.write_files`); the loop's `write_file` went
  straight to disk with no such check. A model regenerating
  `requirements.txt` or `package.json` from memory could therefore replace
  the project's real one with a shorter version, dropping dependencies —
  and every later step would build and test against a different dependency
  set than the project actually has, which is the kind of failure that
  looks like a code bug for a long time.

  The test is create-versus-overwrite, not existence: creating a manifest
  is legitimate and common (5 of 8 benchmark runs did it), and a run that
  wrote one must be able to update it. Within a step the instance tracks
  what it created; across steps the answer comes from FileMemory, because
  `build_step_tools()` builds a fresh `AgentTools` per step. Only a
  manifest that existed before the run began is refused, and the refusal
  names `edit_file` — exact-match and single-occurrence, so it cannot
  silently discard the rest of the file — as the way to add a dependency.
  Ordinary source files are unaffected.

## 0.6.4 — 2026-08-07

The two defects observed alongside the 0.6.3 fence bug, neither of which
caused that failure but both of which would hide or repeat one — plus four
more found by a third A/B iteration over the Pac-Man task, where both modes
passed (loop 4:45 / $0.43, classic 16:25 / $1.34) and every artifact was
checked against an independent wall-invariance drive.

### Fixed

- **A halted pipeline exits non-zero.** The failure branch logged "Pipeline
  failed", wrote the HTML report, and fell through returning `None`, so the
  process exited 0. A benchmark run that stopped at step 11 of 12 after three
  failed diagnosis attempts, having never written its tests, still returned
  `EXIT=0` — indistinguishable from success to CI, a `&&` chain, or any
  harness reading `$?`. `_main_impl` now returns 0/1 from the
  `pipeline_success` flag it already had and `main` exits with it. Returned
  rather than raised, so watcher and executor cleanup still run first.
  `--version`, the `kb` subcommand and an aborted prompt return `None` and
  stay 0; SIGINT stays 130; an unhandled exception is still re-raised.
- **The fuzzy parser no longer truncates at, or invents files from, inner
  fences.** `parse_code_blocks_fuzzy` searched every pattern's body with an
  unanchored non-greedy `` (.*?)``` ``, so a block ended at the first ```
  anywhere. Two consequences. It returned the same truncated document the
  strict parser did on 0.6.3's README — and then, because Pattern 3 takes its
  filename from the line above a fence, the document's own usage examples
  kept matching: a README's install and run instructions were emitted as
  phantom `requirements.txt` and `main.py` files to write. Separately, an
  unanchored search ends a diff block on its own `+``` ` line, so a diff
  touching any Markdown file was cut at the first fence it added. Fence
  detection is now anchored to line starts and shares one span helper with
  the strict parser; the patterns walk top-level blocks only, and Pattern 3
  resumes past each block it takes so a document's interior can never be
  re-read as further filenames.
- **Sanitising a verify gate no longer corrupts `set VAR=value` on
  Windows.** `_declared_verify_cmd` split a gate on `&&`, stripped every
  segment and rejoined with `" && "` unconditionally — rewriting whitespace
  even when it dropped nothing. On `cmd.exe` that is not cosmetic:
  `set VAR=dummy && next` assigns `"dummy "`, trailing space included. A
  planner wrote `set SDL_VIDEODRIVER=dummy&& ...` precisely to avoid that;
  the space put back made SDL look for a display driver named `"dummy "`,
  failed the gate, and cost a diagnosis round that "fixed" it by adding an
  environment-scrubbing function to the *generated* project's `main.py` —
  harness damage shipped in the delivered artifact. The repair now runs
  unconditionally on the resolved gate — the planner writes the spaced form
  about as often as this module reintroduced one, and a first, narrower fix
  that only repaired chains the sanitiser had reassembled left the
  planner-authored case broken (caught by the next benchmark run: the same
  gate failed three times in a row while the generated `main.py` was
  correct all along). Non-assignment segments keep their spacing, so
  `cd app && npm test` is unaffected.
- **A manifest a step just created is tracked in FileMemory.** The
  protected-basename guard exists to stop a hallucinated replacement
  clobbering a real manifest, but it tests `os.path.isfile()` — true the
  moment the file is written — so a manifest the run had just created
  looked pre-existing and was dropped. The content then stayed invisible to
  dependency checks, context injection and the checkpoint for the rest of
  the run, while the log claimed a skip that protected nothing. Fixed on
  both paths: the agent loop's `AgentTools._record` runs *after* its write
  has landed, so it now records unconditionally; the classic path
  distinguishes create-from-overwrite using the pre-write existence set it
  already computed. A genuinely pre-existing manifest is still protected on
  disk and in memory. Seen in 5 of 8 benchmark runs.
- **pytest is not installed for a runner that never runs.**
  `_ensure_pytest_available` ran before the plan-declared suite gate, and
  that gate passed on every run of a unittest-based project, so each run
  paid a pip install and a network round-trip for a runner never invoked.
  Moved to the fallback path that actually needs it.

### Benchmarks

- **`verify_dt_invariance.py` no longer reports a working game as broken.**
  Two of six artifacts in one session got `VERDICT: FAIL - game raised ...`
  from exceptions raised by the harness's *own* probes rather than by the
  game: `set_direction(1, 0)` against a name-based API raised `ValueError`,
  and `pixel_to_tile(px, py)` against a signature taking one sequence raised
  `TypeError`. Derivation is now separated from the drive loop — anything
  raised while deriving an artifact's vocabulary is a refusal (exit 2), and
  only the drive loop can produce a FAIL; the probes try each known
  convention instead of assuming one. Two selection bugs surfaced while
  fixing it: a two-argument `can_move`/`walkable_neighbor` takes
  `(tile, direction)`, not `(col, row)`, and `is_walkable` tied with
  `is_position_walkable` on every ranking term so `dir()` ordering decided
  the winner — "position" usually means pixels, so it answered in the wrong
  coordinate space. Ranking now prefers the plainest canonical name and is
  deterministic. All six artifacts verify: 6 PASS, matching six independent
  hand-written drives (before: 3 PASS, 2 false FAIL, 1 refusal). One of
  those original passes was luck — it had been using
  `get_walkable_neighbors`, which returns a *list*, as a boolean wall test.

## 0.6.3 — 2026-08-07

One fix, found by A/B benchmarking `agent_loop` on/off over the Pac-Man task.
The classic path halted at step 11 of 12 on a bug in the extractor, not in
the model's output — and because the bug is deterministic, no retry could
ever escape it.

Two full iterations were run per mode, folder cleared between every run.
Before the fix: classic halted (8:01 / $0.800), loop passed (5:29 / $0.524).
After: classic passed 16/16 (9:05 / $0.922), loop passed 8/8 (5:23 /
$0.484). All four artifacts were checked against an independent
wall-invariance drive as well as their own suites. Loop read 35-42% of its
input from the prompt cache in both runs; classic read 0% in both.

### Fixed

- **Generated Markdown is no longer truncated at its first inner code
  fence.** `Executor.parse_code_blocks` matched a file's body with a
  non-greedy group (`` (.*?)\n``` ``), which ends at the FIRST fence line
  inside the block. Correct for source files, wrong for any file whose own
  content contains fences: an 808-token README was written as 15 lines /
  417 bytes, cut off mid-sentence at "install the required dependency:",
  because every command the step's verify gate looked for lived inside a
  fence. The gate failed correctly, but all three diagnosis attempts
  regenerated the same document, hit the same truncation, logged "previous
  fix changed nothing", and the pipeline halted having never written the
  tests. Extraction now splits the response by `#### [FILE]:` marker and
  requires a closing fence at least as long as its opener, so a model that
  wraps ``` content in a ```` fence parses exactly. Where both fences are
  three characters the nesting is genuinely ambiguous and the file's own
  format breaks the tie — outermost fence for `.md`/`.rst`/README-likes,
  first closer for source files, so a follow-up example block is still not
  swallowed into the file above it. The same rerun that halted before now
  produces a 158-line README with its 12 inner fences intact and passes the
  gate on the first attempt. The `agent_loop` path was never affected:
  `write_file` receives content as a structured tool argument and no fence
  parsing is involved.

## 0.6.2 — 2026-08-07

Five fixes from continued A/B benchmarking of `agent_loop` on/off over the
Pac-Man task with adversarial delta-time invariants. The theme is the
harness: in every case the model did work the pipeline then discarded,
hid, or graded against the wrong thing. Each fix carries a regression test.

Benchmarked after these changes, one run per mode, both passing their own
suite and an independent wall-invariance check: loop 5:01 / $0.4778 / 49%
prompt-cache hit; classic 7:55 / $0.5872 / 3%. This is the first classic
run on this task to pass ground truth.

### Fixed

- **A failing command is routed by its error, not its shape.** A step whose
  gate was a bare console script (`ruff check messy.py`, not on the child
  process's PATH) was told "the failure is in the code — edit the source",
  though the source was already correct. Routing now matches the failure
  signature itself (not recognized / command not found / No module named /
  executable file not found / cannot find the path), so any command failing
  that way gets environment advice, including invoking an installed-but-
  unPATHed tool as `python -m <tool>`. Assertions and tracebacks keep the
  original wording verbatim. The run this came from reported failure on a
  task ground truth showed had PASSED, after 22 turns and 61.9k tokens.
- **The repeat-command nudge survives across attempts.** The streak reset
  every attempt, so a command re-run once per attempt across loop →
  escalation → recovery never tripped it. It is now seeded from the attempt
  journal; an edit within the attempt still clears it.
- **A step cannot shadow a dependency whose install just failed.** After a
  `pip install pygame` failure, one run wrote a local `pygame/` package
  whose own docstring said it performs no real rendering — shadowing the
  real library, so every later step and test would have passed against a
  no-op renderer. Blocked only when an install of that exact distribution
  failed in the same step, and only for a new top-level module or package.
- **A rejected chunk edit is no longer re-parsed as whole files.** Pattern 5
  attributed indented method bodies to modules on a single symbol match,
  writing three method bodies over three real files. A complete file passes
  the new check trivially.
- **A test suite the project's own runner cannot find is not done.** A run
  shipped a false green: six steps verified early, the pipeline reported
  success, and the delivered project answered `python -m unittest -v` with
  "Ran 0 tests" — nothing had made `tests/` a package. A path-scoped gate
  proves the file runs and says nothing about discovery, so the TEST step
  now re-runs its own gate with the file scoping stripped and fails when
  that collects nothing.
- **A CONFIG_BUG fix may only touch config files.** Observed rewriting five
  source modules under a triage meaning "the test environment is
  misconfigured", taking a suite from 1 failure in 61 tests to 4 failures
  and 3 errors in 64. A repeat CONFIG_BUG verdict that changed nothing now
  routes to the source path.
- **Compound commands are never token-rewritten.** `pip install -U pygame
  && pip freeze > requirements.txt` was reassembled into `pip install -U
  requirements.txt && freeze`, became a step's gate, and killed a run at
  step 1 of 9 with an otherwise correct plan.
- **The plan's own test runner is honoured.** A task whose acceptance was
  `python -m unittest -v` had pytest installed and used anyway. Relatedly,
  `unittest discover -v tests/test_x.py` read the path as a start directory
  and ran zero tests, recording a game with 61 of 63 tests passing as 0/2.
- **Windows: `| head -N` is dropped rather than failing the command.** head
  does not exist there, so the pipeline died before the command ran. pip
  self-upgrades now route through `python -m pip`.

### Performance

- **A loop step ends as soon as its gate goes green.** Steps averaged 7.6
  turns against a max of 8, so they essentially never finished early, and
  the whole conversation is resent every turn (~27k prompt tokens on late
  turns against ~2k at the start). The gate is now checked as soon as an
  edit lands — one subprocess, no tokens. On the same 7-step plan: 305,306
  tokens rather than 821,235 (-63%), 282s rather than 517s, avg 4.4 turns
  rather than 7.6, with the artifact still passing its suite and the
  independent dt-invariance check.
- **The Anthropic chat path asks for a prompt cache.** `agent_loop` keeps
  its system prompt byte-identical *for* prompt caches, but the request
  never set `cache_control`, so a run billed 1,224,846 tokens sent and 0
  cached. Three breakpoints take the hit rate to 65% and cut full-price
  input per loop turn from 18,688 to 7,492. The billed prompt is now
  reported as input + cache_read + cache_write, since `input_tokens` alone
  makes a working cache look like a shrinking prompt.
- **A weak `verify:` is repaired in place instead of regenerating the plan.**
  A re-plan cost 8,214 sent / 3,219 received to fix one line and churned the
  step decomposition; the repair costs ~500 tokens. Shell-level assertions
  count as teeth; `assert True` does not.
- **An empty response that billed output tokens is treated as a reasoning
  burn even when `finish_reason` is "stop".** Only cap-hits were detected,
  so a burn wearing a clean stop fell through to the anti-`<think>`
  preamble, which cannot help server-side reasoning and mutates the prompt,
  losing prompt-cache reuse. Gemini's thinking cap is now persisted in
  `~/.agentchanti/effort_floors.json` alongside the OpenAI floor,
  namespaced so a numeric budget and an effort string cannot be confused.
- **A failing command re-run unchanged is nudged, then `run_command` is
  withheld.** Commands compare with `cd` prefixes and output pipes
  normalised away, since the retry usually arrives redressed rather than
  repeated. One step spent turns 4–7 re-running one gate from four
  directories (~38k tokens) while the defect went untouched.
- **A chunk edit whose line range no longer matches splices the named
  symbol** instead of discarding the fix and spending an attempt on nothing.

### Benchmarks

- `verify_dt_invariance` no longer refuses a game it can drive: a 3-arg
  `is_rect_wall_free` outranked `is_walkable` purely by spelling "wall" and
  carried the opposite polarity. The probe now reads arity and polarity
  before it trusts a name.

## 0.6.1 — 2026-08-05

Correctness and context-hygiene fixes found by an A/B benchmark of
`agent_loop` on/off over eight runs of a Pac-Man task with adversarial
delta-time invariants. Every fix carries a regression test that fails on
the previous code.

### Fixed

- **A package initializer no longer runs before the module it re-exports.**
  A plan gave `src/__init__.py` and `src/pacman.py` the same `exports:` but
  only `depends:1.1`, so they shared a wave. The initializer ran with
  nothing to import and satisfied its gate — `assert all(x is not None
  ...)`, which is true of `class Game: pass` — by writing four stub
  modules. Every gate, the smoke test and 8/8 unit tests passed while
  `from src import Game` returned an empty shell. The missing edge is now
  inferred from `exports:` overlap; `fix_import_dependencies` could not
  help because it derives edges from `imports:`, and that plan declared
  `imports: none`.
- **A gate that contradicts the task no longer costs the work that
  satisfies it.** A plan gated a step on `assert p.can_move()` against a
  `can_move()` meaning "is currently moving", so it demanded a
  freshly-built player already be in motion — while the task demanded
  "2000+ frames without the player moving". Diagnosis made the player
  auto-start, the suite's idle test then failed, its fix regressed the
  gate, and rollback discarded the correct fix. When a test suite is green
  and only inline gates are red, the conflict is now reported and the
  working tree kept. The run still fails: an unresolved red gate is never
  reported as success.
- **KB docs are scoped to the language they are for.** `seeder.py` passed a
  hardcoded `"all"` as the language of every doc it emitted, so the filter
  in `GlobalKBStore` could never exclude anything — "React Component Export
  Instructions" was injected into every step of a Python/Pygame run,
  alongside Django and Vitest material, at 2.8k–3.9k tokens per step. Docs
  now declare their real language, `language:` accepts a comma-separated
  list (`"javascript, typescript"`), and a doc naming a framework the task
  never mentions is dropped — which the language filter alone cannot do,
  since the Django docs are correctly tagged `python`. Measured on the same
  task: foreign-stack docs injected per run went from 5 to 0.
- **Streaming calls survive a reasoning model's silence.** The OpenAI
  streaming POST used a 120s read timeout, which `requests` applies to the
  gap *between* bytes; a reasoning model emits nothing while it thinks. A
  planner call died at exactly 120s, was retried (re-billing the prompt)
  and then downgraded to the non-streaming path — up to 3× the tokens of
  one call. `ollama.py` already carried this lesson; this client did not.
- **No setup-guide web search for a bare language.** The blank-project
  pre-seed asks for frameworks and routinely gets the language back too
  ("Python, Pygame"), spending a web fetch plus a summarisation call
  (~2.7k tokens) to learn `pip install`. Languages are skipped; frameworks
  are still searched.

### Compatibility

`language:` in KB frontmatter now accepts a comma-separated list. Registry
**v1.5.0** uses it to scope React/Vite docs to `"javascript, typescript"`.
Older agentchanti releases compare the field by exact string and will not
match a list, so **0.6.1 or newer is required to see those docs** — pin the
registry version on older clients.

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

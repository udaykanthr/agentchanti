# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentChanti is a multi-agent AI coding CLI tool (`agentchanti` command) and Python library (`agentchanti` package). It takes a plain English task description and autonomously plans, codes, reviews, and tests the solution using a pipeline of specialized LLM-powered agents. Supports local LLMs (Ollama, LM Studio) and cloud providers (OpenAI, Gemini, Anthropic).

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_flow.py -v

# Run the CLI
agentchanti "your task" --provider ollama --model deepseek-coder-v2:16b

# Run via library API
python -c "from agentchanti import run_task; run_task(task='...', auto=True)"
```

## Architecture

### Agent Pipeline

The system runs a sequential pipeline: **Planner -> Coder -> Reviewer -> Tester**. Each agent (`agentchanti/agents/`) extends `Agent` base class and calls `self.llm_client.generate_response(prompt)`. The pipeline is orchestrated in two places:
- **CLI path**: `orchestrator/cli.py:main()` — parses args, builds agents, runs the pipeline
- **Library path**: `api.py:run_task()` — programmatic entry point returning `TaskResult`

Both paths share the same execution engine in `orchestrator/pipeline.py`.

### Step Execution Flow

1. **PlannerAgent** generates numbered steps from the task description
2. `pipeline.py:build_step_waves()` groups steps into dependency waves for parallel execution
3. Each step is classified by `classification.py:_classify_step()` via LLM into: **CMD**, **CODE**, **TEST**, or **IGNORE**
4. Step handlers in `orchestrator/step_handlers.py` execute each type:
   - **CMD**: Runs shell commands via `Executor.run_command()`
   - **CODE**: Coder generates code -> Reviewer checks -> retry loop (up to 3x) -> diagnosis on failure
   - **TEST**: TesterAgent generates tests -> runs them -> Coder fixes failures
5. `orchestrator/diagnosis.py` handles failure analysis and auto-fix

### Language Detection (agentchanti/language.py)

Auto-detects project language by scanning file extensions (`detect_language()`) or parsing task keywords (`detect_language_from_task()`). Maps languages to test frameworks via `TEST_FRAMEWORKS` dict. **Known issue**: defaults to Python/pytest when language is `None`, which causes incorrect test generation for non-Python projects (e.g., TypeScript projects get Python tests). The TesterAgent at lines 10-12 and 41-44 hard-defaults to Python when `language` is None.

### LLM Client Layer (agentchanti/llm/)

`LLMClient` base class with provider implementations: `OllamaClient`, `LMStudioClient`, `OpenAIClient`, `GeminiClient`, `AnthropicClient`. All expose `generate_response(prompt) -> str` with retry and streaming support.

Chat-native entry point: `chat(messages, tools=None) -> ChatResponse` (types in `llm/chat_types.py`: `Message`, `ToolDef`, `ToolCall`, `ChatResponse`). Ollama (`/api/chat`), OpenAI (`/chat/completions`) and Anthropic (Messages API) implement it natively with structured tool calling (`NATIVE_CHAT = True`); other providers fall back to flattening the conversation into a text prompt via `flatten_messages()`. Models that reject tools at runtime raise `ToolsNotSupportedError` and are downgraded to the text path for the session. Check availability with `client.supports_tools()`.

### Agent Tools (agent_tools.py)

`AgentTools` is the agent-computer interface for tool-calling loops: six `ToolDef`s (`list_files`, `read_file`, `write_file`, `edit_file`, `run_command`, `search_code`) scoped to a project root, backed by `Executor`, the KB `Searcher`, and `FileMemory`. `execute(ToolCall) -> str` never raises (errors return as strings for the model); `execute_all()` wraps results as `role="tool"` messages. `edit_file` is exact-match single-occurrence replace with `ast.parse` validation for Python; paths escaping the project root are rejected.

### Agent Loop (orchestrator/agent_loop.py)

Default execution path for CODE/TEST steps when the provider supports native tool calling (`agent_loop: true` by default; set `false` to use the classic generate→review→retry pipeline, which also remains the automatic fallback for providers without tool support). `run_agent_loop()` runs the step as a bounded tool-calling conversation (`agent_loop_max_turns`, default 8): stable byte-identical system prompt (KV-cache friendly), model edits/runs via `AgentTools`, observes real output, self-corrects. Exit rules: the final turn withholds tools to force a text summary; a `verify_cmd` (from `verify_cmd_for_language()`: pytest for Python, `npm test` when package.json defines it, `go test ./...`) must pass before the model's "done" claim is accepted, and passes on exhaustion still count as success. Returns the same `(success, error_info)` contract as the classic handlers; gate is at the top of `_handle_code_step`/`_handle_test_step` via `agent_loop_enabled()`.

Failure recovery: `run_recovery_loop()` gives one bounded loop attempt when a CMD step's planned command fails (`_handle_cmd_step`) or when any step reaches `_run_diagnosis_loop` — with the loop enabled it replaces the diagnose→fix→re-run machinery entirely; `RECOVERY_FAILED_MARKER` in the error prevents double attempts.

Telemetry: every loop run records turns/tool-call counts/outcome/recovery-flag (`get_loop_stats()`, `loop_stats_summary()`); the CLI logs a `[AgentLoop] session:` summary line at the end of each run.

### Benchmarks (benchmarks/)

A/B harness comparing `agent_loop` on vs off over the task set in `benchmarks/tasks.py`. Ground truth is per-task `success_cmds` run in the isolated workdir, independent of the pipeline's own claim. Run from repo root: `python benchmarks/run_ab.py --config <yaml-with-keys> [--tasks id1,id2] [--modes on,off] [--truststore]`. Results land in `benchmarks/results/*.json`. Not part of pytest — it spends real API tokens.

`benchmarks/verify_dt_invariance.py <project-dir>` is an independent ground-truth check for generated tile-maze games: it drives the game at several timestep profiles and asserts no entity ever occupies a wall tile, catching games that only hold together at a fixed 1/60 dt. Exit codes are **0 PASS, 1 FAIL, 2 could-not-verify** — the third is deliberate, because generated projects share no vocabulary and a refusal must never be recorded as a failure.

### Ghost Shadow (orchestrator/ghost.py)

Read-only reconciliation of the plan's *declared postconditions* against the real tree — no LLM calls, no commands run, no verdict changed. `GhostPlan.build()` is called in `cli.py` once the plan is final (after blind-edit routing / dependency fixes / verify repair) and before the first step runs, so its file hashes are a true pre-run baseline. Steps' `target:`/`exports:`/`imports:`/`verify:` become interned `Expectation` nodes (`EXISTS`, `TOUCHED`, `PARSES`, `EXPORTS`, `IMPORT_EDGE`, `PKG_PRESENT`, `GATE_PASSED`) shared across the steps that declare them; verdicts are four-valued (`HOLDS`/`VIOLATED`/`UNKNOWN`/`INAPPLICABLE`) and fold over an append-only observation journal. Resolved per wave next to `_reconcile_plan_graph`, reported once at the end of the run under `[Ghost]`.

`PLAN_ANCHORS` is the step-drift check, and the reason the ghost can repair as well as report. Where the planner supplied a file's body (`PlanStep.inline_code`, i.e. content mode), the structural names that body declares — CSS selectors, Python defs/classes/constants, JS exports — must still be present in the written file. A smaller model plans correctly and then deviates while executing an individual step, and nothing else notices: the file exists, parses, and often satisfies a gate that never names the missing piece.

Run-level findings sit alongside the per-step ones: `unplanned-write`, `filename-case-mismatch`, and `tests-never-collected` — test files the run's own acceptance command will never collect, decided statically from the AST (`unittest` collects only `TestCase` subclasses; `pytest` also collects module-level `test_*` functions). It spans planned targets *and* untracked writes, because the four modules that exposed it were written by the agent loop and declared by no step. Silent when the runner cannot be identified, when the file will not parse, or when the runner is pytest.

`degenerate-long-run` catches the one defect every gate is blind to: a suite that satisfies "run >= 2000 frames and assert the invariants hold" while simulating fifty. `Game.update()` opens with `if self.state is not PLAYING: return`, so once the ghosts catch a parked player every later iteration is a no-op — and the invariant assertions keep passing, against a frozen state, all the way to 2000. Measured on four real artifacts: 50/2100, 246/2100 live frames on two runs whose suites were fully green.

Proved structurally, from two halves that must both hold: the advance method early-returns on a state guard (`guarded_advance_methods`, which scans the whole guard prologue — a real `update()` put `if not math.isfinite(dt): raise` first), and a long `range(N)` loop calls it without ever pinning the state to a live value. Both come from the project's own source, so nothing is hardcoded about what a "game" or a "frame" is. Guards are keyed by **class**, and each call's receiver is resolved back to the class it was constructed from (`_class_bindings`, covering both `game = Game(seed)` and a `self.game = Game()` fixture): `Game.update` is guarded while `Player.update` and `Ghost.update` are not, and a suite driving the player directly for 800 frames skips no frames at all.

The guard itself is read loosely enough to survive real code: the whole prologue is scanned, the guard body may do work before it leaves (one real guard called `self.assert_invariants()` first — precisely why its frozen frames kept passing), and an `or` counts (`if self.state != PLAYING or dt == 0.0: return` leaves whenever the state dies) while an `and` does not.

The silences carry as much of the design as the finding. A tautology is deliberately not a pin — `assertIn(game.state, (PLAYING, WIN, GAME_OVER))` admits the terminal states it was meant to exclude, and a real run wrote exactly that inside a 2100-iteration loop. Silent when the test pins the state (in the loop *or* after it), breaks out, handles termination — `if game.state is GAME_OVER: game.restart()` kept one suite live for 2000 of 2000 frames across six restarts — or extends the run's lifetime (`game.spawn_protection_timer = 1_000_000.0` before the loop). Branching on the *live* state is not handling termination: `if game.state == PLAYING and frame % 11 == 0: send_input()` only gates input, and the loop it appeared in still ran 44 of 650 frames.

Every one of those rules was written against a measured artifact, and the check is validated by replaying all six: it flags the four suites measured at 50/2100, 246/2100, 171/2001 and 44/650 live frames, and stays silent on the three measured at 100%. Two rounds of live running produced two false positives and two false negatives before it matched; re-validate against the archive before changing it.

### Ghost Heal (orchestrator/ghost_heal.py)

Deterministic repair of what the shadow finds — no LLM call. Runs per wave (so later steps benefit) and once at the end, under `[GhostHeal]`. Governed by one rule: **never invent content; freely restore content the plan already specified.** Those are different things — writing an empty `.site-header {}` to satisfy a check makes a real defect undetectable, but writing the `.site-header` rule *the planner put in `inline_code`* invents nothing and enforces a decision already made.

Healers: `PKG_PRESENT` → install into the interpreter `Executor._venv_bin_dir()` resolves (never bare `python`, which is the original bug); `EXISTS` → restore from the plan's body, or create an empty `__init__.py` (empty *is* its correct content); `IMPORT_EDGE` → add the import, only when both files are Python in the same directory, exactly one consumer is a real module, and every declared symbol demonstrably exists in the source; `PLAN_ANCHORS`/`EXPORTS` → restore the plan's body when the written file is a strict regression.

Refusals are as important as repairs. Restoration is declined when the written file declares anything the plan's body does not — the step may have added real work, so the conflict is reported instead of clobbered. `PARSES`, `TOUCHED` and `GATE_PASSED` have no healer at all: no source specifies what they should contain. Every heal is verified by re-resolving its expectation, and source edits are snapshotted and reverted if the gap does not close. Flags: `ghost_heal` (default true), `ghost_heal_source_edits` (narrows to environment actions only).

`PKG_PRESENT` is the one check that looks past the repo at the environment: for any manifest the plan targets (`requirements.txt`, `package.json`) every declared runtime dependency must be present in the environment the app will actually run in — the venv `Executor._venv_bin_dir()` resolves, or `node_modules`. Purely a filesystem comparison, no subprocess. It exists because a plan step wrote `python -m venv venv && python -m pip install pygame`, which creates a venv but never activates it, so pygame installed into the pipeline's interpreter instead; every gate passed (the game modules were headless and imported no pygame) and only `main.py` needed it, crashing at launch under the project venv. No venv / no `node_modules` resolves `UNKNOWN` — a project on the ambient interpreter must never be accused.

Surfaces six disagreement classes the rest of the pipeline is blind to: `violated-*` (a step claimed done while a declared postcondition is false), `planned-untouched` (target's bytes never changed), `unplanned-write` (a file no step declared), `no-checkable-claim` (a step whose expectations are all tautologies — the plan-level analogue of `gate_integrity`), `degenerate-long-run` (a long assertion loop that stops simulating partway and asserts a frozen state), and `failed-but-clean` (run marked failed while everything declared holds). Disable with `ghost_shadow: false`.

### Key Subsystems

- **Config** (`config.py`): Priority resolution: CLI args > env vars > `.agentchanti.yaml` > defaults
- **Executor** (`executor.py`): File I/O, shell command execution, plan/code-block parsing
- **FileMemory** (`orchestrator/memory.py`): Thread-safe tracking of files written during a run
- **Knowledge Base** (`kb/`): Local code graph (tree-sitter based), project orientation, context injection for agents. Subcommand: `agentchanti kb ...`
- **Editing** (`editing/`): Diff-aware code editing with fuzzy matching and syntax validation
- **Plugins** (`plugins/`): Custom step handlers (LINT, DEPLOY, etc.) via `StepPlugin` base class, discovered from config or setuptools entry points
- **Step Cache** (`step_cache.py`): Hash-based LLM response caching with configurable TTL
- **Checkpoint** (`checkpoint.py`): Save/restore pipeline state for resume after interruption
- **Git Utils** (`git_utils.py`): Checkpoint branches, commit on success, rollback on failure

### Test Framework Mapping

Defined in `language.py:TEST_FRAMEWORKS`. The TesterAgent (`agents/tester.py`) builds language-specific prompts:
- `_python_test_rules()` for Python/pytest
- `_js_test_rules()` for JavaScript/TypeScript (Jest-oriented, no Vitest support yet)

The step handler `_handle_test_step()` in `step_handlers.py` detects JS project environment (ESM vs CJS) and auto-installs test runners.

## Configuration

Settings file: `.agentchanti.yaml` (project root or home directory). Key sections: `models` (per-agent model overrides), `prompts` (agent prompt suffixes), `openai`/`gemini`/`anthropic` (cloud API keys), `kb` (knowledge base), `editing` (diff-aware editing), `plugins`.

## Entry Points

- CLI: `agentchanti` -> `agentchanti.orchestrator.cli:main` (defined in `setup.py`)
- Library: `from agentchanti import run_task, TaskResult`
- KB subcommand: `agentchanti kb ...` -> `agentchanti.kb.cli:kb_main`

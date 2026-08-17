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
5. `orchestrator/diagnosis.py` handles failure analysis and auto-fix, driven by `_run_diagnosis_loop` (see **Diagnosis Loop** below for what it keeps and what it ships)

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

### Diagnosis Loop (orchestrator/pipeline.py `_run_diagnosis_loop`)

The classic path's failure recovery: up to `MAX_DIAGNOSIS_RETRIES` (3) rounds of diagnose → apply fix → re-execute the step, the last of which escalates to `models.escalation`. On that final attempt `_apply_fix` also stops enforcing the diff guard's **change-ratio** threshold (the full-file-replacement block still applies): rejecting there means the step fails regardless, so the threshold only buys never learning what the stronger model said — measured, a 70-second `gpt-5.6-sol` fix rejected at 63% against 40%, after which the test-only fallback produced nothing twice. The downside is bounded by the best-snapshot restore below, which did not exist when that threshold was written. Reached only with `agent_loop: false` or on a provider without tool support — `run_recovery_loop()` replaces it entirely otherwise.

It is governed by one rule the loop spent two incidents learning: **an error signature says whether two failures are different, never which one is worse.** Both halves of that blindness shipped a broken artifact from a run that had already produced the fix. Inequality kept regressions — two chunk edits took a suite from 4 failures to 19 errors to 39 errors, each adopted as the new baseline because "the error moved on", and the final restore shipped the 39-error state, a `Game` that could not be constructed. Equality discarded a correction — a test-fix loop rewrote a gate-verified `game.py` so that `if entity.at_center:` became `entity.at_center()` on a `@property`, every `advance()` raised `TypeError`, diagnosis attempt 1 root-caused it exactly, and the loop reverted that one-character fix under *"previous fix changed nothing"*. Every wave commit in the project's own git was clean; only the working tree was broken. Re-applying the reverted fix takes the suite from 9 errors + 1 failure to 1 failure, which is what the revert cost.

The enabling defect sat one layer down. `_parse_test_counts` had no `unittest` branch, so every failing `unittest` run collapsed to the `(0, 1)` fallback and a ten-failure suite scored identically to a one-failure suite — no logic above it could tell whether a fix had helped. It now reads `Ran N tests` with `FAILED (failures=F, errors=E)`, clamping the pass count at zero because `subTest` can report more failures than there are test methods.

`_diagnosis_score` builds on that by counting failing tests, which is the comparison that has a direction. It returns `None` — never a guess — for anything no test-runner parser can read, such as a CODE step's bare traceback, and those cases fall back to the signature exactly as before. An unknown score must never read as an improvement.

The loop then separates two questions that one variable used to answer. What the **next attempt builds on** is still "did this improve", now by count rather than by difference. What the step is **left holding** is `_best_snapshot`: the best-scoring state seen anywhere in the loop, recorded *before* any revert can discard it and re-checked once after the loop ends — because the final attempt is the escalated one, the most likely of the three to be right, and the top-of-loop check never sees it (there is no attempt 4). That is precisely how one run lost a correct `gpt-5.6-sol` fix.

What it does **not** claim is that the loop now converges. A step whose failures are unscorable is unchanged and still directionless, and a scored step that never improves still halts the pipeline — the fix decides what gets shipped when every attempt fails, not whether they fail. Both incidents are replayed by `tests/orchestrator/test_diagnosis_restores_best.py`, which drives the real loop and asserts on what is left on disk; all four fail against the prior code. `test_diagnosis_best_snapshot.py` pins the scoring itself, including that pytest/Jest/Go parsing is untouched.

That left one question open — why the first run's two signatures compared *equal* when the same two states, reconstructed offline, hashed differently — and the debug line added at the decision point answered it on the next occurrence. Two consecutive attempts of one step logged `sig 1a3d09c05029→1a3d09c05029` with `error_info` of **1692 then 1038 characters**: different strings, identical hash. `_error_signature` hashed only `norm[:600]`, and a test runner front-loads whatever is invariant (a constant summary line, then test names in alphabetical order) while leaving the discriminating part — which assertion blew up, and the `FAILED (failures=F, errors=E)` tally — until last. It now hashes **both ends**, `norm[:600] + norm[-600:]`, with anything shorter hashed whole. This matters most for CODE steps, where a bare traceback carries no counts for `_diagnosis_score` to read and the signature is the only signal there is. The remaining bound is deliberate and pinned by a test: a difference confined to the *middle* of a very long error is still invisible, because hashing everything would make cosmetic churn anywhere read as progress.

### Benchmarks (benchmarks/)

A/B harness comparing `agent_loop` on vs off over the task set in `benchmarks/tasks.py`. Ground truth is per-task `success_cmds` run in the isolated workdir, independent of the pipeline's own claim. Run from repo root: `python benchmarks/run_ab.py --config <yaml-with-keys> [--tasks id1,id2] [--modes on,off] [--truststore]`. Results land in `benchmarks/results/*.json`. Not part of pytest — it spends real API tokens.

`benchmarks/verify_dt_invariance.py <project-dir>` is an independent ground-truth check for generated tile-maze games: it drives the game at several timestep profiles and asserts no entity ever occupies a wall tile, catching games that only hold together at a fixed 1/60 dt. Exit codes are **0 PASS, 1 FAIL, 2 could-not-verify** — the third is deliberate, because generated projects share no vocabulary and a refusal must never be recorded as a failure.

### Plan Re-plan Gate Carry-Forward (plan_step.py)

A weak `verify:` sends the plan back to the planner, but a re-plan regenerates *every* step — and a planner asked to strengthen step 4's gate has no reason to preserve the strength of step 3's. `repair_verify_commands` exists to avoid the re-plan entirely by rewriting only the offending line; `carry_forward_strong_gates` covers what happens when that repair yields nothing and the re-plan runs anyway.

Measured: attempt 1 of a Pac-Man run declared `verify: python -c "... g.run_frame(0.02); assert g.player.pos[0] != g.player.prev_pos[0]"` on the game-logic step — a gate that fails against exactly the artifact the run shipped, a `run_frame(dt)` that never reads `dt` and a player that never moves. The re-plan was triggered by a *different* step's import-only gate, and attempt 2 returned `assert len(g.ghosts)==4 and all(not g.map.is_wall(*pos) ...)`, which a stub satisfies. Note what the loss was not: **both gates pass `check_gate_quality`**, so no weakness check could see it — the plan simply traded a gate that would have caught the defect for one that would not.

Steps are matched across attempts by primary **target file**, because that is the one thing a re-plan keeps stable while ids, ordering and dependencies churn (ids do churn: 3.1 became 2.1). Then: a new gate that is absent or judged weak is **replaced** by the old one; two gates that can both fail are **conjoined** with `&&` rather than chosen between, since each was written to catch something. Redundant conjunctions are suppressed — a gate whose assertions are a whitespace/quote-insensitive prefix of the newer one, and two invocations of the same test runner (`unittest -v <path>` and `unittest <path>` are one suite, not two). Both suppressions came from replaying the measured plans, which otherwise carried three gates where only one had been lost.

The bound is applicability: a gate is carried only when every *project* module it imports is still produced by some step of the new plan. A stale gate that names a dropped module would fail a correct step forever — the failure mode `gate_integrity` exists for, where one bad gate cost a run 182k tokens and rejected working code.

### Destructive Gates (orchestrator/gate_safety.py)

Every other gate check reads a `verify:` as a *measurement* and asks how good it is — can it fail on wrong behaviour (`check_gate_quality`), does it run in the right directory (`check_gate_consistency`), does it parse at all (`unrunnable_gate_reason`). None asks what else it does. Measured 2026-08-17 23:42: a planner ended a gate with `... & timeout /t 2 /nobreak & taskkill /im python.exe /f 2>nul || exit /b 0`. `/im` names an *image*, not a process, so it force-killed every `python.exe` on the machine — the pipeline among them. The log stops mid-line at the executor call, with no `Finished`, no ghost reconciliation and no wave snapshot; the next run opened with `[CrashDiag] Previous run (pid=33548) ended abnormally`. The step's rewritten `main.py` (7.4k → 14.7k) was left on disk with nothing having checked it. All three existing checks had passed that gate.

What makes this its own class of defect is repetition: a gate runs on the step, again on the loop's early-exit check, again on each platform-variant retry, and again after every later wave via `GateLedger`. A gate with a side effect does not have it once.

Three seams, because gates arrive by three routes. `check_gate_safety` joins the plan-time gaps so `repair_verify_commands` and the re-plan correction get a chance at a better command (with their own wording — "assert a concrete value" is not the fix for a command that kills processes). `neutralize_destructive_gates` then runs on the *accepted* plan as a backstop, because repair and re-plan can both fail and the branch above them deliberately proceeds with imperfect gates: right for a weak gate, wrong here. `_verify_once` in the agent loop refuses at run time, covering gates no plan declared — `verify_cmd_for_language`, a failed CMD step's recovery gate (the failed command verbatim), and the variants built from either; the refusal string does not start with `exit: success`, so `verify_passed` reads it as failure and no step can exit green on it. `GateLedger.record` refuses entry, since the ledger is where one destructive command becomes many.

Sanitisation truncates at the first destructive segment rather than blanking the gate — the head is usually the check the planner meant (in the incident, a real assertion over the class's constants and public API), and what survives is re-judged by the ordinary quality machinery. Segmentation is quote-aware so a `;` inside a `python -c` payload does not split the command; *matching* deliberately is not, because a false positive costs one gate and a false negative costs the run. Audited over every gate and command in this project's own logs: 1 of 14 unique gates flagged, and it is the incident.

Known hole: this governs gates only. An agent loop can still call `run_command` with anything, which is `AgentTools` sandboxing, not a gate check.

### Evidence Provenance (orchestrator/evidence.py)

Separates two verdicts that used to be one: **completed** (the plan ran without a step failing) and **verified** (something the agent did not write in this run agreed). Measured over six A/B runs of `pacman-strict` against external probes: both agent-loop failures printed `All tasks completed successfully!` over a Pac-Man whose player could not move at 1/60, while both classic-path failures failed their own tests and said so. The loop iterates until the gate is green, and the gate was a suite it wrote in the same run — every declared postcondition genuinely held, and nothing in the pipeline was positioned to notice that all of it was self-authored.

Independent evidence is exactly two things: **user-supplied `acceptance_cmds`** (config/`ACCEPTANCE_CMDS`), the one instrument the model neither wrote nor can edit — and so the only check allowed to fail a run on its own — or a **pre-existing test file the run left byte-identical**, snapshotted by hash before the first step next to the ghost's pre-state. Hashing rather than presence is the point: a seeded test the agent rewrote is the oldest cheat there is, so modifying it forfeits independence and is reported as such.

Absence of independent evidence is **not a failure** and never flips `pipeline_success` on its own — a greenfield build honestly has no pre-existing suite, and failing every such run would be false precision. It changes what the run may *claim*: the banner becomes `~ Tasks completed — but nothing independent verified them`, `TaskResult` carries `verified`/`evidence`, and the log emits a parseable `Evidence: ...` line (the A/B harness now prints it as its own column, because `claim=True` alone was the misleading part). Set `require_independent_evidence: true` to make unverified exit non-zero.

### Ghost Shadow (orchestrator/ghost.py)

Read-only reconciliation of the plan's *declared postconditions* against the real tree — no LLM calls, no commands run, no verdict changed. `GhostPlan.build()` is called in `cli.py` once the plan is final (after blind-edit routing / dependency fixes / verify repair) and before the first step runs, so its file hashes are a true pre-run baseline. Steps' `target:`/`exports:`/`imports:`/`verify:` become interned `Expectation` nodes (`EXISTS`, `TOUCHED`, `PARSES`, `EXPORTS`, `IMPORT_EDGE`, `PKG_PRESENT`, `GATE_PASSED`) shared across the steps that declare them; verdicts are four-valued (`HOLDS`/`VIOLATED`/`UNKNOWN`/`INAPPLICABLE`) and fold over an append-only observation journal. Resolved per wave next to `_reconcile_plan_graph`, reported once at the end of the run under `[Ghost]`.

`PLAN_ANCHORS` is the step-drift check, and the reason the ghost can repair as well as report. Where the planner supplied a file's body (`PlanStep.inline_code`, i.e. content mode), the structural names that body declares — CSS selectors, Python defs/classes/constants, JS exports — must still be present in the written file. A smaller model plans correctly and then deviates while executing an individual step, and nothing else notices: the file exists, parses, and often satisfies a gate that never names the missing piece.

Run-level findings sit alongside the per-step ones: `unplanned-write`, `plan-declares-no-targets`, `filename-case-mismatch`, and `tests-never-collected` — test files the run's own acceptance command will never collect, decided statically from the AST (`unittest` collects only `TestCase` subclasses; `pytest` also collects module-level `test_*` functions). It spans planned targets *and* untracked writes, because the four modules that exposed it were written by the agent loop and declared by no step. Silent when the runner cannot be identified, when the file will not parse, or when the runner is pytest.

`unplanned-write` is only meaningful against a plan that declared something. A 20B planner named its files in prose (`Create map.py that defines a Map class`) and emitted no `target:` on any of six CODE/TEST steps, so the ghost tracked one path — `produces: venv\` — and reported all six source files as unplanned writes: six findings that were each true of the plan, useless about the code, and enough to bury the one write that really was a stray. When no step declares a file target (directory-ness read from disk, so `Makefile` counts and `tests` does not), the run reports `plan-declares-no-targets` once instead, which is the fact that actually matters: the file layer of the check — exports, anchors, content regressions — was never armed.

`degenerate-long-run` catches the one defect every gate is blind to: a test that satisfies "iterate N times and assert the invariants hold" while the thing it drives stopped doing work at iteration fifty. The rule reads a *shape*, not a domain — a class whose advance method early-returns on a state comparison, driven by a long literal loop that never pins that state — which is as true of a scheduler, a retry loop, a workflow engine or a stream consumer as of a game; `tests/orchestrator/test_ghost.py` pins that down with an ingest-worker artifact and a rename-every-identifier test. The artifacts that exposed it were generated games because that is what got benchmarked, and the examples below are quoted from them. `Game.update()` opens with `if self.state is not PLAYING: return`, so once the ghosts catch a parked player every later iteration is a no-op — and the invariant assertions keep passing, against a frozen state, all the way to 2000. Measured on four real artifacts: 50/2100, 246/2100 live frames on two runs whose suites were fully green.

Proved structurally, from two halves that must both hold: the advance method early-returns on a state guard (`guarded_advance_methods`, which scans the whole guard prologue — a real `update()` put `if not math.isfinite(dt): raise` first), and a long `range(N)` loop calls it without ever pinning the state to a live value. Both come from the project's own source, so nothing is hardcoded about what a "game" or a "frame" is. Guards are keyed by **class**, and each call's receiver is resolved back to the class it was constructed from (`_class_bindings`, covering both `game = Game(seed)` and a `self.game = Game()` fixture): `Game.update` is guarded while `Player.update` and `Ghost.update` are not, and a suite driving the player directly for 800 frames skips no frames at all. The loop rarely calls the advance method itself, though — the natural way to write "assert the invariants every frame" is a helper that updates and then asserts — so a call into one of the test module's own functions is followed (`_reached_advancers`, depth-capped), binding the helper's parameters from the arguments at the call site, which is how `game` inside `def step(self, game, dt)` is known to be the `Game` the fixture built. The helper's body is then judged for the silences too: a pin or a restart inside it runs every iteration and belongs to its callers.

The guard itself is read loosely enough to survive real code: the whole prologue is scanned, the guard body may do work before it leaves (one real guard called `self.assert_invariants()` first — precisely why its frozen frames kept passing), and an `or` counts (`if self.state != PLAYING or dt == 0.0: return` leaves whenever the state dies) while an `and` does not. Both spellings of the guard are read — a guard splits an attribute's values into the ones it returns on and the ones it works on, and may name either side (`is not PLAYING` names the working value, `in (WIN, GAME_OVER)` names the halting ones) — so `_EarlyReturnGuard` keeps the names on the side they were written and callers ask `proceeds`/`halts` rather than testing set membership, which is right for only one of the two spellings. Reading just the first spelling left a real `Game.update` unguarded, which made its suite no candidate at all, which is how a `range(700)` acceptance loop that simulated 17 frames went unreported. Ordering comparisons still say nothing this check can read.

The silences carry as much of the design as the finding. A tautology is deliberately not a pin — `assertIn(game.state, (PLAYING, WIN, GAME_OVER))` admits the terminal states it was meant to exclude, and a real run wrote exactly that inside a 2100-iteration loop. Silent when the test pins the state (in the loop *or* after it), breaks out, handles termination — `if game.state is GAME_OVER: game.restart()` kept one suite live for 2000 of 2000 frames across six restarts — or extends the run's lifetime (`game.spawn_protection_timer = 1_000_000.0` before the loop). Branching on the *live* state is not handling termination: `if game.state == PLAYING and frame % 11 == 0: send_input()` only gates input, and the loop it appeared in still ran 44 of 650 frames.

Every one of those rules was written against a measured artifact, and the check is validated by replaying all of them: it flags the five suites measured at 50/2100, 246/2100, 171/2001, 44/650 and 17/700 live frames, and stays silent on the three measured at 100% — each of which pins, restarts, or extends the run's lifetime. Rounds of live running produced false positives and false negatives before it matched; re-validate against the archive before changing it.

`varied-input-ignored` is its sibling, and covers what it cannot see: `degenerate-long-run` asks whether the loop stopped doing work, this asks whether the work ever depended on what the loop was varying. Measured on a run whose task demanded "≥600 frames with dt drawn randomly from 0.008..0.05, not a fixed 1/60": the suite looped 600 and then 2000 times doing exactly that and passed, while `run_frame(self, dt)` never mentioned `dt` in its body — the docstring said the parameter was "accepted to mirror the interface the tests expect". Replaying at dt = 1e-9, 1/60 and 1e9 gave byte-identical results, so the adversarial condition the entire task was written around could not have failed, and no other check had anything to say. Fires only on a complete proof: the argument demonstrably changes per iteration (a `random.*` draw or the loop variable), the receiver resolves to a class built in this project, the method is unambiguous, and the parameter it lands on appears nowhere in the body. A `*args`/`**kwargs` callee is never accused, and a constant argument is not variation — a fixed-value long run is the other check's business.

`unprogressed-long-run` is the third in the family, and covers what both of the others miss: `degenerate-long-run` asks whether the loop stopped doing work, `varied-input-ignored` asks whether the work ever read what was being varied, and this asks whether anything ever asserted that the work *accomplished* anything. Measured on a glm-5.2 run whose `Entity._move` stepped toward the current tile centre before stepping forward — so once past the centre the correction moved it **backward**, and at small dt consumed the whole travel budget (`advance(0.01)` → `x=1.04` → `x=1.0` → `x=1.04`, net zero at any dt below ~0.2, with only `advance(1.0)` ever moving anything). The run shipped exit 0 with 19 green tests while external probes of both dt-scaling and `press()` failed. Neither sibling was wrong to stay silent: that game never terminates (2100 frames at 1/60 are all `playing`, so the loop never stops doing work), and `dt` demonstrably *is* read (`travel = self.speed * dt`).

What made it invisible is that every assertion those loops make holds against a frozen world — `assertFalse(map.is_wall(*e.tile))` is an invariant, `assertLessEqual(cur, prev)` is monotone and `172 <= 172` passes, `assertGreater(pellets, 0)` is strict but against a literal. So progress means a **strict relation between two things that both vary**: non-strict operators admit equality by construction, and a comparison against a literal is fixed at both ends of the run. One such assertion anywhere — in the loop, after it, or in a helper it calls — silences the finding, because it is the whole fix. A `break` is deliberately *not* a silence here, unlike in `degenerate-long-run`: breaking on a terminal state is how a careful test ends a run and says nothing about whether the run achieved anything. The threshold is separate and higher (500 vs 200) because this check fires on far more loops, not requiring the object to be able to stop at all; at 200 it flagged a state-*vocabulary* test from an artifact that passed all nine external probes. Validated across every artifact this benchmark has produced: 3 findings on the broken loop artifact, 0 on all five others.

What these checks do **not** claim is that a flagged loop is measurably degenerate. The discriminator is dynamic and not in the AST: the same artifact that hid 17/700 frames also has a 2100-frame loop that really does run all 2100, and is flagged too, because it stays alive only by accident — the player never moves, so the pellets that would trigger `WIN` are never collected, and the day a ghost-collision `GAME_OVER` lands it degenerates silently. The finding says the endurance claim is unprotected, and the fix for both is the same one line.

### Ghost Heal (orchestrator/ghost_heal.py)

Deterministic repair of what the shadow finds — no LLM call. Runs per wave (so later steps benefit) and once at the end, under `[GhostHeal]`. Governed by one rule: **never invent content; freely restore content the plan already specified.** Those are different things — writing an empty `.site-header {}` to satisfy a check makes a real defect undetectable, but writing the `.site-header` rule *the planner put in `inline_code`* invents nothing and enforces a decision already made.

Healers: `PKG_PRESENT` → install into the interpreter `Executor._venv_bin_dir()` resolves (never bare `python`, which is the original bug); `EXISTS` → restore from the plan's body, or create an empty `__init__.py` (empty *is* its correct content); `IMPORT_EDGE` → add the import, only when both files are Python in the same directory, exactly one consumer is a real module, and every declared symbol demonstrably exists in the source; `PLAN_ANCHORS`/`EXPORTS` → restore the plan's body when the written file is a strict regression.

Refusals are as important as repairs. Restoration is declined when the written file declares anything the plan's body does not — the step may have added real work, so the conflict is reported instead of clobbered. `PARSES`, `TOUCHED` and `GATE_PASSED` have no healer at all: no source specifies what they should contain. Every heal is verified by re-resolving its expectation, and source edits are snapshotted and reverted if the gap does not close. Flags: `ghost_heal` (default true), `ghost_heal_source_edits` (narrows to environment actions only).

`PKG_PRESENT` is the one check that looks past the repo at the environment: for any manifest the plan targets (`requirements.txt`, `package.json`) every declared runtime dependency must be present in the environment the app will actually run in — the venv `Executor._venv_bin_dir()` resolves, or `node_modules`. Purely a filesystem comparison, no subprocess. It exists because a plan step wrote `python -m venv venv && python -m pip install pygame`, which creates a venv but never activates it, so pygame installed into the pipeline's interpreter instead; every gate passed (the game modules were headless and imported no pygame) and only `main.py` needed it, crashing at launch under the project venv. No venv / no `node_modules` resolves `UNKNOWN` — a project on the ambient interpreter must never be accused.

Surfaces the disagreement classes the rest of the pipeline is blind to: `violated-*` (a step claimed done while a declared postcondition is false), `export-drift` (declared exports the code renamed that **no step imports** — one collapsed note rather than one finding each, since a contract with no consumer cannot break anything; the consumed case stays a `violated-exports` finding, because that is the `gate_integrity` shape), `planned-untouched` (target's bytes never changed), `unplanned-write` (a file no step declared), `plan-declares-no-targets` (no step declared any file, so nothing written could be reconciled at all), `no-checkable-claim` (a step whose expectations are all tautologies — the plan-level analogue of `gate_integrity`), `degenerate-long-run` (a long assertion loop that stops simulating partway and asserts a frozen state), `varied-input-ignored` (a loop varies an input the code it drives never reads), `unprogressed-long-run` (a long loop whose every assertion would hold if the object had frozen on iteration one), and `failed-but-clean` (run marked failed while everything declared holds). `failed-but-clean` tells a reader to suspect the harness before the model, so it is the one finding whose false positives are actively harmful, and it is fenced twice. A green **acceptance gate** is required — structural checks prove shape, never behaviour — and when the run halted partway that gate must belong to a step that *did not complete*, because a gate on step 2 is not evidence about step 6. Measured twice blaming the harness for the model: gates 2.1/3.1/4.1 green while step 5 failed having recorded no gate at all, and four green gates while step 6 failed `verify` three times. A gate's verdict also **expires**: `GATE_PASSED` records the hashes of the step's declared files the wave it first goes green, and re-resolves to `UNKNOWN` once any of them changes, since the first run's gate 3.1 was still reported green over a `game.py` that a later step had rewritten into a `TypeError` on every `advance()`. Disable with `ghost_shadow: false`.

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

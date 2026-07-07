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

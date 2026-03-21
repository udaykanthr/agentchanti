"""
CLI entry point — argument parsing and main execution flow.
"""

import argparse
import sys
import time

from ..config import Config
from ..llm.ollama import OllamaClient
from ..llm.lm_studio import LMStudioClient
from ..llm.base import LLMError
from ..agents.planner import PlannerAgent
from ..agents.coder import CoderAgent
from ..agents.reviewer import ReviewerAgent
from ..agents.tester import TesterAgent
from ..executor import Executor
from ..embedding_store import EmbeddingStore
from ..cli_display import CLIDisplay, token_tracker, log
from ..language import (
    detect_language, detect_language_from_task, get_test_framework,
    get_language_name, get_code_block_lang,
)
from ..project_scanner import scan_project, format_scan_for_planner, collect_source_files
from ..checkpoint import (
    save_checkpoint, load_checkpoint, clear_checkpoint,
)
from .. import git_utils
from ..knowledge import KnowledgeBase
from ..step_cache import StepCache
from ..report import generate_html_report, StepReport
from ..plugins.registry import PluginRegistry

from .memory import FileMemory
from .pipeline import build_step_waves, _execute_step, _run_diagnosis_loop
from .plan_step import build_waves as _build_plan_waves
from ..agents.analyser import build_project_context, AnalyseAgent


def _rematch_plan_steps(new_steps, old_plan_steps, dependencies):
    """Re-match edited step descriptions to original PlanStep objects.

    Preserves structured metadata (step_type, target_files, exports,
    imports_from, command, inline_code) when the description is similar
    enough. Falls back to UNCLASSIFIED for steps that can't be matched.
    """
    from .plan_step import PlanStep, from_legacy_steps
    from difflib import SequenceMatcher

    result: list[PlanStep] = []
    used: set[int] = set()  # indices into old_plan_steps already matched

    for new_idx, desc in enumerate(new_steps):
        desc_clean = desc.strip().lower()
        best_score = 0.0
        best_old_idx = -1

        for old_idx, old_ps in enumerate(old_plan_steps):
            if old_idx in used:
                continue
            old_clean = old_ps.description.strip().lower()
            score = SequenceMatcher(None, desc_clean, old_clean).ratio()
            if score > best_score:
                best_score = score
                best_old_idx = old_idx

        if best_score >= 0.6 and best_old_idx >= 0:
            # Re-use the old PlanStep with updated description and index
            old = old_plan_steps[best_old_idx]
            ps = PlanStep(
                id=old.id,
                step_type=old.step_type,
                description=desc,
                depends_on=list(old.depends_on),
                command=old.command,
                target_files=list(old.target_files),
                exports=list(old.exports),
                imports_from={k: list(v) for k, v in old.imports_from.items()},
                inline_code=dict(old.inline_code),
                index=new_idx,
            )
            result.append(ps)
            used.add(best_old_idx)
        else:
            # Can't match — create UNCLASSIFIED placeholder
            dep_ids = [str(d + 1) for d in dependencies.get(new_idx, set())]
            result.append(PlanStep(
                id=str(new_idx + 1),
                step_type="UNCLASSIFIED",
                description=desc,
                depends_on=dep_ids,
                index=new_idx,
            ))

    return result


def _blank_project_scaffold_hint(language: str | None) -> str:
    """Return language-appropriate scaffolding examples for blank-project prompt."""
    lang = (language or "").lower()
    if lang == "python":
        return "e.g. `python -m venv venv`, `pip install <packages>`"
    if lang in ("javascript", "typescript"):
        return "e.g. `npm create vite@latest`, `npm install`, framework setup"
    if lang == "go":
        return "e.g. `go mod init`, `go get <packages>`"
    if lang == "rust":
        return "e.g. `cargo init`, `cargo add <crates>`"
    # Generic fallback
    return "e.g. project init command, package install, framework setup"


def main():
    # Dispatch `agentchanti kb ...` to the KB CLI before argparse sees it,
    # so the KB subcommand tree is fully independent of the main task args.
    if len(sys.argv) > 1 and sys.argv[1] == "kb":
        from ..kb.cli import kb_main
        kb_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="AgentChanti — Multi-Agent Local Coder")
    parser.add_argument("task", nargs="?", help="The coding task to perform")
    parser.add_argument("--prompt-from-file", help="Read prompt from a text file")
    parser.add_argument("--provider", choices=["ollama", "lm_studio", "openai", "gemini", "anthropic"],
                        default=None, help="The LLM provider to use (default: from config or lm_studio)")
    parser.add_argument("--model", default=None,
                        help="The model name to use (default: from config)")
    parser.add_argument("--embed-model", default=None,
                        help="Embedding model name (default: from config)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Disable semantic embeddings")
    parser.add_argument("--language", default=None,
                        help="Override detected language (e.g. python, javascript)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming responses")
    parser.add_argument("--no-git", action="store_true",
                        help="Disable git integration")
    parser.add_argument("--resume", action="store_true",
                        help="Force resume from checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoint and start fresh")
    parser.add_argument("--auto", action="store_true",
                        help="Non-interactive mode: auto-approve plan, "
                             "skip all prompts (for backend/service use)")
    parser.add_argument("--config", default=None,
                        help="Path to .agentchanti.yaml config file")
    parser.add_argument("--no-diff", action="store_true",
                         help="Disable diff preview before writing files")
    parser.add_argument("--no-cache", action="store_true",
                         help="Disable step-level caching")
    parser.add_argument("--clear-cache", action="store_true",
                         help="Clear step cache before running")
    parser.add_argument("--no-knowledge", action="store_true",
                         help="Disable project knowledge base")
    parser.add_argument("--report", action="store_true", default=True,
                         help="Generate HTML report after run (default: on)")
    parser.add_argument("--no-report", action="store_true",
                         help="Disable HTML report generation")
    parser.add_argument("--generate-config", "--generate-yaml", action="store_true",
                         help="Generate a .agentchanti.yaml file with current settings and exit")
    parser.add_argument("--no-search", action="store_true",
                         help="Disable web search agent for planning and error diagnosis")
    parser.add_argument("--no-kb", action="store_true",
                         help="Disable KB context injection (debugging)")
    args = parser.parse_args()

    # ── 0. Load config ──
    cfg = Config.load(args.config)

    # CLI overrides
    model = args.model or cfg.DEFAULT_MODEL
    embed_model = args.embed_model or cfg.EMBEDDING_MODEL

    # Update config object with CLI overrides (for --generate-yaml)
    if args.provider is not None:
        cfg.PROVIDER = args.provider
    if args.model:
        cfg.DEFAULT_MODEL = args.model
    if args.embed_model:
        cfg.EMBEDDING_MODEL = args.embed_model
    if args.no_embeddings:
        cfg.NO_EMBEDDINGS = True
    if args.language:
        cfg.LANGUAGE = args.language
    if args.no_stream:
        cfg.STREAM_RESPONSES = False

    # ── 0.5. Generate YAML and exit ──
    if args.generate_config:
        yaml_content = cfg.to_yaml()
        with open(".agentchanti.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print("\n  ✨ Generated .agentchanti.yaml with current settings.\n")
        return

    # Handle prompt-from-file
    if args.prompt_from_file:
        try:
            with open(args.prompt_from_file, "r", encoding="utf-8") as f:
                args.task = f.read().strip()
        except Exception as e:
            print(f"\n  [ERROR] Could not read prompt file: {e}\n")
            return

    if not args.task:
        parser.print_help()
        return

    # ── 1. Detect language ──
    if args.language:
        language = args.language
    else:
        language = detect_language_from_task(args.task) or detect_language()
    log.info(f"Language: {language} ({get_language_name(language)})")

    # ── 2. Init LLM client ──
    stream_enabled = cfg.STREAM_RESPONSES and not args.no_stream
    llm_kwargs = dict(
        max_retries=cfg.LLM_MAX_RETRIES,
        retry_delay=cfg.LLM_RETRY_DELAY,
        stream=stream_enabled,
    )

    provider = args.provider or cfg.PROVIDER
    if provider == "ollama":
        llm_client = OllamaClient(
            base_url=cfg.OLLAMA_BASE_URL, model=model, **llm_kwargs)
    elif provider == "openai":
        from ..llm.openai_client import OpenAIClient
        api_key = cfg.OPENAI_API_KEY
        if not api_key:
            print("\n  [ERROR] OpenAI provider requires an API key.\n"
                  "  Set OPENAI_API_KEY env var or add it to .agentchanti.yaml.\n")
            return
        llm_client = OpenAIClient(
            base_url=cfg.OPENAI_BASE_URL, model=model,
            api_key=api_key, **llm_kwargs)
    elif provider == "gemini":
        from ..llm.gemini_client import GeminiClient
        api_key = cfg.GEMINI_API_KEY
        if not api_key:
            print("\n  [ERROR] Gemini provider requires an API key.\n"
                  "  Set GEMINI_API_KEY env var or add it to .agentchanti.yaml.\n")
            return
        llm_client = GeminiClient(
            base_url=cfg.GEMINI_BASE_URL, model=model,
            api_key=api_key, **llm_kwargs)
    elif provider == "anthropic":
        from ..llm.anthropic_client import AnthropicClient
        api_key = cfg.ANTHROPIC_API_KEY
        if not api_key:
            print("\n  [ERROR] Anthropic provider requires an API key.\n"
                  "  Set ANTHROPIC_API_KEY env var or add it to .agentchanti.yaml.\n")
            return
        llm_client = AnthropicClient(
            base_url=cfg.ANTHROPIC_BASE_URL, model=model,
            api_key=api_key, **llm_kwargs)
    else:
        llm_client = LMStudioClient(
            base_url=cfg.LM_STUDIO_BASE_URL, model=model,
            reasoning_effort=cfg.LM_STUDIO_REASONING_EFFORT, **llm_kwargs)

    # ── 3. Scan existing project ──
    scan_result = scan_project(".")
    source_files = collect_source_files(".")
    log.info(f"Project scan: {scan_result['file_count']} files detected, "
             f"{len(source_files)} source files collected")
    project_context = format_scan_for_planner(
        scan_result, max_chars=cfg.PLANNER_CONTEXT_CHARS,
        source_files=source_files)

    # ── 4. Init embedding store (SQLite-backed for persistence) ──
    embed_store = None
    if not args.no_embeddings:
        try:
            from ..embedding_store_sqlite import SQLiteEmbeddingStore
            import os
            db_path = os.path.join(cfg.EMBEDDING_CACHE_DIR, "embeddings.db")
            embed_store = SQLiteEmbeddingStore(
                llm_client, embed_model=embed_model, db_path=db_path)
            log.info(f"Embeddings enabled with SQLite cache (model: {embed_model})")
        except Exception as e:
            log.warning(f"SQLite embedding store failed ({e}), falling back to in-memory")
            embed_store = EmbeddingStore(llm_client, embed_model=embed_model)
    else:
        log.info("Embeddings disabled")

    # ── 4b. Init step cache ──
    step_cache = None
    if not args.no_cache:
        import os
        cache_dir = os.path.join(cfg.EMBEDDING_CACHE_DIR, "cache")
        step_cache = StepCache(cache_dir=cache_dir,
                               ttl_hours=cfg.STEP_CACHE_TTL_HOURS)
        if args.clear_cache:
            step_cache.clear()
        log.info(f"Step cache enabled (TTL: {cfg.STEP_CACHE_TTL_HOURS}h)")

    # ── 4c. Init knowledge base ──
    knowledge_base = None
    if not args.no_knowledge:
        import os
        kb_path = os.path.join(cfg.EMBEDDING_CACHE_DIR, "knowledge.json")
        knowledge_base = KnowledgeBase(path=kb_path)
        log.info(f"Knowledge base loaded ({knowledge_base.size} entries)")

    # ── 4c-bis. Import plan optimizer ──
    from .plan_optimizer import optimize_plan, optimize_structured_plan

    # ── 4d. Init plugin registry ──
    plugin_registry = PluginRegistry()
    if cfg.PLUGINS:
        plugin_registry.discover(cfg.PLUGINS)
        log.info(f"Plugins loaded: {plugin_registry.size}")

    # ── 4f. Init search agent ──
    search_agent = None
    if cfg.SEARCH_ENABLED and not args.no_search:
        from ..agents.search import SearchAgent
        search_agent = SearchAgent(
            provider=cfg.SEARCH_PROVIDER,
            api_key=cfg.SEARCH_API_KEY,
            api_url=cfg.SEARCH_API_URL,
            max_results=cfg.SEARCH_MAX_RESULTS,
            max_page_chars=cfg.SEARCH_MAX_PAGE_CHARS,
            llm_client=llm_client,
        )
        log.info(f"Search agent enabled (provider: {cfg.SEARCH_PROVIDER})")
    else:
        log.info("Search agent disabled")

    # ── 4g. Init KB context builder and runtime watcher (Phase 4) ──
    kb_context_builder = None
    kb_runtime_watcher = None
    if cfg.KB_ENABLED and not args.no_kb:
        try:
            import os as _os
            from ..kb.startup import KBStartupManager
            from ..kb.context_builder import ContextBuilder
            from ..kb.runtime_watcher import RuntimeWatcher

            # Smart startup check — handles global KB, local KB
            KBStartupManager().run(project_root=_os.getcwd(), api_client=llm_client)

            kb_context_builder = ContextBuilder(project_root=_os.getcwd(), api_client=llm_client)
            kb_runtime_watcher = RuntimeWatcher(
                debounce_seconds=cfg.KB_WATCHER_DEBOUNCE_SECONDS,
            )
            kb_runtime_watcher.start(project_root=_os.getcwd(), api_client=llm_client)
            log.info("[KB] Context builder and runtime watcher initialised")
        except Exception as kb_exc:
            log.warning(f"[KB] Initialisation failed (non-fatal): {kb_exc}")
            kb_context_builder = None
            kb_runtime_watcher = None
    else:
        log.info("[KB] KB context injection disabled")

    # ── 4e. Step reports (for HTML report) ──
    step_reports: list[StepReport] = []

    # ── 5. Init agents (with per-agent model support) ──
    def _make_llm_for_agent(agent_name: str):
        """Create an LLM client for a specific agent, using per-agent model if configured."""
        agent_model = cfg.get_agent_model(agent_name) or model
        if agent_model == model:
            return llm_client  # reuse the main client
        # Create a separate client with the agent-specific model
        if provider == "ollama":
            return OllamaClient(
                base_url=cfg.OLLAMA_BASE_URL, model=agent_model, **llm_kwargs)
        elif provider == "openai":
            from ..llm.openai_client import OpenAIClient
            return OpenAIClient(
                base_url=cfg.OPENAI_BASE_URL, model=agent_model,
                api_key=cfg.OPENAI_API_KEY, **llm_kwargs)
        elif provider == "gemini":
            from ..llm.gemini_client import GeminiClient
            return GeminiClient(
                base_url=cfg.GEMINI_BASE_URL, model=agent_model,
                api_key=cfg.GEMINI_API_KEY, **llm_kwargs)
        elif provider == "anthropic":
            from ..llm.anthropic_client import AnthropicClient
            return AnthropicClient(
                base_url=cfg.ANTHROPIC_BASE_URL, model=agent_model,
                api_key=cfg.ANTHROPIC_API_KEY, **llm_kwargs)
        else:
            return LMStudioClient(
                base_url=cfg.LM_STUDIO_BASE_URL, model=agent_model,
                reasoning_effort=cfg.LM_STUDIO_REASONING_EFFORT, **llm_kwargs)

    # Custom prompt suffixes from config
    planner_suffix = cfg.PROMPT_SUFFIXES.get("planner_suffix", "")
    coder_suffix = cfg.PROMPT_SUFFIXES.get("coder_suffix", "")
    reviewer_suffix = cfg.PROMPT_SUFFIXES.get("reviewer_suffix", "")
    tester_suffix = cfg.PROMPT_SUFFIXES.get("tester_suffix", "")

    planner = PlannerAgent("Planner", "Senior Software Architect",
                           "Create a step-by-step plan for the coding task and related testcases.",
                           _make_llm_for_agent("planner"),
                           prompt_suffix=planner_suffix)
    coder = CoderAgent("Coder", "Senior Software Developer",
                       f"Write clean {get_language_name(language)} code for a single step.",
                       _make_llm_for_agent("coder"),
                       prompt_suffix=coder_suffix)
    reviewer = ReviewerAgent("Reviewer", "Code Reviewer",
                             "Review code for errors and style issues.",
                             _make_llm_for_agent("reviewer"),
                             prompt_suffix=reviewer_suffix)
    tester = TesterAgent("Tester", "Software Engineer in Test",
                         "Create unit tests for the provided code.",
                         _make_llm_for_agent("tester"),
                         prompt_suffix=tester_suffix)
    executor = Executor()

    # ── 6. Init display ──
    display = CLIDisplay(args.task or "Config Generation")
    
    # Inject pricing into tracker
    token_tracker.pricing = cfg.PRICING
    
    log.info(f"Task: {args.task}")
    log.info(f"Provider: {provider}, Model: {model}")

    # Wire streaming progress callback
    if stream_enabled:
        # We'll set per-step callbacks in the execution loop
        pass

    # ── 7. Check for checkpoint ──
    checkpoint_file = cfg.CHECKPOINT_FILE
    resuming = False
    checkpoint_state = None
    step_results: dict[int, str] = {}
    start_from = 0

    if not args.fresh:
        checkpoint_state = load_checkpoint(checkpoint_file)
        if checkpoint_state:
            if args.resume or args.auto:
                resuming = True
                log.info("Auto-resuming from checkpoint" if args.auto else "Resuming (--resume)")
            else:
                display.stop_spinner()
                resuming = CLIDisplay.prompt_resume(checkpoint_state)

    # ── 8. Restore state or create git checkpoint ──
    checkpoint_branch: str | None = None
    use_git = not args.no_git and git_utils.is_git_repo()

    if resuming and checkpoint_state:
        log.info("Resuming from checkpoint...")
        memory = FileMemory(embedding_store=embed_store, top_k=cfg.EMBEDDING_TOP_K)
        memory.update(checkpoint_state.get("file_memory", {}))
        steps = checkpoint_state["steps"]
        step_results = checkpoint_state.get("step_results", {})
        start_from = checkpoint_state.get("completed_step", -1) + 1

        # Load dependencies if saved, else parse them out of saved strings as a fallback
        loaded_deps = checkpoint_state.get("dependencies")
        if loaded_deps is not None:
            dependencies = {int(k): set(v) for k, v in loaded_deps.items()}
        else:
            _, dependencies = executor.parse_step_dependencies(steps)

        # Restore structured PlanStep objects if checkpoint has them
        from .plan_step import PlanStep, from_legacy_steps
        saved_plan_steps = checkpoint_state.get("plan_steps")
        if saved_plan_steps:
            plan_steps_parsed = [PlanStep.from_dict(d) for d in saved_plan_steps]
            log.info(f"Restored {len(plan_steps_parsed)} structured PlanSteps from checkpoint")
        else:
            # Legacy checkpoint without plan_steps — create wrappers
            plan_steps_parsed = from_legacy_steps(steps, dependencies)

        language = checkpoint_state.get("language", language)

        # Restore ProjectContext if saved (avoids re-running analysis LLM call)
        saved_project_context = checkpoint_state.get("project_context")
        if saved_project_context:
            from ..agents.analyser import ProjectContext
            _resumed_project_context = ProjectContext.from_dict(saved_project_context)
            log.info("[Resume] Restored ProjectContext from checkpoint (0 LLM tokens)")
        else:
            _resumed_project_context = None

        display.set_steps(steps)
        # Mark completed steps
        for idx in range(start_from):
            display.steps[idx]["status"] = "done"

        if "display_state" in checkpoint_state:
            ds = checkpoint_state["display_state"]
            if "elapsed" in ds:
                display.start_time = time.monotonic() - ds["elapsed"]
            if "steps" in ds:
                for i, saved_step in enumerate(ds["steps"]):
                    if i < len(display.steps):
                        display.steps[i].update(saved_step)

        display.render()
    else:
        # Fresh start
        if use_git:
            log.info("Creating git checkpoint branch...")
            checkpoint_branch = git_utils.create_checkpoint_branch(args.task)
            if checkpoint_branch:
                log.info(f"Git checkpoint: {checkpoint_branch}")
            else:
                log.warning("Failed to create git checkpoint branch")

        # ── 9. Plan ──
        display.show_status("Analyzing task and mapping relevant files...")
        log.info("Planning...")

        # Detect blank projects (no package manager / build config files)
        _has_project_config = bool(scan_result.get("key_files"))
        if _has_project_config:
            planner_context = f"Existing project:\n{project_context}"
        else:
            _scaffold_hint = _blank_project_scaffold_hint(language)
            planner_context = (
                f"PROJECT STATE: BLANK / EMPTY directory — no build config files found.\n"
                f"The plan MUST start with project scaffolding / initialization steps "
                f"({_scaffold_hint}) before writing any source code.\n"
            )
            if project_context:
                planner_context += f"\nCurrent directory contents:\n{project_context}"

        # Inject knowledge base context
        if knowledge_base and knowledge_base.size > 0:
            kb_context = knowledge_base.format_for_planner()
            if kb_context:
                planner_context += f"\n\n{kb_context}"
                log.info(f"Injected {knowledge_base.size} knowledge entries into planner")

        # Baseline test analysis before planning — run existing tests to
        # identify which files pass/fail so the planner only touches broken ones.
        # The task intent determines directive strictness: test-fix tasks get
        # strict "don't touch passing files" rules; feature tasks allow updates.
        from ..agents.planner import _classify_task_intent
        _task_intent = _classify_task_intent(args.task)
        test_analysis = ""
        if _has_project_config:
            try:
                from .test_analyzer import perform_baseline_test_analysis
                from .memory import FileMemory as _PreMemory
                _pre_memory = _PreMemory(embedding_store=embed_store, top_k=cfg.EMBEDDING_TOP_K)
                if source_files:
                    _pre_memory.update(source_files)
                test_analysis = perform_baseline_test_analysis(
                    _pre_memory, executor, language,
                    task_intent=_task_intent,
                )
                if test_analysis:
                    log.info("[Planning] Baseline test analysis (intent=%s):\n%s",
                             _task_intent, test_analysis)
            except Exception as _test_exc:
                log.warning("[Planning] Baseline test analysis failed: %s", _test_exc)

        # Pre-analysis: map relevant files, classify intent, enrich context
        analysis_context = planner.pre_analyze(
            args.task,
            source_files=source_files,
            kb_context_builder=kb_context_builder,
            knowledge_base=knowledge_base,
            test_analysis=test_analysis,
            language=language,
        )
        if analysis_context:
            planner_context = analysis_context + "\n\n" + planner_context
            log.info("[Planning] Pre-analysis context injected")

        MAX_PLAN_RETRIES = 3
        plan = None
        raw_steps = None

        for plan_attempt in range(1, MAX_PLAN_RETRIES + 1):
            display.show_status(
                f"Requesting steps from planner...{f' (retry {plan_attempt})' if plan_attempt > 1 else ''}"
            )
            plan = planner.process(args.task, context=planner_context)
            log.info(f"Plan (attempt {plan_attempt}):\n{plan}")

            # ── 10. Parse steps + dependencies ──
            from .plan_step import (
                parse_structured_plan, is_structured_plan, validate_plan,
                fix_import_dependencies,
                steps_as_text_list, steps_dependencies_dict,
                from_legacy_steps, parse_heuristic_plan, PlanStep,
            )
            plan_steps_parsed: list[PlanStep] | None = None

            _is_structured = is_structured_plan(plan)
            log.info(f"[Plan] Structured plan detected: {_is_structured}")
            if _is_structured:
                plan_steps_parsed = parse_structured_plan(plan)
                if plan_steps_parsed:
                    log.info(
                        f"[Plan] Parsed {len(plan_steps_parsed)} structured steps: "
                        f"{[(s.id, s.step_type, s.index) for s in plan_steps_parsed]}"
                    )
                    errors = validate_plan(plan_steps_parsed)
                    if errors:
                        log.warning(f"[Plan] Validation warnings: {errors}")
                    dep_fixes = fix_import_dependencies(plan_steps_parsed)
                    if dep_fixes:
                        log.info(f"[Plan] Auto-fixed import dependencies: {dep_fixes}")
                    raw_steps = steps_as_text_list(plan_steps_parsed)
                else:
                    log.warning("[Plan] Structured parse returned 0 steps, falling back")

            if plan_steps_parsed is None:
                # Heuristic fallback: handles weaker LLMs that output markdown
                # headers with **Key:** value metadata instead of --STEP format.
                heuristic_steps = parse_heuristic_plan(plan)
                if heuristic_steps:
                    log.info(
                        f"[Plan] Heuristic parser extracted {len(heuristic_steps)} "
                        f"steps from non-standard format"
                    )
                    dep_fixes = fix_import_dependencies(heuristic_steps)
                    if dep_fixes:
                        log.info(f"[Plan] Auto-fixed import dependencies: {dep_fixes}")
                    plan_steps_parsed = heuristic_steps
                    raw_steps = steps_as_text_list(plan_steps_parsed)

            if plan_steps_parsed is None:
                log.info("[Plan] Using legacy step parser (no structured plan)")
                raw_steps = executor.parse_plan_steps(plan)

            if not raw_steps:
                log.warning(f"Plan attempt {plan_attempt}: no steps parsed")
                if plan_attempt < MAX_PLAN_RETRIES:
                    continue
                log.error("Could not parse any steps from the plan.")
                print("\n  [ERROR] Could not parse any steps. Check the log file.\n")
                return

            # Validate plan quality
            is_valid, reason = Executor.validate_plan_quality(raw_steps)
            if is_valid:
                break

            log.warning(f"Plan attempt {plan_attempt} rejected: {reason}")
            if plan_attempt < MAX_PLAN_RETRIES:
                display.show_status(f"Plan too vague ({reason}), retrying...")
            else:
                log.warning(f"Proceeding with low-quality plan after {MAX_PLAN_RETRIES} attempts")
                print(f"\n  [WARN] Plan quality is low ({reason}). You may want to replan or edit.\n")

        if plan_steps_parsed is not None:
            steps = steps_as_text_list(plan_steps_parsed)
            dependencies = steps_dependencies_dict(plan_steps_parsed)
        else:
            steps, dependencies = executor.parse_step_dependencies(raw_steps)

        # ── 10b. Post-plan optimization ──
        pre_opt_count = len(steps)
        if plan_steps_parsed is not None:
            # Structured path: optimize directly on PlanStep objects
            plan_steps_parsed = optimize_structured_plan(
                plan_steps_parsed, knowledge_base=knowledge_base,
                kb_context_builder=kb_context_builder,
                language=language)
            steps = steps_as_text_list(plan_steps_parsed)
            dependencies = steps_dependencies_dict(plan_steps_parsed)
        else:
            # Legacy path
            steps, dependencies = optimize_plan(
                steps, knowledge_base=knowledge_base,
                kb_context_builder=kb_context_builder,
                dependencies=dependencies,
                language=language)
            plan_steps_parsed = from_legacy_steps(steps, dependencies)
        if len(steps) < pre_opt_count:
            log.info(f"[Planning] Optimized: {pre_opt_count} → {len(steps)} steps")

        # ── 11. Plan approval loop ──
        if args.auto:
            log.info(f"Auto-approved {len(steps)} steps (--auto mode)")
        while not args.auto:
            display.stop_spinner()
            # Reattach dependency markers so they are visible and editable in TUI
            display_steps = []
            for i, step in enumerate(steps):
                if dependencies.get(i):
                    deps_str = ", ".join(str(d + 1) for d in sorted(dependencies[i]))
                    display_steps.append(f"{step} (depends: {deps_str})")
                else:
                    display_steps.append(f"{step} (depends: none)")

            # Try TUI editor first, fall back to text-based approval
            action, removed, edited_steps = CLIDisplay.prompt_plan_approval(
                display_steps, use_tui=True)
            if action == "approve":
                break
            elif action == "replan":
                display.show_status("Re-planning...")
                plan = planner.process(args.task, context=planner_context)
                log.info(f"Re-plan:\n{plan}")

                if is_structured_plan(plan):
                    plan_steps_parsed = parse_structured_plan(plan)
                    if plan_steps_parsed:
                        dep_fixes = fix_import_dependencies(plan_steps_parsed)
                        if dep_fixes:
                            log.info(f"[Plan] Auto-fixed import dependencies: {dep_fixes}")
                        raw_steps = steps_as_text_list(plan_steps_parsed)
                    else:
                        raw_steps = executor.parse_plan_steps(plan)
                        plan_steps_parsed = None
                else:
                    raw_steps = executor.parse_plan_steps(plan)
                    plan_steps_parsed = None

                if not raw_steps:
                    log.error("Could not parse any steps from re-plan.")
                    print("\n  [ERROR] Could not parse re-plan steps.\n")
                    return

                if plan_steps_parsed is not None:
                    steps = steps_as_text_list(plan_steps_parsed)
                    dependencies = steps_dependencies_dict(plan_steps_parsed)
                else:
                    steps, dependencies = executor.parse_step_dependencies(raw_steps)

                if plan_steps_parsed is not None:
                    plan_steps_parsed = optimize_structured_plan(
                        plan_steps_parsed, knowledge_base=knowledge_base,
                        kb_context_builder=kb_context_builder,
                        language=language)
                    steps = steps_as_text_list(plan_steps_parsed)
                    dependencies = steps_dependencies_dict(plan_steps_parsed)
                else:
                    steps, dependencies = optimize_plan(
                        steps, knowledge_base=knowledge_base,
                        kb_context_builder=kb_context_builder,
                        dependencies=dependencies,
                        language=language)
                    plan_steps_parsed = from_legacy_steps(steps, dependencies)
            elif action == "edit" and edited_steps:
                new_steps, new_deps = executor.parse_step_dependencies(edited_steps)
                # Preserve structured PlanStep metadata when possible
                if plan_steps_parsed and len(new_steps) == len(steps):
                    # Same number of steps — check if descriptions match
                    _old = [s.strip() for s in steps]
                    _new = [s.strip() for s in new_steps]
                    if _old == _new:
                        # No actual changes — keep structured metadata intact
                        log.info("[Plan] Edit returned unchanged steps, preserving structured metadata")
                        steps = new_steps
                        dependencies = new_deps
                    else:
                        # Steps changed — try to re-match by description overlap
                        steps = new_steps
                        dependencies = new_deps
                        plan_steps_parsed = _rematch_plan_steps(
                            steps, plan_steps_parsed, dependencies)
                else:
                    # Step count changed — still try to re-match by description
                    # to preserve structured metadata (type, command, target_files)
                    steps = new_steps
                    dependencies = new_deps
                    if plan_steps_parsed:
                        plan_steps_parsed = _rematch_plan_steps(
                            steps, plan_steps_parsed, dependencies)
                    else:
                        plan_steps_parsed = from_legacy_steps(steps, dependencies)

        display.set_steps(steps)
        display.render()
        log.info(f"Approved {len(steps)} steps.")

        memory = FileMemory(embedding_store=embed_store, top_k=cfg.EMBEDDING_TOP_K)

        # Pre-load existing source files into memory so the coder
        # can see and modify them instead of creating new files
        if source_files:
            memory.update(source_files)
            log.info(f"Pre-loaded {len(source_files)} source files into memory")

    # ── 11b. Project analysis phase ──
    # Build structured ProjectContext from static analysis + LLM enrichment.
    # Gives Coder and Tester awareness of end-to-end goal, installed packages,
    # import patterns, and test strategy.
    # On resume, reuse the saved ProjectContext instead of calling LLM again.
    _resumed_pc = locals().get('_resumed_project_context')
    if resuming and _resumed_pc is not None:
        # Checkpoint has full enriched ProjectContext — reuse it
        project_context = _resumed_pc
        log.info(
            "[Analysis] Reusing ProjectContext from checkpoint (0 LLM tokens): "
            "lang=%s, fw=%s, test_fw=%s, %d pkgs, %d testable units",
            project_context.language, project_context.framework,
            project_context.test_framework,
            len(project_context.installed_packages),
            len(project_context.testable_units),
        )
    elif resuming:
        # Old checkpoint without ProjectContext — use static analysis only (0 LLM tokens)
        project_context = build_project_context(
            args.task, steps,
            source_files=source_files or {},
            language=language,
        )
        log.info(
            "[Analysis] Resume: static analysis only, skipping LLM enrichment "
            "(0 LLM tokens): lang=%s",
            project_context.language,
        )
    else:
        project_context = build_project_context(
            args.task, steps,
            source_files=source_files or {},
            language=language,
        )
        display.show_status("Analysing project...")
        try:
            analyser = AnalyseAgent(
                "Analyser", "Senior Technical Analyst",
                "Analyse the task and project to guide downstream agents.",
                _make_llm_for_agent("analyser"))
            project_context = analyser.enrich_context(
                project_context, args.task, steps, source_files or {})
            log.info(
                "[Analysis] ProjectContext: lang=%s, fw=%s, test_fw=%s, "
                "%d pkgs, %d testable units",
                project_context.language, project_context.framework,
                project_context.test_framework,
                len(project_context.installed_packages),
                len(project_context.testable_units),
            )
        except Exception as analyse_exc:
            log.warning("[Analysis] LLM enrichment failed (non-fatal): %s",
                        analyse_exc)

    # ── 12. Build execution waves ──
    # Use phase-aware wave builder when structured plan steps are available.
    # This ensures all sub-steps of phase N (e.g. 1.1, 1.2) complete before
    # phase N+1 (e.g. 2.1, 2.2) begins, even when explicit depends: is missing.
    if plan_steps_parsed:
        plan_waves = _build_plan_waves(plan_steps_parsed)
        waves = [[s.index for s in w] for w in plan_waves]
    else:
        waves = build_step_waves(steps, dependencies)
    log.info(f"Execution waves: {waves}")

    # Build step reports for HTML output
    step_reports = [StepReport(index=i, text=steps[i]) for i in range(len(steps))]

    # ── 13. Execute waves ──
    pipeline_success = True

    for wave_idx, wave in enumerate(waves):
        # Filter out already-completed steps (for resume)
        pending = [i for i in wave if i >= start_from]
        if not pending:
            continue

        log.info(f"Wave {wave_idx+1}: executing steps {[i+1 for i in pending]}")

        if len(pending) == 1:
            # Single step — execute directly
            idx = pending[0]
            step_text = steps[idx]
            _ps = next((s for s in plan_steps_parsed if s.index == idx), None) if plan_steps_parsed else None
            if _ps is None and plan_steps_parsed:
                log.warning(
                    "[PlanStep] No PlanStep found for idx=%d. "
                    "Available indices: %s",
                    idx, [s.index for s in plan_steps_parsed],
                )
            idx, success, error_info = _execute_step(
                idx, step_text,
                steps=steps,
                llm_client=llm_client, executor=executor,
                coder=coder, reviewer=reviewer, tester=tester,
                task=args.task, memory=memory, display=display,
                language=language, cfg=cfg, auto=args.auto,
                search_agent=search_agent,
                kb_context_builder=kb_context_builder,
                knowledge_base=knowledge_base,
                project_context=project_context,
                plan_step=_ps,
                all_plan_steps=plan_steps_parsed,
            )

            if success:
                step_results[idx] = "done"
                ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                save_checkpoint(checkpoint_file, args.task, steps, idx,
                                memory.as_dict(), step_results, language,
                                display_state=ds,
                                plan_steps=plan_steps_parsed,
                                project_context=project_context)

                # Budget check after step
                if display.budget_check(cfg.BUDGET_LIMIT):
                    log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}). Halting.")
                    pipeline_success = False
                    break
            else:
                # Diagnosis loop
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    steps=steps,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=args.task, memory=memory, display=display,
                    language=language, cfg=cfg, auto=args.auto,
                    search_agent=search_agent,
                    kb_context_builder=kb_context_builder,
                    knowledge_base=knowledge_base,
                    project_context=project_context,
                    plan_step=_ps,
                    all_plan_steps=plan_steps_parsed,
                )
                if fixed:
                    display.complete_step(idx, "done")
                    step_results[idx] = "done"
                    ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                    save_checkpoint(checkpoint_file, args.task, steps, idx,
                                    memory.as_dict(), step_results, language,
                                    display_state=ds,
                                    plan_steps=plan_steps_parsed,
                                project_context=project_context)

                    # Budget check after fix
                    if display.budget_check(cfg.BUDGET_LIMIT):
                        log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}). Halting.")
                        pipeline_success = False
                        break
                else:
                    pipeline_success = False
                    break
        else:
            # Multi-step wave — execute in parallel
            failed_steps: list[tuple[int, str]] = []

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(len(pending), 4)) as pool:
                futures = {}
                for idx in pending:
                    _ps = next((s for s in plan_steps_parsed if s.index == idx), None) if plan_steps_parsed else None
                    if _ps is None and plan_steps_parsed:
                        log.warning(
                            "[PlanStep] No PlanStep found for idx=%d. "
                            "Available indices: %s",
                            idx, [s.index for s in plan_steps_parsed],
                        )
                    f = pool.submit(
                        _execute_step, idx, steps[idx],
                        steps=steps,
                        llm_client=llm_client, executor=executor,
                        coder=coder, reviewer=reviewer, tester=tester,
                        task=args.task, memory=memory, display=display,
                        language=language, cfg=cfg, auto=args.auto,
                        search_agent=search_agent,
                        kb_context_builder=kb_context_builder,
                        knowledge_base=knowledge_base,
                        project_context=project_context,
                        plan_step=_ps,
                        all_plan_steps=plan_steps_parsed,
                    )
                    futures[f] = idx

                for future in as_completed(futures):
                    idx, success, error_info = future.result()
                    if success:
                        step_results[idx] = "done"
                    else:
                        failed_steps.append((idx, error_info))

                # Budget check after wave
                if display.budget_check(cfg.BUDGET_LIMIT):
                    log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}) after parallel wave. Halting.")
                    pipeline_success = False
                    break

            # Save checkpoint for completed steps
            max_completed = max(
                (i for i in step_results if step_results[i] == "done"),
                default=start_from - 1)
            ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
            save_checkpoint(checkpoint_file, args.task, steps, max_completed,
                            memory.as_dict(), step_results, language,
                            display_state=ds,
                            plan_steps=plan_steps_parsed)

            # Handle failures
            for idx, error_info in failed_steps:
                step_text = steps[idx]
                _ps = next((s for s in plan_steps_parsed if s.index == idx), None) if plan_steps_parsed else None
                fixed = _run_diagnosis_loop(
                    idx, step_text, error_info,
                    steps=steps,
                    llm_client=llm_client, executor=executor,
                    coder=coder, reviewer=reviewer, tester=tester,
                    task=args.task, memory=memory, display=display,
                    language=language, cfg=cfg, auto=args.auto,
                    search_agent=search_agent,
                    kb_context_builder=kb_context_builder,
                    knowledge_base=knowledge_base,
                    project_context=project_context,
                    plan_step=_ps,
                    all_plan_steps=plan_steps_parsed,
                )
                if fixed:
                    display.complete_step(idx, "done")
                    step_results[idx] = "done"
                    ds = {"elapsed": time.monotonic() - display.start_time, "steps": display.steps}
                    save_checkpoint(checkpoint_file, args.task, steps, idx,
                                    memory.as_dict(), step_results, language,
                                    display_state=ds,
                                    plan_steps=plan_steps_parsed,
                                project_context=project_context)

                    # Budget check after fix
                    if display.budget_check(cfg.BUDGET_LIMIT):
                        log.error(f"Budget exceeded (${token_tracker.total_cost:.4f}). Halting.")
                        pipeline_success = False
                        break
                else:
                    pipeline_success = False
                    break

            if not pipeline_success:
                break

    # ── 13.5. Final cross-step test verification ──
    # Re-run all session test files together to catch regressions where a
    # source fix in step N broke tests that already passed in step M.
    if pipeline_success:
        from .pipeline import run_final_test_verification
        verif_ok, verif_err = run_final_test_verification(
            memory=memory,
            executor=executor,
            coder=coder,
            display=display,
            language=language,
            task=args.task,
            cfg=cfg,
            project_context=project_context,
            kb_context_builder=kb_context_builder,
        )
        if not verif_ok:
            pipeline_success = False
            log.warning(f"[FinalVerify] Pipeline marked failed: {verif_err[:200]}")

    # ── 14. Populate step reports from display state ──
    for i, sr in enumerate(step_reports):
        if i < len(display.steps):
            ds = display.steps[i]
            sr.status = ds.get("status", sr.status)
            sr.step_type = ds.get("type", sr.step_type)
            tokens = ds.get("tokens", {})
            sr.tokens_sent = tokens.get("sent", 0)
            sr.tokens_recv = tokens.get("recv", 0)
            sr.duration = ds.get("duration", 0.0)

    # ── 15. Extract knowledge (runs on both success and failure) ──
    # Patterns/fixes from completed steps are valuable regardless of
    # overall pipeline outcome — especially fixes learned from failures.
    if knowledge_base:
        try:
            knowledge_base.extract_from_run(
                args.task, steps, memory.as_dict(), llm_client)
        except Exception as e:
            log.warning(f"Knowledge extraction failed: {e}")

    # ── 16. Finish ──
    if pipeline_success:
        display.finish(success=True)
        clear_checkpoint(checkpoint_file)
        log.info(f"Finished. Total tokens: {token_tracker.total_tokens} "
                 f"(sent={token_tracker.total_prompt_tokens}, "
                 f"recv={token_tracker.total_completion_tokens})")

        # Generate HTML report
        if args.report and not args.no_report:
            try:
                token_usage = {
                    "sent": token_tracker.total_prompt_tokens,
                    "recv": token_tracker.total_completion_tokens,
                    "total": token_tracker.total_tokens,
                    "cost": token_tracker.total_cost,
                    "total_time": time.monotonic() - display.start_time,
                }
                report_path = generate_html_report(
                    args.task, step_reports, token_usage,
                    pipeline_success=True, output_dir=cfg.REPORT_DIR)
                log.info(f"Report generated: {report_path}")
                print(f"\n  📄 Report: {report_path}")
            except Exception as e:
                log.warning(f"Report generation failed: {e}")

        # Git: offer commit
        if use_git and git_utils.has_changes():
            if args.auto:
                git_choice = "commit"
                log.info("Auto-committing changes (--auto mode)")
            else:
                display.stop_spinner()
                git_choice = CLIDisplay.prompt_git_action("complete")
            if git_choice == "commit":
                ok, msg = git_utils.commit_changes(
                    f"AgentChanti: {args.task[:60]}")
                print(f"  {'Committed!' if ok else 'Commit failed: ' + msg}")
            if checkpoint_branch:
                git_utils.delete_checkpoint_branch(checkpoint_branch)
    else:
        display.finish(success=False)
        log.info(f"Pipeline failed. Total tokens: {token_tracker.total_tokens}")

        # Generate HTML report even on failure
        if args.report and not args.no_report:
            try:
                token_usage = {
                    "sent": token_tracker.total_prompt_tokens,
                    "recv": token_tracker.total_completion_tokens,
                    "total": token_tracker.total_tokens,
                    "cost": token_tracker.total_cost,
                    "total_time": time.monotonic() - display.start_time,
                }
                report_path = generate_html_report(
                    args.task, step_reports, token_usage,
                    pipeline_success=False, output_dir=cfg.REPORT_DIR)
                log.info(f"Report generated: {report_path}")
                print(f"\n  📄 Report: {report_path}")
            except Exception as e:
                log.warning(f"Report generation failed: {e}")

        # Git: offer rollback
        if use_git and checkpoint_branch:
            if args.auto:
                git_choice = "skip"
                log.info("Auto-skipping git rollback (--auto mode)")
            else:
                display.stop_spinner()
                git_choice = CLIDisplay.prompt_git_action("failed")
            if git_choice == "rollback":
                ok, msg = git_utils.rollback_to_branch(checkpoint_branch)
                print(f"  {'Rolled back!' if ok else 'Rollback failed: ' + msg}")
            elif git_choice == "commit":
                ok, msg = git_utils.commit_changes(
                    f"AgentChanti (partial): {args.task[:50]}")
                print(f"  {'Committed!' if ok else 'Commit failed: ' + msg}")

    # ── 15. Cleanup ──
    if kb_runtime_watcher is not None:
        try:
            kb_runtime_watcher.stop()
        except Exception:
            pass
    executor.cleanup()


if __name__ == "__main__":
    main()

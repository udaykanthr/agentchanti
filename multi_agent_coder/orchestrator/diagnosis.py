"""
Diagnosis and fix helpers — analyze step failures and apply fixes.
"""

import ast
import os
import re

from ..executor import Executor
from ..cli_display import CLIDisplay, token_tracker, log
from ..diff_display import show_diffs, _detect_hazards, HAZARD_BLOCK

from .memory import FileMemory
from .step_handlers import (
    _shell_instructions, _strip_protected_files, _apply_content_fixes,
    _detect_subproject_root, _find_css_conflicts, _detect_target_files,
)
from .classification import _extract_commands_from_text, _looks_like_command, resolve_cmd_placeholders


# ---------------------------------------------------------------------------
# Tier 1: Deterministic pre-investigation (0 LLM tokens)
# ---------------------------------------------------------------------------

# Regex patterns for extracting structured info from error output
_IMPORT_ERROR_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError)[^'\"]*['\"]?([\w.\-/]+)",
    re.IGNORECASE,
)
_CIRCULAR_IMPORT_RE = re.compile(
    r"cannot import name .+ from partially initialized module",
    re.IGNORECASE,
)
_ATTR_ERROR_RE = re.compile(
    r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
)
_TRACEBACK_FILE_RE = re.compile(
    r'File\s+"([^"]+)",\s+line\s+(\d+)',
)


def _pre_investigate(error_info: str, memory: FileMemory,
                     executor: Executor, step_idx: int,
                     language: str | None = None) -> str:
    """Run deterministic (zero-LLM-token) investigation before diagnosis.

    Parses the error output for structured signals — import failures,
    missing attributes, circular imports, file-not-found — and runs
    targeted filesystem checks to discover the root cause.

    Returns a human-readable evidence block to inject into the diagnosis
    prompt, or empty string if nothing useful was found.
    """
    if not error_info:
        return ""

    findings: list[str] = []

    # Check for circular import first — it's a specific sub-case of
    # ImportError that needs different handling than "module not found".
    _is_circular = bool(_CIRCULAR_IMPORT_RE.search(error_info))

    # ── 1. Module/import errors: trace where the module actually lives ────
    import_match = (
        _IMPORT_ERROR_RE.search(error_info) if not _is_circular else None
    )
    if import_match:
        module_name = import_match.group(1).strip("'\"")
        findings.append(f"Import target: '{module_name}'")

        # Convert dotted module to possible file paths
        module_parts = module_name.replace('-', '_').split('.')
        possible_paths = []
        # e.g. "snake_game.effects" → snake_game/effects.py, snake_game/effects/__init__.py
        rel_path = os.path.join(*module_parts)
        possible_paths.append(rel_path + '.py')
        possible_paths.append(os.path.join(rel_path, '__init__.py'))
        # Also try with nested structure: snake_game/snake_game/effects.py
        if len(module_parts) >= 2:
            nested = os.path.join(module_parts[0], *module_parts)
            possible_paths.append(nested + '.py')
            possible_paths.append(os.path.join(nested, '__init__.py'))

        # Check which paths actually exist on disk
        found_at: list[str] = []
        not_found: list[str] = []
        for p in possible_paths:
            if os.path.isfile(p):
                found_at.append(p)
            else:
                not_found.append(p)

        # Also do a broader search for the leaf module file
        leaf_name = module_parts[-1] + '.py'
        try:
            success, find_output = executor.run_command(
                f"find . -name '{leaf_name}' -not -path '*/venv/*' "
                f"-not -path '*/.git/*' -not -path '*/__pycache__/*'",
                timeout=5,
            )
            if success and find_output.strip():
                disk_locations = [
                    p.strip() for p in find_output.strip().splitlines()
                    if p.strip()
                ]
                findings.append(
                    f"File '{leaf_name}' found on disk at: "
                    + ', '.join(disk_locations)
                )
                # Detect nested directory collision
                for loc in disk_locations:
                    parts = loc.replace('\\', '/').lstrip('./').split('/')
                    if (len(parts) >= 3
                            and parts[0] == parts[1]
                            and parts[0] == module_parts[0]):
                        findings.append(
                            f"⚠ NESTED DIRECTORY COLLISION: '{parts[0]}/{parts[1]}/' — "
                            f"Python resolves '{module_parts[0]}' to the outer "
                            f"'{parts[0]}/' directory which has no '{leaf_name}'. "
                            f"The actual file is nested inside "
                            f"'{parts[0]}/{parts[1]}/{parts[2]}'. "
                            f"Fix: move the package contents up one level, or "
                            f"adjust sys.path / PYTHONPATH."
                        )
            elif not found_at:
                findings.append(
                    f"File '{leaf_name}' NOT found anywhere on disk — "
                    f"the module was never created."
                )
        except Exception:
            pass

        if found_at:
            findings.append(f"Module file exists at: {', '.join(found_at)}")
            # Check if there's an __init__.py in the parent dirs
            for fp in found_at:
                parent = os.path.dirname(fp)
                init_path = os.path.join(parent, '__init__.py')
                if parent and not os.path.isfile(init_path):
                    findings.append(
                        f"⚠ Missing __init__.py in '{parent}/' — "
                        f"Python won't treat it as a package."
                    )
        elif not_found and not found_at:
            findings.append(
                f"Expected locations checked (all missing): "
                + ', '.join(not_found)
            )

        # List what's in the top-level module directory
        top_dir = module_parts[0]
        if os.path.isdir(top_dir):
            try:
                entries = sorted(os.listdir(top_dir))
                # Filter out noise
                entries = [
                    e for e in entries
                    if e not in ('__pycache__', 'venv', '.venv',
                                 'node_modules', '.git', '.pytest_cache')
                ]
                findings.append(
                    f"Contents of '{top_dir}/': {', '.join(entries)}"
                )
                # Check for nested same-name dir
                if top_dir in entries and os.path.isdir(
                    os.path.join(top_dir, top_dir)
                ):
                    inner_entries = sorted(os.listdir(
                        os.path.join(top_dir, top_dir)
                    ))
                    inner_entries = [
                        e for e in inner_entries
                        if e not in ('__pycache__',)
                    ]
                    findings.append(
                        f"Contents of '{top_dir}/{top_dir}/': "
                        + ', '.join(inner_entries)
                    )
            except OSError:
                pass

    # ── 2. Circular import detection ──────────────────────────────────────
    if _CIRCULAR_IMPORT_RE.search(error_info):
        # Extract the import chain from the traceback
        tb_files = _TRACEBACK_FILE_RE.findall(error_info)
        if tb_files:
            chain = [f"{f}:{line}" for f, line in tb_files]
            findings.append(
                f"⚠ CIRCULAR IMPORT detected. Import chain:\n  "
                + '\n  → '.join(chain)
            )
            # Read the imports from the first and last file in the chain
            for fpath, _line in [tb_files[0], tb_files[-1]]:
                for mem_path, content in memory.all_files().items():
                    if fpath.endswith(mem_path) or mem_path.endswith(
                        fpath.lstrip('./')
                    ):
                        # Extract import lines
                        imports = [
                            l.strip() for l in content.splitlines()
                            if l.strip().startswith(('import ', 'from '))
                        ]
                        if imports:
                            findings.append(
                                f"Imports in '{mem_path}':\n  "
                                + '\n  '.join(imports[:10])
                            )
                        break

    # ── 3. AttributeError: check if the class/method exists ──────────────
    attr_match = _ATTR_ERROR_RE.search(error_info)
    if attr_match:
        class_name = attr_match.group(1)
        attr_name = attr_match.group(2)
        # Search memory files for the class definition
        found_class = False
        for fpath, content in memory.all_files().items():
            if fpath.startswith('_'):
                continue
            if f'class {class_name}' in content:
                found_class = True
                has_attr = (
                    f'def {attr_name}' in content
                    or f'{attr_name} =' in content
                    or f'{attr_name}:' in content
                )
                if has_attr:
                    findings.append(
                        f"'{class_name}.{attr_name}' EXISTS in '{fpath}' — "
                        f"the error may be from a stale import or wrong class."
                    )
                else:
                    # List what methods the class DOES have
                    methods = re.findall(
                        r'def (\w+)\s*\(', content
                    )
                    findings.append(
                        f"'{class_name}' in '{fpath}' does NOT have "
                        f"'{attr_name}'. Available methods: "
                        + ', '.join(methods[:15])
                    )
                break
        if not found_class:
            findings.append(
                f"Class '{class_name}' not found in any memory file."
            )

    # ── 4. Traceback file analysis: check file existence on disk ─────────
    tb_files = _TRACEBACK_FILE_RE.findall(error_info)
    if tb_files and not import_match and not attr_match:
        for fpath, line_no in tb_files[:5]:
            if not os.path.isfile(fpath):
                findings.append(
                    f"⚠ Traceback references '{fpath}' but file does "
                    f"not exist on disk."
                )

    if not findings:
        return ""

    result = (
        "PRE-INVESTIGATION FINDINGS (automated, 0 LLM tokens):\n"
        + "\n".join(f"• {f}" for f in findings)
        + "\n"
    )
    log.info("Step %d: Pre-investigation found %d finding(s)", step_idx + 1, len(findings))
    return result


def _diagnose_failure(step_text: str, step_type: str, error_info: str,
                      memory: FileMemory, llm_client, display: CLIDisplay,
                      step_idx: int,
                      search_agent=None,
                      language: str | None = None,
                      previous_diagnosis: str | None = None,
                      kb_context_builder=None,
                      error_route=None,
                      intent_spec=None,
                      executor: Executor | None = None) -> str:
    display.step_info(step_idx, "Analyzing failure root cause...")

    # ── Tier 1: Deterministic pre-investigation ──────────────────────────
    # Parse the error for structured signals and run filesystem checks
    # before invoking the LLM.  Zero tokens, catches structural issues
    # (nested dirs, missing __init__.py, wrong import paths) that the
    # diagnosis LLM consistently misdiagnoses.
    pre_investigation = ""
    if executor is not None:
        try:
            pre_investigation = _pre_investigate(
                error_info, memory, executor, step_idx, language=language,
            )
        except Exception as _pre_exc:
            log.debug("Step %d: Pre-investigation failed: %s", step_idx + 1, _pre_exc)

    # ── KB error-fix lookup using actual error output ────
    # Pass the real error_info into build_context so the ErrorDict
    # can match against stack traces / error messages, not just the
    # step description.
    kb_error_context = ""
    if kb_context_builder is not None:
        try:
            kb_ctx = kb_context_builder.build_context(
                task_description=step_text,
                error_output=error_info,
                max_tokens=2000,
                language=language,
                step_type=step_type,
            )
            if kb_ctx.error_fixes or kb_ctx.behavioral_instructions:
                kb_error_context = kb_context_builder.format_context_for_prompt(kb_ctx)
                log.info(f"Step {step_idx+1}: KB matched {len(kb_ctx.error_fixes)} "
                         f"error fix patterns for diagnosis")
                # Persist error-specific KB context to memory so that if
                # the step is re-executed, _handle_cmd_step's command
                # generation prompt includes the error fix patterns.
                memory._kb_context = kb_error_context
        except Exception as exc:
            log.debug(f"Step {step_idx+1}: KB error lookup during diagnosis failed: {exc}")

    # ── Build prior step context (CMD outputs) ───────────────────────────────
    prior_context = ""
    all_files = memory.all_files()
    for i in range(step_idx):
        key = f"_cmd_output/step_{i+1}.txt"
        if key in all_files:
            prior_context += f"Step {i+1} output:\n{all_files[key]}\n\n"
    if prior_context:
        prior_context = "Previously executed steps:\n" + prior_context

    # Extract installed packages from prior CMD outputs so the search agent
    # can include them in its query and avoid recommending conflicting versions.
    _installed_note = ""
    _PKG_INSTALL_RE = re.compile(
        r'(?:npm install|pip install|yarn add|pnpm add)\s+(.+)', re.IGNORECASE)
    _installed_pkgs: list[str] = []
    for i in range(step_idx):
        raw = all_files.get(f"_cmd_output/step_{i+1}.txt", "")
        for line in raw.splitlines():
            m = _PKG_INSTALL_RE.search(line)
            if m:
                # Extract package names (strip flags like --save-dev)
                pkgs = [p for p in m.group(1).split()
                        if not p.startswith('-') and p]
                _installed_pkgs.extend(pkgs)
    if _installed_pkgs:
        _installed_note = f"Installed: {', '.join(dict.fromkeys(_installed_pkgs))}"

    # ── Optional: search the web for error documentation ────
    # Gated by error_route: skip entirely for code-logic or KB-matched errors.
    search_context = ""
    _skip_web = error_route is not None and error_route.skip_web
    if _skip_web:
        log.info(f"Step {step_idx+1}: ErrorRouter skip_web=True "
                 f"({error_route.source_type}, {error_route.reason}) — skipping web search")
    if search_agent is not None and not _skip_web:
        display.step_info(step_idx, "Searching web for error documentation...")
        try:
            _search_kb = getattr(memory, '_kb_context', '')
            if _installed_note:
                _search_kb = f"{_installed_note}\n{_search_kb}" if _search_kb else _installed_note
            _query_override = error_route.query_hint if error_route else None
            search_context = search_agent.search_for_error(
                error_info, step_text, language=language,
                kb_context=_search_kb,
                query_override=_query_override)
            if search_context:
                log.info(f"Step {step_idx+1}: Search agent found documentation")
        except Exception as exc:
            log.warning(f"Step {step_idx+1}: Search agent error: {exc}")

    context_files = memory.related_context(step_text)

    # Detect sub-project root to inform the LLM
    subproject = _detect_subproject_root(memory)

    # Inject task briefing so diagnosis never removes features to bypass failures.
    _diag_briefing = getattr(memory, '_task_briefing', '')
    _briefing_block = (
        "TASK BRIEFING (the overall goal — respect Preserve and Key constraint):\n"
        f"{_diag_briefing}\n\n"
    ) if _diag_briefing else ""

    _intent_block = ""
    if intent_spec is not None and intent_spec.constraints:
        _intent_block = (
            "INTENT CONSTRAINTS (from requirements analysis — respect these when fixing):\n"
            + "\n".join(f"- {c}" for c in intent_spec.constraints)
            + "\n\n"
        )

    prompt = (
        "A step in our automated coding pipeline has FAILED after multiple retries.\n"
        "Analyze the failure and provide a concrete fix.\n\n"
        f"{_briefing_block}"
        f"{_intent_block}"
        "CRITICAL: Do NOT remove, comment out, or skip the feature/component that is failing. "
        "Fix the root cause instead.\n\n"
        f"Step {step_idx+1}: {step_text}\n"
        f"Step type: {step_type}\n\n"
        f"Error details:\n{error_info}\n\n"
    )
    if pre_investigation:
        prompt += (
            f"{pre_investigation}\n"
            "The findings above were gathered by automated analysis (file system "
            "checks, import tracing). Use them to ground your diagnosis — they "
            "are FACTS, not guesses.\n\n"
        )
    if error_route and error_route.source_type in ("code", "kb"):
        _hint_map = {
            "code": "a code/logic bug in the existing files — focus on the written code, not package installation",
            "kb":   "a known pattern covered by KB docs — apply the KB fix below",
        }
        prompt += (
            f"DIAGNOSIS HINT: This is likely {_hint_map[error_route.source_type]}.\n\n"
        )
    if kb_error_context:
        prompt += (
            "The following error-fix patterns from the knowledge base match "
            "this error. Use them to guide your diagnosis:\n\n"
            f"{kb_error_context}\n\n"
        )
    if subproject:
        prompt += (
            f"IMPORTANT: The project source code lives in the '{subproject}/' "
            f"subdirectory. All commands (npm install, pip install, test "
            f"commands, etc.) must target that directory, NOT the repository "
            f"root. File paths should include the '{subproject}/' prefix.\n\n"
        )
    if search_context:
        prompt += (
            "The following web search results may contain relevant documentation,\n"
            "error explanations, or solutions. Use them to inform your fix:\n\n"
            f"{search_context}\n\n"
        )
    if kb_error_context and search_context:
        prompt += (
            "PRIORITY: The KB error-fix patterns above are the authoritative source "
            "for this project. If web search results suggest a different approach "
            "(e.g. different package versions, downgrading), prefer the KB guidance. "
            "Do NOT install different versions of packages that were already installed "
            "in prior steps — work with what is already installed.\n\n"
        )
    if prior_context:
        prompt += f"{prior_context}\n"
    if context_files:
        prompt += f"Relevant project files:\n{context_files}\n\n"

    # ── Django TemplateDoesNotExist → INSTALLED_APPS hint ─────────────────
    # When Django raises TemplateDoesNotExist the most common root cause is a
    # new app that was never added to INSTALLED_APPS (APP_DIRS=True only scans
    # installed apps).  Detect this and inject both the error app name and the
    # current settings.py so the LLM can produce a targeted fix.
    _tpl_dne_m = re.search(
        r'TemplateDoesNotExist[^\n]*?([\w/.-]+/[\w.-]+\.html)',
        error_info or "",
    )
    if _tpl_dne_m:
        _missing_tpl = _tpl_dne_m.group(1)          # e.g. "homepage/index.html"
        _app_name = _missing_tpl.split("/")[0]       # e.g. "homepage"
        # Locate settings.py — look in memory first, then disk
        _settings_content: str | None = None
        _settings_path: str | None = None
        for _fpath, _fcontent in memory.all_files().items():
            if _fpath.endswith("settings.py") and "INSTALLED_APPS" in _fcontent:
                _settings_content = _fcontent
                _settings_path = _fpath
                break
        if _settings_content is None:
            _search_root = subproject or "."
            for _dirpath, _dirnames, _filenames in os.walk(_search_root):
                _dirnames[:] = [d for d in _dirnames
                                if d not in ("venv", ".venv", "node_modules",
                                             "__pycache__", ".git")]
                if "settings.py" in _filenames:
                    _candidate = os.path.join(_dirpath, "settings.py")
                    try:
                        with open(_candidate, encoding="utf-8") as _sf:
                            _sc = _sf.read()
                        if "INSTALLED_APPS" in _sc:
                            _settings_content = _sc
                            _settings_path = _candidate
                            break
                    except OSError:
                        pass
        if _settings_content and _settings_path:
            _app_in_settings = (
                f"'{_app_name}'" in _settings_content
                or f'"{_app_name}"' in _settings_content
            )
            if not _app_in_settings:
                prompt += (
                    f"\nDJANGO INSTALLED_APPS WARNING:\n"
                    f"The error is `TemplateDoesNotExist: {_missing_tpl}`. "
                    f"The app '{_app_name}' is NOT listed in INSTALLED_APPS in "
                    f"`{_settings_path}`. With APP_DIRS=True Django only searches "
                    f"templates inside installed apps — add '{_app_name}' to "
                    f"INSTALLED_APPS to fix the template lookup.\n\n"
                    f"#### [FILE]: {_settings_path}\n"
                    f"```python\n{_settings_content}\n```\n\n"
                )
                log.info(
                    "Step %d: Injected INSTALLED_APPS hint for missing app '%s'",
                    step_idx + 1, _app_name,
                )

    # ── CSS conflict injection ──────────────────────────────────────────────
    # If the step involves a styling/colour change, inject any global CSS files
    # whose selectors may override the component-level change.  This is the
    # same detection used by the Tier-3 code-step path; without it, the
    # diagnosis LLM fixes the component but leaves the overriding rule intact.
    _diag_targets = _detect_target_files(step_text, memory)
    _diag_css_conflicts = _find_css_conflicts(step_text, _diag_targets, memory)
    if _diag_css_conflicts:
        _diag_css_injected: list[str] = []
        for _css_path, _css_content in _diag_css_conflicts.items():
            if f"#### [FILE]: {_css_path}" not in prompt:
                prompt += f"\n#### [FILE]: {_css_path}\n```css\n{_css_content}\n```\n"
            _diag_css_injected.append(_css_path)
        prompt += (
            "\nCSS OVERRIDE WARNING: The CSS file(s) shown above contain global "
            "rules that may override the component-level style change requested. "
            "For example, `.header, .footer { background-color: var(--bg); }` will "
            "cascade over a component's own `background-color` declaration. "
            "Your fix MUST also update those CSS file(s): remove the conflicting "
            "property from the shared rule, scope it more narrowly, or replace it "
            "so the user's intended visual change is actually visible.\n"
            f"CSS files to review and fix: {', '.join(_diag_css_injected)}\n\n"
        )

    # ── Enhancement #3: extract file paths from error output ──────
    # Parse file paths mentioned in the error (e.g. tracebacks, lint output)
    # and inject matching full source files from memory so the LLM sees the
    # actual buggy code, not just what related_context() guesses.
    _FILE_PATH_RE = re.compile(
        r'(?:File\s+["\'](.+?)["\']'
        r'|([\w./\\-]+\.(?:py|js|ts|jsx|tsx|rb|go|rs|java|c|cpp|cs|php)):\d+)'
    )
    _error_file_paths: set[str] = set()
    for _m in _FILE_PATH_RE.finditer(error_info or ""):
        _p = _m.group(1) or _m.group(2)
        if _p:
            _error_file_paths.add(_p)
    if _error_file_paths:
        _injected: list[str] = []
        for fpath, content in memory.all_files().items():
            # Skip files already injected in context_files or earlier
            # sections to avoid duplicating large code blocks in the prompt.
            if f"#### [FILE]: {fpath}" in prompt:
                continue
            for efp in _error_file_paths:
                if fpath.endswith(efp) or efp.endswith(fpath):
                    _injected.append(
                        f"#### [FILE]: {fpath}\n```\n{content}\n```")
                    break
        if _injected:
            prompt += (
                "Source files referenced in the error output:\n\n"
                + "\n\n".join(_injected) + "\n\n"
            )

    prompt += f"All project files: {memory.summary()}\n\n"

    if previous_diagnosis:
        prompt += (
            "IMPORTANT FEEDBACK ON YOUR PREVIOUS ATTEMPT:\n"
            "Your previous diagnosis failed because it did not produce an actionable fix in the correct format. "
            "You MUST use the exact `#### [FILE]:` marker for code fixes, or exact shell commands for CMD fixes.\n\n"
            "Your previous output was:\n"
            f"{previous_diagnosis}\n\n"
        )

    # Step-type-specific fix instructions
    if step_type == "CMD":
        prompt += (
            "This is a COMMAND step.\n"
            "If the failure is purely a command/environment error, respond with a shell command fix.\n"
            "If the failure is a CODE ERROR (e.g. failing tests, lint errors, syntax errors), provide the COMPLETE corrected file(s).\n"
            "You may provide both a file replacement and a shell command if necessary.\n"
            "Use the exact names, paths, and values from previous steps.\n"
            "Respond with:\n"
            "1. ROOT CAUSE: one-line explanation of what went wrong\n"
            "2. FIX: provide the corrected shell command inside a code block (if applicable):\n"
            "```bash\n"
            "command here\n"
            "```\n"
            "3. CODE FIX: If code needs to be modified to fix the issue, provide the COMPLETE corrected file(s). Do NOT use diffs or patches.\n"
            "   Write the ENTIRE file content, not just the changed parts.\n"
            "   CRITICAL FORMAT — the #### [FILE]: marker must be OUTSIDE and BEFORE the code block:\n"
            "   #### [FILE]: path/to/file.py\n"
            "   ```python\n"
            "   # entire file contents here\n"
            "   ```\n"
            "4. SPECIAL CASE: If the command failed because the directory is not empty (e.g. create-react-app .), "
            "the FIX is to create the app in a new subdirectory (e.g. `npx create-react-app my-app ...`) "
            "instead of the current directory.\n"
            "5. DEPRECATED / UNKNOWN COMMAND: If the error says 'Unknown command' or "
            "'command not found', the tool may have been removed or renamed in a newer version. "
            "Provide the modern replacement command. Common examples:\n"
            "   - `npm set-script` removed in npm v7+ → use `npm pkg set scripts.<name>=\"<value>\"`\n"
            "   - `npx tailwindcss init` removed in Tailwind v4 → not needed, configure via CSS\n"
            "   - If no replacement exists, achieve the same goal by editing the config file directly "
            "(e.g. add a script to package.json via a write/sed command).\n"
            f"{_shell_instructions()}"
        )
    else:
        prompt += (
            "Respond with:\n"
            "1. ROOT CAUSE: one-line explanation of what went wrong\n"
            "2. FIX: PREFER chunk format to avoid unintended changes to unrelated code.\n\n"
            "   PREFERRED — chunk format (surgical, only the broken section):\n"
            "   #### [EDIT]: path/to/file.py:function_or_class_name (lines start-end)\n"
            "   ```python\n"
            "   # complete replacement for this chunk only\n"
            "   ```\n\n"
            "   FALLBACK — full file (only when the entire file must be rewritten):\n"
            "   #### [FILE]: path/to/file.py\n"
            "   ```python\n"
            "   # entire file contents here\n"
            "   ```\n\n"
            "CRITICAL: Do NOT use diff/patch format. "
            "Use [EDIT]: for targeted fixes, [FILE]: only when truly needed.\n"
            "The [EDIT]: marker MUST be OUTSIDE and BEFORE the code block.\n"
        )

    sent_before, recv_before = token_tracker.snapshot()
    display.step_info(step_idx, "Requesting diagnosis from LLM...")

    diagnosis = llm_client.generate_response(prompt)

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    display.step_info(step_idx, "Processing diagnosis response...")
    explanation = CLIDisplay.extract_explanation(diagnosis)
    if explanation:
        display.add_llm_log(explanation, source="Diagnosis")

    log.info(f"Step {step_idx+1}: Diagnosis:\n{diagnosis}")
    return diagnosis


def _extract_echo_chain_file_writes(text: str) -> dict[str, str]:
    """Parse bash blocks where an LLM writes a file via chained echo calls.

    Detects the pattern::

        ```bash
        (
        echo line1 && echo line2 && ...
        ) > path/to/file.py
        ```

    Returns ``{filepath: content}`` for any such blocks found.  The file
    content is extracted in Python so that special shell characters (``<``,
    ``>``, ``|``) inside the source code do not cause the echo commands to
    fail on Windows CMD.
    """
    result: dict[str, str] = {}
    for m in re.finditer(r"```(?:bash|shell|cmd|bat|batch)?\n(.*?)```",
                         text, re.DOTALL | re.IGNORECASE):
        block = m.group(1).rstrip()
        # Must end with ) > filepath
        tail_m = re.search(
            r'\)\s*>\s*([^\s\r\n]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|toml|sh|txt))\s*$',
            block, re.IGNORECASE,
        )
        if not tail_m:
            continue
        filepath = tail_m.group(1).replace("\\", "/")
        content_lines: list[str] = []
        for raw_line in block.splitlines():
            stripped = raw_line.strip()
            # Skip structural lines
            if (stripped.startswith("type ") or stripped.startswith("del ")
                    or stripped.startswith("rm ") or stripped == "("
                    or stripped.startswith(") >") or not stripped):
                continue
            # Split on && to handle multiple echo calls per line
            for part in re.split(r"\s*&&\s*", stripped):
                part = part.strip()
                if part in ("echo.", "echo", "echo "):
                    content_lines.append("")
                elif re.match(r"^echo\s", part, re.IGNORECASE):
                    content_lines.append(part[5:])  # strip 'echo '
        if content_lines:
            result[filepath] = "\n".join(content_lines)
    return result


def _strip_file_blocks_for_command_extraction(text: str) -> str:
    """Remove ``#### [FILE]:`` and ``#### [EDIT]:`` code blocks from *text*.

    These blocks contain source code that should be written to disk, not
    executed as shell commands.  Without stripping, Python/JS code with
    ``if``/``for``/``while`` keywords gets misclassified as shell control
    flow by ``_extract_commands_from_text`` and sent to the shell executor.

    Returns the text with file/edit blocks replaced by empty strings so
    that only genuine command blocks remain for command extraction.
    """
    # Match #### [FILE]: or #### [EDIT]: header followed by a code block
    # The pattern: header line, optional blank lines, triple-backtick block
    _FILE_BLOCK_RE = re.compile(
        r'####\s*\[(?:FILE|EDIT)\]:.*?\n'   # header line
        r'(?:.*?\n)*?'                        # optional lines between header and block
        r'```(?:\w*)\n'                       # opening fence
        r'.*?'                                # block content
        r'```',                               # closing fence
        re.DOTALL,
    )
    return _FILE_BLOCK_RE.sub('', text)


def _check_syntax(filepath: str, content: str) -> str | None:
    """Return an error message if *content* has a syntax error, else ``None``.

    Only checks Python files (by extension).  For other languages, returns
    ``None`` (assume valid) so we never block non-Python fixes.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ('.py', '.pyw'):
        return None
    try:
        ast.parse(content, filename=filepath)
        return None
    except SyntaxError as exc:
        return f"{exc.msg} (line {exc.lineno})"


def _apply_fix(diagnosis: str, executor: Executor, memory: FileMemory,
               display: CLIDisplay, step_idx: int,
               step_type: str = "CODE",
               original_error_cmd: str | None = None,
               step_text: str = '',
               task: str = '',
               step_target_files: list[str] | None = None) -> tuple[bool, bool, bool]:
    """Apply fixes from a diagnosis response.

    Returns ``(applied, cmds_succeeded, has_fix_commands)`` where *applied* is True if any
    fix action was taken, *cmds_succeeded* is True if all fix commands
    ran successfully (relevant for CMD steps where the fix command itself
    is the corrected step), and *has_fix_commands* is True if at least one parameter
    was executed as a shell command.

    Parameters
    ----------
    original_error_cmd:
        The original command that failed.  When provided, any extracted
        fix command that exactly matches the failing command is skipped
        to avoid re-running the same broken command.
    """
    applied = False
    cmds_succeeded = True
    has_fix_commands = False
    executed_cmds: list[str] = []

    display.step_info(step_idx, "Applying diagnosis fix...")

    # ── Preferred: parse [EDIT]: chunk markers (surgical, avoids unintended changes) ──
    files: dict[str, str] = {}
    try:
        from ..editing.chunk_editor import ChunkEditor as _CE
        _ce = _CE()
        _chunk_edits = _ce.parse_chunk_response(diagnosis)
        if _chunk_edits:
            for _edit in _chunk_edits:
                _fpath = _edit.file_path
                _existing = memory.get(_fpath)
                if _existing is None:
                    try:
                        with open(_fpath, "r", encoding="utf-8", errors="replace") as _f:
                            _existing = _f.read()
                    except OSError:
                        pass
                if _existing is not None:
                    try:
                        files[_fpath] = _ce.apply_chunk_edits(
                            _existing, [_edit])
                    except Exception as _exc:
                        log.warning(
                            "Step %d: ChunkEdit apply failed for %s: %s",
                            step_idx + 1, _fpath, _exc)
            if files:
                log.info("Step %d: Diagnosis applied %d chunk edit(s): %s",
                         step_idx + 1, len(files), list(files.keys()))
    except ImportError:
        pass

    # ── Fallback: full-file [FILE]: markers ──
    if not files:
        files = executor.parse_code_blocks(diagnosis)

    # Fallback: try fuzzy parsing (handles diff blocks, inline file
    # comments, and other common LLM diagnosis formats)
    if not files:
        files = executor.parse_code_blocks_fuzzy(diagnosis)
        if files:
            log.info(f"Step {step_idx+1}: Standard parser found nothing, "
                     f"fuzzy parser extracted: {list(files.keys())}")

    # Fallback: detect echo-chain file writes — low-IQ LLMs often generate
    # `(echo line && echo line ...) > file.py` instead of #### [FILE]: blocks.
    # Parse the echo content in Python to avoid shell special-char failures.
    if not files:
        files = _extract_echo_chain_file_writes(diagnosis)
        if files:
            log.info(f"Step {step_idx+1}: Detected echo-chain file write(s), "
                     f"extracting content: {list(files.keys())}")

    if files:
        # Strip protected manifest files before any further processing
        files = _strip_protected_files(files)
        # Apply KB-driven content fixes (e.g. Tailwind v3→v4 directives)
        content_fixes = getattr(memory, '_content_fixes', None)
        files = _apply_content_fixes(files, content_fixes)

    if files:
        # ── Syntax guard: reject files with syntax errors ──
        # The LLM (especially during diagnosis) can produce broken Python
        # that survives to disk and compounds across fix attempts.  Check
        # with ast.parse before writing — revert to the original content
        # for any file that fails.
        _syntax_ok: dict[str, str] = {}
        for _fp, _content in files.items():
            _err = _check_syntax(_fp, _content)
            if _err:
                log.warning(
                    "Step %d: Syntax error in diagnosis fix for %s: %s — "
                    "reverting this file",
                    step_idx + 1, _fp, _err,
                )
                display.step_info(
                    step_idx,
                    f"Rejected fix for {os.path.basename(_fp)}: {_err}")
            else:
                _syntax_ok[_fp] = _content
        if len(_syntax_ok) < len(files):
            log.info(
                "Step %d: Syntax guard rejected %d/%d file(s)",
                step_idx + 1, len(files) - len(_syntax_ok), len(files),
            )
        files = _syntax_ok

    if files:
        # ── Structural preservation guard ──
        # Reject diagnosis edits that delete too many functions/classes
        # from the original file.  This catches the common failure mode
        # where the LLM replaces a large chunk and accidentally drops
        # methods like step(), request_direction(), etc.
        _DEF_CLASS_RE = re.compile(r'^\s*(?:def |class |async def )\w+', re.MULTILINE)
        _struct_ok: dict[str, str] = {}
        for _fp, _new_content in files.items():
            _ext = os.path.splitext(_fp)[1].lower()
            if _ext not in ('.py', '.pyw'):
                _struct_ok[_fp] = _new_content
                continue
            # Read original file to compare structure
            _orig = memory.get(_fp)
            if _orig is None:
                try:
                    with open(os.path.join(".", _fp), "r",
                              encoding="utf-8", errors="replace") as _f:
                        _orig = _f.read()
                except OSError:
                    _orig = None
            if _orig is None:
                _struct_ok[_fp] = _new_content
                continue
            _orig_defs = set(_DEF_CLASS_RE.findall(_orig))
            _new_defs = set(_DEF_CLASS_RE.findall(_new_content))
            if len(_orig_defs) >= 3:
                _lost = _orig_defs - _new_defs
                _lost_ratio = len(_lost) / len(_orig_defs)
                if _lost_ratio > 0.5:
                    _lost_names = [d.strip() for d in sorted(_lost)][:5]
                    log.warning(
                        "Step %d: Structural guard rejected fix for %s — "
                        "would delete %d/%d definitions: %s",
                        step_idx + 1, _fp,
                        len(_lost), len(_orig_defs), _lost_names,
                    )
                    display.step_info(
                        step_idx,
                        f"Rejected fix for {os.path.basename(_fp)}: "
                        f"would delete {len(_lost)} definitions")
                    continue
            _struct_ok[_fp] = _new_content
        if len(_struct_ok) < len(files):
            log.info(
                "Step %d: Structural guard rejected %d/%d file(s)",
                step_idx + 1, len(files) - len(_struct_ok), len(files),
            )
        files = _struct_ok

    if files and step_target_files:
        # ── Scope guard: restrict diagnosis fixes to step-relevant files ──
        # The diagnosis LLM receives context files for reference but should
        # only modify files that the step targets or that are closely related
        # (e.g. test files for the target, or files already written by the
        # pipeline).  Without this guard, the LLM can (and does) completely
        # rewrite unrelated files like game_window.py or particles.py when
        # diagnosing a game_state.py failure.
        #
        # Skip the guard when targets are non-file descriptors (e.g. CMD
        # steps with "produces: test results" or "produces: venv/").
        # These are not real file paths and would block ALL fixes.
        def _looks_like_file(path: str) -> bool:
            """Return True if path looks like a real source file target.

            Excludes dotdirs (e.g. .pytest_cache), directories without
            extensions (e.g. venv/, test results), and descriptive text.
            """
            base = os.path.basename(path)
            if base.startswith('.'):
                return False  # dotdir like .pytest_cache
            if '/' in path and not base:
                return False  # trailing slash = directory
            # Must have a real file extension (not just a leading dot)
            _, ext = os.path.splitext(base)
            return bool(ext)

        _real_targets = [
            tf for tf in step_target_files if _looks_like_file(tf)
        ]
        if _real_targets:
            _allowed = set()
            _memory_files = set(memory.all_files().keys()) if hasattr(memory, 'all_files') else set()
            for tf in _real_targets:
                _allowed.add(tf)
                # Also allow the bare filename (steps often use relative paths)
                _allowed.add(os.path.basename(tf))
            # Allow files already tracked by memory (written in earlier steps)
            _allowed.update(_memory_files)
            scoped_files: dict[str, str] = {}
            for filepath, content in files.items():
                _basename = os.path.basename(filepath)
                if filepath in _allowed or _basename in _allowed:
                    scoped_files[filepath] = content
                else:
                    log.warning(
                        "Step %d: Diagnosis fix for '%s' rejected — "
                        "file not in step scope %s",
                        step_idx + 1, filepath, _real_targets,
                    )
            if len(scoped_files) < len(files):
                log.info(
                    "Step %d: Scope guard filtered %d/%d diagnosis file(s)",
                    step_idx + 1,
                    len(files) - len(scoped_files),
                    len(files),
                )
            files = scoped_files

    if files:
        # Filter out files with hazardous diffs.
        # In diagnosis context, the LLM was specifically asked to fix a
        # failure — size reduction is often intentional (e.g. removing
        # broken imports).  Only block truly dangerous hazards like
        # HAZARD_BLOCK or structural loss (export removal, dependency
        # deletion from manifests).  Warn-only hazards like "size
        # reduction" are logged but allowed through.
        safe_files: dict[str, str] = {}
        for filepath, content in files.items():
            full_path = os.path.join(".", filepath)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8",
                              errors="replace") as f:
                        old_content = f.read()
                    hazards = _detect_hazards(filepath, old_content, content)
                    if hazards:
                        # Separate blocking hazards from size-reduction warnings.
                        # Also block catastrophic size reductions (>60%) —
                        # these almost always indicate the diagnosis LLM
                        # replaced source code with a stub to match broken
                        # tests, deleting all the real game logic.
                        blocking = [
                            (sev, msg) for sev, msg in hazards
                            if sev == HAZARD_BLOCK
                            or "export" in msg.lower()
                            or "dependencies" in msg.lower()
                        ]
                        # Promote severe size reductions to blocking
                        old_len = len(old_content)
                        new_len = len(content)
                        if old_len > 200 and new_len < old_len * 0.4:
                            pct = int((1 - new_len / old_len) * 100)
                            blocking.append((
                                HAZARD_BLOCK,
                                f"Catastrophic size reduction ({pct}%: "
                                f"{old_len} -> {new_len} chars). "
                                f"Diagnosis likely replaced source with a stub.",
                            ))
                        warn_only = [
                            (sev, msg) for sev, msg in hazards
                            if (sev, msg) not in blocking
                        ]
                        if blocking:
                            msgs = "; ".join(m for _, m in blocking)
                            log.warning(f"Step {step_idx+1}: Skipping hazardous "
                                        f"fix for {filepath}: {msgs}")
                            display.step_info(step_idx,
                                              f"Skipped unsafe fix for {filepath}")
                            continue
                        if warn_only:
                            msgs = "; ".join(m for _, m in warn_only)
                            log.info(f"Step {step_idx+1}: Diagnosis fix for "
                                     f"{filepath} has warnings (allowed): {msgs}")
                except OSError:
                    pass
            safe_files[filepath] = content

        if safe_files:
            show_diffs(safe_files, log_only=True)
            written = executor.write_files(safe_files)
            memory.update(safe_files)
            display.step_info(step_idx, f"Fixed files: {', '.join(written)}")
            log.info(f"Step {step_idx+1}: Applied code fixes to: "
                     f"{', '.join(written)}")
            applied = True

    # Extract and run fix commands.
    # For CMD steps, only extract from triple-backtick code blocks to avoid
    # picking up the *broken* original command or the new file contents.
    #
    # IMPORTANT: Strip [FILE]: and [EDIT]: code blocks from the diagnosis
    # text before extracting commands.  These blocks contain source code
    # (Python, JS, etc.) that _extract_commands_from_text would incorrectly
    # classify as shell commands when they contain control-flow keywords
    # like `if`, `for`, `while` — causing entire file contents to be
    # executed as shell commands.
    _cmd_text = _strip_file_blocks_for_command_extraction(diagnosis)
    fix_commands = _extract_commands_from_text(
        _cmd_text, code_blocks_only=(step_type == "CMD"))

    # Filter out commands that exactly match the original failing command
    if original_error_cmd and fix_commands:
        _orig = original_error_cmd.strip()
        before = len(fix_commands)
        fix_commands = [c for c in fix_commands if c.strip() != _orig]
        if len(fix_commands) < before:
            log.info(f"Step {step_idx+1}: Filtered out {before - len(fix_commands)} "
                     f"command(s) matching the original failing command")

    # Fallback: if no commands found, look for raw lines that look like commands
    # (e.g. "npx create-react-app ..." sitting on its own line)
    if not fix_commands and step_type == "CMD":
        for line in diagnosis.splitlines():
            line = line.strip()
            # Heuristic: line must start with a known command, contain spaces (args),
            # and not be a numbered list item (e.g. "1. npx ...")
            if not line or len(line.split()) < 2:
                continue
            # Remove leading bullets/numbers if present
            clean_line = line.lstrip('1234567890.-* ').strip()
            if _looks_like_command(clean_line) and clean_line not in fix_commands:
                fix_commands.append(clean_line)
        
        if fix_commands:
            log.info(f"Step {step_idx+1}: Fuzzy command parser found: {fix_commands}")

    # Detect sub-project root so fix commands like `npm install` run in
    # the correct directory instead of the repo root.
    import re as _re_fix
    subproject = _detect_subproject_root(memory)
    _subproject_cmd_patterns = (
        r'\bnpm\s+(install|start|run|test|build|ci)\b',
        r'\bnpx\s+',
        r'\byarn\s+(install|add|start|dev|build|test)\b',
        r'\bpnpm\s+(install|add|start|dev|build|test)\b',
        r'\bnode\s+',
        r'\bng\s+(serve|build|test)\b',
        r'\bpip\s+install\b',
        r'\bpytest\b',
    )

    for cmd in fix_commands:
        # Resolve any <placeholder> tokens the LLM left in the fix command
        cmd = resolve_cmd_placeholders(cmd, step_text=step_text, task=task)

        # Determine if this command should run in the sub-project directory
        fix_cwd = None
        if subproject:
            needs_subproject = any(
                _re_fix.search(p, cmd, _re_fix.IGNORECASE)
                for p in _subproject_cmd_patterns
            )
            already_has_cd = (f'cd {subproject}' in cmd
                              or f'cd ./{subproject}' in cmd)
            if needs_subproject and not already_has_cd:
                fix_cwd = subproject
                log.info(f"Step {step_idx+1}: Running fix command in "
                         f"sub-project: {subproject}/")

        cwd_note = f" (in {fix_cwd}/)" if fix_cwd else ""
        display.step_info(step_idx, f"Running fix: {cmd}{cwd_note}")
        log.info(f"Step {step_idx+1}: Running fix command: {cmd}{cwd_note}")

        # Detect long-running server commands so they are started as background
        # processes rather than blocking until the 120 s timeout expires.
        _BACKGROUND_PATTERNS = (
            r'\bmanage\.py\s+runserver\b',
            r'\bnpm\s+(start|run\s+dev|run\s+start)\b',
            r'\byarn\s+(start|dev)\b',
            r'\bpnpm\s+(start|dev)\b',
            r'\bnpx\s+.*dev\b',
            r'\bgunicorn\b',
            r'\buvicorn\b',
            r'\bflask\s+run\b',
            r'\bfastapi\s+run\b',
        )
        _run_as_bg = any(
            _re_fix.search(p, cmd, _re_fix.IGNORECASE)
            for p in _BACKGROUND_PATTERNS
        )
        success, output = executor.run_command(cmd, cwd=fix_cwd, background=_run_as_bg)
        if output:
            truncated = output[:4000] if len(output) > 4000 else output
            memory.update({
                f"_fix_output/step_{step_idx+1}.txt": f"$ {cmd}\n\n{truncated}"
            })
        if not success:
            cmds_succeeded = False
        applied = True
        has_fix_commands = True
        executed_cmds.append(cmd)

        # For CMD steps the LLM often emits multiple alternative commands
        # (e.g. "try this; if that fails, try this other form").  Stop after
        # the first one that succeeds so a later alternative that's now
        # irrelevant (e.g. mkdir when the dir already exists) can't flip
        # cmds_succeeded to False and suppress the replacement-command check.
        if step_type == "CMD" and success:
            log.info(f"Step {step_idx+1}: CMD fix command succeeded — "
                     f"stopping fix command loop (alternatives not needed).")
            break

    return applied, cmds_succeeded, has_fix_commands, executed_cmds

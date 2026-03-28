"""
Diagnosis and fix helpers — analyze step failures and apply fixes.
"""

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
from .classification import _extract_commands_from_text, _looks_like_command


def _diagnose_failure(step_text: str, step_type: str, error_info: str,
                      memory: FileMemory, llm_client, display: CLIDisplay,
                      step_idx: int,
                      search_agent=None,
                      language: str | None = None,
                      previous_diagnosis: str | None = None,
                      kb_context_builder=None) -> str:
    display.step_info(step_idx, "Analyzing failure root cause...")

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
    search_context = ""
    if search_agent is not None:
        display.step_info(step_idx, "Searching web for error documentation...")
        try:
            _search_kb = getattr(memory, '_kb_context', '')
            if _installed_note:
                _search_kb = f"{_installed_note}\n{_search_kb}" if _search_kb else _installed_note
            search_context = search_agent.search_for_error(
                error_info, step_text, language=language,
                kb_context=_search_kb)
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

    prompt = (
        "A step in our automated coding pipeline has FAILED after multiple retries.\n"
        "Analyze the failure and provide a concrete fix.\n\n"
        f"{_briefing_block}"
        "CRITICAL: Do NOT remove, comment out, or skip the feature/component that is failing. "
        "Fix the root cause instead.\n\n"
        f"Step {step_idx+1}: {step_text}\n"
        f"Step type: {step_type}\n\n"
        f"Error details:\n{error_info}\n\n"
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


def _apply_fix(diagnosis: str, executor: Executor, memory: FileMemory,
               display: CLIDisplay, step_idx: int,
               step_type: str = "CODE",
               original_error_cmd: str | None = None) -> tuple[bool, bool, bool]:
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
                        # Separate blocking hazards from size-reduction warnings
                        blocking = [
                            (sev, msg) for sev, msg in hazards
                            if sev == HAZARD_BLOCK
                            or "export" in msg.lower()
                            or "dependencies" in msg.lower()
                        ]
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
    fix_commands = _extract_commands_from_text(
        diagnosis, code_blocks_only=(step_type == "CMD"))

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
        success, output = executor.run_command(cmd, cwd=fix_cwd)
        if output:
            truncated = output[:4000] if len(output) > 4000 else output
            memory.update({
                f"_fix_output/step_{step_idx+1}.txt": f"$ {cmd}\n\n{truncated}"
            })
        if not success:
            cmds_succeeded = False
        applied = True
        has_fix_commands = True

    return applied, cmds_succeeded, has_fix_commands

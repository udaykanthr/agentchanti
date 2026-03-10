"""
Post-planning optimizer — pure Python (no LLM calls) step merging and pruning.

Applied after the planner generates raw steps but before execution.
Reduces token waste and execution time by:
- Replacing LLM-generated commands with KB-known correct commands
- Merging install commands into single steps
- Combining same-file CODE steps
- Skipping already-installed packages
- Removing meta/no-op steps
"""

from __future__ import annotations

import logging
import os
import re

_logger = logging.getLogger(__name__)


# ── Install command patterns ─────────────────────────────────────

_INSTALL_CMD_RE = re.compile(
    r'`((?:pip3?|pip3?\s+-m\s+pip)\s+install\s+.+?)`'
    r'|`(npm\s+(?:install|i)\s+.+?)`'
    r'|`(yarn\s+add\s+.+?)`'
    r'|`(pnpm\s+add\s+.+?)`'
    r'|`(gem\s+install\s+.+?)`'
    r'|`(cargo\s+add\s+.+?)`'
    r'|`(go\s+get\s+.+?)`',
    re.IGNORECASE,
)

_PKG_MANAGER_RE = re.compile(
    r'^(pip3?(?:\s+-m\s+pip)?|npm|yarn|pnpm|gem|cargo|go)\s+'
    r'(install|i|add|get)\s+',
    re.IGNORECASE,
)

# ── File path extraction from step text ──────────────────────────

_FILE_PATH_RE = re.compile(r'`([^`]+\.[a-zA-Z]{1,5})`')

# ── Meta/no-op step patterns ────────────────────────────────────

_NOOP_PATTERNS = [
    re.compile(r'^\s*(analyze|review|examine|study|understand|read|check|inspect|look at|investigate)\b', re.I),
    re.compile(r'^\s*(plan|design|think|decide|consider|evaluate|assess)\b', re.I),
    re.compile(r'^\s*(open|launch|start)\s+(the\s+)?(ide|editor|browser|terminal|vs\s*code)', re.I),
    re.compile(r'^\s*(verify|confirm|ensure|validate)\s+(that\s+)?(everything|all|code)\s+(works?|is\s+correct)', re.I),
    re.compile(r'^\s*(document|write\s+documentation)\b', re.I),
]

# ── Test pseudo-code fragment patterns ───────────────────────────
# LLMs often append "Related Test Cases" sections with numbered
# pseudo-code items that parse_plan_steps() captures as steps.
# These are NOT actionable pipeline steps.

_FRAGMENT_PATTERNS = [
    # Assertion pseudo-code: "**Assertion**: game.game_over is True"
    re.compile(r'^\s*\*{0,2}Assertion\*{0,2}\s*:', re.I),
    # Bare initialisation: "Initialize `Game`."
    re.compile(r'^\s*Initialize\s+`?\w+`?\.?\s*$', re.I),
    # Bare method call: "Call `tick()`."
    re.compile(r'^\s*Call\s+`[^`]+`\.?\s*$', re.I),
    # Bare variable access: "Get snake segments."
    re.compile(r'^\s*Get\s+\w+', re.I),
    # Loop pseudo-code: "Loop 100 times:"
    re.compile(r'^\s*Loop\s+\d+\s+times', re.I),
    # Set variable pseudo-code: "Set direction to Up"
    re.compile(r'^\s*Set\s+\w+\s+to\s+', re.I),
    # Move pseudo-code: "Move snake head to"
    re.compile(r'^\s*Move\s+\w+.*\bto\b', re.I),
]

# Minimum length for a real plan step (chars). Fragments are usually
# very short single-clause sentences.
_MIN_REAL_STEP_LENGTH = 30


def optimize_plan(
    steps: list[str],
    knowledge_base=None,
    kb_context_builder=None,
    dependencies: dict[int, set[int]] | None = None,
    language: str | None = None,
) -> tuple[list[str], dict[int, set[int]]]:
    """Optimize a raw plan by merging, deduplicating, and pruning steps.

    Returns (optimized_steps, dependencies).
    All operations are pure Python — no LLM calls.

    If *dependencies* is provided (already parsed upstream), it is used
    directly instead of re-parsing — this avoids the bug where markers
    have already been stripped from *steps*.
    """
    if not steps:
        return steps, {}

    original_count = len(steps)

    # Use pre-parsed dependencies if available, otherwise parse from steps
    if dependencies is not None:
        clean_steps = list(steps)
        deps = dict(dependencies)
    else:
        clean_steps, deps = _parse_dependencies(steps)

    # Pass 0: Replace LLM-generated commands with KB-known correct commands.
    # This is the KB-first pass — if the KB has exact commands for a step
    # (e.g. install commands, config setup), use those INSTEAD of what the
    # LLM generated.  The LLM's training data is often outdated; KB docs
    # are curated and current.
    clean_steps = _apply_kb_command_overrides(clean_steps, kb_context_builder, language=language)

    # Pass 1: Remove meta/no-op steps
    clean_steps, deps = _remove_noop_steps(clean_steps, deps)

    # Pass 2: Skip redundant installs (packages already in knowledge base)
    if knowledge_base is not None:
        clean_steps, deps = _skip_redundant_installs(clean_steps, deps, knowledge_base)

    # Pass 3: Merge install commands by package manager
    clean_steps, deps = _merge_install_steps(clean_steps, deps)

    # Pass 4: Merge same-file CODE steps
    clean_steps, deps = _merge_same_file_steps(clean_steps, deps)

    # Pass 5: Re-index and fix dependencies
    clean_steps, deps = _reindex_steps(clean_steps, deps)

    optimized_count = len(clean_steps)
    if optimized_count < original_count:
        _logger.info(
            f"[PlanOptimizer] Reduced {original_count} → {optimized_count} steps "
            f"({original_count - optimized_count} removed/merged)"
        )

    return clean_steps, deps


# ── KB command extraction helpers ─────────────────────────────────

# Patterns to extract bash commands from KB markdown docs
_KB_BASH_BLOCK_RE = re.compile(
    r'```(?:bash|sh|shell)\s*\n(.+?)```',
    re.DOTALL,
)

# Negation/deprecation text that precedes a code block — commands in such
# blocks should be EXCLUDED because they show what NOT to do.
_KB_NEGATION_RE = re.compile(
    r'(?:do\s*n[\'o]t|don\'t|avoid|deprecated|removed|no\s+longer|'
    r'instead\s+of|not\s+needed|not\s+required|obsolete|legacy|'
    r'wrong|incorrect|old\s+way|v3\b)',
    re.IGNORECASE,
)

# Keywords that indicate a step is about installing / setting up
_SETUP_KEYWORDS = re.compile(
    r'\b(install|setup|set\s*up|create|init|scaffold|bootstrap|configure)\b',
    re.IGNORECASE,
)

# Keywords to match KB docs against plan steps
_TECH_KEYWORDS = re.compile(
    r'\b(vite|vitest|jest|react|angular|vue|next|nuxt|django|flask|fastapi|'
    r'express|tailwind|pytest|mocha|webpack|rollup|svelte|solid|astro|'
    r'spring|rails|laravel|gin|echo|fiber)\b',
    re.IGNORECASE,
)

# Mutually exclusive framework groups — if a step/task mentions one
# framework from a group, KB docs about a DIFFERENT framework in the
# same group are irrelevant (e.g. React task should not use Angular docs).
_EXCLUSIVE_FRAMEWORK_GROUPS: list[set[str]] = [
    {"react", "angular", "vue", "svelte", "solid"},
]


def has_framework_conflict(context_techs: set[str], doc_techs: set[str]) -> bool:
    """Return True if the doc targets a different exclusive framework.

    Both *context_techs* and *doc_techs* should be lowercase tech keyword
    sets extracted via ``_TECH_KEYWORDS``.
    """
    for group in _EXCLUSIVE_FRAMEWORK_GROUPS:
        ctx_in = context_techs & group
        doc_in = doc_techs & group
        # Both mention frameworks from the same group, but different ones
        if ctx_in and doc_in and not (ctx_in & doc_in):
            return True
    return False


def _extract_commands_from_kb_doc(content: str) -> list[str]:
    """Extract bash commands from a KB doc's code blocks.

    Returns a list of individual commands (one per line), stripping
    comments and empty lines.  Code blocks preceded by negation or
    deprecation text (e.g. "Do NOT use", "Deprecated", "Old way") are
    skipped so that deprecated commands are never used as overrides.
    """
    commands: list[str] = []
    for match in _KB_BASH_BLOCK_RE.finditer(content):
        # Check the 200 chars before this code block for negation context
        pre_start = max(0, match.start() - 200)
        preceding_text = content[pre_start:match.start()]
        neg_match = _KB_NEGATION_RE.search(preceding_text)
        if neg_match:
            # If a section heading (## ...) appears BETWEEN the negation
            # text and this code block, the negation belongs to a previous
            # section and should NOT suppress this block.
            # e.g. "Do not create TS files...\n## Required Packages\n```bash"
            text_after_neg = preceding_text[neg_match.end():]
            if re.search(r'^#{1,4}\s', text_after_neg, re.MULTILINE):
                _logger.debug(
                    "[PlanOptimizer] Negation separated by heading, "
                    "keeping code block: %s",
                    match.group(1)[:60].replace("\n", " "),
                )
            else:
                _logger.debug(
                    "[PlanOptimizer] Skipping negated code block: %s",
                    match.group(1)[:60].replace("\n", " "),
                )
                continue

        block = match.group(1)
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                commands.append(line)
    return commands


def _find_kb_commands_for_step(step_text: str, kb_context_builder, language: str | None = None) -> list[str]:
    """Search KB docs and error dict for commands relevant to a plan step.

    Returns a list of bash commands that the KB prescribes for this
    kind of step, or empty list if no KB match.

    Sources (in order):
    1. KB doc registry (``categories=["doc"]``)
    2. Error dict ``fix_template`` fields — catches peer dependency info
       stored as error fixes (e.g. tailwindcss needing postcss).
    """
    if kb_context_builder is None:
        return []

    try:
        kb_context_builder._ensure_global()
        if kb_context_builder._global_store is None:
            return []

        # Only use results that share tech keywords with the step
        step_techs = set(w.lower() for w in _TECH_KEYWORDS.findall(step_text))
        if not step_techs:
            return []

        commands: list[str] = []

        # ── Source 1: KB doc registry ──
        results = kb_context_builder._global_store.search(
            query=step_text,
            categories=["doc"],
            language=language,
            top_k=2,
            api_client=kb_context_builder._api_client,
        )

        for result in (results or []):
            doc_content = result.content or ""
            doc_title = (result.title or "").lower()
            doc_tags = " ".join(result.tags) if result.tags else ""

            # Check if this doc is relevant to the step's tech stack
            doc_techs = set(w.lower() for w in _TECH_KEYWORDS.findall(
                doc_title + " " + doc_tags + " " + doc_content[:200]))
            overlap = step_techs & doc_techs
            if not overlap:
                continue

            # Skip docs about a conflicting framework (e.g. Angular
            # doc for a React step)
            if has_framework_conflict(step_techs, doc_techs):
                _logger.debug(
                    f"[PlanOptimizer] Skipping '{result.title}' — "
                    f"framework conflict (step={step_techs & doc_techs}, "
                    f"doc has {doc_techs})"
                )
                continue

            doc_cmds = _extract_commands_from_kb_doc(doc_content)
            if doc_cmds:
                _logger.info(
                    f"[PlanOptimizer] KB doc '{result.title}' matched step "
                    f"(overlap: {overlap}), found {len(doc_cmds)} commands"
                )
                commands.extend(doc_cmds)

        # ── Source 2: Error dict fix_template ──
        # Error fixes often contain the correct install commands with
        # peer dependencies (e.g. TailwindCSSDeprecatedInit has
        # "npm install tailwindcss @tailwindcss/postcss postcss").
        if not commands:
            try:
                error_fixes = kb_context_builder._global_store.search_errors(
                    step_text, language=language)
                for fix in error_fixes:
                    if not fix.fix_template:
                        continue
                    # Check tag overlap with step tech keywords
                    fix_tags = {t.strip().lower()
                                for t in (fix.tags or "").split(",")}
                    if not (step_techs & fix_tags):
                        continue
                    # Extract install commands from fix_template text
                    for m in re.finditer(
                        r'((?:npm|pip3?|yarn|pnpm)\s+install\s+'
                        r'[\w@/\-. ]+)',
                        fix.fix_template,
                    ):
                        cmd = m.group(1).strip()
                        if cmd and cmd not in commands:
                            commands.append(cmd)
                            _logger.info(
                                f"[PlanOptimizer] Error dict "
                                f"'{fix.error_type}' has install "
                                f"command: `{cmd}`")
            except Exception as exc:
                _logger.debug(
                    f"[PlanOptimizer] Error dict lookup failed: {exc}")

        return commands

    except Exception as exc:
        _logger.debug(f"[PlanOptimizer] KB command lookup failed: {exc}")
        return []


def _is_shell_command(text: str) -> bool:
    """Check if backtick content is a shell command (not a file path or name)."""
    first_word = text.split()[0].lower() if text.split() else ""
    # Strip leading path components (e.g. ./node_modules/.bin/vitest → vitest)
    first_word = first_word.rsplit("/", 1)[-1]
    shell_prefixes = {
        "pip", "pip3", "npm", "npx", "yarn", "pnpm", "node", "python",
        "python3", "go", "cargo", "gem", "mkdir", "cd", "echo", "cat",
        "rm", "cp", "mv", "chmod", "curl", "wget", "git", "ng", "nx",
        "bunx", "bun", "deno", "docker", "apt", "brew", "choco",
    }
    return first_word in shell_prefixes


# Command operation type classification for precise matching
_CMD_OP_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("install",   re.compile(r'\b(npm\s+install|npm\s+i\b|pip\s+install|yarn\s+add|pnpm\s+add|gem\s+install|cargo\s+add|go\s+get)\b', re.I)),
    ("create",    re.compile(r'\b(npm\s+create|npx\s+create|yarn\s+create|npm\s+init)\b', re.I)),
    ("pkg_set",   re.compile(r'\b(npm\s+pkg\s+set)\b', re.I)),
    ("run",       re.compile(r'\b(npm\s+run|yarn\s+run|npx\s+|bunx\s+)\b', re.I)),
    ("config",    re.compile(r'\b(npm\s+pkg|ng\s+add|ng\s+generate)\b', re.I)),
]


def _classify_cmd_operation(cmd: str) -> str | None:
    """Classify a shell command into an operation type."""
    for op_type, pattern in _CMD_OP_PATTERNS:
        if pattern.search(cmd):
            return op_type
    return None


def _extract_cmd_packages(cmd: str) -> set[str]:
    """Extract package names from an install command.

    Returns a set of lowered package names (without version specifiers
    or flag arguments).  For scoped packages like ``@tailwindcss/postcss``
    the full scoped name is kept.
    """
    # Strip "cd ... &&" prefix if present (e.g. "cd project && npm install X")
    effective_cmd = cmd
    if re.match(r'cd\s+\S+\s*&&\s*', cmd, re.I):
        effective_cmd = re.sub(r'^cd\s+\S+\s*&&\s*', '', cmd).strip()

    m = _PKG_MANAGER_RE.match(effective_cmd)
    if not m:
        return set()

    pkg_str = effective_cmd[m.end():]
    packages: set[str] = set()
    for token in pkg_str.split():
        token = token.strip()
        if not token or token.startswith("-"):
            continue
        # Strip trailing version specifier: lodash@4.17 → lodash
        # But keep scoped packages: @scope/pkg@ver → @scope/pkg
        if token.startswith("@"):
            # @scope/pkg or @scope/pkg@version
            at_parts = token.split("@")
            # ['', 'scope/pkg'] or ['', 'scope/pkg', 'version']
            name = f"@{at_parts[1]}" if len(at_parts) >= 2 else token
        else:
            name = token.split("@")[0]
        if name:
            packages.add(name.lower())
    return packages


def _is_deprecated_command(cmd: str, kb_context_builder) -> bool:
    """Check if a command matches a deprecated error pattern in the KB.

    Uses the global error dict to catch commands like ``tailwindcss init``
    that have a corresponding ``TailwindCSSDeprecatedInit`` error entry.
    Only considers a match valid if the error pattern actually matches
    the command text (not just keyword overlap).
    """
    try:
        if (
            kb_context_builder is not None
            and kb_context_builder._global_store is not None
        ):
            fixes = kb_context_builder._global_store.search_errors(cmd)
            for fix in fixes:
                etype = (fix.error_type or "").lower()
                if "deprecated" not in etype and "obsolete" not in etype:
                    continue
                # Verify the error pattern actually matches the command
                if fix.pattern:
                    import re as _re
                    try:
                        if _re.search(fix.pattern, cmd):
                            return True
                    except _re.error:
                        pass
    except Exception:
        pass
    return False


def _apply_kb_command_overrides(
    steps: list[str], kb_context_builder, language: str | None = None,
) -> list[str]:
    """Replace LLM-generated commands with KB-known correct commands.

    For each CMD-like step (install, setup, configure), searches the KB
    for matching documentation.  If the KB has concrete bash commands for
    the same operation, the LLM's command is REPLACED with the KB command.

    When a step mentions install/setup + a technology but has no backtick
    command at all, the KB command is INJECTED into the step text so that
    downstream package extraction picks up all required packages
    (including peer dependencies like postcss for tailwindcss).

    Only replaces actual shell commands (not file paths or folder names),
    and only when the KB command matches the same operation type
    (install→install, create→create, etc.).
    """
    if kb_context_builder is None:
        _logger.debug("[PlanOptimizer] KB command overrides skipped: "
                      "no kb_context_builder")
        return steps

    result = list(steps)
    override_count = 0

    for i, step in enumerate(steps):
        # Only override CMD-like steps (installs, setup, config)
        if not _SETUP_KEYWORDS.search(step):
            continue

        # Must mention a specific technology
        step_techs = set(w.lower() for w in _TECH_KEYWORDS.findall(step))
        if not step_techs:
            continue

        # Find all backtick segments that are actual shell commands
        llm_cmd_match = None
        for m in re.finditer(r'`([^`]+)`', step):
            candidate = m.group(1)
            if _is_shell_command(candidate):
                llm_cmd_match = m
                break  # use the first shell command found

        # Search KB for the correct command
        kb_commands = _find_kb_commands_for_step(step, kb_context_builder, language=language)

        if not kb_commands:
            _logger.debug(
                f"[PlanOptimizer] Step {i+1}: no KB commands found for "
                f"{step_techs} step: {step[:60]}...")
            continue

        if not llm_cmd_match:
            # Step has no backtick command (e.g. "Install tailwindcss").
            # Only inject for steps explicitly about installing packages —
            # NOT for "Create folder structure" or "Configure routing" steps
            # that happen to mention a technology name.
            if not re.search(r'\binstall\b', step, re.I):
                _logger.debug(
                    f"[PlanOptimizer] Step {i+1}: no backtick command and "
                    f"not an install step, skipping: {step[:60]}...")
                continue
            # Inject the first matching install command from KB so that
            # downstream package extraction sees the full package list.
            # Verify the KB command's packages relate to the step's tech.
            for kb_cmd in kb_commands:
                kb_op = _classify_cmd_operation(kb_cmd)
                if kb_op != "install":
                    continue
                if _is_deprecated_command(kb_cmd, kb_context_builder):
                    _logger.info(
                        f"[PlanOptimizer] Skipping deprecated KB cmd: "
                        f"`{kb_cmd}`")
                    continue
                # Verify package relevance: at least one package name
                # should contain the step's technology as a whole word.
                # Uses _TECH_KEYWORDS to avoid substring false positives
                # (e.g. "react" matching inside "react-router-dom").
                kb_pkgs = _extract_cmd_packages(kb_cmd)
                if kb_pkgs:
                    kb_pkg_techs = set(
                        w.lower()
                        for w in _TECH_KEYWORDS.findall(
                            " ".join(kb_pkgs))
                    )
                    if not (step_techs & kb_pkg_techs):
                        _logger.debug(
                            f"[PlanOptimizer] Step {i+1}: KB cmd "
                            f"packages {kb_pkgs} unrelated to step "
                            f"tech {step_techs}, skipping")
                        continue
                result[i] = f"{step} with `{kb_cmd}`"
                override_count += 1
                _logger.info(
                    f"[PlanOptimizer] KB inject step {i+1}: "
                    f"added `{kb_cmd}` (no command in step)")
                break
            continue

        llm_cmd = llm_cmd_match.group(1)
        llm_op = _classify_cmd_operation(llm_cmd)

        # Find the best matching KB command by operation type.
        # Only replace when the operation types match exactly AND the
        # KB command's packages are relevant to the step (prevents
        # replacing e.g. tailwindcss packages with react-router-dom).
        llm_pkgs = _extract_cmd_packages(llm_cmd)
        best_kb_cmd = None
        if llm_op is not None:
            for kb_cmd in kb_commands:
                kb_op = _classify_cmd_operation(kb_cmd)
                if kb_op != llm_op:
                    continue
                # Check if this command is deprecated per error dict
                if _is_deprecated_command(kb_cmd, kb_context_builder):
                    _logger.info(
                        f"[PlanOptimizer] Skipping deprecated KB cmd: "
                        f"`{kb_cmd}`"
                    )
                    continue
                # Package relevance check for install commands:
                if llm_op == "install":
                    kb_pkgs = _extract_cmd_packages(kb_cmd)
                    if not llm_pkgs and kb_pkgs:
                        # Bare "npm install" (installs from package.json)
                        # should NOT be replaced with "npm install X Y Z"
                        # — they serve completely different purposes.
                        _logger.debug(
                            f"[PlanOptimizer] Step {i+1}: bare install "
                            f"command, not replacing with "
                            f"package-specific {kb_pkgs}")
                        continue
                    if llm_pkgs and kb_pkgs and not (llm_pkgs & kb_pkgs):
                        # Both have packages but zero overlap — different
                        # technology (e.g. tailwindcss vs react-router-dom)
                        _logger.debug(
                            f"[PlanOptimizer] Step {i+1}: KB packages "
                            f"{kb_pkgs} have no overlap with LLM packages "
                            f"{llm_pkgs}, skipping")
                        continue
                best_kb_cmd = kb_cmd
                break

        # No match — do NOT fallback to an unrelated command
        if not best_kb_cmd:
            _logger.debug(
                f"[PlanOptimizer] Step {i+1}: KB has commands but none "
                f"match operation type '{llm_op}': {step[:60]}...")
            continue

        # Only replace if the commands are actually different
        if best_kb_cmd.strip() == llm_cmd.strip():
            _logger.debug(
                f"[PlanOptimizer] Step {i+1}: KB command identical to "
                f"LLM command, skipping")
            continue

        # Replace the command in the step text
        new_step = step.replace(f'`{llm_cmd}`', f'`{best_kb_cmd}`')
        result[i] = new_step
        override_count += 1
        _logger.info(
            f"[PlanOptimizer] KB override step {i+1}: "
            f"`{llm_cmd}` → `{best_kb_cmd}`"
        )

    if override_count > 0:
        _logger.info(
            f"[PlanOptimizer] Applied {override_count} KB command override(s)")
    else:
        _logger.info(
            "[PlanOptimizer] No KB command overrides applied")

    return result


# ── Internal passes ──────────────────────────────────────────────

def _parse_dependencies(steps: list[str]) -> tuple[list[str], dict[int, set[int]]]:
    """Extract (depends: N, M) markers from steps."""
    cleaned: list[str] = []
    deps: dict[int, set[int]] = {}

    # Match dependency markers in various LLM formats:
    # - Standalone: (depends: 1) or (depends: 1, 3)
    # - Combined with step type: (CMD, depends: 1) or (CODE, depends: 2, 3)
    # - With optional trailing colon: (depends: 1): or (CMD, depends: 1):
    dep_re = re.compile(r'\([^)]*?depends?:\s*([\d,\s]+)\)[:\s]*$', re.I)

    for i, step in enumerate(steps):
        m = dep_re.search(step)
        if m:
            dep_nums = {int(x.strip()) - 1 for x in m.group(1).split(",")
                        if x.strip().isdigit()}
            deps[i] = dep_nums
            step = step[:m.start()].strip()
        cleaned.append(step)

    return cleaned, deps


def _remove_noop_steps(
    steps: list[str], deps: dict[int, set[int]]
) -> tuple[list[str], dict[int, set[int]]]:
    """Remove steps that are not actionable pipeline steps.

    Catches two categories:
    1. Meta/analytical steps: "Analyze the project", "Review code"
    2. Test pseudo-code fragments that LLMs append after the real plan
       (e.g. "Initialize `Game`.", "**Assertion**: x is True")
    """
    keep_indices: list[int] = []
    for i, step in enumerate(steps):
        # Check classic no-op patterns (only if no backtick command)
        is_noop = any(pat.search(step) for pat in _NOOP_PATTERNS)
        if is_noop and '`' not in step:
            _logger.debug(f"[PlanOptimizer] Removing no-op step: {step[:60]}")
            continue

        # Check test pseudo-code fragments — these have backticks but
        # are still not real pipeline steps
        is_fragment = any(pat.search(step) for pat in _FRAGMENT_PATTERNS)
        if is_fragment:
            _logger.debug(f"[PlanOptimizer] Removing fragment: {step[:60]}")
            continue

        # Very short steps without file paths or commands are likely fragments
        # e.g. "Get snake segments." or "Loop 100 times:"
        stripped = re.sub(r'\*{1,2}[^*]+\*{1,2}', '', step).strip()  # remove bold markers
        if len(stripped) < _MIN_REAL_STEP_LENGTH and '`' not in step:
            _logger.debug(f"[PlanOptimizer] Removing short fragment: {step[:60]}")
            continue

        keep_indices.append(i)

    return _filter_steps(steps, deps, keep_indices)


def _skip_redundant_installs(
    steps: list[str], deps: dict[int, set[int]], knowledge_base
) -> tuple[list[str], dict[int, set[int]]]:
    """Skip install steps for packages already recorded in knowledge base."""
    keep_indices: list[int] = []

    for i, step in enumerate(steps):
        # Check if this is an install step
        m = _INSTALL_CMD_RE.search(step)
        if not m:
            keep_indices.append(i)
            continue

        # Hybrid steps (install + config/code) should never be fully
        # skipped — they contain non-install content that must survive.
        is_hybrid = _is_hybrid_install_step(step)

        # Extract the full install command
        cmd = next(g for g in m.groups() if g is not None)
        # Extract packages from the command
        pm_match = _PKG_MANAGER_RE.match(cmd)
        if not pm_match:
            keep_indices.append(i)
            continue

        pkg_str = cmd[pm_match.end():]
        packages = [
            t.strip() for t in pkg_str.split()
            if t.strip() and not t.startswith("-")
        ]

        # Filter out already-installed packages
        new_packages = []
        for pkg in packages:
            # Strip version specifiers
            pkg_name = re.split(r'[=<>~!@]', pkg)[0].lower()
            if not knowledge_base.is_package_installed(pkg_name):
                new_packages.append(pkg)
            else:
                _logger.debug(f"[PlanOptimizer] Skipping installed: {pkg_name}")

        if not new_packages and not is_hybrid:
            # All packages already installed — skip entire step
            # (only for pure install steps; hybrid steps are kept for
            # their config/code content)
            _logger.info(f"[PlanOptimizer] Skipping redundant install: {step[:60]}")
            continue

        if len(new_packages) < len(packages):
            # Some packages already installed — rebuild command
            pm = pm_match.group(1)
            action = pm_match.group(2)
            new_cmd = f"{pm} {action} {' '.join(new_packages)}"
            step = re.sub(r'`[^`]+`', f'`{new_cmd}`', step, count=1)
            steps[i] = step

        keep_indices.append(i)

    return _filter_steps(steps, deps, keep_indices)


def _is_hybrid_install_step(step: str) -> bool:
    """Check if a step has substantial non-install content.

    Returns True for "hybrid" steps that contain an install command AND
    code blocks or multiple file paths (e.g. "Install tailwindcss and
    create postcss.config.mjs with: ```js ...```").  These steps should
    NOT be merged into a single install command because the config/code
    content would be lost.
    """
    # Code fences (```) indicate code generation / configuration content
    if '```' in step:
        return True
    # Multiple file paths suggest file creation/modification instructions
    paths = _FILE_PATH_RE.findall(step)
    real_paths = [p for p in paths if _is_likely_filepath(p)]
    if len(real_paths) > 1:
        return True
    return False


def _merge_install_steps(
    steps: list[str], deps: dict[int, set[int]]
) -> tuple[list[str], dict[int, set[int]]]:
    """Merge multiple install steps for the same package manager."""
    # Group install steps by package manager
    install_groups: dict[str, list[tuple[int, str, list[str]]]] = {}
    non_install_indices: list[int] = []

    for i, step in enumerate(steps):
        m = _INSTALL_CMD_RE.search(step)
        if not m:
            non_install_indices.append(i)
            continue

        # Hybrid steps: contain an install command BUT ALSO have substantial
        # non-install content (code fences, multiple file paths, config
        # instructions).  These are CODE/config steps that happen to include
        # an install sub-command — merging them would lose the config content.
        if _is_hybrid_install_step(step):
            _logger.debug(
                f"[PlanOptimizer] Step {i+1}: hybrid install+config step, "
                f"not merging: {step[:60]}..."
            )
            non_install_indices.append(i)
            continue

        cmd = next(g for g in m.groups() if g is not None)
        pm_match = _PKG_MANAGER_RE.match(cmd)
        if not pm_match:
            non_install_indices.append(i)
            continue

        pm_key = pm_match.group(1).lower().replace(" ", "")
        # Normalize: "pip3 -m pip" → "pip"
        if "pip" in pm_key:
            pm_key = "pip"

        pkg_str = cmd[pm_match.end():]
        packages = [
            t.strip() for t in pkg_str.split()
            if t.strip() and not t.startswith("-")
        ]

        if pm_key not in install_groups:
            install_groups[pm_key] = []
        install_groups[pm_key].append((i, step, packages))

    # If no merging needed, return as-is
    if all(len(group) <= 1 for group in install_groups.values()):
        return steps, deps

    # Build merged steps
    merged_steps: list[str] = []
    merged_deps: dict[int, set[int]] = {}
    idx_map: dict[int, int] = {}  # old index → new index

    # First, add merged install steps
    for pm_key, group in sorted(install_groups.items()):
        if len(group) == 1:
            old_i = group[0][0]
            new_i = len(merged_steps)
            idx_map[old_i] = new_i
            merged_steps.append(group[0][1])
            if old_i in deps:
                merged_deps[new_i] = deps[old_i]
        else:
            # Merge all packages into one command
            all_packages: list[str] = []
            all_deps: set[int] = set()
            seen_pkgs: set[str] = set()

            for old_i, _, packages in group:
                for pkg in packages:
                    pkg_lower = pkg.lower()
                    if pkg_lower not in seen_pkgs:
                        all_packages.append(pkg)
                        seen_pkgs.add(pkg_lower)
                if old_i in deps:
                    all_deps.update(deps[old_i])

            # Build merged command
            pm_cmd = {"pip": "pip install", "npm": "npm install",
                      "yarn": "yarn add", "pnpm": "pnpm add",
                      "gem": "gem install", "cargo": "cargo add",
                      "go": "go get"}.get(pm_key, f"{pm_key} install")

            merged_text = f"Install all dependencies with `{pm_cmd} {' '.join(all_packages)}`"
            new_i = len(merged_steps)
            merged_steps.append(merged_text)

            # Map all old indices to this new one
            for old_i, _, _ in group:
                idx_map[old_i] = new_i

            if all_deps:
                merged_deps[new_i] = all_deps

            _logger.info(
                f"[PlanOptimizer] Merged {len(group)} {pm_key} install steps "
                f"→ {len(all_packages)} packages"
            )

    # Then add non-install steps
    for old_i in non_install_indices:
        new_i = len(merged_steps)
        idx_map[old_i] = new_i
        merged_steps.append(steps[old_i])
        if old_i in deps:
            # Remap dependencies
            new_dep_set = set()
            for d in deps[old_i]:
                if d in idx_map:
                    new_dep_set.add(idx_map[d])
            if new_dep_set:
                merged_deps[new_i] = new_dep_set

    return merged_steps, merged_deps


def _merge_same_file_steps(
    steps: list[str], deps: dict[int, set[int]]
) -> tuple[list[str], dict[int, set[int]]]:
    """Merge CODE steps that target the same file into a single step."""
    # Identify which file each step targets
    file_to_indices: dict[str, list[int]] = {}
    step_files: dict[int, str | None] = {}

    for i, step in enumerate(steps):
        # Only merge CODE-looking steps (not CMD install steps)
        if _INSTALL_CMD_RE.search(step):
            step_files[i] = None
            continue

        paths = _FILE_PATH_RE.findall(step)
        # Filter to actual file paths (not commands)
        real_paths = [p for p in paths if _is_likely_filepath(p)]

        if len(real_paths) == 1:
            fpath = real_paths[0]
            step_files[i] = fpath
            if fpath not in file_to_indices:
                file_to_indices[fpath] = []
            file_to_indices[fpath].append(i)
        else:
            step_files[i] = None

    # Find files with multiple steps
    merge_groups = {f: indices for f, indices in file_to_indices.items()
                    if len(indices) > 1}

    if not merge_groups:
        return steps, deps

    # Build merged result
    merged_indices: set[int] = set()
    merged_replacements: dict[int, str] = {}  # first index → merged text
    merged_dep_updates: dict[int, set[int]] = {}

    for fpath, indices in merge_groups.items():
        first_idx = indices[0]
        # Combine descriptions
        descriptions = []
        all_deps: set[int] = set()

        for idx in indices:
            descriptions.append(steps[idx])
            if idx in deps:
                all_deps.update(deps[idx])
            if idx != first_idx:
                merged_indices.add(idx)

        # Remove self-dependencies
        all_deps -= set(indices)

        merged_text = (
            f"Update `{fpath}` with all changes: "
            + " AND ".join(descriptions)
        )
        merged_replacements[first_idx] = merged_text
        if all_deps:
            merged_dep_updates[first_idx] = all_deps

        _logger.info(
            f"[PlanOptimizer] Merged {len(indices)} steps for `{fpath}`"
        )

    # Build new step list
    new_steps: list[str] = []
    new_deps: dict[int, set[int]] = {}
    idx_map: dict[int, int] = {}

    for i, step in enumerate(steps):
        if i in merged_indices:
            continue  # skip — merged into another step

        new_i = len(new_steps)
        idx_map[i] = new_i

        if i in merged_replacements:
            new_steps.append(merged_replacements[i])
            if i in merged_dep_updates:
                new_deps[new_i] = merged_dep_updates[i]
            elif i in deps:
                new_deps[new_i] = deps[i]
        else:
            new_steps.append(step)
            if i in deps:
                new_deps[new_i] = deps[i]

    # Remap dependencies in new_deps
    remapped_deps: dict[int, set[int]] = {}
    for new_i, dep_set in new_deps.items():
        remapped = set()
        for d in dep_set:
            if d in idx_map:
                remapped.add(idx_map[d])
            elif d in merged_indices:
                # Find which step this was merged into
                for fpath, indices in merge_groups.items():
                    if d in indices:
                        remapped.add(idx_map[indices[0]])
                        break
        if remapped:
            remapped_deps[new_i] = remapped

    return new_steps, remapped_deps


def _reindex_steps(
    steps: list[str], deps: dict[int, set[int]]
) -> tuple[list[str], dict[int, set[int]]]:
    """Re-number steps 0..N-1 and validate dependency references."""
    n = len(steps)
    clean_deps: dict[int, set[int]] = {}

    for i, dep_set in deps.items():
        if i < n:
            valid = {d for d in dep_set if 0 <= d < n and d != i}
            if valid:
                clean_deps[i] = valid

    return steps, clean_deps


# ── Utilities ────────────────────────────────────────────────────

def _filter_steps(
    steps: list[str], deps: dict[int, set[int]], keep_indices: list[int]
) -> tuple[list[str], dict[int, set[int]]]:
    """Filter steps to only keep specified indices, remapping dependencies."""
    if len(keep_indices) == len(steps):
        return steps, deps

    idx_map = {old: new for new, old in enumerate(keep_indices)}
    new_steps = [steps[i] for i in keep_indices]
    new_deps: dict[int, set[int]] = {}

    for old_i in keep_indices:
        new_i = idx_map[old_i]
        if old_i in deps:
            remapped = {idx_map[d] for d in deps[old_i] if d in idx_map}
            if remapped:
                new_deps[new_i] = remapped

    return new_steps, new_deps


def _is_likely_filepath(text: str) -> bool:
    """Check if text looks like a file path (not a command)."""
    # Must have an extension
    if not re.search(r'\.\w{1,5}$', text):
        return False
    # Must not start with a known command
    first_word = text.split()[0].lower() if text.split() else ""
    commands = {"pip", "npm", "yarn", "pnpm", "npx", "node", "python", "go",
                "cargo", "gem", "mkdir", "cd", "echo", "cat", "rm"}
    if first_word in commands:
        return False
    return True

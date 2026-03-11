"""
Post-planning optimizer — pure Python (no LLM calls) step merging and pruning.

Applied after the planner generates raw steps but before execution.
Reduces token waste and execution time by:
- Merging install commands into single steps
- Combining same-file CODE steps
- Skipping already-installed packages
- Removing meta/no-op steps
"""

from __future__ import annotations

import logging
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


# ── Tech keyword matching (used by planner pre-analysis) ─────────

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

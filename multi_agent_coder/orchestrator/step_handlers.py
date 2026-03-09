"""
Step handlers — CMD, CODE, and TEST step execution logic.
"""
# Separate retry limit for test generation (lower than code to avoid pipeline halts)
MAX_TEST_GEN_RETRIES = 2

import json
import os
import re
import shutil

from ..config import Config
from ..agents.coder import CoderAgent
from ..agents.reviewer import ReviewerAgent
from ..agents.tester import TesterAgent
from ..executor import Executor
from ..cli_display import CLIDisplay, token_tracker, log
from ..language import (
    get_code_block_lang, get_test_framework, detect_test_runner,
    detect_language_from_files, EXTENSION_MAP,
)

from .memory import FileMemory
from .classification import _extract_command_from_step, _extract_commands_from_text, _looks_like_command

from ..diff_display import show_diffs, prompt_diff_approval, _detect_hazards


MAX_STEP_RETRIES = 2  # Used for code steps and test run/fix attempts

# Map test runner binary → install command
_RUNNER_INSTALL = {
    "pytest": "pip install pytest",
    "jest": "npm install --save-dev jest",
    "npx": "npm install --save-dev jest",
    "mocha": "npm install --save-dev mocha",
    "vitest": "npm install --save-dev vitest",
    "go": None,  # built-in, no install needed
    "cargo": None,
    "rspec": "gem install rspec",
    "phpunit": "composer require --dev phpunit/phpunit",
}


_DEV_ONLY_PACKAGES = re.compile(
    r'^(@types/|eslint|prettier|@eslint|stylelint|'
    r'jest|vitest|mocha|chai|sinon|nyc|c8|cypress|playwright|'
    r'@testing-library/|@jest/|ts-jest|babel-jest|'
    r'webpack-dev-server|@vitejs/plugin-|vite$|'
    r'typescript$|ts-node|tsx$|nodemon|concurrently)',
    re.IGNORECASE,
)


def _ensure_packages_installed(
    project_context,
    executor: Executor,
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    subproject_cwd: str | None = None,
    language: str | None = None,
) -> None:
    """Proactively install missing packages before code/test/server steps.

    Reads the project manifest (``package.json`` or ``requirements.txt``)
    from the **subproject root** to determine what is currently installed,
    then compares against ``project_context.required_packages`` and runs
    bulk installs for anything missing.

    Designed to be called just-in-time — i.e. after the project scaffold
    has been created by earlier CMD steps but before the first CODE, TEST,
    or dev-server step runs.

    Key behaviours:
    - **One-time**: sets ``memory._packages_preinstalled`` to skip re-runs.
    - **Non-fatal**: failures are logged as warnings, never halt the pipeline.
    - **Subproject-aware**: installs inside the correct ``cwd``.
    - **npm**: splits into ``npm install`` (production) and
      ``npm install --save-dev`` (dev-only) so packages land in the
      correct section of ``package.json``.
    """
    # Already ran for this pipeline execution
    if getattr(memory, '_packages_preinstalled', False):
        return

    if project_context is None:
        return

    required = getattr(project_context, 'required_packages', [])
    if not required:
        return

    # Read the *current* manifest from disk (may differ from the snapshot
    # taken during analysis, because earlier CMD steps installed packages).
    root = subproject_cwd or "."
    pkg_json_path = os.path.join(root, "package.json")
    reqs_txt_path = os.path.join(root, "requirements.txt")

    currently_installed: set[str] = set()
    install_tool: str | None = None

    if os.path.isfile(pkg_json_path):
        try:
            with open(pkg_json_path, "r", encoding="utf-8") as f:
                pkg = json.loads(f.read())
            currently_installed.update(pkg.get("dependencies", {}).keys())
            currently_installed.update(pkg.get("devDependencies", {}).keys())
            install_tool = "npm install"
        except Exception:
            pass
    elif os.path.isfile(reqs_txt_path):
        try:
            with open(reqs_txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        pkg_name = re.split(r'[>=<~!\[]', line)[0].strip()
                        if pkg_name:
                            currently_installed.add(pkg_name)
            install_tool = "pip install"
        except Exception:
            pass

    if not install_tool:
        # No manifest found yet — project may not be scaffolded
        return

    # Determine what's truly missing right now
    missing = [
        pkg for pkg in required
        if pkg not in currently_installed
        and pkg not in ('--save-dev', '--save', '-D', '-g')
    ]

    if not missing:
        memory._packages_preinstalled = True
        return

    log.info(f"[Pre-install] Installing {len(missing)} missing package(s): "
             f"{', '.join(missing)}")
    display.step_info(step_idx,
                      f"Pre-installing {len(missing)} missing package(s)...")

    # For npm: split into production deps and dev-only deps so they land
    # in the correct section of package.json.
    if "npm" in install_tool:
        prod_pkgs = [p for p in missing if not _DEV_ONLY_PACKAGES.match(p)]
        dev_pkgs = [p for p in missing if _DEV_ONLY_PACKAGES.match(p)]

        all_ok = True
        combined_output = ""

        if prod_pkgs:
            cmd = f"npm install {' '.join(prod_pkgs)}"
            log.info(f"[Pre-install] Production: {cmd}")
            ok, output = executor.run_command(cmd, cwd=subproject_cwd)
            all_ok &= ok
            combined_output += output + "\n"

        if dev_pkgs:
            cmd = f"npm install --save-dev {' '.join(dev_pkgs)}"
            log.info(f"[Pre-install] Dev: {cmd}")
            ok, output = executor.run_command(cmd, cwd=subproject_cwd)
            all_ok &= ok
            combined_output += output + "\n"

        ok = all_ok
        output = combined_output
    else:
        cmd = f"{install_tool} {' '.join(missing)}"
        ok, output = executor.run_command(cmd, cwd=subproject_cwd)

    if ok:
        log.info(f"[Pre-install] Successfully installed: {', '.join(missing)}")
        display.step_info(step_idx, f"Pre-installed {len(missing)} package(s)")
        # Update project_context so downstream agents see the new state
        for pkg in missing:
            if pkg not in project_context.installed_packages:
                project_context.installed_packages.append(pkg)
        project_context.missing_packages = [
            p for p in project_context.missing_packages
            if p not in missing
        ]
    else:
        log.warning(f"[Pre-install] Bulk install failed (non-fatal): "
                    f"{output[:300]}")
        display.step_info(step_idx, "Pre-install failed (non-fatal)")

    # Mark as done regardless of success — individual install steps will retry
    memory._packages_preinstalled = True


def _get_runner_install_cmd(runner: str) -> str | None:
    """Return the install command for a test runner binary.

    Returns ``None`` for tools that must be installed manually (e.g. ``go``,
    ``cargo``) — the caller must handle this gracefully.
    """
    return _RUNNER_INSTALL.get(runner, f"pip install {runner}")


def _read_js_project_env(cwd: str | None = None) -> dict:
    """Read package.json and project config to detect JS/TS environment.

    Returns a dict with:
        is_esm: bool       — True if package.json has "type": "module"
        has_jest: bool      — True if jest in dependencies/devDependencies
        has_jest_globals: bool — True if @jest/globals is installed
        has_jest_config: bool — True if jest.config.* exists
        module_type: str    — "module" or "commonjs"
        test_runner: str    — "vitest", "jest", or "jest" (default)
        has_vitest: bool    — True if vitest in devDependencies
        has_vitest_config: bool — True if vitest.config.* exists
        has_tsx: bool       — True if .tsx files exist in project
        has_jsx: bool       — True if .jsx files exist in project
    """
    env = {
        "is_esm": False,
        "has_jest": False,
        "has_jest_globals": False,
        "has_jest_config": False,
        "module_type": "commonjs",
        "test_runner": "jest",
        "has_vitest": False,
        "has_vitest_config": False,
        "has_tsx": False,
        "has_jsx": False,
    }

    # Read package.json
    pkg_path = os.path.join(cwd, "package.json") if cwd else "package.json"
    if os.path.isfile(pkg_path):
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
        except (json.JSONDecodeError, OSError):
            return env

        # ESM detection
        if pkg.get("type") == "module":
            env["is_esm"] = True
            env["module_type"] = "module"

        # Dependency detection
        all_deps = {}
        all_deps.update(pkg.get("dependencies", {}))
        all_deps.update(pkg.get("devDependencies", {}))
        env["has_jest"] = "jest" in all_deps
        env["has_jest_globals"] = "@jest/globals" in all_deps
        env["has_vitest"] = "vitest" in all_deps

    # Vitest config detection
    for config_name in ("vitest.config.ts", "vitest.config.js",
                        "vitest.config.mts", "vitest.config.mjs"):
        cfg_path = os.path.join(cwd, config_name) if cwd else config_name
        if os.path.isfile(cfg_path):
            env["has_vitest_config"] = True
            break

    # Vite config with embedded test section (most common Vite+Vitest setup)
    if not env["has_vitest_config"]:
        for vite_cfg in ("vite.config.ts", "vite.config.js",
                         "vite.config.mts", "vite.config.mjs"):
            vite_path = os.path.join(cwd, vite_cfg) if cwd else vite_cfg
            if os.path.isfile(vite_path):
                try:
                    with open(vite_path, "r", encoding="utf-8") as f:
                        vite_content = f.read()
                    # Check for test config inside vite.config
                    if "test:" in vite_content or "test :" in vite_content:
                        env["has_vitest_config"] = True
                        log.info(f"Detected Vitest config embedded in {vite_cfg}")
                except OSError:
                    pass
                break  # stop after first vite.config found

    # Vite project detection: if project uses Vite, prefer Vitest
    env["has_vite"] = False
    if os.path.isfile(os.path.join(cwd, "package.json") if cwd else "package.json"):
        try:
            with open(os.path.join(cwd, "package.json") if cwd else "package.json",
                      "r", encoding="utf-8") as f:
                _pkg = json.load(f)
            _all = {}
            _all.update(_pkg.get("dependencies", {}))
            _all.update(_pkg.get("devDependencies", {}))
            env["has_vite"] = ("vite" in _all or
                               any(k.startswith("@vitejs/") for k in _all))
        except (json.JSONDecodeError, OSError):
            pass

    # Jest config detection
    for config_name in ("jest.config.js", "jest.config.ts", "jest.config.mjs",
                        "jest.config.cjs", "jest.config.json"):
        cfg_path = os.path.join(cwd, config_name) if cwd else config_name
        if os.path.isfile(cfg_path):
            env["has_jest_config"] = True
            break

    # Determine test runner: prefer Vitest if detected
    if env["has_vitest_config"] or env["has_vitest"]:
        env["test_runner"] = "vitest"
    elif env.get("has_vite") and not env["has_jest_config"]:
        # Vite project without explicit Jest config → default to Vitest
        env["test_runner"] = "vitest"
    elif env["has_jest_config"] or env["has_jest"]:
        env["test_runner"] = "jest"

    # Check for .tsx/.jsx files (useful for React testing guidance)
    scan_dir = cwd or "."
    if os.path.isdir(os.path.join(scan_dir, "src")):
        for root, _dirs, files in os.walk(os.path.join(scan_dir, "src")):
            if any(f.endswith(".tsx") for f in files):
                env["has_tsx"] = True
            if any(f.endswith(".jsx") for f in files):
                env["has_jsx"] = True
            if env["has_tsx"] and env["has_jsx"]:
                break

    return env


import platform


# ---------------------------------------------------------------------------
# KB-driven content fixes
# ---------------------------------------------------------------------------


def _apply_content_fixes(
    files: dict[str, str],
    content_fixes: list | None = None,
) -> dict[str, str]:
    """Apply KB-driven content-fix rules to generated file content.

    Each rule (a :class:`ContentFix` from the global KB) specifies a
    ``file_glob``, a ``content_pattern`` regex, a ``replacement``, and
    optional ``ensure_content`` to prepend when missing.

    Returns *files* unchanged when *content_fixes* is ``None`` or empty.
    """
    import fnmatch as _fnmatch

    if not content_fixes:
        return files

    result = dict(files)
    for filepath, content in list(result.items()):
        basename = filepath.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        original = content

        for fix in content_fixes:
            if not _fnmatch.fnmatch(basename, fix.file_glob):
                continue

            flags = fix.compiled_flags()
            pattern = re.compile(fix.content_pattern, flags)
            if not pattern.search(content):
                continue

            content = pattern.sub(fix.replacement, content)

            if fix.ensure_content:
                if fix.ensure_content.strip() not in content:
                    content = fix.ensure_content + content

            if fix.collapse_blanks:
                content = re.sub(r"\n{3,}", "\n\n", content).strip() + "\n"

            log.info(
                "Content fix '%s': applied to %s (%s)",
                fix.name, filepath, fix.description or "no description",
            )

        if content != original:
            result[filepath] = content

    return result


def _strip_protected_files(files: dict[str, str]) -> dict[str, str]:
    """Handle protected manifest files from parsed LLM output.

    For lock files (package-lock.json, yarn.lock, etc.) — fully blocked.
    For mergeable manifests (package.json, requirements.txt, etc.) — attempts
    smart merge: new dependencies/scripts are merged in, removals and version
    changes are blocked.  Falls back to full block on parse errors.

    Files are only protected when they already exist on disk (new projects
    need to create them initially).
    """
    if not files:
        return files

    # Lock files that should NEVER be touched — only package managers write these
    _LOCK_FILES: set[str] = {
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'go.sum', 'Cargo.lock', 'Gemfile.lock',
        'composer.lock', 'Pipfile.lock', 'poetry.lock',
        'Pipfile',  # managed by pipenv
    }

    # Manifests that support smart merge (additive deps only)
    _MERGEABLE_JSON: set[str] = {
        'package.json', 'composer.json',
    }
    _MERGEABLE_TEXT: set[str] = {
        'requirements.txt',
    }
    _MERGEABLE_TOML: set[str] = {
        'Cargo.toml',
    }
    _MERGEABLE_RUBY: set[str] = {
        'Gemfile',
    }
    _MERGEABLE_GO: set[str] = {
        'go.mod',
    }

    filtered: dict[str, str] = {}
    merged_count = 0
    stripped_count = 0

    for fpath, content in files.items():
        basename = os.path.basename(fpath)

        # Not a protected file — pass through
        if basename not in Executor._PROTECTED_FILENAMES:
            filtered[fpath] = content
            continue

        # File doesn't exist on disk yet — allow creation
        if not os.path.isfile(fpath):
            filtered[fpath] = content
            continue

        # Lock files — always block
        if basename in _LOCK_FILES:
            log.warning(f"[Pipeline] Blocked lock file: {fpath} "
                        f"(only package managers should modify this)")
            stripped_count += 1
            continue

        # Read existing file content
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                existing_content = f.read()
        except OSError:
            log.warning(f"[Pipeline] Cannot read {fpath}, blocking write")
            stripped_count += 1
            continue

        # Attempt smart merge based on file type
        merged = None
        if basename in _MERGEABLE_JSON:
            merged = _smart_merge_json_manifest(existing_content, content, fpath)
        elif basename in _MERGEABLE_TEXT:
            merged = _smart_merge_requirements_txt(existing_content, content, fpath)
        elif basename in _MERGEABLE_GO:
            merged = _smart_merge_go_mod(existing_content, content, fpath)
        elif basename in _MERGEABLE_TOML:
            merged = _smart_merge_line_based(existing_content, content, fpath)
        elif basename in _MERGEABLE_RUBY:
            merged = _smart_merge_line_based(existing_content, content, fpath)

        if merged is not None and merged != existing_content:
            filtered[fpath] = merged
            merged_count += 1
            log.info(f"[Pipeline] Smart-merged additive changes into {fpath}")
        elif merged == existing_content:
            log.info(f"[Pipeline] No new additions for {fpath}, skipping write")
            stripped_count += 1
        else:
            log.warning(f"[Pipeline] Smart merge failed for {fpath}, "
                        f"blocking write (fallback)")
            stripped_count += 1

    if merged_count > 0:
        log.info(f"[Pipeline] Smart-merged {merged_count} protected file(s)")
    if stripped_count > 0:
        log.info(f"[Pipeline] Blocked {stripped_count} protected file(s)")
    return filtered


def _smart_merge_json_manifest(existing: str, llm_output: str,
                                filepath: str) -> str | None:
    """Merge additive changes from LLM output into an existing JSON manifest.

    Merges new keys in ``dependencies``, ``devDependencies``, and ``scripts``.
    Blocks removals and version changes.  Returns merged JSON string, or
    ``None`` on parse failure.
    """
    try:
        old_data = json.loads(existing)
        new_data = json.loads(llm_output)
    except (json.JSONDecodeError, TypeError):
        log.warning(f"[SmartMerge] JSON parse failed for {filepath}")
        return None

    if not isinstance(old_data, dict) or not isinstance(new_data, dict):
        return None

    changed = False
    # Sections where we allow additive merges
    merge_sections = ['dependencies', 'devDependencies', 'scripts',
                      'peerDependencies', 'optionalDependencies']

    for section in merge_sections:
        old_section = old_data.get(section, {})
        new_section = new_data.get(section, {})
        if not isinstance(old_section, dict) or not isinstance(new_section, dict):
            continue

        for key, value in new_section.items():
            if key not in old_section:
                # New key — merge it in
                if section not in old_data:
                    old_data[section] = {}
                old_data[section][key] = value
                changed = True
                log.info(f"[SmartMerge] Added {section}.{key} = {value!r} "
                         f"to {filepath}")
            elif old_section[key] != value:
                # Changed value — block, keep original
                log.info(f"[SmartMerge] Blocked change to {section}.{key} "
                         f"in {filepath}: {old_section[key]!r} → {value!r}")

        # Check for removals — log but don't apply
        for key in old_section:
            if key not in new_section:
                log.info(f"[SmartMerge] Blocked removal of {section}.{key} "
                         f"from {filepath}")

    if not changed:
        return existing

    # Preserve original formatting indent
    indent = 2  # default
    for line in existing.splitlines()[1:5]:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            break

    return json.dumps(old_data, indent=indent, ensure_ascii=False) + "\n"


def _smart_merge_requirements_txt(existing: str, llm_output: str,
                                   filepath: str) -> str | None:
    """Merge new packages from LLM output into existing requirements.txt.

    Appends packages that don't exist yet.  Blocks removals and version
    changes.  Returns merged content string.
    """
    import re

    def _parse_req_name(line: str) -> str | None:
        """Extract package name from a requirements.txt line."""
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            return None
        # Handle: package==1.0, package>=1.0, package~=1.0, package[extra]
        m = re.match(r'^([A-Za-z0-9_][A-Za-z0-9._-]*)', line)
        return m.group(1).lower() if m else None

    existing_lines = existing.splitlines()
    new_lines = llm_output.splitlines()

    # Build map of existing packages: name → full line
    existing_pkgs: dict[str, str] = {}
    for line in existing_lines:
        name = _parse_req_name(line)
        if name:
            existing_pkgs[name] = line.strip()

    # Find new packages to add
    additions: list[str] = []
    for line in new_lines:
        name = _parse_req_name(line)
        if name is None:
            continue
        if name not in existing_pkgs:
            additions.append(line.strip())
            log.info(f"[SmartMerge] Adding new package: {line.strip()} "
                     f"to {filepath}")
        elif existing_pkgs[name] != line.strip():
            log.info(f"[SmartMerge] Blocked version change for {name} "
                     f"in {filepath}: {existing_pkgs[name]} → {line.strip()}")

    if not additions:
        return existing

    # Append new packages at the end
    result = existing.rstrip('\n')
    result += '\n' + '\n'.join(additions) + '\n'
    return result


def _smart_merge_go_mod(existing: str, llm_output: str,
                         filepath: str) -> str | None:
    """Merge new require directives from LLM output into existing go.mod.

    Only adds new ``require`` lines that don't exist yet.
    """
    import re

    def _parse_requires(content: str) -> dict[str, str]:
        """Extract module → version from require directives."""
        reqs: dict[str, str] = {}
        # Single-line: require module/path v1.2.3
        for m in re.finditer(r'^\s*require\s+(\S+)\s+(\S+)', content, re.MULTILINE):
            reqs[m.group(1)] = m.group(2)
        # Block: require ( ... )
        for block in re.finditer(r'require\s*\((.*?)\)', content, re.DOTALL):
            for line in block.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith('//'):
                    parts = line.split()
                    if len(parts) >= 2:
                        reqs[parts[0]] = parts[1]
        return reqs

    existing_reqs = _parse_requires(existing)
    new_reqs = _parse_requires(llm_output)

    additions: list[str] = []
    for mod, ver in new_reqs.items():
        if mod not in existing_reqs:
            additions.append(f"\trequire {mod} {ver}")
            log.info(f"[SmartMerge] Adding require {mod} {ver} to {filepath}")

    if not additions:
        return existing

    result = existing.rstrip('\n')
    result += '\n' + '\n'.join(additions) + '\n'
    return result


def _smart_merge_line_based(existing: str, llm_output: str,
                             filepath: str) -> str | None:
    """Generic line-based merge for Cargo.toml, Gemfile, etc.

    Appends lines from LLM output that don't exist (case-sensitive) in
    the existing file.  This is a conservative catch-all for formats
    we don't deeply parse.
    """
    existing_lines_set = set(existing.splitlines())

    additions: list[str] = []
    for line in llm_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if line not in existing_lines_set and stripped not in {
            l.strip() for l in existing_lines_set
        }:
            additions.append(line)

    if not additions:
        return existing

    log.info(f"[SmartMerge] Appending {len(additions)} new line(s) to {filepath}")
    result = existing.rstrip('\n')
    result += '\n' + '\n'.join(additions) + '\n'
    return result


def _shell_instructions() -> str:
    """Return OS-aware shell command guidance for LLM prompts."""
    if os.name == 'nt':
        base = (
            "Use plain CMD commands that work in Windows cmd.exe.\n"
            "For listing files use: dir /s /b\n"
            "For reading a file use: type <path>\n"
            "For creating a directory use: mkdir <path>\n"
            "For installing Python packages use: pip install <package>\n"
            "For activating a virtual environment use: call venv\\Scripts\\activate\n"
            "Do NOT use Unix commands: source, mkdir -p, touch, rm -rf, cat, ls, chmod, export.\n"
            "Do NOT use PowerShell cmdlets like Get-ChildItem, Select-Object, etc.\n"
        )
    else:
        _sysname = platform.system()
        _os_label = "macOS" if _sysname == "Darwin" else _sysname
        base = (
            f"Use standard shell commands for {_os_label}.\n"
            "For listing files use: find . -type f\n"
            "For reading a file use: cat <path>\n"
            "For creating a directory use: mkdir -p <path>\n"
            "For installing Python packages use: pip install <package>\n"
            "For activating a virtual environment use: source venv/bin/activate\n"
        )
    base += (
        "\nCRITICAL: Commands run non-interactively (no terminal input available).\n"
        "NEVER use commands that prompt for user input. Always add non-interactive flags:\n"
        "  - npx create-next-app: add --yes\n"
        "  - npm init / yarn init: add --yes or -y\n"
        "  - Angular CLI (ng new): add --defaults\n"
        "  - Composer: add --no-interaction\n"
        "  - Any tool with prompts: use --yes, --default, -y, or equivalent flag.\n"
    )
    return base


def _shell_examples() -> str:
    """Return OS-aware example commands for the planner prompt."""
    if os.name == 'nt':
        return "  1. List all project files with `dir /s /b`"
    else:
        return "  1. List all project files with `find . -type f`"


# File extensions and names that don't need code review
_NON_CODE_EXTENSIONS = {
    '.md', '.txt', '.rst', '.log', '.csv',
    '.yml', '.yaml', '.toml', '.ini', '.cfg',
    '.env', '.env.example', '.gitignore', '.dockerignore',
    '.editorconfig',
}
_NON_CODE_FILENAMES = {
    'README', 'README.md', 'README.rst', 'README.txt',
    'LICENSE', 'LICENSE.md', 'LICENSE.txt',
    'CHANGELOG', 'CHANGELOG.md',
    'CONTRIBUTING', 'CONTRIBUTING.md',
    'Makefile', 'Dockerfile', 'Procfile',
    '.gitignore', '.dockerignore', '.editorconfig',
    'requirements.txt', 'setup.cfg',
}


def _all_non_code_files(filenames: list[str]) -> bool:
    """Return True if every file in the list is non-functional (docs, config, etc.)."""
    if not filenames:
        return False
    for f in filenames:
        basename = f.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        _, ext = os.path.splitext(basename)
        if basename not in _NON_CODE_FILENAMES and ext.lower() not in _NON_CODE_EXTENSIONS:
            return False
    return True


def _build_prior_steps_context(memory: FileMemory, step_idx: int) -> str:
    """Collect outputs of prior steps from memory for context."""
    parts: list[str] = []
    all_files = memory.all_files()
    for i in range(step_idx):
        key = f"_cmd_output/step_{i+1}.txt"
        if key in all_files:
            parts.append(f"Step {i+1} output:\n{all_files[key]}")
    if not parts:
        return ""
    return "Previously executed steps:\n" + "\n\n".join(parts) + "\n\n"


def _detect_subproject_root(memory: FileMemory) -> str | None:
    """Detect if all project files share a common subdirectory prefix.

    When an earlier CMD step created a project in a subdirectory (e.g.
    ``npx create-next-app my-app``), subsequent files in memory will all
    live under ``my-app/``.  This function finds that common root.

    Detection strategies (in order):
    1. Parse CMD outputs for project-creation commands that specify a
       subdirectory (e.g. ``npx create-next-app my-app``).
    2. Check if all source files share a single first directory component.
    3. Look for project manifests in memory or on disk.
    4. Majority vote among first-level directories.

    Returns the subdirectory name (e.g. ``my-app``) or ``None``.
    """
    all_files = memory.all_files()

    # ── Fallback 0: Parse CMD outputs for project-creation commands ──
    # This is the EARLIEST detection — it works even when no source files
    # have been written to memory yet, only _cmd_output/ entries exist.
    # We look for commands like "npx create-next-app my-app" and extract
    # the directory name from the command itself.
    _PROJECT_CREATE_PATTERNS = [
        # npx create-next-app <dir>
        re.compile(r'create-next-app(?:@\S+)?\s+(\S+)'),
        # npx create-react-app <dir>
        re.compile(r'create-react-app\s+(\S+)'),
        # npx create-vite <dir>  OR  npm create vite@latest <dir>
        re.compile(r'create-vite(?:@\S+)?\s+(\S+)'),
        re.compile(r'npm\s+create\s+vite(?:@\S+)?\s+(\S+)'),
        # npx create-vue <dir>  OR  npm create vue@latest <dir>
        re.compile(r'create-vue(?:@\S+)?\s+(\S+)'),
        re.compile(r'npm\s+create\s+vue(?:@\S+)?\s+(\S+)'),
        # vue create <dir>
        re.compile(r'vue\s+create\s+(\S+)'),
        # ng new <dir>
        re.compile(r'ng\s+new\s+(\S+)'),
        # rails new <dir>
        re.compile(r'rails\s+new\s+(\S+)'),
        # cargo new <dir>
        re.compile(r'cargo\s+new\s+(\S+)'),
        # django-admin startproject <dir>
        re.compile(r'django-admin\s+startproject\s+(\S+)'),
    ]

    for fpath, content in all_files.items():
        if not fpath.startswith('_cmd_output/') and not fpath.startswith('_fix_output/'):
            continue
        # Only scan the first line (the command itself, prefixed with $)
        first_line = content.split('\n')[0] if content else ''
        for pattern in _PROJECT_CREATE_PATTERNS:
            m = pattern.search(first_line)
            if m:
                candidate = m.group(1).strip().rstrip('/')
                # Skip if the command used ./ (current directory)
                if candidate in ('.', './', ''):
                    continue
                # Strip leading ./ if present
                if candidate.startswith('./'):
                    candidate = candidate[2:]
                if not candidate:
                    continue
                # Verify the directory actually exists on disk with a manifest
                if os.path.isdir(candidate):
                    for manifest in ('package.json', 'Cargo.toml', 'go.mod',
                                     'requirements.txt', 'Gemfile', 'pyproject.toml',
                                     'composer.json', 'manage.py'):
                        if os.path.isfile(os.path.join(candidate, manifest)):
                            log.info(f"[SubProject] Detected sub-project root "
                                     f"from CMD output ({manifest}): {candidate}/")
                            return candidate

    # Only consider real source files, not internal tracking paths.
    # Internal paths use underscore-prefixed directories (_cmd_output/,
    # _fix_output/, _search_context/) and must be excluded from sub-project
    # detection.  Directories like __tests__/ and __mocks__/ are legitimate.
    _internal = ('_cmd_output/', '_fix_output/', '_search_context/')
    source_paths = [
        p for p in all_files
        if not p.startswith(_internal) and '/' in p
    ]
    if not source_paths:
        return None

    # Directories that are NOT sub-project roots — they are conventional
    # source subdirectories within a project.  If the common first component
    # is one of these, we should NOT treat it as a sub-project generator root.
    _NON_SUBPROJECT_DIRS = {
        'src', 'lib', 'app', 'apps', 'api', 'pkg',
        'components', 'pages', 'routes', 'views',
        'utils', 'helpers', 'hooks', 'styles', 'css',
        'public', 'static', 'assets', 'images', 'fonts',
        'tests', 'test', '__tests__', 'spec', 'specs',
        'docs', 'documentation', 'scripts', 'config',
        'dist', 'build', 'out', 'output',
        'models', 'controllers', 'services', 'repositories',
        'migrations', 'fixtures', 'seeds',
        'middleware', 'decorators', 'validators',
    }

    # Extract first path component from each file
    first_components: set[str] = set()
    for p in source_paths:
        parts = p.replace('\\', '/').split('/')
        if len(parts) >= 2:  # must have at least dir/file
            first_components.add(parts[0])

    # If all files share the same single first directory component,
    # that's our sub-project root — unless it's a well-known source directory
    if len(first_components) == 1:
        subproject = first_components.pop()
        # Sanity check: directory must exist on disk and not be a common
        # source directory name (which would be a false positive)
        if os.path.isdir(subproject):
            # Even if the name is in _NON_SUBPROJECT_DIRS (e.g. "app"),
            # treat it as a real sub-project if it contains a manifest
            is_blocked_name = subproject in _NON_SUBPROJECT_DIRS
            if not is_blocked_name:
                log.info(f"[SubProject] Detected sub-project root: {subproject}/")
                return subproject
            # Override the blocklist when a project manifest exists
            for manifest in ('package.json', 'Cargo.toml', 'go.mod',
                             'requirements.txt', 'Gemfile', 'pyproject.toml',
                             'composer.json', 'manage.py'):
                if os.path.isfile(os.path.join(subproject, manifest)):
                    log.info(f"[SubProject] Detected sub-project root "
                             f"(manifest override, {manifest}): {subproject}/")
                    return subproject

    # Fallback 1: if memory contains files from multiple top-level directories
    # (e.g. search provider added files), look for a known project manifest
    from ..executor import Executor
    manifest_dirs = set()
    for p in source_paths:
        if os.path.basename(p) in Executor._PROTECTED_FILENAMES:
            dirname = os.path.dirname(p)
            if dirname:  # Must be a subdirectory
                manifest_dirs.add(dirname)

    if len(manifest_dirs) == 1:
        subproject = manifest_dirs.pop()
        if os.path.isdir(subproject):
            log.info(f"[SubProject] Detected sub-project root via manifest in memory: {subproject}/")
            return subproject

    # Fallback 2: scan immediate subdirectories on disk for project manifests.
    # Protected files (package.json, etc.) are often NOT in memory because
    # _strip_protected_files blocks them.  Check the filesystem directly.
    # Only consider directories that memory files reference.
    candidate_dirs = first_components if first_components else set()
    for candidate in candidate_dirs:
        if not os.path.isdir(candidate):
            continue
        for manifest in ('package.json', 'requirements.txt', 'go.mod',
                         'Cargo.toml', 'Gemfile', 'pyproject.toml',
                         'composer.json'):
            if os.path.isfile(os.path.join(candidate, manifest)):
                log.info(f"[SubProject] Detected sub-project root via "
                         f"disk manifest ({manifest}): {candidate}/")
                return candidate

    # Fallback 3: if memory has files under a common prefix but the primary
    # check failed (e.g. multiple first-components), pick the directory that
    # contains the majority of files.
    if len(first_components) > 1:
        from collections import Counter
        counts = Counter(
            p.replace('\\', '/').split('/')[0]
            for p in source_paths
            if len(p.replace('\\', '/').split('/')) >= 2
        )
        if counts:
            best, best_count = counts.most_common(1)[0]
            total = sum(counts.values())
            # Only use majority if it covers >70% of files
            if best_count > total * 0.7 and os.path.isdir(best):
                log.info(f"[SubProject] Detected sub-project root via "
                         f"majority ({best_count}/{total} files): {best}/")
                return best

    return None


def _prefix_subproject_paths(files: dict[str, str],
                             subproject: str,
                             memory: FileMemory) -> dict[str, str]:
    """Prefix file paths with the sub-project root when missing.

    When the LLM generates paths like ``components/Header.tsx`` but the
    project lives under ``my-app/``, this function rewrites them to
    ``my-app/components/Header.tsx``.

    Files that are already prefixed, already known in memory, or are
    internal tracking paths (``_cmd_output/`` etc.) are left unchanged.
    """
    if not subproject:
        return files

    prefix = subproject.rstrip('/') + '/'
    known_paths = set(memory.all_files().keys())
    corrected: dict[str, str] = {}

    # Internal tracking directories that should never be prefixed
    _INTERNAL_PREFIXES = ('_cmd_output/', '_fix_output/', '_search_context/')

    for fpath, content in files.items():
        # Skip internal tracking paths (but NOT __tests__, __mocks__, etc.)
        if fpath.startswith(_INTERNAL_PREFIXES):
            corrected[fpath] = content
            continue

        # Already has the sub-project prefix
        if fpath.startswith(prefix):
            corrected[fpath] = content
            continue

        # Already a known file in memory (exact match) — don't touch
        if fpath in known_paths:
            corrected[fpath] = content
            continue

        # Prefix with sub-project root
        new_path = prefix + fpath
        log.info(f"[SubProject] Prefixed '{fpath}' → '{new_path}'")
        corrected[new_path] = content

    return corrected


def _handle_search_step(step_text: str, search_agent,
                        memory: FileMemory, display: CLIDisplay,
                        step_idx: int,
                        language: str | None = None) -> tuple[bool, str]:
    """Handle a [SEARCH] step — search the web for documentation / info.

    Results are stored in memory under ``_search_context/step_N.txt`` so
    that subsequent CODE / CMD steps can reference them.  The step is always
    considered successful (best-effort, non-blocking).
    """
    if search_agent is None:
        display.step_info(step_idx, "Search agent not available, skipping.")
        log.warning(f"Step {step_idx+1}: SEARCH step but no search_agent configured.")
        return True, ""

    display.step_info(step_idx, f"Searching: {step_text[:80]}...")
    log.info(f"Step {step_idx+1}: SEARCH — {step_text}")

    try:
        result = search_agent.search_for_task(
            step_text, language=language,
            kb_context=getattr(memory, '_kb_context', ''))
    except Exception as exc:
        log.warning(f"Step {step_idx+1}: Search failed: {exc}")
        display.step_info(step_idx, "Search failed (non-blocking), continuing.")
        return True, ""

    if result:
        # Store in memory so downstream steps see the search context
        memory.update({
            f"_search_context/step_{step_idx+1}.txt": result,
        })
        display.step_info(step_idx, "Search results stored for downstream steps.")
        log.info(f"Step {step_idx+1}: Search returned {len(result)} chars of context.")
    else:
        display.step_info(step_idx, "No relevant results found.")
        log.info(f"Step {step_idx+1}: Search returned no results.")

    return True, ""


def _handle_cmd_step(step_text: str, executor: Executor,
                     llm_client, memory: FileMemory,
                     display: CLIDisplay, step_idx: int,
                     language: str | None = None,
                     project_context=None) -> tuple[bool, str]:
    cmd = _extract_command_from_step(step_text)

    if cmd:
        pass  # use extracted command
    else:
        display.step_info(step_idx, "Generating command...")

        prior_context = _build_prior_steps_context(memory, step_idx)
        file_summary = memory.summary()

        gen_prompt = (
            "You are a shell command generator. Given a task step, output "
            "ONLY the shell command to accomplish it. No explanations, no "
            "markdown, no backticks — just the raw command.\n"
            f"{_shell_instructions()}\n"
        )
        # Tell the LLM about the sub-project directory so it runs commands there
        _subproject_for_prompt = _detect_subproject_root(memory)
        if _subproject_for_prompt:
            gen_prompt += (
                f"IMPORTANT: This project lives in the '{_subproject_for_prompt}/' subdirectory. "
                f"All npm/npx/yarn/pnpm/node commands MUST be prefixed with "
                f"'cd {_subproject_for_prompt} && ' to run in the correct directory.\n"
            )
        kb_ctx = getattr(memory, '_kb_context', '')
        if kb_ctx:
            gen_prompt += f"{kb_ctx}\n"
        if prior_context:
            gen_prompt += (
                f"{prior_context}"
                "IMPORTANT: Use the exact names, paths, and values from the "
                "previous steps above. Do NOT guess or use defaults.\n\n"
            )
        if file_summary != "(no files yet)":
            gen_prompt += f"Project files: {file_summary}\n\n"
        gen_prompt += f"Step: {step_text}\n\nCommand:"
        sent_before, recv_before = token_tracker.snapshot()

        cmd_response = llm_client.generate_response(gen_prompt).strip()

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if cmd_response:
            display.add_llm_log(cmd_response, source="Coder")

        extracted = _extract_commands_from_text(cmd_response)
        if extracted:
            cmd = extracted[0]
        else:
            cmd = cmd_response.strip('`').strip()
            if not _looks_like_command(cmd):
                for line in cmd.splitlines():
                    line = line.strip('`').strip()
                    if _looks_like_command(line):
                        cmd = line
                        break
        if cmd.startswith('```'):
            cmd = cmd.split('\n', 1)[-1].rsplit('```', 1)[0].strip()

        if not cmd:
            display.step_info(step_idx, "Could not generate command, skipping.")
            log.warning(f"Step {step_idx+1}: LLM returned empty command.")
            return True, ""

    # Detect sub-project root so commands like `npm install` run in the
    # correct directory instead of the repo root.
    subproject_cwd = None
    subproject = _detect_subproject_root(memory)
    if subproject:
        import re as _re_sp
        # Commands that should run inside the sub-project directory
        _subproject_cmd_patterns = (
            r'\bnpm\s+(install|start|run|test|build|ci|pkg|exec|init)\b',
            r'\bnpx\s+',
            r'\byarn\s+(install|add|start|dev|build|test)\b',
            r'\bpnpm\s+(install|add|start|dev|build|test)\b',
            r'\bnode\s+',
            r'\bng\s+(serve|build|test|generate|add)\b',
        )
        needs_subproject = any(
            _re_sp.search(p, cmd, _re_sp.IGNORECASE)
            for p in _subproject_cmd_patterns
        )
        # Don't set cwd if the command already includes a `cd` to the subproject
        already_has_cd = f'cd {subproject}' in cmd or f'cd ./{subproject}' in cmd
        if needs_subproject and not already_has_cd:
            subproject_cwd = subproject
            log.info(f"Step {step_idx+1}: Running command in sub-project: "
                     f"{subproject}/")

    # Detect if this should be a background command (e.g. starting a server).
    # Must be specific — broad keywords like "npm" or "run" cause false
    # positives that make install/build commands return before completing.
    import re as _re
    _bg_cmd_patterns = (
        r'\bnpm\s+start\b',               # npm start
        r'\bnpm\s+run\s+(dev|serve|start)\b',  # npm run dev/serve/start
        r'\bnpx\s+(next|vite|nuxt)\s+dev\b',   # npx next dev, npx vite dev
        r'\bnode\s+\S*server\S*',          # node server.js, node src/server.ts
        r'\bpython\s+\S*server\S*',        # python server.py
        r'\bpython\s+-m\s+(http\.server|flask)\b',
        r'\bflask\s+run\b',
        r'\brunserver\b',                   # manage.py runserver (Django)
        r'\buvicorn\b',
        r'\bgunicorn\b',
        r'\bng\s+serve\b',                 # Angular dev server
        r'\byarn\s+(start|dev)\b',
        r'\bpnpm\s+(start|dev)\b',
    )
    is_background = any(_re.search(p, cmd, _re.IGNORECASE) for p in _bg_cmd_patterns)

    # ── Proactive pre-install: ensure packages are installed ──
    # Runs once, just before the first server or test-suite command,
    # when the project scaffold already exists.
    if is_background and project_context is not None:
        _ensure_packages_installed(
            project_context, executor, memory, display, step_idx,
            subproject_cwd=subproject_cwd, language=language,
        )

    cwd_note = f" (in {subproject_cwd}/)" if subproject_cwd else ""
    if is_background:
        display.step_info(step_idx, f"Running background: {cmd}{cwd_note}")
        log.info(f"Step {step_idx+1}: Running background command: {cmd}")
    else:
        display.step_info(step_idx, f"Running: {cmd}{cwd_note}")
        log.info(f"Step {step_idx+1}: Running command: {cmd}")

    success, output = executor.run_command(
        cmd, background=is_background, cwd=subproject_cwd)
    log.info(f"Step {step_idx+1}: Command output:\n{output}")

    if output:
        truncated = output[:4000] if len(output) > 4000 else output
        memory.update({
            f"_cmd_output/step_{step_idx+1}.txt": f"$ {cmd}\n\n{truncated}"
        })

    if success:
        display.step_info(step_idx, "Command succeeded.")
        return True, ""
    else:
        display.step_info(step_idx, "Command failed. See log.")
        log.warning(f"Step {step_idx+1}: Command failed.")
        return False, f"Command `{cmd}` failed.\nOutput:\n{output}"


def _auto_fix_hazards(files: dict[str, str], coder: CoderAgent,
                      executor: Executor, display: CLIDisplay,
                      step_idx: int, step_text: str,
                      language: str | None = None,
                      base_dir: str = ".") -> dict[str, str]:
    """Scan generated files for hazardous diffs and auto-fix them.

    For each file where ``_detect_hazards`` flags problems (e.g. significant
    size reduction, dependency removal), the coder LLM is asked to produce
    a corrected version that preserves the existing content while applying
    only the intended changes.

    Returns the (potentially corrected) file dict.
    """
    fixed_files = dict(files)

    for filepath, new_content in list(files.items()):
        full_path = os.path.join(base_dir, filepath)
        if not os.path.isfile(full_path):
            continue  # new file, nothing to compare

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                old_content = f.read()
        except OSError:
            continue

        hazards = _detect_hazards(filepath, old_content, new_content)
        if not hazards:
            continue

        hazard_descriptions = "\n".join(f"- {msg}" for _, msg in hazards)
        log.warning(f"Step {step_idx+1}: Hazards detected in {filepath}:\n"
                    f"{hazard_descriptions}")
        display.step_info(step_idx, f"Hazard in {filepath}, auto-fixing...")

        fix_prompt = (
            f"You generated a new version of `{filepath}` but it has safety issues:\n"
            f"{hazard_descriptions}\n\n"
            f"EXISTING file content (DO NOT lose any of this):\n"
            f"```\n{old_content}\n```\n\n"
            f"YOUR generated version (has problems):\n"
            f"```\n{new_content}\n```\n\n"
            f"The step was: {step_text}\n\n"
            f"Produce a CORRECTED version of `{filepath}` that:\n"
            f"1. Keeps ALL existing content (dependencies, imports, configs, etc.)\n"
            f"2. Only adds/changes what the step requires\n"
            f"3. Does NOT remove anything that was in the original file\n\n"
            f"#### [FILE]: {filepath}\n"
            f"```\n"
            f"(write the complete corrected file here)\n"
            f"```"
        )

        sent_before, recv_before = token_tracker.snapshot()

        fix_response = coder.process(fix_prompt, context="", language=language)

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        explanation = CLIDisplay.extract_explanation(fix_response)
        if explanation:
            display.add_llm_log(explanation, source="Coder")

        fix_files = executor.parse_code_blocks(fix_response)
        if not fix_files:
            fix_files = executor.parse_code_blocks_fuzzy(fix_response)

        if filepath in fix_files:
            # Verify the fix resolved the hazard
            new_hazards = _detect_hazards(filepath, old_content, fix_files[filepath])
            if len(new_hazards) < len(hazards):
                fixed_files[filepath] = fix_files[filepath]
                log.info(f"Step {step_idx+1}: Auto-fixed hazard in {filepath} "
                         f"({len(hazards)} -> {len(new_hazards)} hazards)")
                display.step_info(step_idx, f"Fixed hazard in {filepath}")
            else:
                log.warning(f"Step {step_idx+1}: Auto-fix did not resolve hazard in "
                            f"{filepath}, keeping original generated version for user review")
        else:
            log.warning(f"Step {step_idx+1}: Auto-fix did not return {filepath}")

    return fixed_files


# Regex to detect at least one test block in JS/TS/Python test files.
# Matches: describe(, it(, test(, def test_, @pytest, @Test
_HAS_TEST_BLOCK_RE = re.compile(
    r'(?:'
    r'\b(?:describe|it|test)\s*\('          # JS/TS: describe(, it(, test(
    r'|\bdef\s+test_'                       # Python: def test_
    r'|\b@(?:pytest\.mark|Test)\b'          # Python/Java annotations
    r'|\bcontext\s+["\']'                   # RSpec: context "..."
    r')',
)

# File names that are typically setup/config, not test suites
_SETUP_FILE_RE = re.compile(
    r'(setup|config|fixture|helper|util|mock|factory|conftest)'
    r'\.test\.',
    re.IGNORECASE,
)


def _strip_empty_test_files(files: dict[str, str]) -> dict[str, str]:
    """Remove test files that contain no actual test blocks.

    LLMs sometimes generate scaffold files like ``vitestSetup.test.js``
    that only contain configuration/setup code (imports, vi.mock calls,
    custom matchers) but no ``describe``/``it``/``test`` blocks.  These
    cause "No test suite found" errors from every test runner.

    Returns the dict with empty test files removed.  Non-test files
    (e.g. source files in a source-bug fix) are always kept.
    """
    result: dict[str, str] = {}
    for fpath, content in files.items():
        basename = fpath.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        # Only check files that look like test files
        is_test_file = (
            ".test." in basename
            or ".spec." in basename
            or basename.startswith("test_")
            or "_test." in basename
            or "_spec." in basename
        )

        if not is_test_file:
            # Not a test file (e.g. source fix) — keep as-is
            result[fpath] = content
            continue

        # Check if the file has at least one test block
        if _HAS_TEST_BLOCK_RE.search(content):
            result[fpath] = content
        else:
            log.warning(
                f"[TestFilter] Stripped '{fpath}' — no test blocks found "
                f"(setup/scaffold file)"
            )

    return result


def _quick_offline_lint(files: dict[str, str]) -> str:
    """Perform a quick offline syntax/linter check on generated files."""
    errors = []
    import os
    
    try:
        from syntax_checker import check_syntax
    except ImportError:
        check_syntax = None

    for filepath, content in files.items():
        ext = os.path.splitext(filepath)[1].lower()
        if ext.startswith('.'):
            ext = ext[1:]

        # ── Built-in Python syntax validation (Enhancement #7) ──
        # Use Python's compile() for .py files — catches syntax errors
        # before writing to disk, saving a full test run cycle.
        if ext == 'py':
            try:
                compile(content, filepath, 'exec')
            except SyntaxError as e:
                line_info = f":{e.lineno}" if e.lineno else ""
                errors.append(f"{filepath}{line_info}: SyntaxError: {e.msg}")
                continue  # skip syntax_checker for this file

        if check_syntax:
            supported_exts = {k.lstrip('.') for k in EXTENSION_MAP.keys()}
            if ext in supported_exts:
                try:
                    result = check_syntax(ext, content)
                    if hasattr(result, 'errors') and result.errors:
                        errors.append(f"{filepath} SyntaxError:\n{result.description}")
                except Exception as e:
                    errors.append(f"{filepath}: Error parsing: {str(e)}")
                    
    if errors:
        return "LINTER ERRORS FOUND:\n" + "\n".join(errors) + "\n\n"
    return ""


# ── Relative-import path resolution extensions ────────────────────
# For JS/TS files, import './Foo' can resolve to Foo.js, Foo.jsx,
# Foo.ts, Foo.tsx, Foo/index.js, Foo/index.jsx, etc.
_JS_IMPORT_EXTENSIONS = (
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    '/index.js', '/index.jsx', '/index.ts', '/index.tsx',
)

# Matches: import ... from '...', import '...', export ... from '...'
_JS_IMPORT_RE = re.compile(
    r'''(?:import|export)\s+.*?from\s+['"](\.\.?/[^'"]+)['"]'''
    r'''|import\s+['"](\.\.?/[^'"]+)['"]''',
)

# Python relative imports: from .foo import bar, from ..pkg import baz
_PY_IMPORT_RE = re.compile(
    r'^from\s+(\.+\w[\w.]*)\s+import\b',
    re.MULTILINE,
)


def _validate_import_paths(
    files: dict[str, str],
    memory: "FileMemory",
) -> str:
    """Check that relative imports in generated files resolve to known files.

    Validates JS/TS/JSX/TSX and Python files.  Returns an error string
    describing broken imports, or empty string if all imports resolve.
    """
    # Build a set of all known file paths (memory + newly generated)
    all_paths: set[str] = set(memory.all_files().keys()) | set(files.keys())
    # Normalise to forward slashes for matching
    all_paths_norm: set[str] = {p.replace("\\", "/") for p in all_paths}

    errors: list[str] = []

    for filepath, content in files.items():
        ext = os.path.splitext(filepath)[1].lower()
        dir_of_file = os.path.dirname(filepath).replace("\\", "/")

        if ext in ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'):
            for m in _JS_IMPORT_RE.finditer(content):
                import_path = m.group(1) or m.group(2)
                if not import_path:
                    continue

                resolved = os.path.normpath(
                    os.path.join(dir_of_file, import_path)
                ).replace("\\", "/")

                # Check if it resolves directly (with extension already)
                if resolved in all_paths_norm:
                    continue

                # Try appending standard extensions
                found = False
                for try_ext in _JS_IMPORT_EXTENSIONS:
                    if (resolved + try_ext) in all_paths_norm:
                        found = True
                        break
                if found:
                    continue

                # Try to find the correct path by basename
                basename = os.path.basename(resolved)
                suggestions = [
                    p for p in all_paths_norm
                    if os.path.basename(p).startswith(basename + '.')
                    or os.path.basename(p) == basename
                    or p.endswith('/' + basename + '.jsx')
                    or p.endswith('/' + basename + '.js')
                    or p.endswith('/' + basename + '.tsx')
                    or p.endswith('/' + basename + '.ts')
                ]
                if suggestions:
                    # Calculate what the correct import should be
                    best = suggestions[0]
                    correct_rel = os.path.relpath(
                        best, dir_of_file
                    ).replace("\\", "/")
                    # Strip extension for JS imports
                    correct_rel_no_ext = re.sub(
                        r'\.(jsx?|tsx?|mjs|cjs)$', '', correct_rel
                    )
                    if not correct_rel_no_ext.startswith('.'):
                        correct_rel_no_ext = './' + correct_rel_no_ext
                    errors.append(
                        f"{filepath}: import '{import_path}' not found. "
                        f"Did you mean '{correct_rel_no_ext}'? "
                        f"(actual file: {best})"
                    )
                else:
                    errors.append(
                        f"{filepath}: import '{import_path}' does not "
                        f"resolve to any known file."
                    )

        elif ext == '.py':
            for m in _PY_IMPORT_RE.finditer(content):
                rel_import = m.group(1)
                # Count leading dots
                dots = len(rel_import) - len(rel_import.lstrip('.'))
                module_path = rel_import[dots:]
                if not module_path:
                    continue

                # Walk up `dots` directories
                base = dir_of_file
                for _ in range(dots - 1):
                    base = os.path.dirname(base)

                py_path = os.path.join(
                    base, module_path.replace('.', '/')
                ).replace("\\", "/")

                # Check module.py or module/__init__.py
                if (py_path + '.py') in all_paths_norm:
                    continue
                if (py_path + '/__init__.py') in all_paths_norm:
                    continue

                errors.append(
                    f"{filepath}: relative import '{rel_import}' does not "
                    f"resolve to any known file."
                )

    if errors:
        return "IMPORT ERRORS FOUND:\n" + "\n".join(errors) + "\n\n"
    return ""


# ── Package import detection and auto-install ─────────────────────

# Matches ALL JS/TS imports (package AND relative):
#   import X from 'pkg'  |  import 'pkg'  |  export { X } from 'pkg'
#   Also handles dynamic: await import('pkg')
_JS_ALL_IMPORT_RE = re.compile(
    r'''(?:import|export)\s+.*?from\s+['"]([^'"]+)['"]'''
    r'''|import\s+['"]([^'"]+)['"]'''
    r'''|import\(\s*['"]([^'"]+)['"]\s*\)''',
)

# Node built-in modules — never need npm install
_NODE_BUILTINS = frozenset({
    'assert', 'async_hooks', 'buffer', 'child_process', 'cluster',
    'console', 'constants', 'crypto', 'dgram', 'diagnostics_channel',
    'dns', 'domain', 'events', 'fs', 'http', 'http2', 'https',
    'inspector', 'module', 'net', 'os', 'path', 'perf_hooks',
    'process', 'punycode', 'querystring', 'readline', 'repl',
    'stream', 'string_decoder', 'sys', 'timers', 'tls', 'trace_events',
    'tty', 'url', 'util', 'v8', 'vm', 'wasi', 'worker_threads', 'zlib',
    # node: protocol prefixed versions are caught by startswith check
})


def _extract_npm_package_name(specifier: str) -> str | None:
    """Extract the npm package name from an import specifier.

    Examples:
        'react'                       → 'react'
        'react-router-dom'            → 'react-router-dom'
        '@heroicons/react/outline'    → '@heroicons/react'
        '@testing-library/react'      → '@testing-library/react'
        './App'                       → None  (relative)
        'fs'                          → None  (built-in)
        'node:path'                   → None  (built-in)
    """
    if not specifier:
        return None

    # Skip relative imports
    if specifier.startswith('.') or specifier.startswith('/'):
        return None

    # Skip node: protocol
    if specifier.startswith('node:'):
        return None

    # Scoped packages: @scope/package/subpath → @scope/package
    if specifier.startswith('@'):
        parts = specifier.split('/')
        if len(parts) >= 2:
            pkg_name = parts[0] + '/' + parts[1]
        else:
            return None  # malformed
    else:
        # Regular packages: package/subpath → package
        pkg_name = specifier.split('/')[0]

    # Skip Node built-ins
    if pkg_name in _NODE_BUILTINS:
        return None

    return pkg_name


def _auto_install_code_imports(
    files: dict[str, str],
    executor: Executor,
    memory: "FileMemory",
    display: CLIDisplay,
    step_idx: int,
) -> None:
    """Scan generated JS/TS files for package imports and auto-install missing ones.

    After the coder generates code, it may import packages that weren't
    listed in ``required_packages`` (e.g. ``@heroicons/react``).  This
    function detects those imports, checks ``package.json``, and runs
    ``npm install`` for anything missing.
    """
    # Only process JS/TS files
    js_exts = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}
    imported_packages: set[str] = set()

    for filepath, content in files.items():
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in js_exts:
            continue

        for m in _JS_ALL_IMPORT_RE.finditer(content):
            specifier = m.group(1) or m.group(2) or m.group(3)
            pkg = _extract_npm_package_name(specifier)
            if pkg:
                imported_packages.add(pkg)

    if not imported_packages:
        return

    # Also scan existing memory files — packages already imported before
    # this step have presumably been installed already, so don't re-install
    for filepath, content in memory.all_files().items():
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in js_exts:
            continue
        for m in _JS_ALL_IMPORT_RE.finditer(content):
            specifier = m.group(1) or m.group(2) or m.group(3)
            pkg = _extract_npm_package_name(specifier)
            if pkg:
                imported_packages.add(pkg)

    # Read package.json to see what's already installed
    subproject_cwd = _detect_subproject_root(memory)
    root = subproject_cwd or "."
    pkg_json_path = os.path.join(root, "package.json")

    installed: set[str] = set()
    if os.path.isfile(pkg_json_path):
        try:
            with open(pkg_json_path, "r", encoding="utf-8") as f:
                pkg_data = json.loads(f.read())
            installed.update(pkg_data.get("dependencies", {}).keys())
            installed.update(pkg_data.get("devDependencies", {}).keys())
            installed.update(pkg_data.get("peerDependencies", {}).keys())
        except Exception:
            pass
    else:
        # No package.json — can't determine what's installed
        return

    missing = sorted(imported_packages - installed)
    if not missing:
        return

    cmd = f"npm install {' '.join(missing)}"
    cwd_note = f" (in {subproject_cwd}/)" if subproject_cwd else ""
    display.step_info(step_idx,
                      f"Auto-installing {len(missing)} imported package(s){cwd_note}")
    log.info(f"Step {step_idx+1}: Auto-installing packages from code imports: "
             f"{', '.join(missing)}")

    ok, output = executor.run_command(cmd, cwd=subproject_cwd)
    if ok:
        log.info(f"Step {step_idx+1}: Auto-install succeeded: {', '.join(missing)}")
        display.step_info(step_idx,
                          f"Auto-installed {len(missing)} package(s): "
                          f"{', '.join(missing)}")
    else:
        log.warning(f"Step {step_idx+1}: Auto-install failed (non-fatal): "
                    f"{output[:300]}")


# ---- ANSI code pattern (shared) ----
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _extract_test_error(output: str, max_chars: int = 1500) -> str:
    """Extract actionable error info from verbose test runner output.

    Strips ANSI escape codes, drops DOM/HTML dumps and deep stack traces,
    and keeps: error type + message, the failing source line, framework
    suggestions, and the test summary.  Works generically across Jest,
    Vitest, pytest, Go test, RSpec, etc.

    Returns at most *max_chars* of useful error context.
    """
    if not output:
        return ""

    # 1. Strip ANSI colour codes
    clean = _ANSI_RE.sub('', output)
    lines = clean.splitlines()

    kept: list[str] = []
    skip_block = False  # True while inside a DOM / HTML dump block

    # Patterns for lines we always want to keep
    _KEEP_PATTERNS = re.compile(
        r'(FAIL\b|FAILED\b|PASS\b|ERROR\b|Error[:\s]|error[:\s]'
        r'|TypeError|ReferenceError|SyntaxError|NameError'
        r'|ModuleNotFoundError|ImportError|AttributeError'
        r'|IndentationError|FileNotFoundError|AssertionError'
        r'|AssertError|KeyError|ValueError|TestingLibrary'
        r'|expect\(|Expected\b|Received\b|Difference:'
        r'|× |✕ |✗ |✓ |✔ '
        r'|Tests:|Test Files|Test Suites|Duration'
        r'|\d+ (failed|passed|skipped|pending)'
        r'|RUNS?\b)',
        re.IGNORECASE
    )

    # Patterns for the source pointer block (e.g. "  > 24 | const x = ...")
    _SOURCE_PTR = re.compile(r'^\s*(>\s*)?\d+\s*\|')

    # Patterns for suggestion / hint lines
    _SUGGESTION = re.compile(
        r'(\(If this is intentional|Hint:|Did you mean'
        r'|To fix|Possible fix|Consider using|use the .+variant)',
        re.IGNORECASE
    )

    # Patterns indicating a DOM / HTML dump to skip
    _DOM_LINE = re.compile(
        r'^\s*<(div|span|a|button|nav|header|footer|section|svg|path|ul|li|ol|img|form|input|p|h[1-6])'
        r'|^\s*\[\d+m<'  # Residual ANSI-prefixed HTML
        r'|^\s*Ignored nodes:',
        re.IGNORECASE
    )

    # Stack trace lines to skip (deep frames only, keep top 2)
    _STACK_FRAME = re.compile(r'^\s*(at |❯ |\.\.\.\s*\d+ more)')
    stack_frame_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            skip_block = False
            stack_frame_count = 0
            continue

        # Detect and skip DOM/HTML dump blocks
        if _DOM_LINE.match(stripped):
            if not skip_block:
                kept.append('    [... DOM/HTML output truncated ...]')
                skip_block = True
            continue

        if skip_block:
            # Still inside a DOM dump — skip unless we hit a keeper
            if _KEEP_PATTERNS.search(stripped) or _SUGGESTION.search(stripped):
                skip_block = False
            else:
                continue

        # Limit stack trace depth: keep first 2 frames, skip the rest
        if _STACK_FRAME.match(stripped):
            stack_frame_count += 1
            if stack_frame_count > 2:
                if stack_frame_count == 3:
                    kept.append('    [... stack frames truncated ...]')
                continue

        # Always keep: error messages, source pointers, suggestions, summaries
        if (_KEEP_PATTERNS.search(stripped)
                or _SOURCE_PTR.match(stripped)
                or _SUGGESTION.search(stripped)):
            kept.append(line)
            continue

        # Keep lines that are short context around errors (e.g. test names)
        if len(stripped) < 120:
            kept.append(line)

    result = '\n'.join(kept).strip()

    # Final cap
    if len(result) > max_chars:
        result = result[:max_chars] + '\n... [truncated]'

    return result if result else output[:max_chars]


# ---- Batch error summary for efficient test fixing ----

# Regex to match common error type lines across test runners
_ERROR_TYPE_RE = re.compile(
    r'(ModuleNotFoundError|ImportError|SyntaxError|NameError|'
    r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
    r'AssertionError|AssertError|KeyError|ValueError|ReferenceError|'
    r'RangeError|RuntimeError|OSError|IOError|PermissionError|'
    r'expect\(received\))'
)

# Test name patterns for pytest, Jest/Vitest, Go, RSpec
_TEST_NAME_PATTERNS = [
    # pytest: "FAILED tests/test_foo.py::test_bar - Error..."
    re.compile(r'FAILED\s+([\w/.]+::\S+)'),
    # pytest short: "tests/test_foo.py::test_bar FAILED"
    re.compile(r'([\w/.]+::\S+)\s+FAILED'),
    # Jest/Vitest: "✕ test name" or "× test name" or "✗ test name"
    re.compile(r'[×✕✗]\s+(.+)'),
    # Jest: "FAIL path/to/test.js"
    re.compile(r'FAIL\s+([\w/.]+\.\w+)'),
    # Go: "--- FAIL: TestName"
    re.compile(r'---\s+FAIL:\s+(\S+)'),
    # RSpec: "rspec ./spec/foo_spec.rb:42"
    re.compile(r'rspec\s+([\w/.]+:\d+)'),
]


def _build_scoped_test_cmd(
    base_cmd: str,
    test_files: dict[str, str],
    subproject_cwd: str | None = None,
) -> str:
    """Build a test command scoped to only the files from this step.

    Appends file paths to the base test command so the runner only
    executes the tests that were written/modified in the current step,
    rather than the entire test suite.

    Supports pytest, vitest, jest, go test, and rspec.
    """
    if not test_files:
        return base_cmd

    # Get file paths relative to the test runner's working directory
    scoped_paths: list[str] = []
    for fpath in test_files:
        # If subproject_cwd is set, file paths in memory are already
        # prefixed with it (e.g. "react-app/src/__tests__/Foo.test.jsx").
        # The test command runs inside subproject_cwd, so strip the prefix.
        if subproject_cwd:
            prefix = subproject_cwd.rstrip("/\\") + "/"
            prefix_back = subproject_cwd.rstrip("/\\") + "\\"
            if fpath.startswith(prefix):
                fpath = fpath[len(prefix):]
            elif fpath.startswith(prefix_back):
                fpath = fpath[len(prefix_back):]
        # Use forward slashes for consistency
        scoped_paths.append(fpath.replace("\\", "/"))

    if not scoped_paths:
        return base_cmd

    # Build the scoped command based on the runner
    base_lower = base_cmd.lower()

    # pytest: `pytest file1.py file2.py`
    # vitest: `npx vitest run file1.test.jsx file2.test.jsx`
    # jest:   `npx jest file1.test.jsx file2.test.jsx`
    # go:     `go test ./path/to/...` (different pattern, skip scoping)
    # rspec:  `rspec file1_spec.rb file2_spec.rb`

    if "go test" in base_lower:
        # Go test uses package paths, not file paths — skip scoping
        return base_cmd

    # For all other runners, append file paths
    path_args = " ".join(scoped_paths)
    return f"{base_cmd} {path_args}"


def _count_test_failures(output: str) -> int:
    """Count the number of individual test failures in test runner output.

    Counts BOTH file-level failures (compilation/import errors that prevent
    the file from running) AND individual test assertion failures — whichever
    is higher — since file-level failures each require a fix.

    Works across pytest, Jest, Vitest, Go test, and RSpec.
    """
    if not output:
        return 0

    clean = _ANSI_RE.sub('', output)
    file_failures = 0
    test_failures = 0

    # Vitest/Jest: "Test Files  X failed" and "Tests  Y failed" on separate lines
    m_files = re.search(r'Test Files?\s+(\d+)\s+failed', clean)
    if m_files:
        file_failures = int(m_files.group(1))
    m_tests = re.search(r'Tests:\s*(\d+)\s+failed', clean)
    if m_tests:
        test_failures = int(m_tests.group(1))

    # If we found Vitest/Jest-style summaries, return the larger count
    if file_failures or test_failures:
        return max(file_failures, test_failures)

    # pytest summary: "X failed"
    m = re.search(r'(\d+)\s+failed', clean)
    if m:
        return int(m.group(1))

    # Go: count "--- FAIL:" lines
    count = len(re.findall(r'---\s+FAIL:', clean))
    if count:
        return count

    # Fallback: count failure marker lines
    count = 0
    for line in clean.splitlines():
        stripped = line.strip()
        if re.match(r'(FAILED\s+|×\s+|✕\s+|✗\s+)', stripped):
            count += 1

    return max(count, 1) if 'FAIL' in clean.upper() else 0


def _build_batch_error_summary(output: str, max_chars: int = 6000) -> str:
    """Build a structured, compact summary of ALL test failures.

    Groups failures by error type so the LLM can identify shared root causes
    (e.g. wrong import path affecting 10 tests → one fix). Works across
    pytest, Jest, Vitest, Go test, and RSpec output formats.

    Returns at most *max_chars* of structured error context.
    """
    if not output:
        return ""

    clean = _ANSI_RE.sub('', output)
    lines = clean.splitlines()

    # Collect individual failure blocks
    failures: list[dict] = []       # [{test, error_type, message, file, line}]
    error_groups: dict[str, list] = {}  # error_type -> [failure indices]

    # Extract test names that failed
    failed_tests: list[str] = []
    for line in lines:
        for pat in _TEST_NAME_PATTERNS:
            m = pat.search(line)
            if m:
                failed_tests.append(m.group(1).strip())
                break

    # Pattern for file:line references
    _file_line_re = re.compile(
        r'(?:File\s+["\'](.+?)["\'],\s+line\s+(\d+)'  # Python
        r'|(\S+\.\w+):(\d+))'                           # JS/Go/general
    )

    # ── Pre-pass: capture Vitest/Jest file-level failures ──────────
    # Vitest/Jest output "FAIL path/to/file.ext" for files that failed
    # to compile or import.  These often don't have a standard error
    # type; instead the error (e.g. "SyntaxError", "Cannot find module")
    # appears on subsequent lines.  We collect them into a dedicated
    # "FileLoadError" group so the LLM sees them.
    _FAIL_FILE_RE = re.compile(r'^(?:FAIL)\s+([\w/.@-]+\.\w+)')
    _seen_fail_files: set[str] = set()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = _FAIL_FILE_RE.match(stripped)
        if m:
            fail_file = m.group(1)
            if fail_file not in _seen_fail_files:
                _seen_fail_files.add(fail_file)
                # Gather the next non-blank lines as the error message
                err_lines: list[str] = []
                j = i + 1
                while j < len(lines) and j < i + 8:
                    s = lines[j].strip()
                    if not s:
                        break
                    err_lines.append(s)
                    j += 1
                err_msg = ' '.join(err_lines).strip()
                if len(err_msg) > 200:
                    err_msg = err_msg[:200] + '...'

                # Try to identify the actual error type from the message
                et_m = _ERROR_TYPE_RE.search(err_msg)
                err_type = et_m.group(1) if et_m else 'FileLoadError'

                failure = {
                    'test': fail_file,
                    'error_type': err_type,
                    'message': err_msg or '(file failed to load/compile)',
                    'file': fail_file,
                    'line': None,
                }
                failures.append(failure)
                error_groups.setdefault(err_type, []).append(
                    len(failures) - 1)
        i += 1

    # ── Main pass: extract error details from assertion/runtime failures ──
    current_test = None
    current_error_type = None
    current_message_lines: list[str] = []
    current_file = None
    current_line_no = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Blank line — flush current failure if we have one
            if current_error_type and current_message_lines:
                msg = ' '.join(current_message_lines).strip()
                if len(msg) > 200:
                    msg = msg[:200] + '...'
                failure = {
                    'test': current_test or '(unknown test)',
                    'error_type': current_error_type,
                    'message': msg,
                    'file': current_file,
                    'line': current_line_no,
                }
                failures.append(failure)
                error_groups.setdefault(current_error_type, []).append(
                    len(failures) - 1)
                current_error_type = None
                current_message_lines = []
                current_file = None
                current_line_no = None
            continue

        # Detect test name context
        for pat in _TEST_NAME_PATTERNS:
            m = pat.search(stripped)
            if m:
                current_test = m.group(1).strip()
                break

        # Detect error type
        et_match = _ERROR_TYPE_RE.search(stripped)
        if et_match:
            # Flush previous if any
            if current_error_type and current_message_lines:
                msg = ' '.join(current_message_lines).strip()
                if len(msg) > 200:
                    msg = msg[:200] + '...'
                failure = {
                    'test': current_test or '(unknown test)',
                    'error_type': current_error_type,
                    'message': msg,
                    'file': current_file,
                    'line': current_line_no,
                }
                failures.append(failure)
                error_groups.setdefault(current_error_type, []).append(
                    len(failures) - 1)

            current_error_type = et_match.group(1)
            # Message is the rest of the line after the error type
            rest = stripped[et_match.end():].lstrip(': ')
            current_message_lines = [rest] if rest else []
            current_file = None
            current_line_no = None

        elif current_error_type:
            # Continuation of error message
            current_message_lines.append(stripped)

        # Detect file:line references
        fl_match = _file_line_re.search(stripped)
        if fl_match and current_error_type:
            current_file = fl_match.group(1) or fl_match.group(3)
            current_line_no = fl_match.group(2) or fl_match.group(4)

    # Flush last failure
    if current_error_type and current_message_lines:
        msg = ' '.join(current_message_lines).strip()
        if len(msg) > 200:
            msg = msg[:200] + '...'
        failure = {
            'test': current_test or '(unknown test)',
            'error_type': current_error_type,
            'message': msg,
            'file': current_file,
            'line': current_line_no,
        }
        failures.append(failure)
        error_groups.setdefault(current_error_type, []).append(
            len(failures) - 1)

    # If we couldn't parse structured failures, fall back to _extract_test_error
    if not failures:
        return _extract_test_error(output, max_chars=max_chars)

    # Build the structured summary grouped by error type
    total = len(failures)
    unique_files = set()
    for f in failures:
        if f.get('file'):
            unique_files.add(f['file'])

    parts: list[str] = []
    parts.append(f"FAILED TESTS SUMMARY ({total} failure(s)"
                 f"{f' across {len(unique_files)} file(s)' if unique_files else ''}):\n")

    # Sort error groups by count (most common first) so LLM fixes root causes
    sorted_groups = sorted(error_groups.items(),
                           key=lambda x: len(x[1]), reverse=True)

    failure_num = 0
    for error_type, indices in sorted_groups:
        count = len(indices)
        parts.append(f"--- {error_type} ({count} occurrence(s)) ---")

        for idx in indices:
            failure_num += 1
            f = failures[idx]
            loc = ""
            if f.get('file'):
                loc = f"  File: {f['file']}"
                if f.get('line'):
                    loc += f":{f['line']}"
            parts.append(f"  {failure_num}. {f['test']}")
            parts.append(f"     Error: {f['message']}")
            if loc:
                parts.append(f"    {loc}")

        parts.append("")  # blank line between groups

    # Add the test summary line if present
    for line in lines:
        stripped = line.strip()
        if re.search(r'\d+\s+(failed|passed|skipped)', stripped):
            parts.append(f"Summary: {stripped}")
            break

    result = '\n'.join(parts).strip()
    if len(result) > max_chars:
        result = result[:max_chars] + '\n... [truncated]'

    return result


def _handle_code_step(step_text: str, coder: CoderAgent, reviewer: ReviewerAgent,
                      executor: Executor, task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      cfg: Config | None = None,
                      auto: bool = False,
                      code_graph=None,
                      project_profile=None,
                      skip_review: bool = False,
                      project_context=None) -> tuple[bool, str]:
    # --- Proactive pre-install: ensure all required packages are installed ---
    # CMD steps scaffold the project first (e.g. npm create vite@latest).
    # By the first CODE step the manifest exists, so we bulk-install any
    # packages from the plan summary that are still missing.
    if project_context is not None:
        subproject_cwd = _detect_subproject_root(memory)
        _ensure_packages_installed(
            project_context, executor, memory, display, step_idx,
            subproject_cwd=subproject_cwd, language=language,
        )

    # --- Tier 1: Diff-aware editing (requires KB graph + high confidence) ---
    if cfg and getattr(cfg, "EDITING_DIFF_MODE", False) and code_graph is not None:
        diff_result = _try_diff_edit(
            step_text=step_text, coder=coder, task=task,
            memory=memory, display=display, step_idx=step_idx,
            language=language, cfg=cfg, code_graph=code_graph,
            project_profile=project_profile,
        )
        if diff_result is not None:
            return diff_result

    # --- Tier 2: Chunk edit (regex-based, no KB graph needed) ---
    if cfg and getattr(cfg, "EDITING_CHUNK_MODE", True):
        chunk_result = _try_chunk_edit(
            step_text=step_text, coder=coder, reviewer=reviewer,
            executor=executor, task=task, memory=memory,
            display=display, step_idx=step_idx,
            language=language, cfg=cfg, auto=auto,
            project_profile=project_profile,
            project_context=project_context,
        )
        if chunk_result is not None:
            return chunk_result

    # --- Tier 3: Full-file flow (fallback) ---
    feedback = ""
    context_window = cfg.CONTEXT_WINDOW if cfg else 8192
    ctx_budget = int(context_window * 0.8)
    prev_files: dict[str, str] = {}  # Track files from previous attempt

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        # Prepend project orientation + knowledge context
        context_prefix = ""
        if project_context is not None:
            coder_analysis = project_context.format_for_coder()
            if coder_analysis:
                context_prefix = coder_analysis + "\n\n"
        if project_profile is not None:
            try:
                context_prefix += project_profile.format_for_prompt() + "\n\n"
            except Exception:
                pass
        kb_ctx = getattr(memory, '_kb_context', '')
        if kb_ctx:
            context_prefix += kb_ctx + "\n\n"

        context = context_prefix + f"Task: {task}"
        # Use slim context for non-target files when enabled
        use_slim = cfg and getattr(cfg, "EDITING_SLIM_CONTEXT", True)
        targets = _detect_target_files(step_text, memory) if use_slim else []
        if use_slim and targets:
            slim = memory.related_context_slim(step_text, max_tokens=ctx_budget)
            if slim:
                context += f"\nProject file structures:\n{slim}"
            for tf in targets:
                tf_content = memory.get(tf)
                if tf_content:
                    context += f"\n\n#### [FILE]: {tf}\n```\n{tf_content}\n```"
        else:
            related = memory.related_context(step_text, max_tokens=ctx_budget)
            if related:
                context += f"\nExisting files (overwrite as needed):\n{related}"
        if memory.summary() != "(no files yet)":
            context += f"\nAll project files: {memory.summary()}"
        if feedback:
            context += f"\nFeedback: {feedback}"
            # On retry, tell the coder to ONLY fix the flagged issues
            context += (
                "\n\nCRITICAL: Only fix the specific issues mentioned in the "
                "feedback above. Do NOT modify any code that is unrelated to "
                "the feedback. Preserve ALL existing content, formatting, and "
                "special characters exactly as they are. Only output the "
                "file(s) that need changes."
            )

        display.step_info(step_idx, f"Coding (attempt {attempt}/{MAX_STEP_RETRIES})...")
        sent_before, recv_before = token_tracker.snapshot()

        response = coder.process(step_text, context=context, language=language)

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        explanation = CLIDisplay.extract_explanation(response)
        if explanation:
            display.add_llm_log(explanation, source="Coder")

        display.step_info(step_idx, "Processing LLM response...")
        files = executor.parse_code_blocks(response)
        if not files:
            files = executor.parse_code_blocks_fuzzy(response)
        if not files:
            feedback = "No file markers found. Use #### [FILE]: path/to/file.py format."
            display.step_info(step_idx, "No files parsed, retrying...")
            log.warning(f"Step {step_idx+1}: No files parsed from coder response.")
            continue

        # Normalize paths: fix LLM-generated paths that are suffixes of
        # known project files (e.g. src/App.js → my-app/src/App.js)
        files = _normalize_fix_paths(files, memory)

        # Prefix files with sub-project root if detected
        # (e.g. components/Header.tsx → my-app/components/Header.tsx)
        subproject = _detect_subproject_root(memory)
        if subproject:
            files = _prefix_subproject_paths(files, subproject, memory)

        # Strip protected manifest files (package.json, etc.) to prevent
        # LLM from overwriting them with corrupted versions
        files = _strip_protected_files(files)

        # Apply KB-driven content fixes (e.g. Tailwind v3→v4 directives)
        content_fixes = getattr(memory, '_content_fixes', None)
        files = _apply_content_fixes(files, content_fixes)

        # On retry, merge: keep previously approved files that weren't
        # re-generated, so the coder doesn't need to regenerate everything
        if attempt > 1 and prev_files:
            merged = dict(prev_files)
            merged.update(files)  # new files override previous
            files = merged

        # Auto-fix hazardous diffs before showing to user
        files = _auto_fix_hazards(files, coder, executor, display, step_idx,
                                  step_text, language=language)

        # Show diffs and wait for approval before writing
        display.stop_spinner()
        approved = prompt_diff_approval(files, auto=auto)
        if not approved:
            feedback = "User rejected the changes. Try a different approach."
            display.step_info(step_idx, "Changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected diff, retrying.")
            continue

        display.step_info(step_idx, "Processing approved changes...")

        # Build review context before updating memory/disk so diffs are captured correctly
        use_diff_review = cfg and getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True)
        if use_diff_review:
            review_ctx = _build_review_context(files, memory, step_text)

        prev_files = dict(files)  # Save for potential merge on retry
        written = executor.write_files(files)
        memory.update(files)
        display.step_info(step_idx, f"Written: {', '.join(written)}")

        # Auto-install any new package imports found in the generated code
        # (e.g. @heroicons/react, lucide-react, etc. that the LLM used
        # but didn't list in required_packages)
        _auto_install_code_imports(files, executor, memory, display, step_idx)

        # Skip review for non-code files (README, LICENSE, configs, etc.)
        if _all_non_code_files(list(files.keys())):
            display.step_info(step_idx, "Non-code files, skipping review ✔")
            log.info(f"Step {step_idx+1}: Skipped review (non-code files: {list(files.keys())})")
            return True, ""

        # ── Review gate ──────────────────────────────────────────
        # When a TEST step follows (skip_review=True), skip the LLM
        # reviewer and rely on test execution to catch real bugs.
        # Always run the free offline lint + import checks regardless.
        lint_errors = _quick_offline_lint(files)
        import_errors = _validate_import_paths(files, memory)
        lint_errors = lint_errors + import_errors

        if skip_review:
            # Lint-only path: skip LLM review, test step will validate
            if lint_errors:
                feedback = f"Lint/syntax errors found:\n{lint_errors}\nFix these issues."
                display.step_info(step_idx, "Lint errors found, retrying...")
                log.warning(f"Step {step_idx+1}: Code lint errors: {lint_errors[:300]}")
                continue
            display.step_info(step_idx, "Review skipped (test step follows) ✔")
            log.info(f"Step {step_idx+1}: Skipped LLM review (test step follows)")
            return True, ""

        # Full LLM review path
        display.step_info(step_idx, "Reviewing code...")
        sent_before, recv_before = token_tracker.snapshot()

        lint_context = f"\n\n{lint_errors}Please fix these errors in your review." if lint_errors else ""

        # Inject KB context so the reviewer has up-to-date framework docs
        # and doesn't reject valid code based on outdated training data.
        reviewer_kb = ""
        if kb_ctx:
            reviewer_kb = (
                f"\n\n[KB Documentation — trust this over your training data]\n"
                f"{kb_ctx}\n"
            )

        # Inject success criteria so the reviewer validates completeness
        criteria_ctx = ""
        if project_context is not None:
            criteria = getattr(project_context, 'success_criteria', [])
            if criteria:
                criteria_ctx = (
                    "\n\nSUCCESS CRITERIA — reject if any are NOT met by the changes:\n"
                    + "\n".join(f"- {c}" for c in criteria)
                )

        use_diff_review = cfg and getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True)
        if use_diff_review:
            review = reviewer.process(
                f"Review this code change for the step: {step_text}\n\n{review_ctx}",
                context=f"Step: {step_text}\nReview ONLY the changes shown.{lint_context}{reviewer_kb}{criteria_ctx}",
                language=language,
                review_mode="diff",
            )
        else:
            review = reviewer.process(
                f"Review this code for the step: {step_text}\n\n{response}",
                context=f"Step: {step_text}\nOnly review changes relevant to this step.{lint_context}{reviewer_kb}{criteria_ctx}",
                language=language,
            )

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        if review:
            display.add_llm_log(review, source="Reviewer")

        log.info(f"Step {step_idx+1}: Review:\n{review}")

        review_lower = review.lower()
        # Accept if the reviewer explicitly approves
        approved = any(phrase in review_lower for phrase in (
            "code looks good",
            "looks good",
            "no issues",
            "no critical issues",
            "no bugs found",
            "code is correct",
            "functionally correct",
            "lgtm",
        ))

        if approved:
            display.step_info(step_idx, "Review passed ✔")
            return True, ""

        # On the last attempt, accept the code if the review only has
        # minor/style suggestions (no keywords indicating actual bugs)
        if attempt == MAX_STEP_RETRIES:
            has_critical = any(kw in review_lower for kw in (
                "error", "bug", "crash", "undefined", "missing import",
                "will fail", "won't work", "does not work", "broken",
                "incorrect", "wrong", "typeerror", "nameerror",
                "syntaxerror", "attributeerror", "keyerror",
                "referenceerror",
            ))
            if not has_critical:
                display.step_info(step_idx, "Review has only minor suggestions, accepting ✔")
                log.info(f"Step {step_idx+1}: Accepted on last attempt "
                         f"(review had no critical keywords)")
                return True, ""

        feedback = review
        display.step_info(step_idx, "Review found issues, retrying...")
        log.warning(f"Step {step_idx+1}: Review issues: {review[:200]}")

    log.error(f"Step {step_idx+1}: Failed after {MAX_STEP_RETRIES} attempts.")
    return False, f"Code step failed after {MAX_STEP_RETRIES} attempts.\nLast review feedback:\n{feedback}"



def _normalize_fix_paths(fix_files: dict[str, str],
                        memory: FileMemory) -> dict[str, str]:
    """Correct LLM-generated paths that are suffixes of known project paths.

    Example: if memory has ``my-app/src/index.js`` and the LLM outputs
    ``src/index.js``, remap to the full path.
    """
    known_paths = set(memory.all_files().keys())
    if not known_paths:
        return fix_files

    corrected: dict[str, str] = {}
    for fpath, content in fix_files.items():
        if fpath in known_paths:
            corrected[fpath] = content
            continue

        # Check if fpath is a suffix of an existing known path
        matched = None
        for known in known_paths:
            if known.endswith('/' + fpath) or known.endswith('\\' + fpath):
                matched = known
                break

        if matched:
            log.warning(f"[PathFix] Remapped '{fpath}' → '{matched}' "
                        f"(matched existing project file)")
            corrected[matched] = content
        else:
            corrected[fpath] = content

    return corrected


def _filter_test_only_files(fix_files: dict[str, str],
                            test_files: dict[str, str],
                            memory: FileMemory) -> dict[str, str]:
    """Filter fix files to only allow test files during test fix loop.

    Blocks writes to:
    - Protected manifest files (package.json, etc.)
    - Source files that already exist in memory (prevents overwrite)

    Allows writes to:
    - Files that were part of the original test_files
    - Files in test directories (__tests__/, tests/, spec/, test/)
    - Files with test naming patterns (test_*, *.test.*, *_test.*, *_spec.*)
    """
    import re
    import os

    allowed: dict[str, str] = {}
    known_source_files = set(memory.all_files().keys())
    test_paths = set(test_files.keys())

    # Patterns for test files
    _TEST_DIR_PATTERNS = {'__tests__', 'tests', 'test', 'spec'}
    _TEST_NAME_RE = re.compile(
        r'(^test_|[./]test[./]|\.test\.|_test\.|_spec\.|spec[./])',
        re.IGNORECASE
    )

    for fpath, content in fix_files.items():
        basename = os.path.basename(fpath)

        # Block: protected manifest files
        if basename in Executor._PROTECTED_FILENAMES:
            log.warning(f"[TestFix] Blocked write to protected file: {fpath}")
            continue

        # Allow: file was in original test_files
        if fpath in test_paths:
            allowed[fpath] = content
            continue

        # Allow: file is in a test directory
        path_parts = set(fpath.replace('\\', '/').split('/'))
        if path_parts & _TEST_DIR_PATTERNS:
            allowed[fpath] = content
            continue

        # Allow: file matches test naming pattern
        if _TEST_NAME_RE.search(fpath):
            allowed[fpath] = content
            continue

        # Block: file is a known source file in memory
        if fpath in known_source_files:
            log.warning(f"[TestFix] Blocked write to source file during "
                        f"test fix: {fpath}")
            continue

        # Default: allow unknown new files (might be test helpers)
        allowed[fpath] = content

    blocked_count = len(fix_files) - len(allowed)
    if blocked_count > 0:
        log.info(f"[TestFix] Blocked {blocked_count} non-test file(s) "
                 f"from test fix write")

    return allowed


def _cleanup_stale_js_test_files(
    test_files: dict[str, str],
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    subproject_cwd: str | None,
) -> None:
    """Remove stale .test.js files when a .test.jsx version was just written.

    Vite/Rollup cannot parse JSX in .js files.  When the TesterAgent generates
    .test.jsx files, any leftover .test.js with the same base name will still
    be picked up by the test runner and cause parse errors.
    """
    jsx_files = [f for f in test_files if f.endswith(('.test.jsx', '.test.tsx'))]
    if not jsx_files:
        return

    removed: list[str] = []
    for jsx_path in jsx_files:
        # Derive the .js / .ts counterpart
        if jsx_path.endswith('.test.jsx'):
            stale = jsx_path.replace('.test.jsx', '.test.js')
        else:
            stale = jsx_path.replace('.test.tsx', '.test.ts')

        abs_stale = (os.path.join(subproject_cwd, stale) if subproject_cwd
                     else stale)
        if os.path.isfile(abs_stale):
            try:
                os.remove(abs_stale)
                removed.append(stale)
                log.info(f"Step {step_idx+1}: Removed stale test file "
                         f"{stale} (replaced by {jsx_path})")
            except OSError as e:
                log.warning(f"Step {step_idx+1}: Failed to remove "
                            f"stale {stale}: {e}")

    if removed:
        display.step_info(step_idx,
                          f"Cleaned up stale .js test files: "
                          f"{', '.join(removed)}")


def _cleanup_ghost_test_files(
    test_files: dict[str, str],
    prev_step_test_files: set[str],
    display: CLIDisplay,
    step_idx: int,
    subproject_cwd: str | None,
) -> None:
    """Remove ghost test files left by previous gen attempts of the SAME step.

    When the TesterAgent retries generation (gen_attempt loop), the new attempt
    may produce a different set of files than the previous attempt.  Files from
    the old attempt that aren't in the new ``test_files`` become "ghosts" —
    the test runner picks them up and they fail with stale import paths.

    ``prev_step_test_files`` should contain the set of file paths written by
    earlier gen attempts **of the same step** (NOT files from other steps).
    This prevents accidentally deleting test files produced by earlier steps
    in the pipeline.
    """
    if not prev_step_test_files:
        return

    current_paths = set(test_files.keys())
    ghosts = prev_step_test_files - current_paths
    if not ghosts:
        return

    removed: list[str] = []
    base = subproject_cwd or "."

    for ghost_path in ghosts:
        abs_path = os.path.join(base, ghost_path) if not os.path.isabs(ghost_path) else ghost_path
        if os.path.isfile(abs_path):
            try:
                os.remove(abs_path)
                removed.append(ghost_path)
                log.info(f"Step {step_idx+1}: Removed ghost test file "
                         f"{ghost_path}")
            except OSError as e:
                log.warning(f"Step {step_idx+1}: Failed to remove "
                            f"ghost {ghost_path}: {e}")

    if removed:
        display.step_info(step_idx,
                          f"Cleaned up {len(removed)} ghost test file(s): "
                          f"{', '.join(removed)}")


# ── Enhancement #6: Import-trace context injection ──────────────
def _extract_imported_sources(test_files: dict[str, str],
                              memory: FileMemory) -> dict[str, str]:
    """Parse import statements in test files to identify tested source files.

    Returns a dict of {filepath: content} for source files that are
    directly imported by the test files.
    """
    import re as _re

    # Python: from src.snake_game import ... / import snake_game
    _PY_IMPORT_RE = _re.compile(
        r'(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))')
    # JS/TS: import ... from '../src/app' / require('../src/app')
    _JS_IMPORT_RE = _re.compile(
        r'''(?:from\s+['"](.+?)['"]|require\s*\(\s*['"](.+?)['"]\s*\))''')

    all_files = memory.all_files()
    imported_sources: dict[str, str] = {}

    for _tpath, tcontent in test_files.items():
        # Collect candidate module paths from imports
        candidates: set[str] = set()
        for m in _PY_IMPORT_RE.finditer(tcontent):
            mod = m.group(1) or m.group(2)
            if mod:
                candidates.add(mod.replace('.', '/'))
        for m in _JS_IMPORT_RE.finditer(tcontent):
            rel = m.group(1) or m.group(2)
            if rel:
                # Strip leading ./ or ../
                clean = _re.sub(r'^\.\.?/', '', rel)
                candidates.add(clean)

        # Match candidates against known memory files
        for fpath, content in all_files.items():
            if fpath in imported_sources:
                continue
            # Skip test files themselves
            if 'test' in fpath.lower():
                continue
            for cand in candidates:
                if cand in fpath or fpath.endswith(cand) or fpath.endswith(cand + '.py'):
                    imported_sources[fpath] = content
                    break

    return imported_sources


# ── Enhancement #9: Spec-driven test generation ─────────────────
def _extract_source_specs(code_summary: str) -> str:
    """Extract function/class signatures and docstrings from source code.

    Returns a compact spec block the tester can use to understand
    what behaviors to test.
    """
    import re as _re

    specs: list[str] = []
    # Parse #### [FILE]: ... ```...``` blocks
    for match in _re.finditer(
            r'####\s+\[FILE\]:\s*(.+?)\n```\w*\n(.*?)```',
            code_summary, _re.DOTALL):
        filepath, content = match.groups()
        if 'test' in filepath.lower():
            continue
        file_specs: list[str] = []
        # Extract class/function definitions + docstrings
        for func_match in _re.finditer(
                r'((?:class|def)\s+\w+[^:]*:)\s*\n(\s+(?:""".*?"""|\'\'\'.*?\'\'\'))?',
                content, _re.DOTALL):
            sig = func_match.group(1).strip()
            doc = func_match.group(2)
            if doc:
                file_specs.append(f"  {sig}\n    {doc.strip()}")
            else:
                file_specs.append(f"  {sig}")
        if file_specs:
            specs.append(f"# {filepath}\n" + "\n".join(file_specs))

    return "\n\n".join(specs) if specs else ""


# ── Enhancement #5: Bidirectional bug detection ──────────────────
def _triage_test_failure(error_detail: str, source_summary: str,
                         test_summary: str, llm_client,
                         display: CLIDisplay, step_idx: int) -> str:
    """Determine whether a test failure is a TEST_BUG or SOURCE_BUG.

    Returns 'TEST_BUG' or 'SOURCE_BUG'.
    """
    triage_prompt = (
        "A test has failed. Analyze the error and determine the root cause.\n"
        "Answer with ONLY one word: TEST_BUG or SOURCE_BUG\n\n"
        "- TEST_BUG = the test assertion, setup, or import is incorrect\n"
        "- SOURCE_BUG = the source code under test has a logic, syntax, "
        "or implementation error\n\n"
        f"Test output:\n{error_detail[:3000]}\n\n"
    )
    if source_summary:
        triage_prompt += f"Source files:\n{source_summary[:4000]}\n\n"
    if test_summary:
        triage_prompt += f"Test files:\n{test_summary[:4000]}\n"

    display.step_info(step_idx, "Analyzing failure origin (test vs source)...")

    sent_before, recv_before = token_tracker.snapshot()
    response = llm_client.generate_response(triage_prompt).strip().upper()
    sent_after, recv_after = token_tracker.snapshot()
    display.step_tokens(step_idx, sent_after - sent_before,
                        recv_after - recv_before)

    if "SOURCE_BUG" in response:
        log.info(f"Step {step_idx+1}: Triage result: SOURCE_BUG")
        return "SOURCE_BUG"
    log.info(f"Step {step_idx+1}: Triage result: TEST_BUG")
    return "TEST_BUG"

def _handle_test_step(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      auto: bool = False,
                      search_agent=None,
                      project_context=None,
                      kb_context_builder=None) -> tuple[bool, str]:
    # Detect sub-project (if the test targets a nested folder)
    subproject_cwd = _detect_subproject_root(memory)

    # ── Proactive pre-install: ensure packages are installed ──
    # Runs once, just before the first test suite command,
    # when the project scaffold already exists.
    if project_context is not None:
        _ensure_packages_installed(
            project_context, executor, memory, display, step_idx,
            subproject_cwd=subproject_cwd, language=language,
        )

    # Infer language from memory file paths when not explicitly provided
    if language is None:
        mem_files = list(memory.all_files().keys())
        if mem_files:
            language = detect_language_from_files(mem_files)
            if language:
                log.info(f"Step {step_idx+1}: Inferred language '{language}' "
                         f"from memory files")

    # Detect JS/TS project environment for ESM-aware test generation
    js_env: dict | None = None
    test_runner: str | None = None
    if language in ("javascript", "typescript"):
        js_env = _read_js_project_env(subproject_cwd)
        test_runner = js_env.get("test_runner")
        log.info(f"Step {step_idx+1}: JS project env: {js_env}")

    # Use language-aware defaults (fall back to Python only as last resort)
    lang_tag = get_code_block_lang(language) if language else "python"
    fw = get_test_framework(language, test_runner=test_runner) if language else get_test_framework("python")
    test_cmd = fw["command"]

    # Ensure the test runner binary is installed before attempting to run tests
    parts = test_cmd.split()
    runner = parts[0]
    # For "npx <tool>", the binary to check is "npx" itself
    if not shutil.which(runner):
        actual_tool = parts[1] if runner == "npx" and len(parts) > 1 else runner
        install_cmd = _get_runner_install_cmd(actual_tool)
        if install_cmd is None:
            # System-level tool (go, cargo, etc.) — can't auto-install
            msg = (f"`{runner}` is not installed. It must be installed manually "
                   f"(it cannot be installed via pip/npm).")
            display.step_info(step_idx, msg)
            log.error(f"Step {step_idx+1}: {msg}")
            return False, msg
        display.step_info(step_idx, f"`{runner}` not found, installing...")
        log.info(f"Step {step_idx+1}: Auto-installing: {install_cmd}")
        ok, out = executor.run_command(install_cmd, cwd=subproject_cwd)
        if ok:
            display.step_info(step_idx, f"Installed `{actual_tool}`")
        else:
            log.warning(f"Step {step_idx+1}: Failed to install "
                        f"{actual_tool}: {out[:200]}")

    # Auto-setup for Jest-based JS/TS projects (skip when Vitest is the runner)
    if js_env and test_runner != "vitest":
        # Auto-setup for ESM projects: install @jest/globals if needed
        if js_env.get("is_esm") and not js_env.get("has_jest_globals"):
            display.step_info(step_idx, "ESM project detected, installing @jest/globals...")
            ok, out = executor.run_command("npm install --save-dev @jest/globals", cwd=subproject_cwd)
            if ok:
                js_env["has_jest_globals"] = True
                display.step_info(step_idx, "Installed @jest/globals")
            else:
                log.warning(f"Step {step_idx+1}: Failed to install @jest/globals: {out[:200]}")

        # Create minimal jest.config for ESM if missing
        if js_env.get("is_esm") and not js_env.get("has_jest_config"):
            jest_config_content = (
                "// Auto-generated for ESM compatibility\n"
                "export default {\n"
                "  transform: {},\n"
                "};\n"
            )
            config_path = os.path.join(subproject_cwd, "jest.config.js") if subproject_cwd else "jest.config.js"
            if not os.path.isfile(config_path):
                try:
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(jest_config_content)
                    js_env["has_jest_config"] = True
                    display.step_info(step_idx, "Created jest.config.js for ESM")
                    log.info(f"Step {step_idx+1}: Auto-created jest.config.js for ESM")
                except OSError as e:
                    log.warning(f"Step {step_idx+1}: Failed to create jest.config.js: {e}")

    code_summary = ""
    for fname, content in memory.all_files().items():
        code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"

    feedback = ""
    last_test_output = ""
    prev_gen_error = None  # Track errors across gen attempts for early exit
    prev_step_test_files: set[str] = set()  # Files from earlier gen attempts of THIS step

    for gen_attempt in range(1, MAX_TEST_GEN_RETRIES + 1):
        display.step_info(step_idx, f"Generating tests (attempt {gen_attempt}/{MAX_TEST_GEN_RETRIES})...")
        kb_ctx = getattr(memory, '_kb_context', '')
        gen_context = ""
        # Inject structured project analysis (gives tester awareness of
        # end-to-end goal, installed packages, import patterns, and
        # assertion guidance — prevents vacuum-based test generation)
        if project_context is not None:
            analysis_block = project_context.format_for_tester()
            if analysis_block:
                gen_context += analysis_block + "\n\n"
        if kb_ctx:
            gen_context += kb_ctx + "\n\n"
        gen_context += f"Code:\n{code_summary}"

        # ── Enhancement #9: spec-driven test generation ──
        source_specs = _extract_source_specs(code_summary)
        if source_specs:
            gen_context += (
                "\nSOURCE CODE SPECIFICATIONS (test these behaviors):\n"
                f"{source_specs}\n"
            )

        if feedback:
            gen_context += f"\nFeedback: {feedback}"
        # Add JS/TS environment info to context
        if js_env:
            env_note = f"\nProject environment: {js_env}"
            if js_env.get('is_esm') and test_runner != "vitest":
                env_note += (
                    "\nCRITICAL: This is an ES Module project. "
                    "Tests MUST import from '@jest/globals'.\n"
                )
            gen_context += env_note

        sent_before, recv_before = token_tracker.snapshot()

        test_response = tester.process(
            step_text, context=gen_context, language=language,
            env_info=js_env)

        sent_after, recv_after = token_tracker.snapshot()
        sent_delta = sent_after - sent_before
        recv_delta = recv_after - recv_before
        display.step_tokens(step_idx, sent_delta, recv_delta)

        explanation = CLIDisplay.extract_explanation(test_response)
        if explanation:
            display.add_llm_log(explanation, source="Tester")

        test_files = executor.parse_code_blocks(test_response)
        if not test_files:
            test_files = executor.parse_code_blocks_fuzzy(test_response)
        if not test_files:
            feedback = "No test files found. Use #### [FILE]: format."
            display.step_info(step_idx, "No test files parsed, retrying...")
            continue

        # Strip protected manifest files before they reach memory
        test_files = _strip_protected_files(test_files)

        # Apply KB-driven content fixes (e.g. jest-dom → jest-dom/vitest)
        content_fixes = getattr(memory, '_content_fixes', None)
        test_files = _apply_content_fixes(test_files, content_fixes)

        # Normalize paths: fix LLM-generated paths that are suffixes of known files
        test_files = _normalize_fix_paths(test_files, memory)

        # Filter: only allow test files (block any source/config files)
        test_files = _filter_test_only_files(test_files, test_files, memory)
        if not test_files:
            feedback = "Generated files were all non-test files. Generate ONLY test files."
            display.step_info(step_idx, "No valid test files after filtering, retrying...")
            continue

        # Strip scaffold/setup files that have no actual test blocks.
        # LLMs sometimes generate "vitestSetup.test.js" or similar files
        # that contain only config/setup but no describe/it/test blocks,
        # causing "No test suite found" errors from the runner.
        test_files = _strip_empty_test_files(test_files)
        if not test_files:
            feedback = ("All generated files were setup/scaffold files with no test blocks. "
                        "Generate ONLY test files containing describe/it/test blocks.")
            display.step_info(step_idx, "No test suites found in generated files, retrying...")
            log.warning(f"Step {step_idx+1}: All generated test files stripped — no test blocks")
            continue

        # Quick offline lint check (free, no LLM cost) — catches syntax
        # errors before writing to disk.  Real test failures are caught by
        # the execution loop below, which provides concrete error output
        # that the coder can act on.
        lint_errors = _quick_offline_lint(test_files)
        if lint_errors:
            feedback = f"Lint/syntax errors found:\n{lint_errors}\nFix these issues."
            display.step_info(step_idx, "Lint errors found, regenerating...")
            log.warning(f"Step {step_idx+1}: Test lint errors: {lint_errors[:300]}")
            continue

        # Show diffs and wait for approval before writing test files
        display.stop_spinner()
        approved = prompt_diff_approval(test_files, auto=auto)
        if not approved:
            feedback = "User rejected the test changes. Try a different approach."
            display.step_info(step_idx, "Test changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected test diff, retrying.")
            continue

        display.step_info(step_idx, "Processing approved test files...")

        # Prefix test file paths with sub-project root (same as CODE steps)
        if subproject_cwd:
            test_files = _prefix_subproject_paths(
                test_files, subproject_cwd, memory)

        written = executor.write_files(test_files)
        memory.update(test_files)
        display.step_info(step_idx, f"Tests written: {', '.join(written)}")

        # ── Cleanup stale .js test files when .jsx versions exist ──
        # Vite/Rollup cannot parse JSX in .js files.  When the LLM generates
        # .test.jsx files, any leftover .test.js file with the same base name
        # causes persistent "Parse failure: Expression expected" errors.
        _cleanup_stale_js_test_files(test_files, memory, display, step_idx,
                                     subproject_cwd)

        # ── Cleanup ghost test files from earlier gen attempts of THIS step ──
        # If this is a retry (gen_attempt > 1), the previous attempt may have
        # written different files.  Those ghosts get picked up by the runner
        # and fail with stale imports.  Only cleans files from THIS step's
        # earlier attempts — never touches files from other pipeline steps.
        _cleanup_ghost_test_files(test_files, prev_step_test_files,
                                  display, step_idx, subproject_cwd)

        # Track files from this gen attempt so later attempts can clean them
        prev_step_test_files = set(test_files.keys())

        # Safety check: if generated test files import from 'vitest' but
        # the detected test command is Jest, override to vitest.
        # This catches cases where the LLM generates Vitest tests but the
        # framework detection still defaults to Jest.
        if "jest" in test_cmd.lower():
            uses_vitest = any(
                "from 'vitest'" in content or 'from "vitest"' in content
                for content in test_files.values()
            )
            if uses_vitest:
                test_cmd = "npx vitest run"
                test_runner = "vitest"
                log.info(f"Step {step_idx+1}: Overriding test command to "
                         f"'{test_cmd}' — test files import from 'vitest'")
                display.step_info(step_idx, "Detected vitest imports → using vitest runner")

        # Scope the test command to only run files from THIS step.
        # This prevents failures in unrelated test files from blocking
        # the current step's fix loop.
        scoped_test_cmd = _build_scoped_test_cmd(
            test_cmd, test_files, subproject_cwd)
        if scoped_test_cmd != test_cmd:
            log.info(f"Step {step_idx+1}: Scoped test command: {scoped_test_cmd}")

        prev_output = None
        prev_fail_count = None
        for run_attempt in range(1, MAX_STEP_RETRIES + 1):
            display.step_info(step_idx, f"Running: {scoped_test_cmd} (attempt {run_attempt})...")
            log.info(f"Step {step_idx+1}: Running test command: {scoped_test_cmd}" + (f" in {subproject_cwd}" if subproject_cwd else ""))
            success, output = executor.run_tests(scoped_test_cmd, cwd=subproject_cwd)
            log.info(f"Step {step_idx+1}: Test run output:\n{output or '(no output)'}")

            last_test_output = output

            if success:
                display.step_info(step_idx, "Tests passed ✔")
                return True, ""

            # Extract error classes and their messages for smarter deduplication
            import re as _re
            _error_classes = set(line.strip() for line in _re.findall(
                r'((?:ModuleNotFoundError|ImportError|SyntaxError|NameError|'
                r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
                r'AssertionError|KeyError|ValueError|ReferenceError)[^\n]*)',
                output or ""
            ))

            # Detect stuck loop: compare error classes, not just exact output
            if prev_output and run_attempt > 1:
                prev_error_classes = set(line.strip() for line in _re.findall(
                    r'((?:ModuleNotFoundError|ImportError|SyntaxError|NameError|'
                    r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
                    r'AssertionError|KeyError|ValueError|ReferenceError)[^\n]*)',
                    prev_output or ""
                ))
                if _error_classes and _error_classes == prev_error_classes:
                    display.step_info(step_idx,
                                      "Same error types repeating — fix not working, stopping.")
                    log.warning(f"Step {step_idx+1}: Same error classes on attempt "
                                f"{run_attempt}: {_error_classes}, breaking retry loop.")
                    break
                if output == prev_output:
                    display.step_info(step_idx,
                                      "Same error repeating — not a code issue, stopping retry loop.")
                    log.warning(f"Step {step_idx+1}: Identical test output on attempt "
                                f"{run_attempt}, breaking retry loop.")
                    break
            prev_output = output

            # Early exit on no progress: if failure count is not decreasing,
            # the fix attempts aren't helping — stop wasting LLM calls
            current_fail_count = _count_test_failures(output or "")
            if (prev_fail_count is not None
                    and current_fail_count > 0
                    and current_fail_count >= prev_fail_count
                    and run_attempt > 1):
                display.step_info(step_idx,
                                  f"No progress ({current_fail_count} failures, "
                                  f"was {prev_fail_count}) — stopping retry loop.")
                log.warning(f"Step {step_idx+1}: Failure count not decreasing "
                            f"({prev_fail_count} → {current_fail_count}), "
                            f"breaking retry loop at attempt {run_attempt}.")
                break
            prev_fail_count = current_fail_count

            # Early exit for unfixable error types
            _unfixable = {e for e in _error_classes if e.startswith(('SyntaxError', 'IndentationError'))}
            if _unfixable and run_attempt > 1:
                _unfixable_names = {e.split(':')[0] for e in _unfixable}
                display.step_info(step_idx,
                                  f"Persistent {', '.join(_unfixable_names)} — likely unfixable, stopping.")
                log.warning(f"Step {step_idx+1}: Unfixable errors after "
                            f"{run_attempt} attempts: {_unfixable}")
                break

            # Detect system-level / environment failures early — these
            # can't be fixed by editing code (e.g. missing Gemfile,
            # missing runtime, missing package manager).
            from .pipeline import _detect_system_level_failure
            sys_issue = _detect_system_level_failure(output)
            if sys_issue:
                msg = (f"System dependency missing: {sys_issue}. "
                       f"Cannot fix by editing code.")
                display.step_info(step_idx, msg)
                log.error(f"Step {step_idx+1}: {msg}")
                return False, msg

            # If test runner itself is not installed, try to install it
            if "not installed" in output or "not on PATH" in output:
                runner_parts = test_cmd.split()
                actual_tool = runner_parts[1] if runner_parts[0] == "npx" and len(runner_parts) > 1 else runner_parts[0]
                install_cmd = _get_runner_install_cmd(actual_tool)
                if install_cmd is None:
                    # System-level tool — can't auto-install, stop retrying
                    display.step_info(step_idx,
                                      f"`{actual_tool}` must be installed manually.")
                    log.error(f"Step {step_idx+1}: `{actual_tool}` is not installed "
                              f"and cannot be auto-installed.")
                    break
                display.step_info(step_idx, f"Installing `{actual_tool}`...")
                log.info(f"Step {step_idx+1}: Installing test runner: {install_cmd}")
                ok, out = executor.run_command(install_cmd, cwd=subproject_cwd)
                if ok:
                    display.step_info(step_idx, f"Installed `{actual_tool}`, re-running...")
                    success, output = executor.run_tests(scoped_test_cmd, cwd=subproject_cwd)
                    last_test_output = output
                    if success:
                        display.step_info(step_idx, "Tests passed after runner install ✔")
                        return True, ""
                continue  # retry with coder fix if runner install + rerun still failed

            # Auto-install missing packages before asking coder to fix
            missing_pkgs = executor.detect_missing_packages(output)
            if missing_pkgs:
                install_tool = "npm install --save-dev" if language in ("javascript", "typescript") else ("pip install")
                
                display.step_info(step_idx, f"Installing missing packages: {', '.join(missing_pkgs)}")
                log.info(f"Step {step_idx+1}: Auto-installing: {missing_pkgs} with {install_tool}")
                install_ok, install_out = executor.install_packages(missing_pkgs, tool=install_tool, cwd=subproject_cwd)
                if install_ok:
                    display.step_info(step_idx, "Packages installed, re-running tests...")
                    log.info(f"Step {step_idx+1}: Re-running test command: {scoped_test_cmd}")
                    success, output = executor.run_tests(scoped_test_cmd, cwd=subproject_cwd)
                    log.info(f"Step {step_idx+1}: Test re-run after install:\n{output or '(no output)'}")
                    last_test_output = output
                    if success:
                        display.step_info(step_idx, "Tests passed after package install ✔")
                        return True, ""
                else:
                    log.warning(f"Step {step_idx+1}: Package install failed: {install_out}")

            # Early exit: if the same error type repeats across gen attempts,
            # the fix isn't working — break to avoid wasting LLM calls
            if prev_gen_error:
                prev_gen_classes = set(_re.findall(
                    r'(ModuleNotFoundError|ImportError|SyntaxError|NameError|'
                    r'TypeError|AttributeError|IndentationError|FileNotFoundError|'
                    r'AssertionError|KeyError|ValueError)',
                    prev_gen_error or ""
                ))
                if (output == prev_gen_error or
                        (_error_classes and _error_classes == prev_gen_classes)):
                    display.step_info(step_idx,
                                      "Same test error repeating across attempts — stopping.")
                    log.warning(f"Step {step_idx+1}: Same error types across gen "
                                f"attempts: {_error_classes}, breaking retry loop.")
                    break
            prev_gen_error = output

            display.step_info(step_idx, "Tests failed, analyzing failure origin...")

            # ── Enhancement #6: import-trace context injection ──
            imported_sources = _extract_imported_sources(test_files, memory)
            source_context = ""
            for fpath, content in imported_sources.items():
                source_context += f"#### [FILE]: {fpath}\n```{lang_tag}\n{content}\n```\n\n"

            # Use batch error summary for structured, grouped view of ALL failures
            batch_summary = _build_batch_error_summary(output) if output else ""
            error_detail = batch_summary or (
                f"(command `{test_cmd}` produced no output — it may have "
                f"crashed or the test framework may not be installed)"
            )

            # ── Enhancement #5: bidirectional bug detection ──
            # Determine whether the bug is in the test or the source code.
            test_summary_for_triage = ""
            for fpath, content in test_files.items():
                test_summary_for_triage += f"{fpath}:\n{content[:1000]}\n\n"
            bug_origin = _triage_test_failure(
                error_detail, source_context, test_summary_for_triage,
                coder.llm_client, display, step_idx)

            is_source_bug = (bug_origin == "SOURCE_BUG")
            if is_source_bug:
                display.step_info(step_idx,
                                  "Bug detected in SOURCE code, asking coder to fix source...")

            # ── KB error-fix lookup for test failures ──
            # Consult the global KB for known error patterns (e.g.
            # TestingLibraryElementError, assertion count mismatches,
            # mock callback issues) to give the coder concrete guidance.
            kb_fix_context = ""
            if kb_context_builder is not None:
                try:
                    kb_ctx = kb_context_builder.build_context(
                        task_description=step_text,
                        error_output=output,
                        max_tokens=2000,
                    )
                    if kb_ctx.error_fixes:
                        kb_fix_context = kb_context_builder.format_context_for_prompt(kb_ctx)
                        log.info(f"Step {step_idx+1}: KB matched "
                                 f"{len(kb_ctx.error_fixes)} error fix patterns "
                                 f"for test fix loop")
                except Exception as exc:
                    log.debug(f"Step {step_idx+1}: KB error lookup in test "
                              f"fix loop failed: {exc}")

            # Search the web for error documentation to help the coder fix
            search_context = ""
            if search_agent is not None:
                display.step_info(step_idx, "Searching web for test error fix...")
                try:
                    search_context = search_agent.search_for_error(
                        error_detail, step_text, language=language,
                        kb_context=getattr(memory, '_kb_context', ''))
                    if search_context:
                        log.info(f"Step {step_idx+1}: Search agent found "
                                 f"test error documentation")
                except Exception as exc:
                    log.warning(f"Step {step_idx+1}: Search agent error "
                                f"during test fix: {exc}")

            # Build fix context with the batch error summary
            test_file_list = ", ".join(test_files.keys())
            fix_context = (
                f"Test command: `{test_cmd}`\n\n"
                f"{error_detail}\n\n"
            )
            if is_source_bug:
                # Include imported source files for source bug fixes
                fix_context += (
                    f"Source files under test:\n{source_context}\n\n"
                    f"Test files (for reference): {test_file_list}\n"
                )
            else:
                fix_context += (
                    f"Test files that need fixing: {test_file_list}\n"
                )
            fix_context += f"Project files:\n{code_summary}"
            if kb_fix_context:
                fix_context += (
                    f"\n\n{kb_fix_context}"
                )
            if search_context:
                fix_context += (
                    f"\n\nThe following web search results contain relevant "
                    f"documentation and solutions for this error. Use them "
                    f"to inform your fix:\n\n{search_context}"
                )
            jsx_hint = ""
            if test_runner == "vitest" and language in ("javascript", "typescript"):
                jsx_ext = ".jsx" if language == "javascript" else ".tsx"
                js_ext = ".js" if language == "javascript" else ".ts"
                jsx_hint = (
                    f"IMPORTANT: Vite/Rollup CANNOT parse JSX in {js_ext} files. "
                    f"If a test file contains JSX (e.g. <Component />, render(<App />)), "
                    f"it MUST use the .test{jsx_ext} extension, NOT .test{js_ext}. "
                    f"If you need to rename a file from .test{js_ext} to .test{jsx_ext}, "
                    f"output the corrected content under the new .test{jsx_ext} path. "
                )

            # ── Enhancement #8: targeted patching + #5: bidirectional fix ──
            if is_source_bug:
                fix_prompt = (
                    "The tests have revealed a BUG IN THE SOURCE CODE, not in the tests.\n"
                    "Analyze the test failures and fix the SOURCE files to make the tests pass.\n"
                    "Do NOT modify test files. Fix ONLY the source implementation files.\n"
                    f"{jsx_hint}"
                    "Return ALL corrected source file(s) using #### [FILE]: format."
                )
            else:
                fix_prompt = (
                    "Below is a structured summary of ALL test failures grouped by error type. "
                    "Analyze the patterns — if multiple failures share a root cause "
                    "(e.g. wrong import path, missing mock, incorrect API usage), "
                    "fix the root cause once rather than patching each test individually.\n"
                    "Focus on fixing ONLY the failing tests. Do NOT rewrite passing tests.\n"
                    f"{jsx_hint}"
                    "Fix ONLY the test files. "
                    "Do NOT modify source files, package.json, or any config files. "
                    "Do NOT add new dependencies. "
                    "Return ALL corrected test file(s) using #### [FILE]: format."
                )

            display.step_info(step_idx, "Generating fix from LLM...")
            sent_before, recv_before = token_tracker.snapshot()

            fix_response = coder.process(
                fix_prompt, context=fix_context, language=language)

            sent_after, recv_after = token_tracker.snapshot()
            sent_delta = sent_after - sent_before
            recv_delta = recv_after - recv_before
            display.step_tokens(step_idx, sent_delta, recv_delta)

            display.step_info(step_idx, "Processing fix response...")
            explanation = CLIDisplay.extract_explanation(fix_response)
            if explanation:
                display.add_llm_log(explanation, source="Coder")

            fix_files = executor.parse_code_blocks(fix_response)
            if not fix_files:
                fix_files = executor.parse_code_blocks_fuzzy(fix_response)
            if fix_files:
                # Strip protected manifest files
                fix_files = _strip_protected_files(fix_files)
                # Normalize paths
                fix_files = _normalize_fix_paths(fix_files, memory)

                if is_source_bug:
                    # For source bugs: write source fixes directly, skip test-only filter
                    show_diffs(fix_files, log_only=True)
                    executor.write_files(fix_files)
                    memory.update(fix_files)
                    log.info(f"Step {step_idx+1}: Applied SOURCE fix to: "
                             f"{', '.join(fix_files.keys())}")
                else:
                    # For test bugs: filter to test-only files
                    fix_files = _filter_test_only_files(
                        fix_files, test_files, memory)
                    if fix_files:
                        show_diffs(fix_files, log_only=True)
                        executor.write_files(fix_files)
                        memory.update(fix_files)
                        # Clean up stale .js files if fix produced .jsx
                        _cleanup_stale_js_test_files(
                            fix_files, memory, display, step_idx,
                            subproject_cwd)
                        # Update scoped command if fix introduced new test files
                        merged_test_files = {**test_files, **fix_files}
                        scoped_test_cmd = _build_scoped_test_cmd(
                            test_cmd, merged_test_files, subproject_cwd)
                    else:
                        log.warning(f"Step {step_idx+1}: All fix files were "
                                    f"blocked by test-only filter")
                code_summary = ""
                for fname, content in memory.all_files().items():
                    code_summary += f"#### [FILE]: {fname}\n```{lang_tag}\n{content}\n```\n\n"

        log.error(f"Step {step_idx+1}: Tests still failing after {MAX_STEP_RETRIES} fixes.")
        return False, f"Tests still failing after {MAX_STEP_RETRIES} fix attempts.\nLast test output:\n{last_test_output}"

    log.error(f"Step {step_idx+1}: Could not generate valid tests after {MAX_TEST_GEN_RETRIES} attempts.")
    return False, f"Could not generate valid tests after {MAX_TEST_GEN_RETRIES} attempts.\nLast feedback:\n{feedback}"


# ---------------------------------------------------------------------------
# Diff-aware editing (Phase 5)
# ---------------------------------------------------------------------------

def _try_diff_edit(
    *,
    step_text: str,
    coder: CoderAgent,
    task: str,
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    language: str | None,
    cfg: Config,
    code_graph,
    project_profile=None,
) -> tuple[bool, str] | None:
    """Attempt a diff-aware edit.  Returns ``(success, error_info)`` on
    success, or ``None`` to signal the caller should fall back to the
    full-file flow.
    """
    import re as _re

    try:
        from ..editing.scope_resolver import ScopeResolver
        from ..editing.context_slicer import ContextSlicer
        from ..editing.diff_parser import DiffParser
        from ..editing.patch_applier import PatchApplier
        from ..editing.metrics import log_edit_metric
    except ImportError as exc:
        log.debug("[DiffEdit] editing module not available: %s", exc)
        return None

    # Identify target file from step text or memory
    target_file = _detect_target_file(step_text, memory)
    if not target_file:
        log.debug("[DiffEdit] No target file identified from step text")
        return None

    # Check the file actually exists on disk
    if not os.path.isfile(target_file):
        log.debug("[DiffEdit] Target file does not exist: %s", target_file)
        return None

    display.step_info(step_idx, f"[DiffEdit] Resolving scope for {os.path.basename(target_file)}...")

    # 1. Resolve scope
    resolver = ScopeResolver(code_graph)
    scope = resolver.resolve(step_text, target_file, code_graph)

    min_conf = getattr(cfg, "EDITING_MIN_CONFIDENCE", 0.60)
    if scope.confidence < min_conf:
        log.warning(
            "[DiffEdit] Confidence %.2f < %.2f for %s, falling back",
            scope.confidence, min_conf, target_file,
        )
        _log_fallback_metric(cfg, target_file, step_text, scope, "low_confidence")
        return None

    # 2. Slice files
    ctx_lines = getattr(cfg, "EDITING_CONTEXT_LINES", 5)
    slicer = ContextSlicer()

    scopes_map: dict = {}
    for af in scope.affected_files:
        scopes_map[af] = scope

    slices = slicer.slice_files(scopes_map)
    formatted = slicer.format_for_prompt(slices)

    # Prepend project orientation + knowledge context to sliced context
    prefix_parts = []
    if project_profile is not None:
        try:
            prefix_parts.append(project_profile.format_for_prompt())
        except Exception:
            pass
    kb_ctx = getattr(memory, '_kb_context', '')
    if kb_ctx:
        prefix_parts.append(kb_ctx)
    if prefix_parts:
        formatted = "\n\n".join(prefix_parts) + "\n\n" + formatted

    # Compute token stats
    full_file_lines = 0
    sliced_lines = 0
    for fslice in slices.values():
        full_file_lines += fslice.total_lines
        sliced_lines += sum(b.line_end - b.line_start + 1 for b in fslice.slices)
        if fslice.imports_block:
            sliced_lines += fslice.imports_block.count("\n") + 1

    display.step_info(step_idx, f"[DiffEdit] Sending {sliced_lines}/{full_file_lines} lines to LLM...")

    # 3. Build diff prompt and call LLM
    diff_prompt = _build_diff_prompt(step_text, formatted)
    sent_before, recv_before = token_tracker.snapshot()

    llm_response = coder.llm_client.generate_response(diff_prompt)

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    # 4. Parse diff
    parser = DiffParser()
    parsed = parser.parse(llm_response)

    if parsed is None:
        log.warning("[DiffEdit] LLM did not return valid diff format, falling back")
        _log_fallback_metric(cfg, target_file, step_text, scope, "parse_failed")
        return None

    # Validate hunks against actual file content
    file_contents: dict[str, list[str]] = {}
    for patch in parsed.file_patches:
        try:
            with open(patch.file_path, "r", encoding="utf-8", errors="replace") as f:
                file_contents[patch.file_path] = f.readlines()
        except OSError:
            pass

    parsed = parser.validate(parsed, file_contents)
    if parsed is None or not parsed.parse_successful:
        log.warning("[DiffEdit] >50%% hunks invalid, falling back")
        _log_fallback_metric(cfg, target_file, step_text, scope, "validation_failed")
        return None

    # 5. Apply patches
    fuzzy_window = getattr(cfg, "EDITING_FUZZY_MATCH_WINDOW", 3)
    validate_syntax = getattr(cfg, "EDITING_VALIDATE_SYNTAX", True)
    fallback_syntax = getattr(cfg, "EDITING_FALLBACK_ON_SYNTAX_ERROR", True)

    applier = PatchApplier(
        fuzzy_match_window=fuzzy_window,
        validate_syntax=validate_syntax,
        fallback_on_syntax_error=fallback_syntax,
    )
    result = applier.apply(parsed)

    if not result.success:
        log.warning("[DiffEdit] Patch apply failed: %s", result.error)
        _log_fallback_metric(cfg, target_file, step_text, scope, f"apply_failed: {result.error}")
        return None

    # Update memory with modified files
    for fpath in result.files_modified:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                memory.update({fpath: f.read()})
        except OSError:
            pass

    display.step_info(
        step_idx,
        f"[DiffEdit] Applied {result.hunks_applied} hunk(s) to "
        f"{len(result.files_modified)} file(s)"
    )

    # Log metrics
    if getattr(cfg, "EDITING_TRACK_METRICS", True):
        reduction = (
            round((1 - sliced_lines / full_file_lines) * 100, 1)
            if full_file_lines > 0 else 0
        )
        log_edit_metric({
            "file": target_file,
            "task_length_chars": len(step_text),
            "resolution_method": scope.resolution_method,
            "confidence": round(scope.confidence, 2),
            "full_file_lines": full_file_lines,
            "sliced_lines_sent": sliced_lines,
            "token_reduction_pct": reduction,
            "hunks_applied": result.hunks_applied,
            "hunks_failed": result.hunks_failed,
            "fallback_used": False,
            "syntax_valid": result.syntax_valid,
            "affected_files_count": len(scope.affected_files),
        })

    return True, ""


def _detect_target_file(step_text: str, memory: FileMemory) -> str | None:
    """Try to identify the target file for editing from the step text."""
    import re as _re

    # Look for explicit file path mentions in the step text
    known_files = list(memory.all_files().keys())

    # Direct mention of a known file path
    for fpath in known_files:
        if fpath in step_text:
            return fpath

    # Check for basename mention
    for fpath in known_files:
        basename = os.path.basename(fpath)
        if basename and basename in step_text and basename.count(".") > 0:
            return fpath

    # Look for file-path-like patterns in the step text
    path_pattern = _re.compile(r'[\w/\\]+\.\w{1,5}')
    for m in path_pattern.finditer(step_text):
        candidate = m.group().replace("\\", "/")
        for fpath in known_files:
            if fpath.endswith(candidate) or candidate.endswith(fpath):
                return fpath

    # If only one file in memory, use it
    if len(known_files) == 1:
        return known_files[0]

    return None


def _detect_target_files(step_text: str, memory: FileMemory,
                         max_files: int = 3) -> list[str]:
    """Identify ALL target files for editing from the step text."""
    import re as _re

    known_files = list(memory.all_files().keys())
    found: list[str] = []
    found_set: set[str] = set()

    def _add(fpath: str) -> None:
        if fpath not in found_set and len(found) < max_files:
            if os.path.isfile(fpath):
                found.append(fpath)
                found_set.add(fpath)

    for fpath in known_files:
        if fpath in step_text:
            _add(fpath)

    for fpath in known_files:
        basename = os.path.basename(fpath)
        if basename and basename in step_text and basename.count(".") > 0:
            _add(fpath)

    path_pattern = _re.compile(r'[\w/\\]+\.\w{1,5}')
    for m in path_pattern.finditer(step_text):
        candidate = m.group().replace("\\", "/")
        for fpath in known_files:
            if fpath.endswith(candidate) or candidate.endswith(fpath):
                _add(fpath)

    if not found and len(known_files) == 1:
        _add(known_files[0])

    return found


def _build_chunk_prompt(
    task_description: str,
    formatted_chunks: str,
    slim_context: str,
    language: str | None = None,
) -> str:
    """Build the LLM prompt for chunk-level editing."""
    lang_tag = language or "python"
    return f"""You are editing existing code. You receive CHUNKS of files, not full files.
You MUST respond ONLY with the chunks you want to change.
Do NOT output unchanged chunks. Do NOT output full files.
Do NOT use #### [FILE]: markers. Use #### [EDIT]: markers instead.

Task: {task_description}

{slim_context}

{formatted_chunks}

For EACH chunk you want to change, use EXACTLY this format:

#### [EDIT]: {{file_path}}:{{function_or_class_name}} (lines {{start}}-{{end}})
```{lang_tag}
// complete replacement for this chunk only
```

For adding a NEW function/class, use:

#### [NEW]: {{file_path}} (after line {{line_number}})
```{lang_tag}
// new code to insert
```

Rules:
1. Only output chunks that ACTUALLY CHANGE
2. Include the COMPLETE replacement chunk (not a partial diff)
3. Preserve the function/class signature unless the task requires changing it
4. Match the existing indentation style exactly
5. Use the exact file paths shown in the context above
6. The line numbers MUST match the line ranges shown in EDITABLE markers
"""


def _build_review_context(
    new_files: dict[str, str],
    memory: FileMemory,
    step_text: str,
) -> str:
    """Build a compact review context showing only what changed."""
    import difflib

    parts: list[str] = []

    for fpath, new_content in new_files.items():
        old_content = memory.get(fpath)
        if old_content is None and os.path.isfile(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            except OSError:
                pass

        if old_content is not None:
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{fpath}", tofile=f"b/{fpath}",
                n=3,
            )
            diff_text = "".join(diff)
            if diff_text.strip():
                parts.append(f"#### Changes in {fpath}:\n```diff\n{diff_text}```")
            else:
                parts.append(f"#### {fpath}: no changes")
        else:
            parts.append(f"#### [NEW FILE]: {fpath}\n```\n{new_content}\n```")

    return "\n\n".join(parts)


def _try_chunk_edit(
    *,
    step_text: str,
    coder: CoderAgent,
    reviewer: ReviewerAgent,
    executor: Executor,
    task: str,
    memory: FileMemory,
    display: CLIDisplay,
    step_idx: int,
    language: str | None,
    cfg: Config,
    auto: bool = False,
    project_profile=None,
    project_context=None,
) -> tuple[bool, str] | None:
    """Attempt chunk-level editing. Returns (success, error) or None for fallback."""
    try:
        from ..editing.chunk_editor import ChunkEditor
    except ImportError:
        return None

    max_files = getattr(cfg, "EDITING_MAX_CHUNK_FILES", 3)
    target_files = _detect_target_files(step_text, memory, max_files=max_files)
    if not target_files:
        log.debug("[ChunkEdit] No target files identified")
        return None

    chunk_editor = ChunkEditor()
    all_chunks: list = []
    all_target_ids: list[str] = []

    for fpath in target_files:
        content = memory.get(fpath)
        if content is None:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue

        chunks = chunk_editor.chunk_file(fpath, content)
        if not chunks:
            continue

        targets = chunk_editor.identify_target_chunks(chunks, step_text)
        if not targets:
            targets = [c.chunk_id for c in chunks if c.chunk_type != "imports"]

        all_chunks.extend(chunks)
        all_target_ids.extend(targets)

    if not all_chunks:
        log.debug("[ChunkEdit] No chunks extracted")
        return None

    formatted = chunk_editor.format_chunks_for_prompt(all_chunks, all_target_ids)

    slim_ctx = ""
    if getattr(cfg, "EDITING_SLIM_CONTEXT", True):
        slim_ctx = memory.related_context_slim(
            step_text,
            max_tokens=int((cfg.CONTEXT_WINDOW if cfg else 8192) * 0.3),
        )
        for tf in target_files:
            pattern = f"#### [FILE_STRUCTURE]: {tf}"
            if pattern in slim_ctx:
                start = slim_ctx.find(pattern)
                next_entry = slim_ctx.find("\n\n#### [FILE_STRUCTURE]:", start + 1)
                if next_entry != -1:
                    slim_ctx = slim_ctx[:start] + slim_ctx[next_entry + 2:]
                else:
                    slim_ctx = slim_ctx[:start].rstrip()

    prompt_prefix = ""
    if project_profile is not None:
        try:
            prompt_prefix = project_profile.format_for_prompt() + "\n\n"
        except Exception:
            pass
    kb_ctx = getattr(memory, '_kb_context', '')
    if kb_ctx:
        prompt_prefix += kb_ctx + "\n\n"

    chunk_prompt = prompt_prefix + _build_chunk_prompt(
        step_text, formatted, slim_ctx, language=language,
    )

    display.step_info(step_idx, f"[ChunkEdit] Sending {len(all_target_ids)} target chunks...")
    sent_before, recv_before = token_tracker.snapshot()

    llm_response = coder.llm_client.generate_response(chunk_prompt)

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    edits = chunk_editor.parse_chunk_response(llm_response)
    if edits is None:
        log.info("[ChunkEdit] LLM used full-file format or no edits parsed, falling back")
        return None

    # Detect dropped edits: count [EDIT] markers in the raw response and
    # compare against parsed edits.  If the LLM produced EDIT markers that
    # failed to parse (e.g. multi-word chunk names before the regex fix),
    # fall back to full-file mode rather than applying partial edits.
    raw_edit_count = len(re.findall(
        r"####\s*\[EDIT\]:", llm_response,
    ))
    if raw_edit_count > len(edits):
        log.warning(
            "[ChunkEdit] Parsed %d edits but found %d [EDIT] markers in "
            "response — some edits were dropped, falling back to full-file",
            len(edits), raw_edit_count,
        )
        return None

    edits_by_file: dict[str, list] = {}
    for edit in edits:
        edits_by_file.setdefault(edit.file_path, []).append(edit)

    result_files: dict[str, str] = {}
    original_files: dict[str, str | None] = {}
    for fpath, file_edits in edits_by_file.items():
        original = memory.get(fpath)
        if original is None:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    original = f.read()
            except OSError:
                log.info("[ChunkEdit] Treating %s as a new file", fpath)
                original = None

        original_files[fpath] = original

        try:
            # Filter chunks for this file so the editor can resolve
            # mismatched line numbers from the LLM.
            file_chunks = [c for c in all_chunks if c.file_path == fpath]
            result_files[fpath] = chunk_editor.apply_chunk_edits(
                original or "", file_edits, known_chunks=file_chunks,
            )
        except Exception as exc:
            log.warning("[ChunkEdit] Failed to apply edits to %s: %s", fpath, exc)
            return None

    if not result_files:
        return None

    # Build review context before updating memory/disk so diffs are captured correctly
    review_ctx = _build_review_context(result_files, memory, step_text)

    display.stop_spinner()
    approved = prompt_diff_approval(result_files, auto=auto)
    if not approved:
        display.step_info(step_idx, "Changes rejected by user")
        return False, "User rejected chunk edits."

    display.step_info(step_idx, "Processing approved changes...")
    written = executor.write_files(result_files)
    memory.update(result_files)
    display.step_info(step_idx, f"[ChunkEdit] Written: {', '.join(written)}")

    if _all_non_code_files(list(result_files.keys())):
        display.step_info(step_idx, "Non-code files, skipping review")
        return True, ""

    reviewer_mode = "diff" if getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True) else "full"

    display.step_info(step_idx, "Reviewing changes...")
    sent_before, recv_before = token_tracker.snapshot()

    lint_errors = _quick_offline_lint(result_files)
    lint_context = f"\n\n{lint_errors}Please fix these errors in your review." if lint_errors else ""

    # Inject KB context so the reviewer has up-to-date framework docs
    reviewer_kb = ""
    if kb_ctx:
        reviewer_kb = (
            f"\n\n[KB Documentation — trust this over your training data]\n"
            f"{kb_ctx}\n"
        )

    # Inject success criteria so the reviewer validates completeness
    criteria_ctx = ""
    if project_context is not None:
        criteria = getattr(project_context, 'success_criteria', [])
        if criteria:
            criteria_ctx = (
                "\n\nSUCCESS CRITERIA — reject if any are NOT met by the changes:\n"
                + "\n".join(f"- {c}" for c in criteria)
            )

    review = reviewer.process(
        f"Review this code change for the step: {step_text}\n\n{review_ctx}",
        context=f"Step: {step_text}\nReview ONLY the changes shown.{lint_context}{reviewer_kb}{criteria_ctx}",
        language=language,
        review_mode=reviewer_mode,
    )

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    if review:
        display.add_llm_log(review, source="Reviewer")

    review_lower = review.lower()
    approved = any(phrase in review_lower for phrase in (
        "code looks good", "looks good", "no issues", "no critical issues",
        "no bugs found", "code is correct", "functionally correct", "lgtm",
    ))

    if approved:
        display.step_info(step_idx, "Review passed")
        return True, ""

    log.info("[ChunkEdit] Review found issues, reverting and falling back to full-file for retry")
    
    # Revert memory and disk since the review failed
    for fpath, orig_content in original_files.items():
        if orig_content is not None:
            memory.update({fpath: orig_content})
            try:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(orig_content)
            except OSError:
                pass
        else:
            # It was a new file, remove it
            try:
                if fpath in memory.all_files():
                    del memory.all_files()[fpath]  # Depending on FileMemory implementation
                os.remove(fpath)
            except Exception:
                pass

    return None


def _build_diff_prompt(task_description: str, formatted_slices: str) -> str:
    """Build the LLM prompt requesting a structured diff response."""
    return f"""You are editing existing code. You will receive minimal file slices, not full files.
You MUST respond with ONLY a diff in the exact format specified.
NEVER rewrite the full file. NEVER include unchanged lines in your response.
ALWAYS use the exact line numbers shown in the slice annotations.

Task: {task_description}

{formatted_slices}

Respond with ONLY a diff in this exact format — nothing else:

@@DIFF_START@@
FILE: {{file_path}}
<<<<<<< ORIGINAL (line {{start_line}})
{{exactly as lines appear in the slice}}
=======
{{your replacement lines}}
>>>>>>> UPDATED
@@DIFF_END@@

Rules:
- Use line numbers exactly as they appear in the ORIGINAL slice block annotations. First line of ORIGINAL must match {{start_line}}.
- Only include blocks that actually change.
- Preserve indentation exactly.
- If no changes are needed for a file, omit it entirely.
- The ORIGINAL block must EXACTLY MATCH the slice content you want to replace, including all whitespace.
- For multiple changes in the same file, include multiple ORIGINAL/UPDATED blocks under the same FILE header.
- For changes across multiple files, include multiple FILE sections."""


def _log_fallback_metric(
    cfg: Config,
    target_file: str,
    step_text: str,
    scope,
    reason: str,
) -> None:
    """Log a fallback metric entry."""
    if not getattr(cfg, "EDITING_TRACK_METRICS", True):
        return
    try:
        from ..editing.metrics import log_edit_metric
        log_edit_metric({
            "file": target_file,
            "task_length_chars": len(step_text),
            "resolution_method": scope.resolution_method,
            "confidence": round(scope.confidence, 2),
            "full_file_lines": 0,
            "sliced_lines_sent": 0,
            "token_reduction_pct": 0,
            "hunks_applied": 0,
            "hunks_failed": 0,
            "fallback_used": True,
            "fallback_reason": reason,
            "syntax_valid": True,
            "affected_files_count": len(scope.affected_files),
        })
    except Exception:
        pass

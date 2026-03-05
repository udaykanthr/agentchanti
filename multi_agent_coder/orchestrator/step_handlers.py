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

    # Check for .tsx files (useful for React testing guidance)
    scan_dir = cwd or "."
    if os.path.isdir(os.path.join(scan_dir, "src")):
        for root, _dirs, files in os.walk(os.path.join(scan_dir, "src")):
            if any(f.endswith(".tsx") for f in files):
                env["has_tsx"] = True
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
        result = search_agent.search_for_task(step_text, language=language)
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
                     language: str | None = None) -> tuple[bool, str]:
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
            r'\bnpm\s+(install|start|run|test|build|ci)\b',
            r'\bnpx\s+',
            r'\byarn\s+(install|add|start|dev|build|test)\b',
            r'\bpnpm\s+(install|add|start|dev|build|test)\b',
            r'\bnode\s+',
            r'\bng\s+(serve|build|test)\b',
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


def _build_batch_error_summary(output: str, max_chars: int = 4000) -> str:
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
                      skip_review: bool = False) -> tuple[bool, str]:
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
        if project_profile is not None:
            try:
                context_prefix = project_profile.format_for_prompt() + "\n\n"
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
        approved = prompt_diff_approval(files, auto=auto)
        if not approved:
            feedback = "User rejected the changes. Try a different approach."
            display.step_info(step_idx, "Changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected diff, retrying.")
            continue

        # Build review context before updating memory/disk so diffs are captured correctly
        use_diff_review = cfg and getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True)
        if use_diff_review:
            review_ctx = _build_review_context(files, memory, step_text)

        prev_files = dict(files)  # Save for potential merge on retry
        written = executor.write_files(files)
        memory.update(files)
        display.step_info(step_idx, f"Written: {', '.join(written)}")

        # Skip review for non-code files (README, LICENSE, configs, etc.)
        if _all_non_code_files(list(files.keys())):
            display.step_info(step_idx, "Non-code files, skipping review ✔")
            log.info(f"Step {step_idx+1}: Skipped review (non-code files: {list(files.keys())})")
            return True, ""

        # ── Review gate ──────────────────────────────────────────
        # When a TEST step follows (skip_review=True), skip the LLM
        # reviewer and rely on test execution to catch real bugs.
        # Always run the free offline lint check regardless.
        lint_errors = _quick_offline_lint(files)

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

        use_diff_review = cfg and getattr(cfg, "EDITING_REVIEWER_DIFF_MODE", True)
        if use_diff_review:
            review = reviewer.process(
                f"Review this code change for the step: {step_text}\n\n{review_ctx}",
                context=f"Step: {step_text}\nReview ONLY the changes shown.{lint_context}{reviewer_kb}",
                language=language,
                review_mode="diff",
            )
        else:
            review = reviewer.process(
                f"Review this code for the step: {step_text}\n\n{response}",
                context=f"Step: {step_text}\nOnly review changes relevant to this step.{lint_context}{reviewer_kb}",
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


def _handle_test_step(step_text: str, tester: TesterAgent, coder: CoderAgent,
                      reviewer: ReviewerAgent, executor: Executor,
                      task: str, memory: FileMemory,
                      display: CLIDisplay, step_idx: int,
                      language: str | None = None,
                      auto: bool = False,
                      search_agent=None) -> tuple[bool, str]:
    # Detect sub-project (if the test targets a nested folder)
    subproject_cwd = _detect_subproject_root(memory)

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

    for gen_attempt in range(1, MAX_TEST_GEN_RETRIES + 1):
        display.step_info(step_idx, f"Generating tests (attempt {gen_attempt}/{MAX_TEST_GEN_RETRIES})...")
        kb_ctx = getattr(memory, '_kb_context', '')
        gen_context = ""
        if kb_ctx:
            gen_context += kb_ctx + "\n\n"
        gen_context += f"Code:\n{code_summary}"
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

        # Normalize paths: fix LLM-generated paths that are suffixes of known files
        test_files = _normalize_fix_paths(test_files, memory)

        # Filter: only allow test files (block any source/config files)
        test_files = _filter_test_only_files(test_files, test_files, memory)
        if not test_files:
            feedback = "Generated files were all non-test files. Generate ONLY test files."
            display.step_info(step_idx, "No valid test files after filtering, retrying...")
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
        approved = prompt_diff_approval(test_files, auto=auto)
        if not approved:
            feedback = "User rejected the test changes. Try a different approach."
            display.step_info(step_idx, "Test changes rejected by user, retrying...")
            log.info(f"Step {step_idx+1}: User rejected test diff, retrying.")
            continue

        # Prefix test file paths with sub-project root (same as CODE steps)
        if subproject_cwd:
            test_files = _prefix_subproject_paths(
                test_files, subproject_cwd, memory)

        written = executor.write_files(test_files)
        memory.update(test_files)
        display.step_info(step_idx, f"Tests written: {', '.join(written)}")

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

        prev_output = None
        prev_fail_count = None
        for run_attempt in range(1, MAX_STEP_RETRIES + 1):
            display.step_info(step_idx, f"Running: {test_cmd} (attempt {run_attempt})...")
            log.info(f"Step {step_idx+1}: Running test command: {test_cmd}" + (f" in {subproject_cwd}" if subproject_cwd else ""))
            success, output = executor.run_tests(test_cmd, cwd=subproject_cwd)
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
                    success, output = executor.run_tests(test_cmd, cwd=subproject_cwd)
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
                    log.info(f"Step {step_idx+1}: Re-running test command: {test_cmd}")
                    success, output = executor.run_tests(test_cmd, cwd=subproject_cwd)
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

            display.step_info(step_idx, "Tests failed, asking coder to fix...")

            # Use batch error summary for structured, grouped view of ALL failures
            batch_summary = _build_batch_error_summary(output) if output else ""
            error_detail = batch_summary or (
                f"(command `{test_cmd}` produced no output — it may have "
                f"crashed or the test framework may not be installed)"
            )

            # Search the web for error documentation to help the coder fix
            search_context = ""
            if search_agent is not None:
                display.step_info(step_idx, "Searching web for test error fix...")
                try:
                    search_context = search_agent.search_for_error(
                        error_detail, step_text, language=language)
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
                f"Test files that need fixing: {test_file_list}\n"
                f"Project files:\n{code_summary}"
            )
            if search_context:
                fix_context += (
                    f"\n\nThe following web search results contain relevant "
                    f"documentation and solutions for this error. Use them "
                    f"to inform your fix:\n\n{search_context}"
                )
            fix_prompt = (
                "Below is a structured summary of ALL test failures grouped by error type. "
                "Analyze the patterns — if multiple failures share a root cause "
                "(e.g. wrong import path, missing mock, incorrect API usage), "
                "fix the root cause once rather than patching each test individually. "
                "Fix ONLY the test files. "
                "Do NOT modify source files, package.json, or any config files. "
                "Do NOT add new dependencies. "
                "Return ALL corrected test file(s) using #### [FILE]: format."
            )

            sent_before, recv_before = token_tracker.snapshot()

            fix_response = coder.process(
                fix_prompt, context=fix_context, language=language)

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
            if fix_files:
                # Strip protected manifest files
                fix_files = _strip_protected_files(fix_files)
                # Normalize paths and filter to test-only files
                fix_files = _normalize_fix_paths(fix_files, memory)
                fix_files = _filter_test_only_files(
                    fix_files, test_files, memory)
                if fix_files:
                    show_diffs(fix_files, log_only=True)
                    executor.write_files(fix_files)
                    memory.update(fix_files)
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

    approved = prompt_diff_approval(result_files, auto=auto)
    if not approved:
        display.step_info(step_idx, "Changes rejected by user")
        return False, "User rejected chunk edits."

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

    review = reviewer.process(
        f"Review this code change for the step: {step_text}\n\n{review_ctx}",
        context=f"Step: {step_text}\nReview ONLY the changes shown. {lint_context}{reviewer_kb}",
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

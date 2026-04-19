"""
Language detection and test framework mapping for multi-language support.
"""

import json
import os
from collections import Counter


# ── Extension → Language mapping ──

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".php": "php",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".lua": "lua",
    ".sh": "bash",
    ".bat": "batch",
    ".ps1": "powershell",
    ".css": "css",
    ".html": "html",
}


# ── Test framework configs per language ──

TEST_FRAMEWORKS = {
    "python": {
        "command": "python -m pytest",
        "dir": "tests",
        "ext": ".py",
        "prefix": "test_",
    },
    "javascript": {
        "command": "npx jest --forceExit --watchAll=false",
        "dir": "__tests__",
        "ext": ".js",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev jest",
        "config_note": (
            "Jest must be listed in devDependencies. "
            "If package.json has `\"type\": \"module\"`, tests must use "
            "`import/export` syntax. Otherwise, use `require()/module.exports`. "
            "Use relative paths from the test file to the source file."
        ),
    },
    "javascript:vitest": {
        "command": "npx vitest run",
        "dir": "__tests__",
        "ext": ".jsx",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev vitest",
        "config_note": (
            "Vitest is the test runner. Use `import { describe, it, expect } from 'vitest';` "
            "in every test file. Use ES `import` syntax for all imports. "
            "IMPORTANT: Use .jsx extension for test files containing JSX (Vite cannot parse JSX in .js files)."
        ),
    },
    "typescript": {
        "command": "npx jest --forceExit --watchAll=false",
        "dir": "__tests__",
        "ext": ".ts",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev jest ts-jest @types/jest",
        "config_note": (
            "TypeScript projects need ts-jest or @swc/jest configured. "
            "Ensure jest.config has `transform` set for .ts files. "
            "Use `import/export` syntax in test files."
        ),
    },
    "typescript:vitest": {
        "command": "npx vitest run",
        "dir": "__tests__",
        "ext": ".ts",
        "prefix": "",
        "suffix": ".test",
        "setup_cmd": "npm install --save-dev vitest",
        "config_note": (
            "Vitest is the test runner. Use `import { describe, it, expect } from 'vitest';` "
            "in every test file. Use ES `import` syntax for all imports. "
            "For React components (.tsx), use @testing-library/react."
        ),
    },
    "go": {
        "command": "go test ./...",
        "dir": "",
        "ext": ".go",
        "prefix": "",
        "suffix": "_test",
    },
    "rust": {
        "command": "cargo test",
        "dir": "tests",
        "ext": ".rs",
        "prefix": "test_",
    },
    "java": {
        "command": "mvn test",
        "dir": "src/test/java",
        "ext": ".java",
        "prefix": "Test",
    },
    "ruby": {
        "command": "bundle exec rspec",
        "dir": "spec",
        "ext": ".rb",
        "prefix": "",
        "suffix": "_spec",
    },
}


# ── Keyword → Language mapping for task string detection ──

_TASK_KEYWORDS = {
    "python":     ["python", "flask", "django", "fastapi", "pip", "pytest", "pandas", "numpy"],
    "javascript": ["javascript", "node", "express", "react", "vue", "npm", "webpack", "jest",
                   "next.js", "nextjs", "next js", "create-next-app", "nuxt", "vite",
                   "svelte", "gatsby", "remix"],
    "typescript": ["typescript", "angular", "nest", "tsx", "tsc", "nestjs"],
    "go":         ["golang", "go module", "gin", "echo framework"],
    "rust":       ["rust", "cargo", "tokio", "actix"],
    "java":       ["java", "spring", "maven", "gradle", "junit"],
    "ruby":       ["ruby", "rails", "sinatra", "rspec", "bundler"],
    "csharp":     ["c#", "csharp", "dotnet", ".net", "asp.net"],
    "cpp":        ["c++", "cpp", "cmake"],
}

_LANGUAGE_NAMES = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "rust": "Rust",
    "java": "Java",
    "ruby": "Ruby",
    "csharp": "C#",
    "cpp": "C++",
    "c": "C",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "php": "PHP",
    "scala": "Scala",
    "r": "R",
    "lua": "Lua",
    "bash": "Bash",
    "batch": "Batch",
    "powershell": "PowerShell",
}

_CODE_BLOCK_LANGS = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "ruby": "ruby",
    "csharp": "csharp",
    "cpp": "cpp",
    "c": "c",
    "swift": "swift",
    "kotlin": "kotlin",
    "php": "php",
    "scala": "scala",
}

# Directories to skip when scanning for language detection
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv",
              "env", "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
              "target", "bin", "obj", ".idea", ".vscode", ".agentchanti"}


def detect_language(directory: str = ".") -> str:
    """Scan file extensions in *directory* and return the most common language.

    Checks project manifest files first (package.json → JavaScript, go.mod → Go,
    etc.) before falling back to extension counting.  This prevents untracked or
    leftover files of another language from overriding the true project language.

    Returns ``"python"`` as default when no recognized files are found.
    """
    # Ordered list of manifest filenames → language.  First match wins.
    _MANIFEST_LANGS = [
        ("package.json", "javascript"),
        ("go.mod", "go"),
        ("Cargo.toml", "rust"),
        ("requirements.txt", "python"),
        ("setup.py", "python"),
        ("pyproject.toml", "python"),
        ("manage.py", "python"),   # Django projects
        ("pom.xml", "java"),
        ("build.gradle", "java"),
        ("Gemfile", "ruby"),
        ("composer.json", "php"),
    ]

    # 1. Root-level manifests are the strongest signal.
    # Collect all matches — if multiple languages found, they conflict and we
    # fall through to extension counting rather than picking the wrong one.
    root_langs: list[str] = []
    for manifest, lang in _MANIFEST_LANGS:
        if os.path.isfile(os.path.join(directory, manifest)):
            if lang not in root_langs:
                root_langs.append(lang)
    if len(root_langs) == 1:
        return root_langs[0]
    # Multiple manifests → ambiguous; fall through to extension counting.
    # (Zero manifests also falls through.)

    # 2. One level deep — covers sub-projects (e.g. responsive-app/package.json).
    # Only used when root level was unambiguous-free.
    if not root_langs:
        try:
            for entry in sorted(os.listdir(directory)):
                subdir = os.path.join(directory, entry)
                if not os.path.isdir(subdir) or entry in _SKIP_DIRS:
                    continue
                for manifest, lang in _MANIFEST_LANGS:
                    if os.path.isfile(os.path.join(subdir, manifest)):
                        return lang
        except OSError:
            pass

    # 3. Fallback: extension counting (original behaviour).
    ext_counts: Counter = Counter()
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext in EXTENSION_MAP:
                ext_counts[EXTENSION_MAP[ext]] += 1

    if not ext_counts:
        return "python"
    return ext_counts.most_common(1)[0][0]


def detect_language_from_task(task: str) -> str | None:
    """Try to infer the language from keywords in the task string.

    Returns ``None`` if no match is found (caller should fall back to
    ``detect_language``).
    """
    task_lower = task.lower()
    for lang, keywords in _TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in task_lower:
                return lang
    return None


def get_test_framework(language: str, test_runner: str | None = None) -> dict:
    """Return test framework config for *language*, defaulting to pytest.

    Uses the pluggable LanguageBackend system first, falls back to the
    static TEST_FRAMEWORKS dict for backward compatibility.

    When *test_runner* is ``"vitest"`` and the language is JS or TS, returns
    the Vitest-specific framework config instead of Jest.
    """
    # Try the pluggable backend first
    from .language_backend import get_backend
    backend = get_backend(language)
    if backend.language == language or backend.language == language.split(":")[0]:
        return backend.get_test_framework(test_runner)
    # Fallback to static dict
    if test_runner == "vitest" and language in ("javascript", "typescript"):
        return TEST_FRAMEWORKS[f"{language}:vitest"]
    return TEST_FRAMEWORKS.get(language, TEST_FRAMEWORKS["python"])


def get_language_name(language: str) -> str:
    """Human-readable name for a language key."""
    return _LANGUAGE_NAMES.get(language, language.capitalize())


def get_code_block_lang(language: str) -> str:
    """Markdown fence language tag for a language key."""
    return _CODE_BLOCK_LANGS.get(language, language)


def detect_language_from_files(file_paths: list[str]) -> str | None:
    """Infer the project language from a list of file paths.

    Typically called with paths extracted from ``#### [FILE]:`` markers in
    context strings.  Returns ``None`` when no recognised extensions are found.
    """
    ext_counts: Counter = Counter()
    for path in file_paths:
        _, ext = os.path.splitext(path)
        lang = EXTENSION_MAP.get(ext)
        if lang:
            ext_counts[lang] += 1
    if not ext_counts:
        return None
    return ext_counts.most_common(1)[0][0]


def detect_test_runner(directory: str | None = None) -> str | None:
    """Detect whether a JS/TS project uses Vitest or Jest.

    Checks for ``vitest.config.*`` files first, then ``vite.config.*`` with
    embedded ``test:`` section, then ``jest.config.*`` files, then falls back
    to inspecting ``package.json`` devDependencies.  Vite-based projects
    without an explicit Jest config default to Vitest.

    Returns ``"vitest"``, ``"jest"``, or ``None``.
    """
    cwd = directory or "."

    # Check for vitest config files
    for name in ("vitest.config.ts", "vitest.config.js", "vitest.config.mts",
                 "vitest.config.mjs"):
        if os.path.isfile(os.path.join(cwd, name)):
            return "vitest"

    # Check for vite.config with embedded test section
    for name in ("vite.config.ts", "vite.config.js", "vite.config.mts",
                 "vite.config.mjs"):
        vite_path = os.path.join(cwd, name)
        if os.path.isfile(vite_path):
            try:
                with open(vite_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "test:" in content or "test :" in content:
                    return "vitest"
            except OSError:
                pass
            break  # stop after first vite.config found

    # Check for jest config files
    for name in ("jest.config.js", "jest.config.ts", "jest.config.mjs",
                 "jest.config.cjs", "jest.config.json"):
        if os.path.isfile(os.path.join(cwd, name)):
            return "jest"

    # Fall back to package.json devDependencies
    pkg_path = os.path.join(cwd, "package.json")
    if os.path.isfile(pkg_path):
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            all_deps = {}
            all_deps.update(pkg.get("dependencies", {}))
            all_deps.update(pkg.get("devDependencies", {}))
            if "vitest" in all_deps:
                return "vitest"
            # Vite project without Jest → default to Vitest
            has_vite = ("vite" in all_deps or
                        any(k.startswith("@vitejs/") for k in all_deps))
            if has_vite and "jest" not in all_deps:
                return "vitest"
            if "jest" in all_deps:
                return "jest"
        except (json.JSONDecodeError, OSError):
            pass

    return None


def detect_language_llm(
    file_paths: list[str],
    task: str,
    llm_client,
    current_language: str | None = None,
) -> str | None:
    """Ask the LLM to determine the primary programming language of the project.

    Uses a representative sample of file paths and the task description as
    evidence.  Returns a language key (e.g. ``"python"``) or ``None`` if the
    LLM response cannot be mapped to a known language.

    Intended to be called during pre-analysis when the heuristic detection may
    have picked the wrong language (e.g. a ``go.mod`` in a Django project).
    """
    import logging
    _log = logging.getLogger(__name__)

    # Build a representative sample — cap at 40 paths to keep the prompt small.
    # Prefer well-known manifest/config filenames first.
    _PRIORITY_NAMES = {
        "manage.py", "settings.py", "requirements.txt", "setup.py",
        "pyproject.toml", "package.json", "go.mod", "Cargo.toml",
        "pom.xml", "build.gradle", "Gemfile", "composer.json",
    }
    prioritised = sorted(
        file_paths,
        key=lambda p: (os.path.basename(p) not in _PRIORITY_NAMES, p),
    )
    sample = prioritised[:40]

    known_langs = ", ".join(sorted(set(EXTENSION_MAP.values()) | set(_LANGUAGE_NAMES)))
    current_hint = (
        f"\nCurrent auto-detected language (may be wrong): {current_language}"
        if current_language else ""
    )

    prompt = (
        "You are a language detection expert. Given the file paths of a software "
        "project and a task description, identify the PRIMARY programming language "
        "of the project.\n"
        f"{current_hint}\n\n"
        "File paths:\n"
        + "\n".join(f"  {p}" for p in sample)
        + f"\n\nTask: {task}\n\n"
        f"Known language keys: {known_langs}\n\n"
        "Reply with ONLY the language key (e.g. python, javascript, go, rust, java). "
        "No explanation, no punctuation — just the single language key."
    )

    try:
        response = llm_client.generate_response(prompt).strip().lower()
        # Strip any stray punctuation/words the LLM may add
        response = response.split()[0].rstrip(".,;:")
        # Map common aliases
        _ALIASES = {"py": "python", "js": "javascript", "ts": "typescript",
                    "golang": "go", "c#": "csharp", "c++": "cpp"}
        response = _ALIASES.get(response, response)
        if response in _LANGUAGE_NAMES or response in set(EXTENSION_MAP.values()):
            _log.info("[LangDetect] LLM identified language: %s", response)
            return response
        _log.warning("[LangDetect] LLM returned unrecognised language: %r", response)
    except Exception as exc:
        _log.warning("[LangDetect] LLM language detection failed: %s", exc)

    return None

"""
Post-step dependency validation — detects orphaned exports, broken imports,
and missing connections between files after each pipeline step.

Uses fast regex-based import/export extraction (no tree-sitter dependency)
and a single LLM call to fix all detected gaps.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

# ── Extension → language family mapping ──────────────────────────

_EXT_TO_LANG_FAMILY: dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
}

# ── Test-file detection patterns ─────────────────────────────────

_TEST_FILE_SUFFIXES = (
    ".test.js", ".test.ts", ".test.jsx", ".test.tsx",
    ".spec.js", ".spec.ts", ".spec.jsx", ".spec.tsx",
    "_test.go", "_test.py",
)


def _is_test_file(file_path: str) -> bool:
    """Return ``True`` if *file_path* looks like a test file."""
    basename = os.path.basename(file_path).lower()
    if basename.startswith("test_"):
        return True
    for suffix in _TEST_FILE_SUFFIXES:
        if basename.endswith(suffix):
            return True
    return False


# ── Data structures ──────────────────────────────────────────────

@dataclass
class FileDeps:
    """Import/export information for a single file."""
    file_path: str
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)


@dataclass
class IntegrationGap:
    """A detected integration problem between files."""
    gap_type: str               # "orphaned_export" | "broken_import" | "missing_connection"
    source_file: str            # The file where the problem originates
    target_file: str | None     # The file that should reference/be referenced
    symbol: str                 # The symbol name(s) involved
    description: str            # Human-readable explanation


@dataclass
class DependencySnapshot:
    """Snapshot of all import/export relationships at a point in time."""
    file_deps: dict[str, FileDeps] = field(default_factory=dict)


# ── Import / export regex patterns ───────────────────────────────

# Python
_PY_IMPORT_PATTERNS = [
    re.compile(r"^\s*from\s+([\w.]+)\s+import\s+", re.MULTILINE),
    re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE),
]
_PY_EXPORT_PATTERNS = [
    re.compile(r"^(?:class|def)\s+(\w+)", re.MULTILINE),
    re.compile(r"__all__\s*=\s*\[([^\]]+)\]"),
]

# JavaScript / TypeScript
_JS_IMPORT_PATTERNS = [
    re.compile(r"""^\s*import\s+.*?\s+from\s+['"](.*?)['"]""", re.MULTILINE),
    re.compile(r"""^\s*import\s+['"](.*?)['"]""", re.MULTILINE),
    re.compile(r"""(?:require|import)\s*\(\s*['"](.*?)['"]\s*\)"""),
]
_JS_EXPORT_PATTERNS = [
    re.compile(r"^\s*export\s+default\s+(?:function|class)\s+(\w+)", re.MULTILINE),
    re.compile(r"^\s*export\s+(?:function|class|const|let|var|type|interface|enum)\s+(\w+)", re.MULTILINE),
    re.compile(r"^\s*export\s*\{([^}]+)\}", re.MULTILINE),
    re.compile(r"module\.exports\s*=\s*(\w+|\{[^}]+\})"),
    re.compile(r"^\s*export\s+default\s+(\w+)\s*;?\s*$", re.MULTILINE),
]

# Go
_GO_IMPORT_PATTERNS = [
    re.compile(r'^\s*"(.*?)"$', re.MULTILINE),
    re.compile(r'^\s*import\s+"(.*?)"$', re.MULTILINE),
]
_GO_EXPORT_PATTERNS = [
    re.compile(r"^(?:func|type|var|const)\s+([A-Z]\w*)", re.MULTILINE),
    re.compile(r"^func\s+\(\w+\s+\*?\w+\)\s+([A-Z]\w*)", re.MULTILINE),
]

# Java
_JAVA_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+([\w.]+);", re.MULTILINE),
]
_JAVA_EXPORT_PATTERNS = [
    re.compile(
        r"^\s*(?:public|protected)\s+(?:static\s+)?(?:abstract\s+)?"
        r"(?:class|interface|enum)\s+(\w+)",
        re.MULTILINE,
    ),
]

# Rust
_RUST_IMPORT_PATTERNS = [
    re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE),
]
_RUST_EXPORT_PATTERNS = [
    re.compile(r"^\s*pub\s+(?:fn|struct|enum|trait|type|const|static|mod)\s+(\w+)", re.MULTILINE),
]

_LANG_PATTERNS: dict[str, tuple[list, list]] = {
    "python":     (_PY_IMPORT_PATTERNS, _PY_EXPORT_PATTERNS),
    "javascript": (_JS_IMPORT_PATTERNS, _JS_EXPORT_PATTERNS),
    "typescript": (_JS_IMPORT_PATTERNS, _JS_EXPORT_PATTERNS),
    "go":         (_GO_IMPORT_PATTERNS, _GO_EXPORT_PATTERNS),
    "java":       (_JAVA_IMPORT_PATTERNS, _JAVA_EXPORT_PATTERNS),
    "rust":       (_RUST_IMPORT_PATTERNS, _RUST_EXPORT_PATTERNS),
}


# ── Python stdlib modules (for external import detection) ────────

_PYTHON_STDLIB = frozenset({
    "abc", "argparse", "ast", "asyncio", "base64", "bisect", "calendar",
    "collections", "contextlib", "copy", "csv", "ctypes", "dataclasses",
    "datetime", "decimal", "difflib", "email", "enum", "errno",
    "fileinput", "fnmatch", "fractions", "ftplib", "functools", "gc",
    "getpass", "glob", "gzip", "hashlib", "heapq", "hmac", "html",
    "http", "imaplib", "importlib", "inspect", "io", "itertools",
    "json", "keyword", "linecache", "locale", "logging", "lzma",
    "math", "mimetypes", "multiprocessing", "numbers", "operator",
    "os", "pathlib", "pickle", "pkgutil", "platform", "pprint",
    "profile", "queue", "random", "re", "readline", "reprlib",
    "secrets", "select", "shelve", "shlex", "shutil", "signal",
    "site", "smtplib", "socket", "sqlite3", "ssl", "stat",
    "statistics", "string", "struct", "subprocess", "sys", "sysconfig",
    "tempfile", "textwrap", "threading", "time", "timeit", "token",
    "tokenize", "trace", "traceback", "tracemalloc", "types", "typing",
    "unicodedata", "unittest", "urllib", "uuid", "venv", "warnings",
    "wave", "weakref", "webbrowser", "xml", "xmlrpc", "zipfile",
    "zipimport", "zlib", "_thread",
})


# ── Core functions ───────────────────────────────────────────────

def extract_file_deps(
    file_path: str, content: str, language: str | None = None,
) -> FileDeps:
    """Extract import and export information from a single file.

    Uses language-specific regex patterns. If *language* is ``None`` it is
    inferred from the file extension.
    """
    if not language:
        ext = os.path.splitext(file_path)[1].lower()
        language = _EXT_TO_LANG_FAMILY.get(ext)

    if not language or language not in _LANG_PATTERNS:
        return FileDeps(file_path=file_path)

    import_patterns, export_patterns = _LANG_PATTERNS[language]

    imports: list[str] = []
    for pat in import_patterns:
        for m in pat.finditer(content):
            source = m.group(1).strip().strip("'\"")
            if source and source not in imports:
                imports.append(source)

    exports: list[str] = []
    for pat in export_patterns:
        for m in pat.finditer(content):
            raw = m.group(1).strip()
            if "," in raw or raw.startswith("{"):
                # Comma-separated: export { X, Y } or module.exports = { X, Y }
                inner = raw.strip("{}")
                for name in inner.split(","):
                    name = name.strip().split(" as ")[0].split(":")[0].strip()
                    name = name.strip("'\"")  # strip quotes from __all__ entries
                    if name and name not in exports:
                        exports.append(name)
            else:
                if raw and raw not in exports:
                    exports.append(raw)

    # Filter private Python symbols (leading underscore)
    if language == "python":
        exports = [e for e in exports if not e.startswith("_")]

    return FileDeps(file_path=file_path, imports=imports, exports=exports)


def build_snapshot(
    memory_files: dict[str, str], language: str | None = None,
) -> DependencySnapshot:
    """Build a dependency snapshot from all files currently in memory.

    Skips test files (they import source but are not expected to be imported)
    and non-code files.
    """
    snapshot = DependencySnapshot()
    for fpath, content in memory_files.items():
        ext = os.path.splitext(fpath)[1].lower()
        if ext not in _EXT_TO_LANG_FAMILY:
            continue
        if _is_test_file(fpath):
            continue
        snapshot.file_deps[fpath] = extract_file_deps(fpath, content, language)
    return snapshot


# ── Import path resolution helpers ───────────────────────────────

def _normalize_import_path(import_source: str, importer_file: str) -> str:
    """Resolve relative import paths to a comparable form.

    JS/TS: ``./components/Header`` from ``src/App.tsx`` → ``src/components/Header``
    Python: ``.models`` from ``app/views.py`` → ``app.models``
    """
    if import_source.startswith("./") or import_source.startswith("../"):
        importer_dir = os.path.dirname(importer_file)
        resolved = os.path.normpath(os.path.join(importer_dir, import_source))
        return resolved.replace("\\", "/")
    if import_source.startswith("."):
        # Python relative import
        importer_dir = os.path.dirname(importer_file).replace("/", ".").replace("\\", ".")
        dots = len(import_source) - len(import_source.lstrip("."))
        relative_module = import_source[dots:]
        parts = importer_dir.split(".")
        base = ".".join(parts[:max(0, len(parts) - dots + 1)])
        if relative_module:
            return f"{base}.{relative_module}" if base else relative_module
        return base
    return import_source


def _file_matches_import(file_path: str, import_source: str) -> bool:
    """Check if *file_path* could be the target of *import_source*.

    Handles extension-less imports, Python dotted paths, and index files.
    """
    fp = file_path.replace("\\", "/")
    imp = import_source.replace("\\", "/")

    fp_noext = os.path.splitext(fp)[0]
    imp_noext = os.path.splitext(imp)[0] if "." in os.path.basename(imp) else imp

    # Direct match
    if fp_noext == imp_noext or fp_noext.endswith("/" + imp_noext):
        return True

    # Python dotted module → path (app.models → app/models)
    imp_as_path = imp.replace(".", "/")
    if fp_noext == imp_as_path or fp_noext.endswith("/" + imp_as_path):
        return True

    # Index file (./components → ./components/index)
    if fp_noext.endswith("/index") and fp_noext.rsplit("/index", 1)[0] == imp_noext:
        return True

    return False


def _is_external_import(import_source: str, importer_file: str) -> bool:
    """Return ``True`` if *import_source* refers to an external package.

    JS/TS: anything not starting with ``./`` or ``../`` (node_modules).
    Python: stdlib modules.
    Go: anything that looks like a stdlib or remote module.
    """
    ext = os.path.splitext(importer_file)[1].lower()

    if ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        if import_source.startswith("./") or import_source.startswith("../"):
            return False
        if import_source.startswith("@/") or import_source.startswith("~/"):
            return False
        return True

    if ext == ".py":
        if import_source.startswith("."):
            return False
        top_module = import_source.split(".")[0]
        return top_module in _PYTHON_STDLIB

    if ext == ".go":
        # Local relative imports start with "."; stdlib has no "."
        if import_source.startswith("."):
            return False
        return True

    if ext in (".java",):
        # java.* and javax.* are stdlib
        if import_source.startswith("java.") or import_source.startswith("javax."):
            return True
        return False

    if ext == ".rs":
        # std::, core::, alloc:: are stdlib
        if import_source.startswith(("std::", "core::", "alloc::")):
            return True
        return False

    return True


# ── Parent file guessing heuristic ───────────────────────────────

def _guess_parent_file(
    new_file: str, step_text: str, memory_files: dict[str, str],
) -> str | None:
    """Heuristic to guess which file should import *new_file*.

    Strategies (tried in order):
    1. Both file stems are mentioned in the step text.
    2. An index file exists in the same or parent directory.
    3. Common root files (App.tsx, main.py, etc.).
    """
    stem = os.path.splitext(os.path.basename(new_file))[0]
    ext = os.path.splitext(new_file)[1].lower()

    # Strategy 1: step text mentions both the new file and another file
    step_lower = step_text.lower()
    if stem.lower() in step_lower:
        for fpath in memory_files:
            if fpath == new_file:
                continue
            other_stem = os.path.splitext(os.path.basename(fpath))[0]
            if other_stem.lower() in step_lower:
                return fpath

    # Strategy 2: index file in the same or parent directory
    parent_dir = os.path.dirname(new_file)
    for index_name in ("index.tsx", "index.ts", "index.jsx", "index.js", "__init__.py"):
        index_path = os.path.join(parent_dir, index_name).replace("\\", "/")
        if index_path in memory_files:
            return index_path

    # Strategy 3: common root files
    if ext in (".tsx", ".jsx", ".ts", ".js"):
        for root_name in ("App.tsx", "App.jsx", "App.ts", "App.js",
                          "main.tsx", "main.ts", "main.jsx", "main.js"):
            for fpath in memory_files:
                if fpath.endswith(root_name) and fpath != new_file:
                    return fpath

    if ext == ".py":
        for fpath in memory_files:
            if fpath.endswith("__init__.py") and fpath != new_file:
                fdir = os.path.dirname(fpath)
                if new_file.startswith(fdir):
                    return fpath

    return None


def _find_file_by_name(name: str, memory_files: dict[str, str]) -> str | None:
    """Find a file in memory whose stem matches *name* (case-insensitive)."""
    name_lower = name.lower()
    for fpath in memory_files:
        stem = os.path.splitext(os.path.basename(fpath))[0].lower()
        if stem == name_lower:
            return fpath
    return None


# ── Gap detection ────────────────────────────────────────────────

def find_gaps(
    before: DependencySnapshot,
    after: DependencySnapshot,
    new_files: list[str],
    step_text: str,
    memory_files: dict[str, str],
) -> list[IntegrationGap]:
    """Compare before/after snapshots to detect integration gaps.

    Detects:
    1. **Orphaned exports** — newly created file exports symbols but nothing
       imports it.
    2. **Broken imports** — a file imports a path that doesn't resolve to
       any known file (only NEW imports are flagged).
    3. **Missing connections** — step text mentions connecting two entities
       but no import edge exists between them.
    """
    gaps: list[IntegrationGap] = []
    all_known_files = set(memory_files.keys())

    # ── 1. Orphaned exports ──
    for nf in new_files:
        nf_deps = after.file_deps.get(nf)
        if not nf_deps or not nf_deps.exports:
            continue

        nf_basename = os.path.splitext(os.path.basename(nf))[0].lower()

        is_imported = False
        for other_path, other_deps in after.file_deps.items():
            if other_path == nf:
                continue
            for imp_src in other_deps.imports:
                resolved = _normalize_import_path(imp_src, other_path)
                if _file_matches_import(nf, resolved) or _file_matches_import(nf, imp_src):
                    is_imported = True
                    break
                # Also check if the import's basename matches the new file's
                # basename.  This handles the common pattern where the same
                # symbol is imported from a sibling directory (e.g. App.jsx
                # imports './components/Homepage' while the new file lives at
                # 'pages/Homepage.jsx').  A basename match means the symbol
                # is already wired — the path just differs.
                imp_basename = os.path.splitext(
                    os.path.basename(imp_src)
                )[0].lower()
                if imp_basename == nf_basename and not _is_external_import(imp_src, other_path):
                    is_imported = True
                    break
            if is_imported:
                break

        if not is_imported and len(after.file_deps) > 1:
            likely_parent = _guess_parent_file(nf, step_text, memory_files)
            gaps.append(IntegrationGap(
                gap_type="orphaned_export",
                source_file=nf,
                target_file=likely_parent,
                symbol=", ".join(nf_deps.exports[:5]),
                description=(
                    f"File '{nf}' exports [{', '.join(nf_deps.exports[:5])}] "
                    f"but no other file imports it."
                    + (f" Likely parent: '{likely_parent}'." if likely_parent else "")
                ),
            ))

    # ── 2. Broken imports ──
    for fpath, deps in after.file_deps.items():
        for imp_src in deps.imports:
            if _is_external_import(imp_src, fpath):
                continue

            resolved = _normalize_import_path(imp_src, fpath)
            found = any(
                _file_matches_import(known, resolved) or _file_matches_import(known, imp_src)
                for known in all_known_files
            )

            if not found:
                # Only flag imports that were NOT present before this step
                before_deps = before.file_deps.get(fpath)
                was_present_before = (
                    before_deps is not None and imp_src in before_deps.imports
                )
                if not was_present_before:
                    gaps.append(IntegrationGap(
                        gap_type="broken_import",
                        source_file=fpath,
                        target_file=None,
                        symbol=imp_src,
                        description=f"File '{fpath}' imports '{imp_src}' but no matching file exists.",
                    ))

    # ── 3. Missing connections ──
    connection_patterns = [
        re.compile(
            r"(?:add|integrate|include|use|import|wire|connect)\s+(\w+)"
            r"\s+(?:to|into|in|within)\s+(\w+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(\w+)\s+(?:component|module|service|class)\s+.*?"
            r"(?:to|into|in)\s+(\w+)",
            re.IGNORECASE,
        ),
    ]
    for pat in connection_patterns:
        m = pat.search(step_text)
        if not m:
            continue
        source_name, target_name = m.group(1), m.group(2)
        source_file = _find_file_by_name(source_name, memory_files)
        target_file = _find_file_by_name(target_name, memory_files)
        if source_file and target_file and source_file != target_file:
            target_deps = after.file_deps.get(target_file)
            if target_deps:
                has_import = any(
                    _file_matches_import(source_file, _normalize_import_path(imp, target_file))
                    or _file_matches_import(source_file, imp)
                    for imp in target_deps.imports
                )
                if not has_import:
                    already = any(
                        g.source_file == source_file and g.target_file == target_file
                        for g in gaps
                    )
                    if not already:
                        gaps.append(IntegrationGap(
                            gap_type="missing_connection",
                            source_file=source_file,
                            target_file=target_file,
                            symbol=source_name,
                            description=(
                                f"Step mentions connecting '{source_name}' to "
                                f"'{target_name}' but '{target_file}' does not "
                                f"import '{source_file}'."
                            ),
                        ))

    return gaps


# ── LLM fix prompt ───────────────────────────────────────────────

_FIX_PROMPT_TEMPLATE = """\
You are a dependency integration specialist. Files were just created or modified \
but have integration gaps that must be fixed. Fix ALL gaps with minimal changes.

## Step Context
{step_text}

## Integration Gaps Found
{gaps_formatted}

## Current File Contents
{files_formatted}

## Module System
{module_system_note}

## Rules
- Output ONLY files that need changes (not unchanged files).
- Use #### [FILE]: path format followed by a code block.
- Include COMPLETE file contents for changed files.
- Match existing import style (ESM/CJS, relative/absolute, single/double quotes).
- For component imports, use the correct relative path from importer to importee.
- Do NOT modify package.json, go.mod, or other config/manifest files.
- Do NOT add unnecessary imports — only fix the gaps listed above.
- Preserve ALL existing code, comments, and formatting in modified files.
"""


def build_fix_prompt(
    gaps: list[IntegrationGap],
    memory_files: dict[str, str],
    step_text: str,
    language: str | None,
) -> str:
    """Build a single LLM prompt to fix all integration gaps."""
    gap_lines: list[str] = []
    for i, gap in enumerate(gaps, 1):
        tag = gap.gap_type.upper().replace("_", " ")
        gap_lines.append(f"{i}. [{tag}] {gap.description}")
    gaps_formatted = "\n".join(gap_lines)

    # Collect only relevant files
    relevant_files: set[str] = set()
    for gap in gaps:
        relevant_files.add(gap.source_file)
        if gap.target_file:
            relevant_files.add(gap.target_file)

    file_parts: list[str] = []
    for fpath in sorted(relevant_files):
        content = memory_files.get(fpath, "")
        if content:
            file_parts.append(f"#### [FILE]: {fpath}\n```\n{content}\n```")
    files_formatted = "\n\n".join(file_parts)

    # Detect module system
    module_system_note = "Unknown module system."
    if language in ("javascript", "typescript"):
        has_esm = any("import " in c and " from " in c for c in memory_files.values())
        has_cjs = any("require(" in c for c in memory_files.values())
        if has_esm:
            module_system_note = "Project uses ES Modules (import/export). Use ESM syntax."
        elif has_cjs:
            module_system_note = "Project uses CommonJS (require/module.exports). Use CJS syntax."
    elif language == "python":
        module_system_note = "Python project. Use standard import syntax."
    elif language == "go":
        module_system_note = "Go project. Use standard import syntax."

    return _FIX_PROMPT_TEMPLATE.format(
        step_text=step_text,
        gaps_formatted=gaps_formatted,
        files_formatted=files_formatted,
        module_system_note=module_system_note,
    )


# ── Main entry point ─────────────────────────────────────────────

def run_dependency_check(
    step_idx: int,
    step_text: str,
    new_files: list[str],
    dep_before: DependencySnapshot,
    dep_after: DependencySnapshot,
    memory,
    llm_client,
    executor,
    display,
    language: str | None,
    cfg=None,
) -> dict[str, str]:
    """Run post-step dependency validation and return fixes.

    Returns ``{}`` immediately if the feature is disabled, there are too few
    files in memory, or no integration gaps are detected (zero LLM overhead).
    When gaps *are* found, makes a **single** LLM call to fix all of them.
    """
    # Feature toggle
    if cfg and not getattr(cfg, "DEPENDENCY_CHECK_ENABLED", True):
        return {}

    memory_files = memory.all_files()

    # Need at least 2 files to have a dependency relationship
    if len(memory_files) <= 1:
        return {}

    if not new_files:
        return {}

    # Detect gaps
    display.step_info(step_idx, "[DepCheck] Scanning dependencies...")
    try:
        gaps = find_gaps(dep_before, dep_after, new_files, step_text, memory_files)
    except Exception as exc:
        _logger.warning("[DepCheck] Gap detection failed: %s", exc)
        return {}

    if not gaps:
        _logger.debug("[DepCheck] No integration gaps for step %d", step_idx + 1)
        display.step_info(step_idx, "[DepCheck] All dependencies connected.")
        return {}

    # Report gaps
    display.step_info(
        step_idx,
        f"[DepCheck] Found {len(gaps)} integration gap(s), generating fixes...",
    )
    for gap in gaps:
        _logger.info("[DepCheck] %s: %s", gap.gap_type, gap.description)

    # Single LLM call
    try:
        prompt = build_fix_prompt(gaps, memory_files, step_text, language)
        display.step_info(step_idx, "[DepCheck] Fixing dependency gaps (LLM)...")
        response = llm_client.generate_response(prompt)
    except Exception as exc:
        _logger.warning("[DepCheck] LLM fix call failed: %s", exc)
        display.step_info(step_idx, "[DepCheck] Fix generation failed, continuing without fixes")
        return {}

    # Parse response
    display.step_info(step_idx, "[DepCheck] Applying fixes...")
    fix_files = executor.parse_code_blocks(response)
    if not fix_files:
        fix_files = executor.parse_code_blocks_fuzzy(response)

    if not fix_files:
        _logger.warning("[DepCheck] Could not parse fix response")
        display.step_info(step_idx, "[DepCheck] No parseable fixes in response")
        return {}

    # Validate: only accept files relevant to the gaps
    relevant_files: set[str] = set()
    for gap in gaps:
        relevant_files.add(gap.source_file)
        if gap.target_file:
            relevant_files.add(gap.target_file)

    validated: dict[str, str] = {}
    for fpath, content in fix_files.items():
        matched = fpath
        if fpath not in memory_files:
            for known in memory_files:
                if known.endswith(fpath) or fpath.endswith(os.path.basename(known)):
                    matched = known
                    break
        if matched in relevant_files or matched in memory_files:
            validated[matched] = content
        else:
            _logger.warning("[DepCheck] Ignoring unexpected file in fix: %s", fpath)

    if validated:
        _logger.info("[DepCheck] Generated fixes for %d file(s)", len(validated))
    return validated

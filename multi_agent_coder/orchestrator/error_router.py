"""
ErrorRouter — classifies a step failure to decide whether to search the web,
analyse local code, rely on KB docs, or skip web search entirely.

Design goals:
- Zero LLM tokens for the common cases (regex + ProjectContext).
- One tiny single-shot LLM call only for genuinely ambiguous errors.
- Result is reused across all diagnosis retry attempts (computed once).
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error classes that are *always* code bugs — web search cannot help
# ---------------------------------------------------------------------------
_CODE_ONLY_EXCEPTIONS = frozenset({
    "TypeError", "AttributeError", "NameError", "UnboundLocalError",
    "SyntaxError", "IndentationError", "TabError",
    "KeyError", "ValueError", "IndexError", "StopIteration",
    "RecursionError", "OverflowError", "ZeroDivisionError",
    "AssertionError", "NotImplementedError",
    # JS equivalents (Jest/Node output)
    "ReferenceError", "RangeError", "EvalError",
})

# Error patterns that signal a missing package / wrong CLI version → need web
_WEB_PATTERNS: list[re.Pattern] = [
    re.compile(r'No module named\s+[\'"]?([\w\-\.]+)', re.IGNORECASE),
    re.compile(r'ModuleNotFoundError', re.IGNORECASE),
    re.compile(r'Cannot find module\s+[\'"]?([\w\-\./@]+)', re.IGNORECASE),
    re.compile(r'Module not found', re.IGNORECASE),
    re.compile(r'command not found', re.IGNORECASE),
    re.compile(r'Unknown command', re.IGNORECASE),
    re.compile(r'is not recognized as an internal or external command', re.IGNORECASE),
    re.compile(r'Operation cance[l]+ed', re.IGNORECASE),   # Vite/npm interactive prompt aborted
    re.compile(r'Operation was aborted', re.IGNORECASE),
    re.compile(r'npm ERR!\s+code E404', re.IGNORECASE),
    re.compile(r'pip.*could not find a version', re.IGNORECASE),
    re.compile(r'No matching distribution found', re.IGNORECASE),
    re.compile(r'ENOENT.*node_modules', re.IGNORECASE),
    re.compile(r'deprecated|removed in|not supported in version', re.IGNORECASE),
]

# Patterns that are clearly local/path/env issues → code, not web
_CODE_PATTERNS: list[re.Pattern] = [
    re.compile(r'PermissionError', re.IGNORECASE),
    re.compile(r'FileNotFoundError', re.IGNORECASE),
    re.compile(r'IsADirectoryError', re.IGNORECASE),
    re.compile(r'No such file or directory', re.IGNORECASE),
    re.compile(r'ENOENT(?!.*node_modules)', re.IGNORECASE),  # local path, not npm
]

# Extract package name from "No module named 'foo'" / "Cannot find module 'foo'"
_PKG_NAME_RE = re.compile(
    r"(?:No module named|Cannot find module|ModuleNotFoundError[^'\"]*)['\"]?([\w\-\./@]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ErrorRoute:
    """Routing decision for a single step failure."""
    source_type: str          # "code" | "web" | "kb" | "web+code"
    skip_web: bool            # True → do not call search_agent at all
    query_hint: str = ""      # pre-built search query (replaces regex extraction)
    confidence: str = "regex" # "regex" | "llm"
    reason: str = ""          # short human-readable explanation (for logs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_error(
    error_info: str,
    step_type: str = "CODE",
    project_context=None,
    kb_matched: bool = False,
    llm_client=None,
) -> ErrorRoute:
    """Classify a step failure to route context gathering.

    Args:
        error_info:      Full error output / traceback.
        step_type:       "CMD" | "CODE" | "TEST"
        project_context: ProjectContext from AnalyseAgent (may be None).
        kb_matched:      True when KB error_fixes already matched this error.
        llm_client:      Used only for ambiguous cases (single-shot call).

    Returns:
        ErrorRoute with routing decision.  Never raises.
    """
    try:
        return _classify(error_info, step_type, project_context, kb_matched, llm_client)
    except Exception as exc:
        _logger.debug("[ErrorRouter] classify_error failed: %s — defaulting to web+code", exc)
        return ErrorRoute(source_type="web+code", skip_web=False,
                          confidence="regex", reason="exception fallback")


# ---------------------------------------------------------------------------
# Internal logic
# ---------------------------------------------------------------------------

def _classify(
    error_info: str,
    step_type: str,
    project_context,
    kb_matched: bool,
    llm_client,
) -> ErrorRoute:
    text = (error_info or "").strip()

    # ── 1. KB already has the answer ────────────────────────────────────────
    if kb_matched:
        _logger.info("[ErrorRouter] KB matched — skipping web search")
        return ErrorRoute(source_type="kb", skip_web=True,
                          reason="KB error_fixes matched")

    # ── 2. Extract Python/JS exception class from first matching line ────────
    exc_class = _extract_exception_class(text)
    if exc_class and exc_class in _CODE_ONLY_EXCEPTIONS:
        _logger.info("[ErrorRouter] %s → code-only (no web search)", exc_class)
        return ErrorRoute(source_type="code", skip_web=True,
                          reason=f"{exc_class} is a code logic error")

    # ── 3. Module/package not found — check installed_packages ──────────────
    pkg_match = _PKG_NAME_RE.search(text)
    if pkg_match:
        pkg_name = _clean_pkg_name(pkg_match.group(1))
        installed = _get_installed(project_context)
        if pkg_name and _is_installed(pkg_name, installed):
            # Package exists but import path is wrong → code fix
            _logger.info("[ErrorRouter] %r in installed_packages → code (wrong import)", pkg_name)
            return ErrorRoute(
                source_type="code", skip_web=True,
                reason=f"'{pkg_name}' is installed — likely wrong import path",
            )
        # Check if package is a local project module (directory or .py file)
        # before triggering web search.  Local project modules like
        # "snake_game" are not pip-installable — searching the web for
        # them wastes tokens.
        import os as _os_er
        _is_local = (
            _os_er.path.isdir(pkg_name)
            or _os_er.path.isfile(f"{pkg_name}.py")
            or _os_er.path.isfile(f"{pkg_name}/__init__.py")
        )
        if _is_local:
            _logger.info(
                "[ErrorRouter] %r is a local project module → code (skip web)", pkg_name)
            return ErrorRoute(
                source_type="code", skip_web=True,
                reason=f"'{pkg_name}' is a local project directory/module",
            )
        if pkg_name:
            # Genuinely missing external package → web search
            query = _build_package_query(pkg_name, project_context)
            _logger.info("[ErrorRouter] %r not installed → web (query: %s)", pkg_name, query)
            return ErrorRoute(
                source_type="web", skip_web=False,
                query_hint=query,
                reason=f"'{pkg_name}' not found in installed packages",
            )

    # ── 4. Known web patterns (CLI errors, version issues) ──────────────────
    for pat in _WEB_PATTERNS:
        if pat.search(text):
            _logger.info("[ErrorRouter] Web pattern matched: %s", pat.pattern[:40])
            return ErrorRoute(source_type="web", skip_web=False,
                              reason=f"matched web pattern: {pat.pattern[:40]}")

    # ── 5. Known local/env patterns → code only ──────────────────────────────
    for pat in _CODE_PATTERNS:
        if pat.search(text):
            _logger.info("[ErrorRouter] Code pattern matched: %s", pat.pattern[:40])
            return ErrorRoute(source_type="code", skip_web=True,
                              reason=f"matched code pattern: {pat.pattern[:40]}")

    # ── 6. LLM fallback for ambiguous errors ─────────────────────────────────
    if llm_client is not None:
        return _llm_classify(text, step_type, project_context, llm_client)

    # ── 7. Safe default: try web (don't suppress search) ────────────────────
    _logger.info("[ErrorRouter] Ambiguous — defaulting to web+code")
    return ErrorRoute(source_type="web+code", skip_web=False,
                      confidence="regex", reason="ambiguous, no LLM client")


def _llm_classify(text: str, step_type: str, project_context, llm_client) -> ErrorRoute:
    """Single-shot LLM call (~150 tokens in / ~30 out) for ambiguous errors."""
    installed = _get_installed(project_context)
    installed_str = ", ".join(installed[:15]) if installed else "unknown"
    language = getattr(project_context, "language", "") if project_context else ""

    # Keep error snippet short — we only need the key line
    error_snippet = _first_meaningful_lines(text, max_lines=5, max_chars=300)

    prompt = (
        "You are a triage assistant. Given the error below, decide if the fix "
        "requires searching the web for docs/packages, or only analysing the "
        "existing code.\n\n"
        f"Language: {language or 'unknown'}\n"
        f"Step type: {step_type}\n"
        f"Installed packages: {installed_str}\n"
        f"Error:\n{error_snippet}\n\n"
        "Reply with EXACTLY one of: WEB, CODE, WEB+CODE\n"
        "Optionally append on the same line: | query: <concise search query>\n"
        "Example: WEB | query: animejs v4 import syntax esm\n"
        "No other text."
    )

    try:
        response = llm_client.generate_response(prompt)
        return _parse_llm_route(response.strip())
    except Exception as exc:
        _logger.warning("[ErrorRouter] LLM classify failed: %s", exc)
        return ErrorRoute(source_type="web+code", skip_web=False,
                          confidence="llm", reason="LLM classify exception")


def _parse_llm_route(response: str) -> ErrorRoute:
    """Parse 'WEB | query: ...' style LLM response."""
    upper = response.upper()
    query_hint = ""
    query_m = re.search(r'\|\s*query:\s*(.+)', response, re.IGNORECASE)
    if query_m:
        query_hint = query_m.group(1).strip()[:150]

    if upper.startswith("CODE"):
        return ErrorRoute(source_type="code", skip_web=True,
                          query_hint=query_hint, confidence="llm",
                          reason="LLM classified as code")
    elif upper.startswith("WEB+CODE") or upper.startswith("WEB+") or upper.startswith("BOTH"):
        return ErrorRoute(source_type="web+code", skip_web=False,
                          query_hint=query_hint, confidence="llm",
                          reason="LLM classified as web+code")
    elif upper.startswith("WEB"):
        return ErrorRoute(source_type="web", skip_web=False,
                          query_hint=query_hint, confidence="llm",
                          reason="LLM classified as web")
    else:
        # Unparseable — default safe
        return ErrorRoute(source_type="web+code", skip_web=False,
                          confidence="llm", reason="LLM response unparseable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_exception_class(text: str) -> str:
    """Extract the Python/JS exception class name from a traceback."""
    # Python: "SomeError: message" at the end of a traceback
    for line in reversed(text.splitlines()):
        line = line.strip()
        m = re.match(r'^([A-Z][A-Za-z]+(?:Error|Exception|Warning)):', line)
        if m:
            return m.group(1)
    # JS: "TypeError: ..." at the start
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r'^(TypeError|ReferenceError|SyntaxError|RangeError|'
                     r'URIError|EvalError|Error):', line)
        if m:
            return m.group(1)
    return ""


def _clean_pkg_name(raw: str) -> str:
    """Strip submodule paths: 'foo.bar.baz' → 'foo', '@scope/pkg' kept as-is."""
    raw = raw.strip(" '\"")
    if raw.startswith("@"):
        # Scoped npm package — keep scope+name
        return raw
    # Python: take the top-level name only
    return raw.split(".")[0].split("/")[0]


def _get_installed(project_context) -> list[str]:
    """Safely extract installed_packages list from ProjectContext."""
    if project_context is None:
        return []
    installed = getattr(project_context, "installed_packages", None)
    if not isinstance(installed, list):
        return []
    return [str(p).lower() for p in installed]


def _is_installed(pkg_name: str, installed: list[str]) -> bool:
    """Case-insensitive check — also matches 'pkg-name' vs 'pkg_name'."""
    name_lower = pkg_name.lower()
    name_norm = name_lower.replace("-", "_")
    for p in installed:
        p_norm = p.replace("-", "_")
        if p == name_lower or p_norm == name_norm:
            return True
    return False


def _build_package_query(pkg_name: str, project_context) -> str:
    """Build a targeted search query for a missing package."""
    language = getattr(project_context, "language", "") if project_context else ""
    framework = getattr(project_context, "framework", "") if project_context else ""
    lang_suffix = {
        "python": "python pip install",
        "javascript": "npm install javascript",
        "typescript": "npm install typescript",
        "go": "golang go get",
        "ruby": "ruby gem install",
    }.get((language or "").lower(), "")
    parts = [p for p in [pkg_name, lang_suffix, framework] if p]
    query = " ".join(parts)
    if len(query) > 150:
        query = query[:150].rsplit(" ", 1)[0]
    return query


def _first_meaningful_lines(text: str, max_lines: int = 5, max_chars: int = 300) -> str:
    """Return the first N non-empty lines up to max_chars."""
    lines = [l for l in text.splitlines() if l.strip()]
    snippet = "\n".join(lines[:max_lines])
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars]
    return snippet

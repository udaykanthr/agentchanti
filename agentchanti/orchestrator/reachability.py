"""Is a step's target actually part of the product?

A gate can be runnable, assert real values, and still certify nothing —
because the file it asserts about is never loaded. Editing an unimported
stylesheet changes the repository and not the application, and every
check downstream agrees it went fine: the tests render components rather
than styles, the build does not error on a file it never bundles, and the
smoke test only proves the build succeeded.

WHY THIS EXISTS
---------------
Observed on a Vite/React project. `src/main.jsx` carried the app's only
stylesheet import::

    import './index.css'

while `src/App.jsx` imported no CSS at all. `src/App.css` — a leftover
from the original scaffold — was therefore never bundled. Successive runs
asked to restyle the header, the planner targeted `App.css` every time,
and the model dutifully wrote twelve `.site-header` rules including a
full dark palette. The built bundle contained **one** `.site-header`
occurrence; the source file contained twelve. Nothing in the browser ever
changed, across many runs, and no gate could see it — the last of them
asserted seven separate strings about `App.css` and passed on all seven.

SCOPE
-----
Stylesheets only, and only when reachability can be established with
confidence. A CSS file that no entry point can reach is inert by
construction — that is a fact about the module graph, not a heuristic.
The equivalent claim about a JS module is far weaker (dynamic import,
route-level lazy loading, `import.meta.glob`, re-export barrels), so it
is deliberately not made here.

Silence is the safe answer everywhere: no recognisable entry point, an
unreadable file, anything ambiguous — the step is simply not judged. A
false accusation here sends the planner to rewrite a plan that was right.
"""

from __future__ import annotations

import logging
import posixpath
import re
from typing import Callable, Iterable, Optional

_logger = logging.getLogger(__name__)

STYLESHEET_SUFFIXES = (".css", ".scss", ".sass", ".less", ".styl")

# Conventional entry modules, relative to a project root.
_ENTRY_CANDIDATES = (
    "src/main.jsx", "src/main.tsx", "src/main.js", "src/main.ts",
    "src/index.jsx", "src/index.tsx", "src/index.js", "src/index.ts",
    "index.jsx", "index.tsx", "index.js", "index.ts",
)
_ROOT_MARKERS = ("package.json", "index.html")

# `import './x.css'`, `import x from "./y"`, `require('./z')`
_IMPORT_RE = re.compile(
    r"""(?:^|\s)import\s+(?:[^'"]*?\sfrom\s+)?['"]([^'"]+)['"]"""
    r"""|require\(\s*['"]([^'"]+)['"]\s*\)""",
    re.MULTILINE,
)
# `@import "x.css"` / `@import url("x.css")` inside a stylesheet.
_CSS_IMPORT_RE = re.compile(
    r"""@import\s+(?:url\(\s*)?['"]([^'"]+)['"]""", re.IGNORECASE)
# `<link rel="stylesheet" href="...">` and `<script src="...">`
_HTML_REF_RE = re.compile(
    r"""<(?:link[^>]+href|script[^>]+src)\s*=\s*['"]([^'"]+)['"]""",
    re.IGNORECASE)

_RESOLVE_SUFFIXES = ("", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
                     "/index.js", "/index.jsx", "/index.ts", "/index.tsx")
_MAX_VISITS = 400          # a bound, not a limit anyone should hit


Reader = Callable[[str], Optional[str]]


def _norm(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/")).lstrip("./")


def _project_root_of(target: str, read: Reader) -> Optional[str]:
    """Walk up from *target* to the directory that owns the app."""
    parts = _norm(target).split("/")
    for depth in range(len(parts) - 1, -1, -1):
        base = "/".join(parts[:depth])
        for marker in _ROOT_MARKERS:
            if read(posixpath.join(base, marker) if base else marker):
                return base
    return None


def _resolve(spec: str, from_file: str, read: Reader) -> Optional[str]:
    """Resolve a relative specifier against the importing file."""
    if not spec.startswith("."):
        return None                      # package import, not a repo file
    base = posixpath.dirname(_norm(from_file))
    joined = posixpath.normpath(posixpath.join(base, spec))
    for suffix in _RESOLVE_SUFFIXES:
        candidate = joined + suffix
        if read(candidate) is not None:
            return _norm(candidate)
    return None


def reachable_files(root: str, read: Reader) -> Optional[set[str]]:
    """Every file reachable from *root*'s entry points, or None.

    None means "could not establish", which callers must treat as "do not
    judge" rather than "nothing is reachable".
    """
    entries: list[str] = []
    html = posixpath.join(root, "index.html") if root else "index.html"
    html_src = read(html)
    if html_src:
        for ref in _HTML_REF_RE.findall(html_src):
            resolved = _resolve(
                ref if ref.startswith(".") else "./" + ref.lstrip("/"),
                html, read)
            if resolved:
                entries.append(resolved)
    for rel in _ENTRY_CANDIDATES:
        path = posixpath.join(root, rel) if root else rel
        if read(path) is not None:
            entries.append(_norm(path))

    if not entries:
        return None                      # unrecognisable layout — refuse

    seen: set[str] = set()
    queue = list(dict.fromkeys(entries))
    while queue and len(seen) < _MAX_VISITS:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        source = read(current)
        if not source:
            continue
        pattern = (_CSS_IMPORT_RE if current.endswith(STYLESHEET_SUFFIXES)
                   else _IMPORT_RE)
        for match in pattern.finditer(source):
            spec = next((g for g in match.groups() if g), None)
            if not spec:
                continue
            resolved = _resolve(spec, current, read)
            if resolved and resolved not in seen:
                queue.append(resolved)
    return seen


def unreachable_stylesheet_reason(step, all_steps: Iterable,
                                  read: Reader) -> Optional[str]:
    """Explain why *step*'s stylesheet target cannot affect the app.

    Returns None whenever the answer is not certain — including when the
    plan itself intends to wire the file up, which is the ordinary shape
    of "create a component, then import it".
    """
    targets = [t for t in (getattr(step, "target_files", None) or [])
               if t.lower().endswith(STYLESHEET_SUFFIXES)]
    if not targets:
        return None

    # The plan may be about to import it — that is not an orphan, it is
    # work in progress ("create the component, then wire it up").
    #
    # The two declarations point OPPOSITE ways and must not be pooled:
    # `imports_from` names the file a step consumes, so its keys are
    # candidate targets; `imported_by` names the consumers of the step's
    # OWN target, so a non-empty list wires this step's files rather than
    # naming them. Treating both as "paths someone mentioned" let a step
    # declaring `imported_by: main.jsx` be reported as an orphan.
    if getattr(step, "imported_by", None):
        return None
    declared: set[str] = set()
    for other in all_steps or ():
        for spec in (getattr(other, "imports_from", None) or {}):
            declared.add(_norm(spec))

    orphans = []
    for target in targets:
        norm = _norm(target)
        if norm in declared:
            continue
        root = _project_root_of(target, read)
        if root is None:
            continue                      # no recognisable project — refuse
        reachable = reachable_files(root, read)
        if reachable is None:
            continue                      # no entry point — refuse
        if norm not in reachable:
            orphans.append(target)

    if not orphans:
        return None
    return (f"{', '.join(orphans)} is not reachable from this project's "
            f"entry point — no file the app loads imports it, so editing "
            f"it changes the repository and not the application. The tests, "
            f"the build and the smoke test will all still pass. Either "
            f"target the stylesheet that IS loaded, or add a step that "
            f"imports this one")

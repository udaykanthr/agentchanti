"""Dependency graph of a plan's *declared* components.

Why this exists
---------------
``fix_import_dependencies`` used to resolve a step's ``imports:`` against
other steps' ``target:`` by comparing strings. That is brittle in exactly
the way free-form planner output always is — the same dependency arrived
spelled at least six ways across consecutive runs::

    src/map.py     path                          matched
    src\\map.py     Windows separators            missed until normalised
    src.map        dotted module                 missed until converted
    src.map.py     dotted path + extension       missed until stripped
    map.py         bare filename                 missed until basename
    src.map.Map    module path plus symbol       still missed

Every unrecognised spelling silently drops a dependency edge, which puts
producer and consumer in the same wave. Observed twice: a player step
overwrote the map step's target mid-execution, and later a ghost step
clobbered two sibling steps' files in a three-way race.

The fix is to stop matching strings and resolve against a graph of what
the plan *declares*, keyed on several identities at once — most usefully
the exported **symbol**, which is notation-independent. ``src.map.Map``,
``src/map.py`` and ``map.py`` are hopeless to unify as text but all carry
``Map``, and exactly one step exports it.

Planned vs. real
----------------
Nodes here are *intent*, not fact: at plan time the project is usually
blank, so a graph built from files on disk would be empty and could not
order anything. This graph is deliberately kept separate from the KB code
graph (``kb/local/graph.py``), which is ground truth for search and
embeddings — mixing speculative nodes into it would return files that may
never exist. Instead each node carries a status (``planned`` →
``building`` → ``built``) and :meth:`PlanGraph.reconcile` compares the
declared exports against what a step actually produced.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Source extensions an import spec may or may not carry.
SOURCE_EXTS = (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
               ".go", ".rs", ".java", ".rb", ".php", ".vue", ".svelte")

_STATUS_PLANNED = "planned"
_STATUS_BUILDING = "building"
_STATUS_BUILT = "built"

# Directories that group source inside a project rather than containing a
# project. Everything living under `src/` does not make `src` a subproject.
_NOT_PROJECT_ROOTS = frozenset({
    "src", "lib", "app", "apps", "test", "tests", "source", "sources",
    "pkg", "internal", "cmd", "api", "components", "static", "public",
    "assets", "docs", "doc", "scripts", "config", "core", "common",
})


def normalize_path(path: str) -> str:
    """Collapse separators and strip a leading ``./``."""
    return re.sub(r"[\\/]+", "/", (path or "").strip()).lstrip("./")


def strip_source_ext(path: str) -> str:
    low = path.lower()
    for ext in SOURCE_EXTS:
        if low.endswith(ext):
            return path[: -len(ext)]
    return path


def module_key(spec: str) -> str:
    """Canonical identity for a file path or module spec.

    ``src/map.py``, ``src\\map.py``, ``src.map`` and ``src.map.py`` all
    reduce to ``src/map``. A spec that already contains a separator keeps
    its dots (``src/my.utils.py`` is a filename, not a package path).
    """
    stem = strip_source_ext(normalize_path(spec))
    if "/" not in stem:
        stem = stem.replace(".", "/")
    return stem


_DEFAULT_PREFIXED = re.compile(r"^default\s+(\S+)$", re.IGNORECASE)
_AS_DEFAULT = re.compile(r"^(\S+)\s+as\s+default$", re.IGNORECASE)
# Identifier-shaped words inside a prose export description.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")

# Ways a planner spells "this step exports nothing". The prompt says to omit
# the line entirely, but models answer the question instead of skipping it.
_NO_DECLARATION = frozenset({
    "none", "n/a", "na", "nil", "null", "nothing", "noexports", "tbd",
})


def _declares_nothing(spec: str) -> bool:
    """Is *spec* a way of saying "no exports" rather than a symbol name?

    Observed as `exports: (none)` — both export warnings in an otherwise
    clean run were that, hunting for a symbol literally called "(none)".
    Punctuation and case are stripped so `(none)`, `None` and `-` all land
    together; an empty result (a bare dash, `()`, `*`) counts too, since
    there is no name in it to look for.

    Checked only AFTER the exact-match test, so a file that really does
    export something called `none` still matches it first.
    """
    cleaned = re.sub(r"[^\w/]+", "", spec or "").lower()
    return not cleaned or cleaned in _NO_DECLARATION


def _canonical_name(name: str) -> str:
    """Collapse an identifier to a convention-free key.

    ``runHeadless``, ``run_headless``, ``RunHeadless`` and ``run-headless``
    all land on ``runheadless``. Only case and separators are removed, so
    two genuinely different names never collide.
    """
    return re.sub(r"[^0-9a-z]+", "", (name or "").lower())


def _export_satisfied(spec: str, actual: set[str]) -> bool:
    """Does *actual* provide the declared export *spec*?

    The two sides speak different vocabularies, and the mismatch was pure
    noise: planners write ``exports: Footer, default Footer`` while the JS
    extractor reports ``['Footer', 'default']``, so every run warned that
    ``default Footer`` was missing from a file exporting exactly that.
    Six-plus consecutive runs, never once correct — and a warning that is
    always wrong is worse than none, because it trains the reader to skip
    the line that will one day be right.
    """
    spec = (spec or "").strip()
    if not spec or spec in actual:
        return True

    # "no exports", spelled any of the ways a planner spells it. Deliberately
    # after the exact match above, so a genuine symbol named `none` wins.
    if _declares_nothing(spec):
        return True

    # "default Foo" / "Foo as default" — a default export that also has a
    # name. Either spelling in *actual* satisfies it.
    m = _DEFAULT_PREFIXED.match(spec) or _AS_DEFAULT.match(spec)
    if m:
        return "default" in actual or m.group(1) in actual

    # A bare name against a file whose ONLY export is the default. That
    # shape is `function Foo() {}; export default Foo` — the default IS
    # Foo, but the extractor flattens it to "default" and loses the name.
    # Deliberately narrow: when the file exports other names and simply
    # not this one, the warning still stands.
    if actual == {"default"}:
        return True

    # Same name, different naming convention. Every `exports:` example in
    # the planner prompt is JavaScript (`app, startServer`), so a planner
    # naming a *Python* file's symbols answers in camelCase — `exports:
    # runHeadless` against a file that defines `run_headless`. Measured on
    # a Pac-Man run: two violated-exports findings against an artifact that
    # passed all nine behavioural probes, which is the shape this function
    # exists to stop, since findings that are always wrong bury the one
    # that is right.
    #
    # Case and separators are the only difference forgiven. A planner that
    # invented a different name — `runHeadlessSimulation` where the file
    # defines `run_headless` — still falls through to the warning, because
    # that is a real disagreement and not a spelling of the same word.
    canonical = {_canonical_name(a) for a in actual}
    canonical.discard("")
    if _canonical_name(spec) in canonical:
        return True

    # The prompt asks for `exports: <Symbol1>, <Symbol2>`, but a weaker
    # planner answers in English:
    #
    #     exports: main function that prints "Hello World" when executed
    #
    # The file DID export `main`. Comparing the whole sentence to the
    # symbol set finds nothing, so a correct step was warned about — twice
    # in one clean run, both wrong. Look for any identifier in the prose
    # that the file really exports.
    words = _IDENTIFIER_RE.findall(spec)
    if any(w in actual or _canonical_name(w) in canonical for w in words):
        return True

    # Still prose, and nothing in it matched. A sentence is not a symbol
    # reference, so there is no claim here that can be checked — and per
    # this function's own history, a warning that cannot be right is worse
    # than no warning. A bare name that is genuinely absent has no
    # whitespace and still falls through to the warning below.
    if len(words) > 1 or " " in spec:
        return True

    return False


@dataclass
class PlanNode:
    """One file the plan promises to produce."""

    step_id: str
    path: str                                   # normalised target path
    key: str                                    # canonical module identity
    exports: list[str] = field(default_factory=list)
    status: str = _STATUS_PLANNED
    actual_exports: list[str] = field(default_factory=list)

    @property
    def basename(self) -> str:
        return os.path.basename(self.key)


# Directories that hold INSTALLED code rather than planned output. A path
# under one of these names a dependency the project consumes, never a
# module the plan is building, so it must not make an import look
# unresolvable. Kept deliberately short: a leading `env/` is a plausible
# real source directory, so only the unambiguous virtualenv names and the
# package roots themselves are listed.
_DEPENDENCY_DIRS = frozenset({
    "site-packages", "dist-packages", "node_modules", "vendor",
    "__pypackages__", "bower_components",
})
_ENV_ROOTS = frozenset({"venv", ".venv", "virtualenv", ".virtualenv"})


def _is_dependency_path(key: str) -> bool:
    """True when *key* lives inside an installed-dependency tree."""
    parts = [p for p in key.replace("\\", "/").split("/") if p]
    if not parts:
        return False
    if any(part in _DEPENDENCY_DIRS for part in parts):
        return True
    return parts[0].lower() in _ENV_ROOTS


class PlanGraph:
    """Resolution index over a plan's declared targets and exports."""

    def __init__(self, steps: Iterable) -> None:
        self.nodes: list[PlanNode] = []
        self._by_path: dict[str, PlanNode] = {}
        self._by_key: dict[str, PlanNode] = {}
        self._by_symbol: dict[str, PlanNode] = {}
        self._by_basename: dict[str, PlanNode] = {}
        self._symbol_counts: Counter = Counter()
        self._basename_counts: Counter = Counter()

        for step in steps:
            for target in getattr(step, "target_files", None) or []:
                path = normalize_path(target)
                if not path or path.lower() == "none":
                    continue
                node = PlanNode(step_id=step.id, path=path,
                                key=module_key(path),
                                exports=list(getattr(step, "exports", None)
                                             or []))
                self.nodes.append(node)
                # Last writer wins for exact keys; ambiguity is handled by
                # the *_counts guards below, which is what actually gates
                # the fuzzy lookups.
                self._by_path[path] = node
                self._by_key[node.key] = node
                self._by_basename[node.basename] = node
                self._basename_counts[node.basename] += 1
                for sym in node.exports:
                    sym = sym.strip()
                    if not sym or sym.lower() == "none":
                        continue
                    self._by_symbol[sym] = node
                    self._symbol_counts[sym] += 1

    # ── resolution ────────────────────────────────────────────────────

    def resolve(self, spec: str,
                symbols: Optional[Iterable[str]] = None) -> Optional[PlanNode]:
        """Find the node a step's ``imports:`` entry refers to.

        Tried in descending order of confidence:

        1. exact normalised path
        2. canonical module key (handles every path/dotted spelling)
        3. a uniquely-exported symbol from *symbols* — notation-independent,
           and the only thing that rescues ``src.map.Map``
        4. unambiguous bare filename, only when the spec names no directory
           of its own (an explicit ``src/public/index.js`` must never bind
           to ``src/admin/index.js``)
        """
        if not spec:
            return None

        path = normalize_path(spec)
        node = self._by_path.get(path)
        if node is not None:
            return node

        key = module_key(spec)
        node = self._by_key.get(key)
        if node is not None:
            return node

        for sym in symbols or ():
            sym = (sym or "").strip()
            if sym and self._symbol_counts.get(sym, 0) == 1:
                return self._by_symbol[sym]

        # A trailing symbol glued onto the module path: `src.map.Map`.
        # Only when the last segment is not itself a known module.
        if "/" not in normalize_path(spec):
            head, _, tail = key.rpartition("/")
            if head and self._symbol_counts.get(tail, 0) == 1:
                return self._by_symbol[tail]
            if head:
                node = self._by_key.get(head)
                if node is not None:
                    return node

        base = os.path.basename(key)
        if "/" not in key and self._basename_counts.get(base, 0) == 1:
            return self._by_basename.get(base)
        return None

    def producer_of(self, spec: str,
                    symbols: Optional[Iterable[str]] = None) -> Optional[str]:
        """Step id that produces *spec*, or None."""
        node = self.resolve(spec, symbols)
        return node.step_id if node is not None else None

    def has_module(self, key: str) -> bool:
        """True when *key* names a planned target relative to the repo root.

        Strict — no symbol or basename fallback. Used to judge whether a
        command run from the repo root can actually import that module.
        """
        return key in self._by_key

    def prefix_for(self, key: str) -> Optional[str]:
        """Directory that *key* would need as cwd to resolve, if any.

        Returns the leading path a planned target carries that *key* lacks.
        A plan targeting ``pacman_clone/src/config.py`` while its verify
        says ``from src.config import ...`` yields ``pacman_clone`` — the
        gate is unsatisfiable from the repo root, which is where it runs.

        Installed dependencies are not planned targets. A CMD step that
        declares ``produces: venv\\Lib\\site-packages\\pygame`` is naming
        where pip put a third-party package, but the key still ends with
        ``/pygame``, so every gate that imports pygame matched here and
        was reported unsatisfiable. Observed twice: it discarded an
        already-repaired gate and forced a full re-plan, once on the very
        first run of a benchmark session and again on the most expensive
        run of that session (253,706 tokens).
        """
        suffix = "/" + key
        for node_key in self._by_key:
            if node_key.endswith(suffix) and not _is_dependency_path(node_key):
                return node_key[: -len(suffix)]
        return None

    def common_root(self) -> Optional[str]:
        """Single directory every planned target sits under, if any.

        ``None`` when targets are spread across the repo root, or when the
        shared directory is a conventional source folder rather than a
        project container — all files living under ``src/`` does not make
        ``src`` a subproject.
        """
        firsts = set()
        for node in self.nodes:
            head, sep, _ = node.path.partition("/")
            if not sep:
                return None          # a target sits at the repo root
            firsts.add(head)
        if len(firsts) != 1:
            return None
        root = firsts.pop()
        if root.lower() in _NOT_PROJECT_ROOTS:
            return None
        return root

    # ── lifecycle ─────────────────────────────────────────────────────

    def mark_building(self, step_id: str) -> None:
        for node in self.nodes:
            if node.step_id == step_id and node.status == _STATUS_PLANNED:
                node.status = _STATUS_BUILDING

    def mark_built(self, step_id: str,
                   actual_exports: Optional[Iterable[str]] = None) -> None:
        actual = [s for s in (actual_exports or []) if s]
        for node in self.nodes:
            if node.step_id == step_id:
                node.status = _STATUS_BUILT
                if actual:
                    node.actual_exports = list(actual)

    def pending_paths(self) -> list[str]:
        """Targets the plan still owes — nothing has produced them yet."""
        return [n.path for n in self.nodes if n.status != _STATUS_BUILT]

    def reconcile(self, step_id: str) -> list[str]:
        """Declared exports a built step did not actually produce.

        Empty when the step kept its promises, when nothing was declared,
        or when no actual exports were recorded (absence of evidence is
        not evidence of a missing export).
        """
        missing: list[str] = []
        for node in self.nodes:
            if node.step_id != step_id or node.status != _STATUS_BUILT:
                continue
            if not node.exports or not node.actual_exports:
                continue
            actual = set(node.actual_exports)
            missing.extend(s for s in node.exports
                           if not _export_satisfied(s, actual))
        return missing

    # ── diagnostics ───────────────────────────────────────────────────

    def unresolved_imports(self, steps: Iterable) -> list[tuple[str, str]]:
        """``(step_id, spec)`` for imports no planned target satisfies.

        Not an error on its own — the import may be an existing project
        file or a third-party package. Useful for logging.
        """
        gaps: list[tuple[str, str]] = []
        for step in steps:
            for spec, syms in (getattr(step, "imports_from", None) or {}).items():
                if self.resolve(spec, syms) is None:
                    gaps.append((step.id, spec))
        return gaps

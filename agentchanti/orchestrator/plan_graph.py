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
            missing.extend(s for s in node.exports if s not in actual)
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

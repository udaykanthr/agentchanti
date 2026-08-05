"""Merge a partial module rewrite into the original, by symbol.

Models routinely answer "fix ``is_walkable``" with a ``#### [FILE]:``
marker and a body containing only the handful of definitions they
touched. Taken literally that deletes the rest of the module, so the
structural guard in :mod:`agentchanti.orchestrator.diagnosis` refuses it —
correctly, but the step then has no fix at all and the pipeline halts.
Observed on consecutive Pac-Man runs: "would delete 25/30 definitions",
twice, then halt, with a perfectly good replacement for those 5.

The replacement is not a file, it is a set of definitions. Merging by
NAME keeps everything the model did not mention and applies everything it
did — which is what it meant, and what a human reviewer would do by hand.

Refuses (returns ``None``) whenever it cannot prove the result is sound:
either side failing to parse, or a merged result that does not compile.
"""

from __future__ import annotations

import ast
import textwrap

from ..py_syntax import check_python_syntax

__all__ = ["merge_module_symbols", "merge_class_members",
           "replace_symbol_anywhere"]


def _defined_name(node: ast.stmt) -> str | None:
    """The single top-level name *node* binds, or None."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if isinstance(node, ast.Assign) and len(node.targets) == 1 \
            and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _span(node: ast.stmt) -> tuple[int, int]:
    """1-indexed inclusive line span of *node*, decorators included."""
    start = node.lineno
    for dec in getattr(node, "decorator_list", []) or []:
        start = min(start, dec.lineno)
    return start, (getattr(node, "end_lineno", None) or node.lineno)


def _index_body(body: list[ast.stmt]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for node in body:
        name = _defined_name(node)
        if name:
            out[name] = _span(node)
    return out


def _nodes(source: str) -> list[ast.stmt] | None:
    try:
        return ast.parse(source).body
    except (SyntaxError, ValueError):
        return None


def _index(source: str) -> dict[str, tuple[int, int]] | None:
    body = _nodes(source)
    return None if body is None else _index_body(body)


def _by_name(body: list[ast.stmt]) -> dict[str, ast.stmt]:
    out: dict[str, ast.stmt] = {}
    for node in body:
        name = _defined_name(node)
        if name:
            out[name] = node
    return out


def _import_lines(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines(True)
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            s, e = _span(node)
            out.append("".join(lines[s - 1:e]))
    return out


def _last_import_end(source: str) -> int:
    """1-indexed last line of the module's import block, or 0."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return 0
    end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = max(end, _span(node)[1])
    return end


def merge_class_members(original: str, class_name: str,
                        fragment: str) -> str | None:
    """Replace *class_name*'s members with the ones *fragment* redefines.

    A diagnosis fix for `map.py:Map` is usually a couple of methods at
    class-body indentation and nothing else — no `class` line. It is a
    REPLACEMENT of those methods, so the additive path refuses it (rightly:
    appending would leave two definitions with the last one winning), the
    chunk aligner has no textual anchor, and the fuzzy fallback then treats
    the indented fragment as a whole file, which cannot even parse
    ("unexpected indent (line 1)"). Observed three times across runs; the
    fix was correct every time and never landed.

    Names settle it, as everywhere else here: every member the fragment
    defines is swapped for its namesake, everything unmentioned is kept.
    Returns ``None`` unless the result is provably sound.
    """
    body = _nodes(original)
    if body is None or not class_name:
        return None
    target = next((n for n in body
                   if isinstance(n, ast.ClassDef) and n.name == class_name),
                  None)
    if target is None:
        return None

    # The fragment is a bare class body: dedent it and re-parse under a
    # synthetic class so member spans can be located.
    text = textwrap.dedent(fragment)
    if not text.strip():
        return None
    wrapped = "class _F:\n" + textwrap.indent(text, "    ")
    holder = _nodes(wrapped)
    if not holder or not isinstance(holder[0], ast.ClassDef):
        return None
    frag_lines = wrapped.splitlines(True)
    frag_members = _index_body(holder[0].body)
    orig_members = _index_body(target.body)
    if not frag_members:
        return None
    # Only a replacement of EXISTING members is unambiguous. A fragment
    # introducing new names belongs to the additive path, which knows
    # where to append them.
    if not set(frag_members) <= set(orig_members):
        return None

    orig_lines = original.splitlines(True)
    edits: list[tuple[int, int, str]] = []
    for name, (fs, fe) in frag_members.items():
        chunk = "".join(frag_lines[fs - 1:fe])
        if not chunk.endswith("\n"):
            chunk += "\n"
        os_, oe = orig_members[name]
        edits.append((os_ - 1, oe, chunk))

    merged = list(orig_lines)
    for start, end, chunk in sorted(edits, key=lambda t: -t[0]):
        merged[start:end] = [chunk]
    result = "".join(merged)
    if check_python_syntax(result, "<merged>"):
        return None
    return result


def merge_module_symbols(original: str, replacement: str) -> str | None:
    """Apply *replacement*'s top-level definitions onto *original*.

    Definitions present in both are replaced; definitions only in
    *replacement* are appended; everything else in *original* is kept.
    Imports introduced by *replacement* are added after the original's
    import block, since a rewritten definition frequently needs one.

    Returns the merged source, or ``None`` when the merge cannot be shown
    to be safe — the caller should then keep refusing.
    """
    orig_body = _nodes(original)
    repl_body = _nodes(replacement)
    if orig_body is None or repl_body is None:
        return None
    orig_idx, repl_idx = _index_body(orig_body), _index_body(repl_body)
    if not orig_idx or not repl_idx:
        return None
    # Nothing shared means this is not a partial rewrite of the same
    # module; merging unrelated content would be a guess, not a fix.
    if not (set(orig_idx) & set(repl_idx)):
        return None

    orig_lines = original.splitlines(True)
    repl_lines = replacement.splitlines(True)
    orig_nodes, repl_nodes = _by_name(orig_body), _by_name(repl_body)

    # Edits in ABSOLUTE original-file coordinates: (start_idx, end_idx,
    # text). An empty range (i, i) inserts. Applied bottom-up at the end.
    edits: list[tuple[int, int, str]] = []
    appended: list[str] = []

    def _text(rs: int, re_: int) -> str:
        t = "".join(repl_lines[rs - 1:re_])
        return t if t.endswith("\n") else t + "\n"

    for name, (rs, re_) in repl_idx.items():
        if name not in orig_idx:
            appended.append(_text(rs, re_))
            continue
        o_node, r_node = orig_nodes.get(name), repl_nodes.get(name)
        # Two classes of the same name: merge their MEMBERS. Replacing the
        # class wholesale is what loses the methods the model never
        # mentioned — the majority of the "deleted definitions" the
        # structural guard counts are methods, not top-level functions.
        if isinstance(o_node, ast.ClassDef) and isinstance(r_node,
                                                           ast.ClassDef):
            o_members = _index_body(o_node.body)
            r_members = _index_body(r_node.body)
            if o_members and set(o_members) & set(r_members):
                new_members: list[str] = []
                for m_name, (mrs, mre) in r_members.items():
                    if m_name in o_members:
                        mos, moe = o_members[m_name]
                        edits.append((mos - 1, moe, _text(mrs, mre)))
                    else:
                        new_members.append(_text(mrs, mre))
                if new_members:
                    at = _span(o_node)[1]      # end of the original class
                    edits.append((at, at, "\n" + "\n".join(new_members)))
                continue
        os_, oe = orig_idx[name]
        edits.append((os_ - 1, oe, _text(rs, re_)))

    merged = list(orig_lines)
    # Bottom-up so earlier spans keep their indices.
    for start, end, text in sorted(edits, key=lambda t: (-t[0], -t[1])):
        merged[start:end] = [text]

    if appended:
        if merged and merged[-1].strip():
            merged.append("\n")
        merged.append("\n\n".join(appended))

    # New imports the rewritten definitions may depend on.
    existing = {ln.strip() for ln in _import_lines(original)}
    new_imports = [ln for ln in _import_lines(replacement)
                   if ln.strip() and ln.strip() not in existing]
    if new_imports:
        at = _last_import_end(original)
        merged[at:at] = new_imports

    result = "".join(merged)
    if check_python_syntax(result, "<merged>"):
        return None
    return result


def replace_symbol_anywhere(original: str, fragment: str) -> str | None:
    """Replace the symbol *fragment* defines, wherever it lives in *original*.

    The chunk editor addresses a chunk as ``maze.py:_generate_grid`` and
    splices it back by line range. When the range no longer matches — the
    file moved on since the chunk was cut — it refuses, correctly: an
    ambiguous splice would corrupt the file. But refusing loses the fix
    AND burns one of three diagnosis attempts, so a single unplaceable
    edit can consume the whole retry budget. Observed on a Pac-Man run:
    attempts 1 and 3 both died as "Cannot place partial edit for
    maze.py:_generate_grid ... refusing to overwrite it", and the pipeline
    halted with the model's fix in hand.

    A named symbol does not need a line range. Find it by name — at module
    level or inside any class — and replace its span, re-indented to the
    definition it replaces. ``merge_class_members`` covers the sibling case
    where the fragment redefines several members of a known class; this
    covers one definition whose owner is not known up front, which is what
    a chunk id like ``file.py:method`` gives you.

    Returns ``None`` unless the result parses, so a guess is never applied.
    """
    body = _nodes(original)
    if body is None:
        return None

    dedented = textwrap.dedent(fragment)
    repl_body = _nodes(dedented)
    if not repl_body:
        return None
    # One definition only: with several, which owner to search for each is
    # a guess, and merge_class_members / merge_module_symbols handle that
    # case with more context than this has.
    named = [n for n in repl_body if _defined_name(n)]
    if len(named) != 1 or len(repl_body) != 1:
        return None
    name = _defined_name(named[0])

    # Candidate owners: module level first, then each class body. A method
    # name that also exists at module level is replaced at module level —
    # the shallower binding is the one an unqualified name refers to.
    candidates: list[tuple[int, int]] = []
    module_idx = _index_body(body)
    if name in module_idx:
        candidates.append(module_idx[name])
    else:
        for node in body:
            if isinstance(node, ast.ClassDef):
                inner = _index_body(node.body)
                if name in inner:
                    candidates.append(inner[name])
    # Exactly one home, or we cannot say which one was meant.
    if len(candidates) != 1:
        return None

    start, end = candidates[0]
    lines = original.splitlines(True)
    if start < 1 or end > len(lines):
        return None

    # Re-indent the fragment to the definition it is replacing.
    existing = lines[start - 1]
    indent = existing[:len(existing) - len(existing.lstrip())]
    new_text = textwrap.indent(dedented.rstrip("\n"), indent) + "\n"

    merged = "".join(lines[:start - 1]) + new_text + "".join(lines[end:])
    if check_python_syntax(merged, "<merged>"):
        return None
    return merged

"""Answer the wiring question from the code graph before paying an LLM.

`run_wiring_verification` makes one LLM call per run asking "are these
files wired together correctly". Measured 2026-09-26 over three
pre-scaffolded Next.js runs, it cost **6,719 prompt tokens per run -
20.7% of the whole run** - and said `No wiring issues found` every time.
Across every real run on the machine it fired 3 times in 10.

Three of the four things its own docstring says it checks are decidable
from data the pipeline already computes:

    broken imports              -> import edges vs the files that exist
    default-vs-named mismatch   -> FileDeps.has_default_export
    missing entry-point mounts  -> _detect_router_mount_missing

Only *wrong prop shapes* genuinely needs a model on untyped code, and on
TypeScript `tsc --noEmit` settles even that for nothing.

So the LLM becomes the escalation rather than the first resort: when the
graph is clean the call is skipped, and when it is not, the model is
handed NAMED findings instead of a pile of files. That second half
matters as much as the tokens - "here are 4 files, find problems" is the
prompt shape that produced a rewrite using `json.laods` and the
CommonJS-to-ESM rewrite that turned a green gate red. A model asked to
find problems in correct code will find some.

The refusal to judge is the load-bearing part. A graph that cannot see
the project is not a clean graph, and reporting one as the other is the
same mistake `empty_suite_reason` exists to prevent: an exit code that
cannot tell "nothing was wrong" from "nothing was checked". Whenever
this module cannot see enough, `can_judge` is False and the caller runs
the LLM exactly as before.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .dependency_check import _EXT_TO_LANG_FAMILY, extract_file_deps

# Extensions an unsuffixed JS/TS import may resolve to, in resolution order.
_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json")
_PY_EXTS = (".py",)


@dataclass
class WiringSuspect:
    """One thing the graph can say is wrong, in the LLM's own vocabulary."""
    kind: str            # broken-import | missing-default-export | router-mount
    file: str
    detail: str

    def __str__(self) -> str:                      # for the prompt block
        return f"- [{self.kind}] {self.file}: {self.detail}"


@dataclass
class WiringReport:
    suspects: list[WiringSuspect] = field(default_factory=list)
    files_analysed: int = 0
    edges_checked: int = 0
    # False => this module could not see enough to have an opinion, and
    # the caller must fall through to the LLM. Never conflate with clean.
    can_judge: bool = False
    unjudged: str = ""

    @property
    def clean(self) -> bool:
        return self.can_judge and not self.suspects

    def summary(self) -> str:
        base = (f"{self.files_analysed} file(s), {self.edges_checked} import "
                f"edge(s), {len(self.suspects)} suspect(s)")
        return f"{base}; {self.unjudged}" if self.unjudged else base


def _is_project_import(source: str) -> bool:
    """Only a path INTO this project can be a broken project import.

    `react`, `next/image` and `os` are the package manager's problem and
    are covered elsewhere (`_npm_package_of`, `_missing_third_party_module`).
    Judging them here would report every third-party import as broken.
    """
    return source.startswith((".", "/")) and not source.startswith("//")


def _py_relative_base(source: str, importer: str) -> str | None:
    """Resolve a Python relative import, which is NOT a filesystem path.

    `from .models import X` means the `models` module of THIS package, and
    each extra leading dot climbs one package. Treating the dots as `./`
    resolved it to `.models.py`, which exists nowhere — so every ordinary
    relative import in a Python project would have been reported broken,
    and the model sent to rewrite correct code. Exactly the false positive
    this whole gate exists to avoid.
    """
    dots = len(source) - len(source.lstrip("."))
    if not dots:
        return None
    rest = source[dots:]
    parts = os.path.dirname(importer).replace("\\", "/").split("/")
    parts = [p for p in parts if p]
    climb = dots - 1
    if climb:
        if climb > len(parts):
            return None
        parts = parts[:-climb]
    if rest:
        parts += rest.split(".")
    return "/".join(parts)


def _candidates(source: str, importer: str) -> list[str]:
    """Every path a project-relative import could legally resolve to."""
    if importer.endswith(".py"):
        base = _py_relative_base(source, importer)
        if base is None:
            return []
        exts = _PY_EXTS
    else:
        base = os.path.normpath(
            os.path.join(os.path.dirname(importer), source)).replace("\\", "/")
        exts = _JS_EXTS
    out = [base]
    for ext in exts:
        out.append(base + ext)
        out.append(f"{base}/index{ext}")
        out.append(f"{base}/__init__{ext}")
    return out


def _exists(path: str, known: set[str], project_root: str) -> bool:
    """On disk OR in memory.

    Memory alone is not enough: it holds what this run touched, so a
    perfectly good import of an untouched file would read as broken. A
    false "broken import" is worse than no finding at all - it would send
    the model to rewrite correct code.
    """
    if path in known:
        return True
    if project_root:
        try:
            if os.path.exists(os.path.join(project_root, path)):
                return True
        except (OSError, ValueError):
            return False
    return False


def wiring_suspects(
    memory_files: dict,
    *,
    language: str | None = None,
    project_root: str = "",
    router_mismatch: dict | None = None,
) -> WiringReport:
    """What the code graph can say about wiring, without an LLM call."""
    report = WiringReport()
    if not memory_files:
        report.unjudged = "no files resolved"
        return report

    deps, unreadable = {}, 0
    for path, content in memory_files.items():
        ext = os.path.splitext(path)[1].lower()
        if ext not in _EXT_TO_LANG_FAMILY:
            unreadable += 1
            continue
        if not isinstance(content, str):
            unreadable += 1
            continue
        deps[path] = extract_file_deps(path, content, language)

    if not deps:
        # Nothing the graph understands - a Go project, a config-only
        # change, a language without patterns. Not clean: unjudged.
        report.unjudged = (f"no file in a language this graph parses "
                           f"({unreadable} skipped)")
        return report

    report.files_analysed = len(deps)
    report.can_judge = True
    known = set(memory_files)

    for path, fd in deps.items():
        for source in fd.imports:
            if not _is_project_import(source):
                continue
            report.edges_checked += 1
            cands = _candidates(source, path)
            # No candidate path at all means this module could not be
            # resolved, which is not the same as proven missing. Only an
            # import we could have found and did not is a finding.
            if not cands:
                continue
            if any(_exists(c, known, project_root) for c in cands):
                continue
            report.suspects.append(WiringSuspect(
                "broken-import", path,
                f"imports '{source}', which resolves to no file in the "
                f"project"))

        # A default import from a module that exports no default is the
        # `missing_default_export` gap, decided from the target's own
        # source. Judged only when that source is in hand - guessing
        # would manufacture the finding.
        for source in getattr(fd, "default_imports", []) or []:
            if not _is_project_import(source):
                continue
            target = next((c for c in _candidates(source, path)
                           if c in deps), None)
            if target is None:
                continue
            if not deps[target].has_default_export:
                report.suspects.append(WiringSuspect(
                    "missing-default-export", path,
                    f"default-imports from '{source}' but {target} declares "
                    f"no default export"))

    if router_mismatch:
        report.suspects.append(WiringSuspect(
            "router-mount", str(router_mismatch.get("file", "entry point")),
            str(router_mismatch.get("description",
                                    "router primitives used with no Router "
                                    "mounted"))))

    if unreadable:
        report.unjudged = (f"{unreadable} file(s) not parsed by this graph "
                           f"(prop shapes are not checked here)")
    else:
        report.unjudged = "prop shapes are not checked here"
    return report


def findings_block(suspects: list[WiringSuspect]) -> str:
    """The named findings handed to the model, instead of a hunting brief."""
    if not suspects:
        return ""
    lines = "\n".join(str(s) for s in suspects)
    return (
        "\nThe code graph already identified these specific problems. Fix "
        "EXACTLY these and nothing else — do not look for other issues, and "
        "do not rewrite files that are not named here:\n" + lines + "\n")

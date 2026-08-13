"""Shadow reconciliation of a plan's *declared effects* against reality.

What this is
------------
After the planner produces a structured plan, every step already declares
what it will do to the project: ``target:`` names files, ``exports:``
names symbols, ``imports:`` names cross-file edges, ``verify:`` names an
acceptance command. Those declarations are checkable **postconditions**,
and checking them costs no LLM call — only a stat, a hash, and an AST
parse.

This module builds that set of postconditions up front ("the ghost"),
records the project's real pre-state, and afterwards compares what the
plan promised against what the filesystem actually shows. It reports
where the pipeline's own verdict and the evidence disagree.

Read-only by construction
-------------------------
*This module* never writes to the project, never runs a command, and
never changes a verdict — so it cannot slow a run down or fail one.
``GATE_PASSED`` is resolved by asking the
:class:`~.wave_snapshots.GateLedger` what already passed, never by
re-running anything. Every entry point swallows its own exceptions.

Repair lives next door, in :mod:`ghost_heal`, which acts on what is
found here: installing a declared dependency into the interpreter that
actually runs the app, creating an absent package marker, adding a
declared-but-missing import. It is bounded by one rule — heal *state*,
never fabricate *content* — because a healer that invents a CSS rule or
a function body to satisfy a check turns a detectable defect into an
undetectable one. Detection stays honest here regardless: every heal is
verified by re-resolving the expectation it targeted, so a repair that
did not work leaves the verdict red.

Deliberately three-valued
-------------------------
Verdicts are ``HOLDS`` / ``VIOLATED`` / ``UNKNOWN`` / ``INAPPLICABLE``,
never a boolean. The rest of this package learned that the hard way:
``GateLedger._sample_gate`` separates crash and harness errors from real
failures, and ``verify_dt_invariance`` reserves exit code 2 for
"could not verify". Collapsing "no evidence" into "failed" manufactures
regressions out of silence, so an unreadable file, an unknown language
or a missing extractor all resolve to ``UNKNOWN`` and are counted
nowhere.

What it can see that nothing else does
--------------------------------------
* **Planned but untouched** — a step reports success and its target
  file's bytes never changed.
* **Touched but unplanned** — a recovery or agent loop rewrote a file no
  step ever claimed.
* **No checkable claim** — a step whose expectations are all
  tautologies (only "the file we ourselves wrote from the plan exists"),
  i.e. a step that certifies nothing. This mirrors ``gate_integrity``'s
  rule for acceptance commands, applied one layer up to the plan.
* **Failed but clean** — the run was marked failed while every declared
  postcondition holds, which historically has meant a harness defect
  rather than a model failure.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterable, Optional

# `_export_satisfied` encodes hard-won knowledge about the many ways a
# planner spells an export ("default Footer", prose, "(none)"). Reusing it
# is the whole point — a second, naive comparison here would reproduce the
# false-warning history that function exists to end.
#
# `plan_graph.normalize_path` is deliberately NOT used: its `lstrip("./")`
# strips any leading dot, so `.agentchanti/log.txt` becomes
# `agentchanti/log.txt` and `.env` becomes `env`. That is harmless for a
# graph of planned modules, but this module stats real files and compares
# against FileMemory keys (which keep their dots), so it needs a
# normaliser that only collapses separators and a leading `./`.
from .plan_graph import _export_satisfied, module_key

_logger = logging.getLogger(__name__)

# ── Verdict lattice ──────────────────────────────────────────────────
UNKNOWN = "UNKNOWN"
HOLDS = "HOLDS"
VIOLATED = "VIOLATED"
INAPPLICABLE = "INAPPLICABLE"

# ── Expectation kinds ────────────────────────────────────────────────
KIND_EXISTS = "EXISTS"
KIND_TOUCHED = "TOUCHED"
KIND_PARSES = "PARSES"
KIND_EXPORTS = "EXPORTS"
KIND_IMPORT_EDGE = "IMPORT_EDGE"
KIND_PKG_PRESENT = "PKG_PRESENT"
KIND_PLAN_ANCHORS = "PLAN_ANCHORS"
KIND_GATE_PASSED = "GATE_PASSED"

# Manifests whose declared runtime dependencies can be checked against the
# environment the app will actually run in.
_MANIFESTS = ("requirements.txt", "package.json")

# How much each kind counts toward a step having asserted anything real.
# EXISTS is scored at build time instead (0 for a file the plan itself
# supplies the bytes for — the pipeline writing its own inline content and
# then observing that it landed proves nothing about the task).
_WEIGHTS = {
    KIND_TOUCHED: 1,
    KIND_PARSES: 1,
    KIND_EXPORTS: 2,
    KIND_IMPORT_EDGE: 3,
    KIND_PKG_PRESENT: 4,
    KIND_PLAN_ANCHORS: 4,
    KIND_GATE_PASSED: 5,
}

# A step whose resolved evidence weighs less than this asserted nothing
# that could have failed. One EXISTS on a command-produced file clears it;
# one EXISTS on a file we pasted from the plan does not.
MIN_STEP_STRENGTH = 1

_TEXT_READ_LIMIT = 2_000_000     # don't hash a stray binary/asset blob


def _norm(path: str) -> str:
    """Collapse separators and strip a leading ``./`` — dots survive."""
    p = re.sub(r"[\\/]+", "/", (path or "").strip())
    while p.startswith("./"):
        p = p[2:]
    return p


def _looks_like_path(target: str) -> bool:
    """Is *target* a file path, or prose the planner wrote instead?

    ``produces:`` is where planners put whatever they consider the step's
    output, and a weaker model answers it in English. Observed on a
    20B-model run: ``produces: pygame package`` became a "planned target"
    that could never exist on disk, so the shadow reported a missing file
    and then scored the step at zero evidence for good measure — two
    fabricated findings from one prose line.

    Whitespace is the giveaway. Real paths in generated projects do not
    contain spaces, while a prose answer almost always does; ``venv`` and
    ``tests`` stay valid because a bare directory name is a legitimate
    target.
    """
    t = (target or "").strip()
    if not t or t.lower() in ("none", "n/a", "na", "-"):
        return False
    return not any(ch.isspace() for ch in t)


def _actual_spelling(full_path: str) -> Optional[str]:
    """The name as the directory really spells it, or ``None``."""
    directory, name = os.path.split(full_path)
    try:
        entries = os.listdir(directory or ".")
    except OSError:
        return None
    if name in entries:
        return name
    lowered = name.lower()
    for entry in entries:
        if entry.lower() == lowered:
            return entry
    return None


def _near_miss(full_path: str) -> Optional[str]:
    """A sibling whose name differs from *full_path*'s only trivially.

    Answers "the plan said `board.py` and it is not there — is something
    almost-that there?" Case differences and a changed extension are the
    two spellings a generated project actually gets wrong, and on Windows
    the case variant still imports, so the mismatch stays invisible until
    the project is checked out somewhere case-sensitive.
    """
    directory, name = os.path.split(full_path)
    if not name:
        return None
    try:
        entries = os.listdir(directory or ".")
    except OSError:
        return None
    stem, _ = os.path.splitext(name)
    lowered = name.lower()
    for entry in entries:
        if entry == name:
            continue
        if entry.lower() == lowered:
            return entry
    for entry in entries:
        e_stem, e_ext = os.path.splitext(entry)
        if e_ext and e_stem.lower() == stem.lower():
            return entry
    return None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _read(root: str, rel: str) -> Optional[str]:
    """File contents, or ``None`` when it cannot be read as text."""
    try:
        full = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(full):
            return None
        if os.path.getsize(full) > _TEXT_READ_LIMIT:
            return None
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


# ── Data model ───────────────────────────────────────────────────────


@dataclass
class Expectation:
    """One checkable postcondition of the plan.

    The ``id`` is canonical and interned, so two steps declaring the same
    fact share a single node — which is what makes a cross-step
    contradiction (step 3 exports ``Board``, step 7 imports it) visible
    as one object rather than two independent opinions.
    """

    id: str
    kind: str
    subject: str                              # path, cmd, or "a.py->b.py"
    detail: str = ""                          # symbol name, etc.
    weight: int = 1
    claimed_by: list[str] = field(default_factory=list)   # producing steps
    required_by: list[str] = field(default_factory=list)  # consuming steps
    # IMPORT_EDGE only: every file the declaring step produces. `imports:`
    # is a step-level declaration, so any one of them satisfying it is
    # enough — see `_check_edge`.
    consumers: list[str] = field(default_factory=list)
    verdict: str = UNKNOWN
    evidence: str = ""
    # True once this postcondition has been observed broken, even if a
    # later pass found it repaired. Kept separate from `verdict`, which
    # always describes the CURRENT state — see `observe`.
    ever_violated: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "subject": self.subject,
            "detail": self.detail, "weight": self.weight,
            "claimed_by": list(self.claimed_by),
            "required_by": list(self.required_by),
            "verdict": self.verdict, "evidence": self.evidence[:400],
            "ever_violated": self.ever_violated,
        }


@dataclass
class GhostFile:
    """A file the plan intends to change, and its real pre-state."""

    path: str
    pre_hash: Optional[str]                   # None = did not exist
    writers: list[str] = field(default_factory=list)
    inline: bool = False                      # bytes came from the plan
    post_hash: Optional[str] = None

    @property
    def touched(self) -> bool:
        return self.post_hash != self.pre_hash


@dataclass(frozen=True)
class Observation:
    """One append-only journal entry. Verdicts are a fold over these."""

    exp_id: str
    verdict: str
    evidence: str
    stage: str


@dataclass(frozen=True)
class Disagreement:
    """A place where the evidence and the pipeline's verdict differ."""

    kind: str                                 # kebab-case slug
    step_id: str
    detail: str


# ── The ghost ────────────────────────────────────────────────────────


class GhostPlan:
    """Declared postconditions of a plan, reconciled against the tree."""

    def __init__(self, project_root: str = ".") -> None:
        self.root = os.path.abspath(project_root)
        self.expectations: dict[str, Expectation] = {}
        self.files: dict[str, GhostFile] = {}
        self.steps: dict[str, dict] = {}      # step id -> {produces, requires}
        self.journal: list[Observation] = []
        # path -> the spelling actually on disk, when it differs only by
        # case. Case-insensitive filesystems resolve the plan's spelling
        # happily, so this never fails EXISTS; it is reported separately.
        self.case_mismatches: dict[str, str] = {}
        # What the PLAN itself says each file should contain. This is not
        # a guess — the planner wrote it — so repairing a drifted file
        # from it invents nothing. Empty under intent mode, where the
        # plan deliberately supplies goals instead of bodies.
        self.plan_content: dict[str, str] = {}
        self.plan_edits: dict[str, list[tuple[str, str]]] = {}
        # Every command the plan names — verify gates and CMD bodies —
        # so the run's own test runner can be identified.
        self.declared_commands: list[str] = []
        self._lock = Lock()

    # -- construction --------------------------------------------------

    @classmethod
    def build(cls, steps: Iterable, project_root: str = ".") -> "GhostPlan":
        """Derive expectations from a finalized plan and snapshot pre-state.

        Must be called after every plan-repair pass (blind-edit routing,
        verify repair, reclassification) and before the first step runs —
        the pre-state hashes are only meaningful if nothing has executed.
        """
        ghost = cls(project_root)
        for step in steps or ():
            ghost._add_step(step)
        ghost._capture_pre_state()
        return ghost

    def _add_step(self, step) -> None:
        sid = getattr(step, "id", "?")
        for _cmd in (getattr(step, "verify_cmd", None),
                     getattr(step, "command", None)):
            if _cmd:
                self.declared_commands.append(_cmd)
        node = self.steps.setdefault(sid, {"produces": set(), "requires": set()})
        targets = [_norm(t) for t in
                   (getattr(step, "target_files", None) or []) if t]
        targets = [t for t in targets if _looks_like_path(t)]
        inline = {_norm(p) for p in
                  (getattr(step, "inline_code", None) or {})}
        inline |= {_norm(p) for p in
                   (getattr(step, "inline_edits", None) or {})}

        for path in targets:
            gf = self.files.setdefault(
                path, GhostFile(path=path, pre_hash=None))
            if sid not in gf.writers:
                gf.writers.append(sid)
            gf.inline = gf.inline or path in inline

            # A file whose bytes the plan supplies is one the pipeline
            # writes itself; observing that it then exists is circular.
            self._claim(node, Expectation(
                id=f"file:{path}#exists", kind=KIND_EXISTS, subject=path,
                weight=0 if path in inline else 1))
            self._claim(node, Expectation(
                id=f"file:{path}#touched", kind=KIND_TOUCHED, subject=path,
                weight=_WEIGHTS[KIND_TOUCHED]))
            if _parseable(path):
                self._claim(node, Expectation(
                    id=f"file:{path}#parses", kind=KIND_PARSES, subject=path,
                    weight=_WEIGHTS[KIND_PARSES]))
            # A manifest is a promise about the environment, not just a
            # file. The dependency list inside it does not exist yet at
            # plan time, so the node is created now and the list is read
            # when it is resolved.
            if os.path.basename(path).lower() in _MANIFESTS:
                self._claim(node, Expectation(
                    id=f"pkg:{path}#deps-installed", kind=KIND_PKG_PRESENT,
                    subject=path, weight=_WEIGHTS[KIND_PKG_PRESENT]))

        # Where the plan supplied the body itself, the written file must
        # still contain what that body declared. This is the step-drift
        # check: the planner got it right and the per-step model wrote
        # something else — common with smaller models, and invisible to
        # every other check because the file exists, parses, and may even
        # satisfy a weak gate.
        inline_code = getattr(step, "inline_code", None) or {}
        for raw_path, body in inline_code.items():
            path = _norm(raw_path)
            if not path or not (body or "").strip():
                continue
            self.plan_content[path] = body
            anchors = plan_anchors(path, body)
            if anchors:
                self._claim(node, Expectation(
                    id=f"plan:{path}#anchors", kind=KIND_PLAN_ANCHORS,
                    subject=path, detail=",".join(sorted(anchors)),
                    weight=_WEIGHTS[KIND_PLAN_ANCHORS]))
        for raw_path, pairs in (getattr(step, "inline_edits", None) or {}).items():
            self.plan_edits.setdefault(_norm(raw_path), []).extend(pairs)

        # Exports attach to the step's first target: that is the file the
        # plan format means them to describe.
        primary = targets[0] if targets else ""
        for sym in (getattr(step, "exports", None) or []):
            sym = (sym or "").strip()
            if not sym or not primary:
                continue
            self._claim(node, Expectation(
                id=f"file:{primary}#exports:{sym}", kind=KIND_EXPORTS,
                subject=primary, detail=sym,
                weight=_WEIGHTS[KIND_EXPORTS]))

        # One edge per (step, source) — NOT per target file. `imports:` is
        # declared once for the whole step, so fanning it out across every
        # target accuses the step's incidental files of failing to import
        # something they were never going to. Observed: a TEST step with
        # `target: tests/__init__.py, tests/test_game_invariants.py`
        # produced three "the import edge was never wired" findings
        # against the package marker, while the sibling test file next to
        # it imported all three symbols correctly.
        for src, syms in (getattr(step, "imports_from", None) or {}).items():
            src_n = _norm(src)
            if not src_n or not targets:
                continue
            exp = Expectation(
                id=f"edge:{src_n}->step:{sid}", kind=KIND_IMPORT_EDGE,
                subject=src_n,
                detail=",".join(s.strip() for s in (syms or []) if s),
                consumers=list(targets),
                weight=_WEIGHTS[KIND_IMPORT_EDGE])
            self._require(node, exp)

        verify = (getattr(step, "verify_cmd", None) or "").strip()
        if not verify:
            # A CMD step's command IS its acceptance criterion when the
            # command is a test suite. Observed: a step whose entire body
            # was `set SDL_VIDEODRIVER=dummy && python -m unittest -v`
            # declared no target and no verify, so it carried no
            # expectations at all and was reported as a step that
            # "asserted nothing that could have failed" — while running
            # the project's whole acceptance suite.
            command = (getattr(step, "command", None) or "").strip()
            if command:
                try:
                    from .wave_snapshots import is_suite_gate
                    if is_suite_gate(command):
                        verify = command
                except Exception:
                    pass
        if verify:
            self._claim(node, Expectation(
                id=f"gate:{sid}:{_digest(verify)}", kind=KIND_GATE_PASSED,
                subject=verify, weight=_WEIGHTS[KIND_GATE_PASSED]))

    def _intern(self, exp: Expectation) -> Expectation:
        existing = self.expectations.get(exp.id)
        if existing is not None:
            return existing
        self.expectations[exp.id] = exp
        return exp

    def _claim(self, node: dict, exp: Expectation) -> None:
        exp = self._intern(exp)
        node["produces"].add(exp.id)

    def _require(self, node: dict, exp: Expectation) -> None:
        exp = self._intern(exp)
        node["requires"].add(exp.id)

    def _capture_pre_state(self) -> None:
        """Hash every target file as it exists *before* the run."""
        for gf in self.files.values():
            content = _read(self.root, gf.path)
            gf.pre_hash = _digest(content) if content is not None else None

    # -- observation ---------------------------------------------------

    def observe(self, exp_id: str, verdict: str, evidence: str = "",
                stage: str = "") -> None:
        """Append one journal entry and fold it into the node's verdict.

        ``verdict`` always reflects the LATEST observation, because the
        question this module answers is what the run actually shipped.
        An earlier failure is preserved in ``ever_violated`` and in the
        journal rather than in the verdict.

        This was originally the other way round — ``VIOLATED`` was sticky
        — and it produced a confidently false report. Observed: a plan
        created a venv and installed into the wrong interpreter, so at
        wave 2 the declared dependency genuinely was absent and the
        shadow said so correctly; at wave 6 the agent loop's env
        self-heal reinstalled it into the project venv, and the run
        finished green with the package present. The stale ``VIOLATED``
        was still reported at the end, contradicting a run that was by
        then entirely correct. "Was broken once" and "is broken" are
        different claims, and only the second belongs in a verdict.
        """
        with self._lock:
            exp = self.expectations.get(exp_id)
            if exp is None:
                return
            self.journal.append(Observation(exp_id, verdict, evidence, stage))
            if verdict == VIOLATED:
                exp.ever_violated = True
            exp.verdict = verdict
            exp.evidence = evidence

    # -- resolution ----------------------------------------------------

    def resolve(self, step_ids: Iterable[str], *, language: str | None = None,
                gate_cmds: Iterable[str] = (), stage: str = "") -> None:
        """Check every expectation owned by *step_ids* against the tree.

        Safe to call repeatedly (per wave, then once at the end): each
        pass re-reads the files and appends fresh observations.
        """
        wanted: set[str] = set()
        for sid in step_ids:
            node = self.steps.get(sid)
            if node:
                wanted |= node["produces"] | node["requires"]
        gates = list(gate_cmds)
        cache: dict[str, Optional[str]] = {}

        def content(path: str) -> Optional[str]:
            if path not in cache:
                cache[path] = _read(self.root, path)
            return cache[path]

        for exp_id in sorted(wanted):
            exp = self.expectations.get(exp_id)
            if exp is None:
                continue
            try:
                verdict, evidence = self._check(exp, content, gates, language)
            except Exception as exc:            # never fail a run
                _logger.debug("[Ghost] check raised for %s: %s", exp_id, exc)
                continue
            self.observe(exp_id, verdict, evidence, stage)

    def _check(self, exp: Expectation, content, gates: list[str],
               language: str | None) -> tuple[str, str]:
        if exp.kind == KIND_GATE_PASSED:
            return _check_gate(exp.subject, gates)

        if exp.kind == KIND_IMPORT_EDGE:
            return _check_edge(exp.subject, exp.detail,
                               [(c, content(c)) for c in exp.consumers])

        if exp.kind == KIND_PKG_PRESENT:
            return _check_packages(self.root, exp.subject,
                                   content(exp.subject))

        if exp.kind == KIND_PLAN_ANCHORS:
            return _check_plan_anchors(exp.subject, exp.detail,
                                       content(exp.subject))

        path = exp.subject
        full = os.path.join(self.root, path.replace("/", os.sep))
        text = content(path)

        # A plan target is not always a file. CMD steps legitimately
        # declare `produces: venv` or `produces: src/assets`, and judging
        # a directory by `isfile` reports a target that is plainly there
        # as missing — which then drags the step's evidence weight to
        # zero and manufactures a second, equally false "asserted
        # nothing" finding on top of it.
        is_dir = os.path.isdir(full)

        if exp.kind == KIND_EXISTS:
            if text is not None or is_dir:
                # The file is usable here, so this is not a missing
                # target — but on a case-insensitive filesystem it can be
                # usable under a DIFFERENT spelling than the plan asked
                # for, and that difference is invisible until the project
                # is checked out somewhere case-sensitive. Record it as
                # its own finding rather than distorting this verdict.
                actual = _actual_spelling(full)
                if actual and actual != os.path.basename(path):
                    self.case_mismatches[path] = actual
                return HOLDS, "directory" if is_dir else ""
            if os.path.exists(full):
                return UNKNOWN, "exists but could not be read as text"
            near = _near_miss(full)
            if near:
                # A rename is deliberately NOT attempted: every other
                # file that references either spelling would have to move
                # with it, and on Windows the wrong case still imports
                # fine, so the mismatch only breaks on someone else's
                # machine. Naming the candidate makes it a one-line fix
                # for a human without risking a repo-wide edit here.
                return VIOLATED, (
                    f"planned target does not exist, but `{near}` does — "
                    f"filename mismatch (case or extension)")
            return VIOLATED, "planned target does not exist on disk"

        if is_dir:
            # Hashing or parsing a directory is not a question with an
            # answer; existence is the only claim it can settle.
            return INAPPLICABLE, "target is a directory"

        if text is None:
            return UNKNOWN, "file unreadable — no evidence either way"

        gf = self.files.get(path)
        if exp.kind == KIND_TOUCHED:
            if gf is None:
                return UNKNOWN, ""
            gf.post_hash = _digest(text)
            if gf.pre_hash is None:
                return HOLDS, "created"
            if gf.post_hash == gf.pre_hash:
                return VIOLATED, ("bytes identical to the pre-run state — "
                                  "the step changed nothing")
            return HOLDS, "modified"

        if exp.kind == KIND_PARSES:
            return _check_parses(path, text)

        if exp.kind == KIND_EXPORTS:
            return _check_exports(text, exp.detail, path, language)

        return UNKNOWN, ""

    # -- reporting -----------------------------------------------------

    def unplanned_writes(self, tracked: Iterable[str]) -> list[str]:
        """Files the run wrote that no step ever claimed as a target."""
        from .memory import _should_skip_for_context

        planned = set(self.files)
        out: list[str] = []
        for path in tracked or ():
            norm = _norm(path)
            if not norm or norm in planned:
                continue
            if _should_skip_for_context(norm):
                continue
            out.append(norm)
        return sorted(out)

    def step_strength(self, step_id: str) -> int:
        """Weight of the evidence a step actually produced.

        Only ``HOLDS`` counts. A step can be strong on paper and weak in
        fact — an unreadable file resolves ``UNKNOWN`` and proves nothing,
        which is exactly the situation this number exists to expose.
        """
        node = self.steps.get(step_id)
        if not node:
            return 0
        total = 0
        for exp_id in node["produces"] | node["requires"]:
            exp = self.expectations.get(exp_id)
            if exp is not None and exp.verdict == HOLDS:
                total += exp.weight
        return total

    def declared_strength(self, step_id: str) -> int:
        """Weight of what the step CLAIMED, whatever came of checking it.

        Distinct from :meth:`step_strength`, which counts only confirmed
        evidence. "Did this step assert anything that could have failed?"
        is a question about the declaration, not about whether we managed
        to confirm it.

        Observed: a CMD step declared
        ``verify: python -c "import pygame; assert pygame.version.verstr
        .startswith('2.6')"`` — a genuinely falsifiable gate — but the
        gate never entered the ledger, so it resolved UNKNOWN, banked no
        evidence, and the step was reported as asserting nothing that
        could have failed. The step's claim was real; only our record of
        it was missing.
        """
        node = self.steps.get(step_id)
        if not node:
            return 0
        total = 0
        for exp_id in node["produces"] | node["requires"]:
            exp = self.expectations.get(exp_id)
            if exp is not None and exp.verdict != INAPPLICABLE:
                total += exp.weight
        return total

    def tally(self) -> dict[str, int]:
        counts = {HOLDS: 0, VIOLATED: 0, UNKNOWN: 0, INAPPLICABLE: 0}
        for exp in self.expectations.values():
            counts[exp.verdict] = counts.get(exp.verdict, 0) + 1
        return counts

    def disagreements(self, done_step_ids: Iterable[str], *,
                      tracked_files: Iterable[str] = (),
                      pipeline_success: bool = True) -> list[Disagreement]:
        """Places the evidence contradicts the pipeline's own verdict."""
        done = list(done_step_ids)
        out: list[Disagreement] = []

        for sid in done:
            node = self.steps.get(sid)
            if not node:
                continue
            for exp_id in sorted(node["produces"] | node["requires"]):
                exp = self.expectations.get(exp_id)
                if exp is None or exp.verdict != VIOLATED:
                    continue
                out.append(Disagreement(
                    kind=f"violated-{exp.kind.lower().replace('_', '-')}",
                    step_id=sid,
                    detail=f"{exp.subject}"
                           + (f" [{exp.detail}]" if exp.detail else "")
                           + (f" — {exp.evidence}" if exp.evidence else "")))
            if self.declared_strength(sid) < MIN_STEP_STRENGTH:
                out.append(Disagreement(
                    kind="no-checkable-claim", step_id=sid,
                    detail=("step reported done but declared nothing that "
                            "could have failed — no target, no gate, and "
                            "any file it names is one the plan supplied "
                            "the contents of")))

        for path in self.unplanned_writes(tracked_files):
            out.append(Disagreement(
                kind="unplanned-write", step_id="-",
                detail=f"{path} was written but no step declared it"))

        # Test files the run's own acceptance command will never collect.
        # Deliberately spans planned targets AND untracked writes: the
        # four modules that exposed this were written by the agent loop
        # and declared by no step, so a per-step check would have missed
        # every one of them.
        _runner = declared_runner(self.declared_commands)
        _candidates = set(self.files) | {
            _norm(p) for p in (tracked_files or ())}
        for path, reason in uncollected_test_files(
                self.root, _candidates, _runner):
            out.append(Disagreement(
                kind="tests-never-collected", step_id="-",
                detail=f"{path}: {reason}"))

        for planned, actual in sorted(self.case_mismatches.items()):
            out.append(Disagreement(
                kind="filename-case-mismatch", step_id="-",
                detail=(f"the plan targets `{planned}` but the file on "
                        f"disk is `{actual}` — this resolves on a "
                        f"case-insensitive filesystem and breaks on a "
                        f"case-sensitive one")))

        if not pipeline_success:
            counts = self.tally()
            # Every other kind here proves SHAPE — the file exists, parses,
            # declares the right names, matches the plan's body. None of
            # them can see behaviour, so on their own they are no basis at
            # all for telling someone their failure is the harness's fault.
            #
            # Observed: a 20B run produced eight structurally perfect files
            # — 41 postconditions green, every plan-declared anchor present
            # — whose suite failed with "Ghost out of map bounds at (5, 7)".
            # A real logic bug, a correctly-failed run, and this check
            # confidently blamed the harness. A confirmed-green acceptance
            # gate is the only evidence that speaks to behaviour, so
            # without one the honest answer is silence.
            behavioural = [
                e for e in self.expectations.values()
                if e.kind == KIND_GATE_PASSED and e.verdict == HOLDS
            ]
            if counts[VIOLATED] == 0 and behavioural:
                out.append(Disagreement(
                    kind="failed-but-clean", step_id="-",
                    detail=(f"run marked FAILED while all {counts[HOLDS]} "
                            f"resolved postcondition(s) hold, including "
                            f"{len(behavioural)} acceptance gate(s) that "
                            f"went green — suspect the harness before the "
                            f"model")))
        return out

    def report(self, done_step_ids: Iterable[str], *,
               tracked_files: Iterable[str] = (),
               pipeline_success: bool = True) -> list[Disagreement]:
        """Log the shadow summary and any disagreements. Returns them."""
        counts = self.tally()
        gaps = self.disagreements(done_step_ids,
                                  tracked_files=tracked_files,
                                  pipeline_success=pipeline_success)
        strength = sum(self.step_strength(s) for s in self.steps)
        _logger.info(
            "[Ghost] shadow: %d expectation(s) over %d step(s) — "
            "%d hold, %d violated, %d unknown; evidence weight %d; "
            "%d disagreement(s)",
            len(self.expectations), len(self.steps), counts[HOLDS],
            counts[VIOLATED], counts[UNKNOWN], strength, len(gaps))
        # Repaired-in-flight is not a defect, but it is worth seeing: it
        # is the trace of a self-heal or fix round doing its job, and a
        # postcondition that keeps needing repair is a plan smell.
        repaired = [e for e in self.expectations.values()
                    if e.ever_violated and e.verdict == HOLDS]
        if repaired:
            _logger.info(
                "[Ghost] %d postcondition(s) were broken mid-run and are "
                "green now (repaired in flight): %s", len(repaired),
                ", ".join(f"{e.kind}:{e.subject}" for e in repaired[:5]))
        for gap in gaps:
            _logger.warning("[Ghost] %s (step %s): %s",
                            gap.kind, gap.step_id, gap.detail)
        return gaps

    def to_dict(self) -> dict:
        """Serialize for checkpoints, benchmarks, and offline comparison."""
        return {
            "root": self.root,
            "expectations": [e.to_dict()
                             for e in self.expectations.values()],
            "files": [{"path": f.path, "pre_hash": f.pre_hash,
                       "post_hash": f.post_hash, "inline": f.inline,
                       "writers": list(f.writers)}
                      for f in self.files.values()],
            "steps": {s: {"produces": sorted(n["produces"]),
                          "requires": sorted(n["requires"])}
                      for s, n in self.steps.items()},
            "tally": self.tally(),
        }


# ── Individual checks ────────────────────────────────────────────────


_PARSEABLE_EXTS = (".py", ".json")


def _parseable(path: str) -> bool:
    return path.lower().endswith(_PARSEABLE_EXTS)


def _check_parses(path: str, text: str) -> tuple[str, str]:
    """Syntax-only check for the formats we can judge without a toolchain."""
    low = path.lower()
    try:
        if low.endswith(".py"):
            ast.parse(text)
        elif low.endswith(".json"):
            json.loads(text)
        else:
            return INAPPLICABLE, ""
    except SyntaxError as exc:
        return VIOLATED, f"SyntaxError line {exc.lineno}: {exc.msg}"
    except (json.JSONDecodeError, ValueError) as exc:
        return VIOLATED, f"invalid JSON: {exc}"
    except (RecursionError, MemoryError):
        return UNKNOWN, "parser gave up"
    return HOLDS, ""


def _python_class_members(text: str) -> set[str]:
    """Names defined one level inside a class body.

    The language backend reports module-level names only, so a constant
    living on the class it belongs to reads as missing. Observed:
    ``class Map:`` declares ``TILE_SIZE = 32`` and `game.py` uses
    ``Map.TILE_SIZE`` — the plan's ``exports: TILE_SIZE`` was correct and
    was reported as a broken promise. Both the bare and qualified
    spellings are returned, since a plan may write either.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return names
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            targets: list[str] = []
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                targets = [item.name]
            elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name):
                targets = [item.target.id]
            elif isinstance(item, ast.Assign):
                targets = [t.id for t in item.targets
                           if isinstance(t, ast.Name)]
            for name in targets:
                names.add(name)
                names.add(f"{node.name}.{name}")
    return names


def _export_evidence(module_level: Iterable[str],
                     members: Iterable[str] = (),
                     limit: int = 12) -> str:
    """The 'file exports …' evidence for a violated export claim.

    Plain ``sorted(actual)[:8]`` over the merged set was actively
    misleading. ``_python_class_members`` contributes both ``member`` and
    ``Class.member`` spellings, and an alphabetical head could be spent
    entirely on those: for the ``entities.py`` that really declared
    ``Player``, ``GridMover``, ``add_direction`` and 11 other module-level
    names, the evidence showed eight entries, three of them
    ``Ghost.__init__``-style, and none of the names a reader would look
    for. A correct finding read like a false positive.

    So the file's own module-level names come first — a plan's ``exports:``
    almost always names one — class members fill only the room left over,
    and the elided count is always stated, because a silent truncation is
    what made the old line read as a complete list.
    """
    module_level = sorted(set(module_level))
    members = sorted(set(members) - set(module_level))
    shown = module_level[:limit]
    if len(shown) < limit:
        shown += members[:limit - len(shown)]
    omitted = (len(module_level) + len(members)) - len(shown)
    listing = ", ".join(shown) if shown else "nothing"
    if omitted > 0:
        listing += f" (+{omitted} more)"
    return listing


def _check_exports(text: str, symbol: str, path: str,
                   language: str | None) -> tuple[str, str]:
    """Is *symbol* actually exported by the file's real contents?

    An extractor that cannot run yields ``UNKNOWN``: per
    ``plan_graph._export_satisfied``'s history, a confident claim built on
    a missing extractor is wrong far more often than it is right.
    """
    try:
        from ..language_backend import get_backend
        backend = get_backend(_language_for(path, language))
        exported = set(backend.extract_exports(text) or [])
    except Exception:
        return UNKNOWN, "no export extractor for this file"
    members: set[str] = set()
    if path.lower().endswith(".py"):
        members = _python_class_members(text)
    # The verdict is decided on the union, exactly as before; only the
    # evidence distinguishes the two sources, so the reader sees the
    # file's own top-level names before one class's method list.
    actual = exported | members
    if not actual:
        return UNKNOWN, "extractor found no exports at all — inconclusive"
    if _export_satisfied(symbol, actual):
        return HOLDS, ""
    return VIOLATED, (f"declared export not found; file exports "
                      f"{_export_evidence(exported, members)}")


_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby",
}


def _language_for(path: str, fallback: str | None) -> str | None:
    return _EXT_LANG.get(os.path.splitext(path)[1].lower()) or fallback


def _is_unbound_use(text: str, symbol: str) -> bool:
    """Does *text* read *symbol* without anything binding it?

    That combination is a genuine defect — a NameError the moment the
    line runs — and it is the only condition under which writing an
    import is a repair rather than decoration.

    Conservative in the safe direction: any binding anywhere in the
    module (import, assignment, def, class, parameter, comprehension
    target) counts, even in an inner scope. A false "it is bound"
    declines a repair; a false "it is unbound" would write a duplicate
    or shadowing import, which is the more damaging error.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False

    used = False
    bound = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id == symbol:
                used = True
            elif node.id == symbol:
                bound = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == symbol:
                    bound = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            if node.name == symbol:
                bound = True
        elif isinstance(node, ast.arg) and node.arg == symbol:
            bound = True
        elif isinstance(node, ast.Attribute) and node.attr == symbol:
            # `mod.Symbol` is a use that an import of `mod` already
            # satisfies — not evidence that `Symbol` itself is needed.
            pass
    return used and not bound


def _check_edge(src: str, symbols: str,
                consumers: list[tuple[str, Optional[str]]]) -> tuple[str, str]:
    """Does ANY file the step produces reference the module it imports?

    Deliberately weak twice over. It looks for the producer's module stem
    *or* any declared symbol anywhere in the consumer — barrel
    re-exports, dynamic imports and aliasing all make a stricter test
    wrong. And one satisfying file settles it for the whole step, because
    `imports:` is a step-level declaration: a step that produces a
    package marker and a test module has wired the import if the test
    module imports it.

    Only when every one of the step's files is readable and none of them
    mentions the module is the edge called unwired.
    """
    readable = [(p, t) for p, t in consumers if t is not None]
    if not readable:
        return UNKNOWN, "no consumer file could be read"
    stem = os.path.basename(module_key(src))
    declared = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    needles = [stem] + declared

    # Checked BEFORE the mention test, because it is strictly stronger.
    # A file that reads a declared symbol while nothing binds it is a
    # NameError waiting to run — yet it "mentions" the symbol, so the
    # mention test below would have called it correctly wired.
    for path, text in readable:
        if not path.endswith(".py"):
            continue
        unbound = [s for s in declared if _is_unbound_use(text, s)]
        if unbound:
            return VIOLATED, (
                f"{path} uses {', '.join(unbound)} but nothing imports or "
                f"defines it — the import edge is missing and the file "
                f"raises NameError when that line runs")

    for path, text in readable:
        for needle in needles:
            if needle and re.search(rf"\b{re.escape(needle)}\b", text):
                return HOLDS, f"wired in {path}"
    if len(readable) < len(consumers):
        return UNKNOWN, "some of the step's files could not be read"
    return VIOLATED, (
        f"none of the step's file(s) ({', '.join(p for p, _ in readable)}) "
        f"mentions `{stem}` or any declared symbol — the import edge was "
        f"never wired")


# ── Declared dependencies vs. the environment that will run the app ──
#
# WHY THIS EXISTS
# A plan step wrote `python -m venv venv && python -m pip install pygame`.
# `venv` was created but never activated, so the second `python` was still
# the pipeline's interpreter: pygame landed in the pipeline's environment
# and never in the project's. Every gate then passed — the game modules
# were cleanly headless and imported no pygame — and the suite went fully
# green. Only `main.py`, which imports pygame inside `main()`, ever needed
# it, and it ran under the project venv where it was absent. Both the
# classic and the agent-loop arm of a benchmark shipped an application
# that could not start, with every check green.
#
# The claim checked here is the one nothing else made: every dependency
# the plan's own manifest declares must be present in the environment the
# app will actually run in. Purely a filesystem question — no subprocess,
# no import — so the shadow stays read-only.


def _pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def _requirements_names(text: str) -> list[str]:
    """Distribution names from a requirements.txt, specifiers stripped."""
    names: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        # Options (-r, -e, --index-url) and direct URLs name no
        # distribution we can look up by directory.
        if not line or line.startswith("-") or "://" in line:
            continue
        name = re.split(r"[\[<>=!~;\s]", line, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return names


def _site_packages(venv_bin: str) -> Optional[str]:
    """The site-packages dir belonging to *venv_bin*'s environment."""
    root = os.path.dirname(venv_bin)
    win = os.path.join(root, "Lib", "site-packages")
    if os.path.isdir(win):
        return win
    lib = os.path.join(root, "lib")
    if os.path.isdir(lib):
        try:
            for entry in sorted(os.listdir(lib)):
                cand = os.path.join(lib, entry, "site-packages")
                if os.path.isdir(cand):
                    return cand
        except OSError:
            return None
    return None


def _installed_names(site_dir: str) -> Optional[set[str]]:
    """Every distribution and top-level module name under *site_dir*.

    Both are collected because the two vocabularies differ: a dependency
    is declared by distribution name (``beautifulsoup4``) and imported by
    module name (``bs4``). Matching either is enough to say it is there —
    the question being asked is presence, not spelling.

    Returns ``None`` only when the directory cannot be read. An empty set
    is a real answer: a readable site-packages with nothing in it means
    the dependencies are genuinely absent, which is precisely the state
    this check exists to catch.
    """
    found: set[str] = set()
    try:
        entries = os.listdir(site_dir)
    except OSError:
        return None
    for entry in entries:
        if entry.endswith((".dist-info", ".egg-info")):
            stem = entry.rsplit(".", 1)[0]
            found.add(_pep503(stem.rsplit("-", 1)[0]))
            continue
        if entry.endswith(".py"):
            found.add(_pep503(entry[:-3]))
            continue
        if not entry.startswith("_") and "." not in entry:
            found.add(_pep503(entry))
    return found


def _check_packages(root: str, manifest: str,
                    text: Optional[str]) -> tuple[str, str]:
    """Are the manifest's declared dependencies in the app's environment?

    ``UNKNOWN`` whenever the environment cannot be identified — no
    project venv, no readable site-packages, an unsupported manifest.
    Absence of an environment is not absence of a dependency, and this
    check must never accuse a project that simply runs on the ambient
    interpreter.
    """
    if text is None:
        return UNKNOWN, "manifest unreadable"
    base = os.path.basename(manifest).lower()

    if base == "package.json":
        try:
            deps = list((json.loads(text).get("dependencies") or {}).keys())
        except (ValueError, AttributeError):
            return UNKNOWN, "manifest is not readable JSON"
        if not deps:
            return INAPPLICABLE, "no runtime dependencies declared"
        node_modules = os.path.join(root, "node_modules")
        if not os.path.isdir(node_modules):
            return UNKNOWN, "no node_modules — cannot tell what is installed"
        missing = [d for d in deps
                   if not os.path.exists(os.path.join(node_modules, *d.split("/")))]
        if missing:
            return VIOLATED, (
                f"declared but not installed in node_modules: "
                f"{', '.join(sorted(missing))}")
        return HOLDS, f"{len(deps)} dependency(ies) present"

    names = _requirements_names(text)
    if not names:
        return INAPPLICABLE, "no dependencies declared"

    try:
        from ..executor import Executor
        venv_bin = Executor._venv_bin_dir(root)
    except Exception:
        return UNKNOWN, "could not locate the project interpreter"
    if not venv_bin:
        # No project venv: the app runs on whatever interpreter is
        # ambient, which this check cannot inspect from disk.
        return UNKNOWN, "no project venv — app runs on the ambient interpreter"

    site_dir = _site_packages(venv_bin)
    if not site_dir:
        return UNKNOWN, f"no site-packages under {venv_bin}"
    installed = _installed_names(site_dir)
    if installed is None:
        return UNKNOWN, f"could not read {site_dir}"

    missing = [n for n in names if _pep503(n) not in installed]
    if missing:
        return VIOLATED, (
            f"declared in {manifest} but absent from the environment the "
            f"app runs in ({site_dir}): {', '.join(sorted(missing))} — "
            f"gates can still pass if no tested module imports them")
    return HOLDS, f"{len(names)} dependency(ies) present in {site_dir}"


def _canon_cmd(cmd: str) -> str:
    """Whitespace-free form of a command, for identity comparison only.

    The pipeline rewrites a gate's spacing before running it, so the
    plan's string and the ledger's string are routinely different bytes
    for the same command. Observed: the plan declared
    ``set SDL_VIDEODRIVER=dummy && python -m unittest -v`` while the
    ledger recorded ``set SDL_VIDEODRIVER=dummy&& python -m unittest -v``
    — the space is deliberately removed, because on Windows cmd.exe
    ``set VAR=dummy `` assigns the trailing space into the variable and
    breaks SDL. Comparing raw strings called a gate that had just passed
    "never passed". Two commands that differ only in whitespace are the
    same command for this purpose.
    """
    return re.sub(r"\s+", "", cmd or "")


# ── Plan-declared anchors: did the step build what the plan said? ────
#
# An "anchor" is a name the PLAN's own body for a file declares — a CSS
# class, a Python def/class, a JS export. Checking that the written file
# still contains them catches per-step drift: the planner specified the
# right thing and the model that executed the step produced something
# else. Nothing else in the pipeline notices, because the file exists,
# parses, and often satisfies a gate that never names the missing piece.
#
# Only structural names are used, never whole bodies. The claim being
# made is "the plan said this file declares `.site-header` and it does
# not", which is checkable and true or false — not "the styling is
# wrong", which is a judgement.

_CSS_SELECTOR_RE = re.compile(r"(?:^|[\s,{}>+~])([.#][A-Za-z_][\w-]*)")
_JS_EXPORT_RE = re.compile(
    r"export\s+(?:default\s+)?(?:async\s+)?"
    r"(?:function|class|const|let|var)\s+([A-Za-z_$][\w$]*)")
_CSS_SUFFIXES = (".css", ".scss", ".sass", ".less", ".styl")


def plan_anchors(path: str, body: str) -> set[str]:
    """Structural names the plan's own body for *path* declares."""
    low = path.lower()
    if low.endswith(_CSS_SUFFIXES):
        return {m.group(1) for m in _CSS_SELECTOR_RE.finditer(body)}
    if low.endswith(".py"):
        names: set[str] = set()
        try:
            tree = ast.parse(body)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return names
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets
                             if isinstance(t, ast.Name) and t.id.isupper())
        return names
    if low.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
        return {m.group(1) for m in _JS_EXPORT_RE.finditer(body)}
    return set()


def _check_plan_anchors(path: str, declared: str,
                        text: Optional[str]) -> tuple[str, str]:
    """Does the written file still declare what the plan's body did?"""
    if text is None:
        return UNKNOWN, "file unreadable"
    wanted = [a for a in (declared or "").split(",") if a]
    if not wanted:
        return INAPPLICABLE, "the plan body declared no structural names"
    missing = []
    for anchor in wanted:
        # A CSS selector carries its own sigil; identifiers get word
        # boundaries so `Board` does not match `Dashboard`.
        pattern = (re.escape(anchor) if anchor[0] in ".#"
                   else rf"\b{re.escape(anchor)}\b")
        if not re.search(pattern, text):
            missing.append(anchor)
    if not missing:
        return HOLDS, f"all {len(wanted)} plan-declared name(s) present"
    return VIOLATED, (
        f"the step drifted from the plan — the plan's own body for this "
        f"file declares {', '.join(sorted(missing))}, and the written "
        f"file does not")


# ── Test files the declared runner will never collect ────────────────
#
# WHY THIS EXISTS
# An agent loop wrote four test modules — 18KB across test_player.py,
# test_main.py, test_ghost.py and test_game_map.py — in pytest style:
# module-level `def test_x(Player, tmp_path)` with fixtures. The
# project's own acceptance command was `python -m unittest -v`, which
# collects only TestCase subclasses, so all four contributed exactly
# zero tests. `python -m unittest` reported 2 tests and passed; pytest
# on the same directory reported 22. Twenty tests were invisible to the
# command the task was graded on, the files imported cleanly so nothing
# errored, and every check in the pipeline stayed green.
#
# Collection is decided statically here rather than by running anything:
# the rules are simple enough to read off the AST, and this module does
# not execute commands.

_TEST_NAME_RE = re.compile(r"(^|[/_.])test[_s]?\d*\.py$|conftest\.py$",
                           re.IGNORECASE)

_RUNNERS = (
    ("unittest", re.compile(r"\bpython[0-9.]*\s+-m\s+unittest\b|\bunittest\b")),
    ("pytest", re.compile(r"\bpytest\b")),
)


def is_python_test_file(path: str) -> bool:
    base = os.path.basename(path.replace("\\", "/"))
    if not base.endswith(".py") or base == "conftest.py":
        return False
    return base.startswith("test_") or base.endswith("_test.py")


def declared_runner(commands: Iterable[str]) -> Optional[str]:
    """Which Python test runner the plan's own commands name.

    ``pytest`` wins a tie: it collects everything unittest does plus
    module-level functions, so a project running both is only in trouble
    when the unittest-only command is the acceptance gate.
    """
    found: set[str] = set()
    for cmd in commands:
        for name, pattern in _RUNNERS:
            if pattern.search(cmd or ""):
                found.add(name)
    if "pytest" in found:
        return "pytest"
    if "unittest" in found:
        return "unittest"
    return None


def _python_test_counts(text: str) -> tuple[int, int]:
    """``(unittest_visible, pytest_only)`` test counts for one module."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return -1, -1                      # unparseable: no opinion
    unittest_visible = 0
    pytest_only = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Attribute):
                    bases.append(b.attr)
                elif isinstance(b, ast.Name):
                    bases.append(b.id)
            if any(b.endswith("TestCase") for b in bases):
                unittest_visible += sum(
                    1 for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name.startswith("test"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                pytest_only += 1
    return unittest_visible, pytest_only


def uncollected_test_files(root: str, paths: Iterable[str],
                           runner: Optional[str]) -> list[tuple[str, str]]:
    """``(path, reason)`` for test files the *runner* will never collect.

    Silent when the runner cannot be identified, when a file will not
    parse, or when the runner is pytest (which collects both styles) —
    absence of a clear rule is not evidence of a broken test file.
    """
    if runner != "unittest":
        return []
    out: list[tuple[str, str]] = []
    for path in sorted(set(paths)):
        if not is_python_test_file(path):
            continue
        text = _read(root, path)
        if text is None:
            continue
        visible, pytest_style = _python_test_counts(text)
        if visible < 0:
            continue                        # unparseable
        if visible > 0:
            continue
        if pytest_style > 0:
            out.append((path, (
                f"{pytest_style} test(s) are written pytest-style "
                f"(module-level functions), and `unittest` collects only "
                f"TestCase subclasses — none of them run under the "
                f"project's own acceptance command")))
        else:
            out.append((path, "defines no tests the declared runner collects"))
    return out


def _check_gate(cmd: str, gates: list[str]) -> tuple[str, str]:
    """Did this step's acceptance command ever go green?

    Answered purely from the ledger of gates that already passed — this
    module never runs a command. Matching is whitespace-insensitive and
    allows containment in either direction, because the pipeline may
    respell a gate or prefix it with a ``cd`` before recording it.
    """
    if not gates:
        return UNKNOWN, "no gate ledger available"
    want = _canon_cmd(cmd)
    if not want:
        return UNKNOWN, "empty verify command"
    for recorded in gates:
        got = _canon_cmd(recorded)
        if got == want or want in got or got in want:
            return HOLDS, ""
    # UNKNOWN, not VIOLATED. The ledger records gates that passed through
    # the normal step path; it is NOT a complete log of every gate that
    # ever ran. Observed: a CMD step's `python -m unittest -v` passed
    # inside the agent loop's recovery path and again in BulkTest, and
    # never entered the ledger — reporting "never passed" about a suite
    # that had just gone green twice is exactly the confident falsehood
    # this module's three-valued discipline exists to prevent. Absence
    # from an incomplete record is absence of evidence.
    return UNKNOWN, ("not found in the gate ledger — it may have passed "
                     "outside the ledger's recording path (agent-loop "
                     "recovery, BulkTest), so this is inconclusive")


# ── Module-level handle (mirrors get_gate_ledger) ────────────────────

_ghost: Optional[GhostPlan] = None


def start_ghost(steps, project_root: str = ".") -> Optional[GhostPlan]:
    """Build and install the run's ghost. Returns ``None`` on any failure."""
    global _ghost
    try:
        _ghost = GhostPlan.build(steps, project_root)
        _logger.info(
            "[Ghost] tracking %d expectation(s) across %d step(s) and "
            "%d file(s)", len(_ghost.expectations), len(_ghost.steps),
            len(_ghost.files))
    except Exception as exc:
        _ghost = None
        _logger.debug("[Ghost] disabled — build failed: %s", exc)
    return _ghost


def get_ghost() -> Optional[GhostPlan]:
    return _ghost


def reset_ghost() -> None:
    global _ghost
    _ghost = None

"""An exact version pin nobody asked for is a guess about the host.

Measured 2026-09-13, "create a 2 snake game in python include production
ready features". The IntentAgent's REQUIREMENTS_SPEC listed, under
``Packages/versions:``, a line the user never wrote::

    - `pygame==2.6.0`

Both readers of that spec inherited it. The planner declared
``verify: ... read_text().strip() == 'pygame==2.6.0'`` and the seeded
contract asserted the same line. pygame 2.6.0 ships no wheel for the
Python 3.13 the project venv ran, so ``pip install -r requirements.txt``
tried a source build and died on ``No module named
'distutils.msvccompiler'``. The install step then took 16 turns and an
escalation — about 123k prompt tokens, over half the run — searching
winget for Visual Studio Build Tools and finally **recreating the venv on
Python 3.11** to make the pin installable. 2.6.1, one patch later, had the
wheel all along.

A model's pin is its training cut-off talking, not a requirement: it
cannot know which versions have binaries for the interpreter this run will
meet. So an unrequested ``name==X.Y.Z`` is relaxed to the compatible
release ``name~=X.Y`` (or ``~=0.Y.Z`` below 1.0, where minors break), which
keeps the version the model meant as a floor and admits the patch the host
can actually install.

``~=`` rather than ``>=X,<Y`` on purpose: the same text lands in shell
commands, and an unquoted ``pip install pygame>=2.6,<3`` is a redirect.

Pins the user DID ask for are left exactly as written — by package and
version named together in the task, or by an explicit request to pin.
"""

from __future__ import annotations

import re

# `name==X.Y.Z…`. Three components are required, and that is what keeps this
# away from code: `speed==1.5` is valid Python and must never be touched,
# while `x==1.2.3` is not valid Python at all. The lookbehind stops a match
# starting mid-identifier or mid-attribute (`g.speed==…`).
_PIN_RE = re.compile(
    r"(?<![\w.\-])"
    r"(?P<name>[A-Za-z][A-Za-z0-9_.\-]*?(?:\[[\w,.\-]+\])?)"
    r"==(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?P<rest>(?:\.\d+|[a-z]+\d*|\+[\w.]+)*)"
    r"(?!\w)(?!\.\d)")

# The user asking for exact versions in general.
_PIN_REQUEST_RE = re.compile(
    r"\b(?:pin(?:ned|ning)?|exact(?:ly)?\s+versions?|lock(?:ed)?\s+versions?)\b",
    re.IGNORECASE)


def _canonical(s: str) -> str:
    """PEP 503 normalisation: case and runs of `-_.` are insignificant."""
    return re.sub(r"[-_.]+", "-", s).lower()


def _user_requested(name: str, version: str, user_task: str) -> bool:
    """Did the user's own words name this package at this version?"""
    text = user_task or ""
    if _PIN_REQUEST_RE.search(text):
        return True
    if version not in text:
        return False
    bare = _canonical(name.split("[", 1)[0])
    # `=`, `[` and whitespace all end a word, so `pygame==2.6.0`,
    # `pygame 2.6.0` and `pygame[extra]==2.6.0` all yield the word `pygame`.
    words = {_canonical(w) for w in re.findall(r"[A-Za-z0-9][\w.\-]*", text)}
    return bare in words


def relax_unrequested_pins(text: str, user_task: str | None
                           ) -> tuple[str, list[tuple[str, str]]]:
    """Relax exact pins in *text* that *user_task* did not ask for.

    Returns the new text and the ``(old, new)`` substitutions made, so the
    caller can say what it changed. Every occurrence of one pin is rewritten
    identically, which is what keeps a plan's gate (`== 'pygame~=2.6'`)
    consistent with the manifest the same plan tells a step to write.
    """
    if not text or "==" not in text:
        return text, []
    changes: list[tuple[str, str]] = []

    def _sub(m: re.Match) -> str:
        name = m.group("name")
        major, minor, patch = m.group("major"), m.group("minor"), m.group("patch")
        version = f"{major}.{minor}.{patch}{m.group('rest')}"
        if m.group("rest"):
            return m.group(0)      # pre-releases / local builds are deliberate
        if _user_requested(name, version, user_task or ""):
            return m.group(0)
        if major == "0":
            new = f"{name}~={major}.{minor}.{patch}"
        else:
            new = f"{name}~={major}.{minor}"
        old = m.group(0)
        if (old, new) not in changes:
            changes.append((old, new))
        return new

    return _PIN_RE.sub(_sub, text), changes

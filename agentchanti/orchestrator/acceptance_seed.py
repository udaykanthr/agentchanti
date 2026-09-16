"""Acceptance tests written from the TASK, before any code exists.

`evidence.py` defines independence as a test the agent did not author:
user-supplied `acceptance_cmds`, or a pre-existing test file the run left
byte-identical. A greenfield build honestly has neither, so every such run
is judged by a suite it wrote itself — and three measured runs shipped
`exit 0` over artifacts that failed every external probe while their own
tests were green.

The point is not who writes the check, it is *when*. A test written after
`game.py` exists is written by an agent that has just read `game.py`, and
it agrees with the code by construction. A test written from the task text
alone cannot be shaped to fit an implementation that does not exist yet.

So this runs once, after the plan is final and before the first step:
generate a suite from the task, write it, and let `snapshot_test_files`
record it a moment later as pre-existing. Everything downstream is already
built — if a later step rewrites it, the hash changes, independence is
forfeited and the run says so.

What it deliberately does NOT do:

* Invent behaviour. The prompt asks only for assertions the task states in
  so many words; a task that says nothing checkable yields no file, and no
  file is an honest "nothing independent verified this".
* Overwrite anyone else's suite. A test file this module did not write is
  better evidence than anything generated here, and is never touched.
* Decide the run. A failing seed test is reported through the normal
  acceptance path; it never silently rewrites a verdict.

WHEN THE TASK CHANGES
---------------------
"A suite already exists, so skip" was too coarse by exactly one case: a
suite this module seeded **for a different task**. Measured 2026-08-17,
across four runs in one directory. Run 1 seeded a contract from a
"Panda3D cube collector" prompt. The prompt was then rewritten into a
Snake game, and every subsequent run logged ``skipped: 1 test file(s)
already predate the run and are stronger evidence`` and then
``Evidence: independent (pre-existing-tests)``. Both statements were true
under the old rule and the conclusion was worthless: the surviving file
asserts only that ``main.py`` does not exit within five seconds, which
holds for any Panda3D script that starts at all, Snake or otherwise. The
banner read exactly the same as it had when the check genuinely matched
the task.

So the seed stamps its own header — the task it was written for, and a
hash of the body it wrote — and re-seeds when the task no longer matches.
Both halves of the header are load-bearing. Without the task hash there
is no way to tell a current contract from a stale one; without the body
hash, regenerating would silently discard edits someone made by hand.
A file whose body no longer matches is treated as adopted by its editor
and left alone, the same refusal `ghost_heal` makes when a written file
declares something the plan's body does not.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re

log = logging.getLogger("agentchanti")

# Written where every runner this project supports will collect it, and
# named so `_is_test_file` recognises it as a test.
SEED_BASENAME = "test_acceptance_contract.py"

# How many times to ask again when the first contract cannot fail. Two,
# because one attempt is lost outright to an unusable response, and the
# generation is cheap next to the run it is supposed to judge.
_REPAIR_ATTEMPTS = 2

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# First line of a seeded file. A comment, so it costs the suite nothing,
# and self-describing, so it survives a wiped .agentchanti directory —
# the state that decides whether to re-seed must live with the artifact
# it describes, not beside it.
_HEADER_RE = re.compile(
    r"^#\s*agentchanti:acceptance-seed\s+task=([0-9a-f]+)\s+body=([0-9a-f]+)\s*$"
)


def _fingerprint(text: str) -> str:
    """Stable short hash, insensitive to whitespace-only edits.

    Reflowing a prompt must not count as changing the task — the check
    is meant to fire when what is being ASKED FOR changes, not when a
    line was rewrapped.
    """
    normalized = " ".join((text or "").split())
    return hashlib.sha1(normalized.encode("utf-8", "replace")).hexdigest()[:16]


def _header(task: str, body: str) -> str:
    return (f"# agentchanti:acceptance-seed task={_fingerprint(task)} "
            f"body={_fingerprint(body)}\n")


def seed_state(path: str) -> tuple[str, str, str] | None:
    """``(task_hash, body_hash, body)`` for a file this module wrote.

    None means "not ours" — no header, or an unreadable file — and is the
    answer that makes the module leave a user's own suite alone.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
            body = fh.read()
    except OSError:
        return None
    match = _HEADER_RE.match(first.strip())
    if not match:
        return None
    return match.group(1), match.group(2), body

_PROMPT = """\
You are writing the ACCEPTANCE TEST for a task, BEFORE any code exists.

TASK:
{task}

Write a single self-contained Python `unittest` file that checks ONLY the
behaviour the task states explicitly. Rules, all of them load-bearing:

1. Assert ONLY what the task says. Do not invent requirements, do not
   guess at internals, do not assume a file layout beyond what the task
   names. If the task states an exact API, use exactly those names.
2. Where the task gives a RANGE or a QUANTITY, exercise it properly. A
   range like "any dt from 0.001 to 0.5" means loop enough iterations at
   the SMALL end for the behaviour to actually be observable — hundreds of
   steps, not a handful. A claim about an invariant is only as good as the
   number of chances it had to fail.
3. Where the task states something must CHANGE, assert a strict relation
   between two observed values (`assertNotEqual`, `assertLess`). Never
   assert only invariants and monotonicity: `assertLessEqual(a, b)` passes
   against a frozen system.
4. Import the modules the task names inside each test method and let an
   ImportError fail that test. Do not guard with try/except or skip.
5. No mocks, no stubs, no reaching into private attributes, no assigning
   to state. Drive the system only through the public API the task names.
6. NEVER call `sys.exit()`, `quit()`, `os._exit()` or a framework's exit
   helper (`userExit()`, `destroy()` that exits) — not in the test, not
   in a `finally:`, not in `tearDown`. They raise SystemExit, which
   unittest records as an ERROR *after* your assertions have already
   passed, turning a green contract red. Let objects be garbage
   collected instead.
7. Never assert on the program's SOURCE TEXT. No `inspect.getsource`, no
   `ast.parse` of the module under test, no reading `__file__` and
   searching it for identifiers. "The source mentions `reset`" passes a
   stub that mentions it and fails correct code that calls the method
   `restart`; you would be testing vocabulary, not behaviour. Build the
   objects and call the methods.

Output ONLY the Python file in one ``` fenced block. No commentary.
"""

# The prompt above is deliberately the 0.7.0 prompt, verbatim. Between 0.8.0
# and 2026-09-15 it grew from 7 rules / 2,132 chars to 11 rules / 4,143 chars
# (no human, no README wording, where the file lives, the platform, no
# climbing above __file__, no desktop inspection) — one rule per observed
# failure — while the failure rate on the measured prompt did not improve,
# and the contract after the Windows platform note walked the Win32 desktop
# for the first time. Every one of those defects is now caught by a detector
# that costs nothing unless it fires, and its guidance is sent only then, in
# the repair note for the contract that actually made the mistake.


# Rule 5 of the prompt says no mocks, and a suite that ignores it is not
# an acceptance contract — it is a description of the code's shape that
# agrees with itself. Measured 2026-08-17: a seeded contract patched
# `panda3d.core.ShowBase.__init__`, an attribute that does not exist, and
# 22 of its 23 tests ERRORED. It was still counted as this run's
# independent evidence, because nothing ran it (see evidence.classify).
# Refusing is the honest outcome: no file means "nothing independent
# verified this", which is true, where an unrunnable file means the same
# thing while looking like proof.
_MOCK_MARKERS = (
    "unittest.mock",
    "from mock import",
    "import mock",
    "MagicMock",
    "mock.patch",
    "@patch",
    "patch(",
)


def mocking_reason(src: str) -> str | None:
    """Which forbidden stubbing construct this suite uses, or None."""
    for marker in _MOCK_MARKERS:
        if marker in src:
            return marker
    return None


# A contract that needs a PERSON cannot pass an unattended run, whatever
# the code does — the same category as a gate this platform's shell cannot
# execute, which `unrunnable_gate_reason` already refuses at plan time. The
# weakness, mocking and source-grep screens all ask whether the contract
# CHECKS the right thing; none asked whether it can ever finish.
#
# Measured 2026-09-10, a 3D pinball run. The seeded contract opened a
# Tkinter prompt window and asked a human to observe the game and type
# scores in:
#
#     score_before = self.integer("Score before the final scoring hit")
#     self.action("Press Esc in the game. Do not close it with the window
#                  manager.")
#     print("\nThe game is still open; press Esc in its window ...")
#
# The run had every gate green, a suite that passed and a game that
# demonstrably worked — verified afterwards by screenshot, 1090 fps and a
# full-power launch that cleared the lane — and still exited non-zero,
# because nobody was there to answer. Worse, it is self-concealing: it
# leaves a real game window parked on the desktop, which reads as a hung
# application rather than as a contract waiting for input.
#
# Matched anywhere in the source, not line-anchored: the measured contract
# imported tkinter lazily inside a method (`    import tkinter as tk`), so
# an `^import` scan would have missed the one construct that mattered.
_INTERACTIVE_MARKERS = (
    "tkinter",
    "input(",
    "raw_input(",
    "getpass",
    "msvcrt.getch",
    "sys.stdin.read",        # also covers .readline()
    "PySimpleGUI",
    "pyautogui",
)


# The contract is written to the PROJECT ROOT and run from there, and the
# prompt never said so. Measured 2026-09-15, a 7-step Snake run: every gate
# green, smoke test launched, ghost 0 violated — and exit 1, because the
# seeded contract began
#
#     PROJECT_ROOT = Path(__file__).resolve().parents[1]
#
# which is the conventional line for a file in `tests/`, and one directory
# ABOVE the project for a file at the root. `snake_game.py` was reported
# missing while it sat beside the contract, and the game was launched with
# cwd outside the project, exiting 2. Both surfaced as assertion FAILURES,
# which the runnability repair rightly never touches — so nothing caught it.
#
# Deterministic from the source alone: a file in the root that climbs from
# `__file__` always leaves the project. Names bound to a `__file__`
# expression are followed, so `HERE = Path(__file__).resolve()` then
# `HERE.parents[1]` is seen too.
def root_escape_reason(src: str) -> str | None:
    """The construct that resolves a path above the project root, or None."""
    if not src or "__file__" not in src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    file_names: set[str] = set()

    def _from_file(node) -> bool:
        return any((isinstance(n, ast.Name) and
                    (n.id == "__file__" or n.id in file_names))
                   for n in ast.walk(node))

    for _ in range(2):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _from_file(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        file_names.add(target.id)

    def _is_dirname(call) -> bool:
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else "")
        return name == "dirname"

    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "parents"
                and _from_file(node.value.value)):
            idx = node.slice
            if (isinstance(idx, ast.Constant) and isinstance(idx.value, int)
                    and idx.value >= 1):
                return f"parents[{idx.value}]"
        if (isinstance(node, ast.Attribute) and node.attr == "parent"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "parent"
                and _from_file(node.value.value)):
            return ".parent.parent"
        if (isinstance(node, ast.Call) and _is_dirname(node) and node.args
                and isinstance(node.args[0], ast.Call)
                and _is_dirname(node.args[0])
                and _from_file(node.args[0])):
            return "os.path.dirname(os.path.dirname(__file__))"
    return None


# The same run's contract carried a second defect under the first: once its
# root was corrected, the KeyboardInterrupt test raised
#
#     process.send_signal(signal.SIGINT)
#     ValueError: Unsupported signal: 2
#
# On Windows `Popen.send_signal` accepts only SIGTERM, CTRL_C_EVENT and
# CTRL_BREAK_EVENT, and several POSIX names do not exist at all. The prompt
# never said which OS the contract runs on, so the model wrote POSIX. A
# construct the running platform rejects is an instrument that cannot
# execute — `posix_only_idiom_reason`'s argument for gates, one layer over.
_WIN_SEND_SIGNAL_OK = {"SIGTERM", "CTRL_C_EVENT", "CTRL_BREAK_EVENT"}
_POSIX_ONLY_SIGNAL_ATTRS = {
    "SIGKILL", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGALRM",
    "SIGCHLD", "SIGPIPE", "SIGSTOP", "SIGCONT", "SIGTSTP", "alarm",
    "setitimer", "pause", "pthread_kill", "sigwait"}
_POSIX_ONLY_OS_ATTRS = {"killpg", "setsid", "getpgid", "setpgrp", "fork"}


def _platform_note(platform: str | None = None) -> str:
    import sys
    platform = platform or sys.platform
    if platform != "win32":
        return f"PLATFORM: {platform}."
    return (
        "PLATFORM: Windows. `subprocess.Popen.send_signal` accepts ONLY "
        "`signal.SIGTERM`, `signal.CTRL_C_EVENT` and `signal.CTRL_BREAK_EVENT` "
        "(sending `signal.SIGINT` raises ValueError); `signal.SIGKILL`, "
        "`SIGHUP`, `SIGUSR1`, `signal.alarm`, `os.killpg`, `os.setsid` and "
        "`preexec_fn=` do not exist here. To stop a program you launched, "
        "call `process.terminate()` and then `process.wait()`.")


def platform_signal_reason(src: str, platform: str | None = None) -> str | None:
    """A process/signal construct the running platform rejects, or None."""
    import sys
    if (platform or sys.platform) != "win32" or not src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "send_signal"
                    and node.args and isinstance(node.args[0], ast.Attribute)
                    and isinstance(node.args[0].value, ast.Name)
                    and node.args[0].value.id == "signal"
                    and node.args[0].attr not in _WIN_SEND_SIGNAL_OK):
                return f"send_signal(signal.{node.args[0].attr})"
            for kw in node.keywords:
                if kw.arg == "preexec_fn":
                    return "preexec_fn="
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "signal" and node.attr in _POSIX_ONLY_SIGNAL_ATTRS:
                return f"signal.{node.attr}"
            if node.value.id == "os" and node.attr in _POSIX_ONLY_OS_ATTRS:
                return f"os.{node.attr}"
    return None


# Measured 2026-09-15, the next Snake run after the two fixes above: every
# gate green, the smoke test launched the game — and the contract failed on
# `snake_game.py did not open a visible pygame window within 8.0 seconds`.
# It launched the game with `sys.executable`, then walked the desktop with
# `ctypes.WinDLL("user32").EnumWindows` for a visible window owned by
# `process.pid`, and meant to scrape its pixels through gdi32. Measured by
# hand, the window existed and was titled 'Two Player Snake' — owned by pid
# 22824 (`C:\\Python313\\python.exe`), whose parent was pid 13136, the
# `venv\\Scripts\\python.exe` launcher stub `Popen` returned. A Windows venv
# interpreter re-executes the base one, so the check could never match.
#
# That is one instance of a class: asserting on the DESKTOP rather than the
# program — window enumeration, screen capture, GUI automation. It depends
# on a logged-in session, on how the interpreter launches, on focus and DPI,
# and none of it is behaviour the task describes. Blind-written framework
# introspection was already the dominant runtime failure; OS introspection
# is the same thing one layer further out.
_DESKTOP_DLLS = {"user32", "gdi32", "dwmapi", "user32.dll", "gdi32.dll",
                 "dwmapi.dll"}
_DESKTOP_MODULES = {"win32gui", "win32ui", "win32api", "win32con",
                    "pywinauto", "mss", "pyscreeze", "Xlib", "Quartz"}


def desktop_introspection_reason(src: str) -> str | None:
    """A construct that inspects the OS window system or screen, or None."""
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _DESKTOP_MODULES or alias.name == "PIL.ImageGrab":
                    return f"import {alias.name}"
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in _DESKTOP_MODULES:
                return f"from {node.module} import ..."
            if node.module == "PIL" and any(a.name == "ImageGrab"
                                            for a in node.names):
                return "PIL.ImageGrab"
        if isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else "")
            first = node.args[0]
            if (name in ("WinDLL", "CDLL", "OleDLL", "LoadLibrary")
                    and isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value.lower() in _DESKTOP_DLLS):
                return f'ctypes.{name}("{first.value}")'
        if (isinstance(node, ast.Attribute) and node.attr in ("user32", "gdi32")
                and isinstance(node.value, (ast.Attribute, ast.Name))
                and (getattr(node.value, "attr", None) == "windll"
                     or getattr(node.value, "id", None) == "windll")):
            return f"windll.{node.attr}"
    return None


# A surface exists the instant `set_mode` returns, and is blank until the
# program's next frame reaches it. So these two questions have different
# answers, and only the second one is about the artifact:
#   * has the program opened a window?      `get_surface() is not None`
#   * has the program DRAWN anything?       pixels, after a frame
_PIXEL_READ_ATTRS = frozenset({
    "tostring", "tobytes",                       # pygame.image.*
    "get_at", "get_at_mapped", "get_buffer",     # Surface.*
    "array2d", "array3d", "pixels2d", "pixels3d",           # surfarray
    "array_red", "array_green", "array_blue", "array_alpha",
})
# Only `get_surface`: a contract that called `set_mode` itself owns the
# surface and may draw on it and read it back in the same breath. The race
# belongs to the observer of a surface some other thread is drawing.
_SURFACE_ACQUIRE_ATTRS = frozenset({"get_surface"})
# Anything that lets the program's loop make progress first.
_SETTLE_ATTRS = frozenset({
    "sleep",                                     # time.sleep
    "wait", "delay", "tick", "tick_busy_loop",   # pygame.time.*
    "join", "wait_for", "acquire",               # threads and events
    "flip", "update",                            # presenting a frame
})


def _flat_stmts(body):
    """Statements in order, descending through `try`/`with` but not loops.

    A loop's body is deliberately opaque: the `time.sleep(0.01)` inside a
    poll-for-the-surface loop is part of ACQUIRING the surface, not a wait
    that stands between acquiring it and reading it.
    """
    for stmt in body:
        if isinstance(stmt, (ast.Try, ast.With, ast.AsyncWith)):
            inner = list(stmt.body)
            inner += list(getattr(stmt, "orelse", []) or [])
            inner += list(getattr(stmt, "finalbody", []) or [])
            yield from _flat_stmts(inner)
        else:
            yield stmt


def _calls_any(node, attrs) -> str | None:
    """The first call in this subtree whose name is in *attrs*."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else "")
        if name in attrs:
            return name
    return None


def render_race_reason(src: str) -> str | None:
    """Why this contract samples the window before anything is drawn, or None.

    Measured 2026-09-16 in `test1`, a clean-slate pinball run on the fixed
    tree. The seeded contract polled until `pygame.display.get_surface()`
    returned non-None — which happens the moment the game calls
    `set_mode`, before its first frame — and then, in the very next
    statement, read the pixels back and required more than one colour:

        rendered_pixels = pygame.image.tostring(surface, "RGB")
        colors = {...}
        self.assertGreater(len(colors), 1, "...must render visible content")

    A fresh surface is uniformly black, so `len(colors) == 1` and the
    contract failed a game that was entirely correct. Measured on that
    artifact with the contract's own code: 1 colour immediately, **364**
    after 0.25s, 364 after 1.0s.

    This is the same family as everything else `structural_defect_reason`
    screens — a contract written before the code exists, anchoring on the
    only vocabulary it can predict, and getting the framework's timing
    wrong — but it is the first one that is a RACE rather than a wrong
    call, so it fails intermittently and reads as a real defect.

    The rule is about ordering, not about sleeping: between acquiring a
    surface the program owns and reading its pixels there must be
    something that lets that program run — a sleep, a frame wait, a join
    — or the read must sit in a loop that retries until it settles.
    Reading a surface the contract drew on itself is untouched, and so is
    any contract that never reads pixels at all.
    """
    if not src:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        stmts = list(_flat_stmts(fn.body))
        acquired = None
        for i, stmt in enumerate(stmts):
            if _calls_any(stmt, _SURFACE_ACQUIRE_ATTRS):
                acquired = i
        if acquired is None:
            continue
        for j in range(acquired, len(stmts)):
            read = _calls_any(stmts[j], _PIXEL_READ_ATTRS)
            if read is None:
                continue
            if isinstance(stmts[j], (ast.For, ast.AsyncFor, ast.While)):
                return None          # retried until it settles
            if any(_calls_any(stmts[k], _SETTLE_ATTRS)
                   for k in range(acquired + 1, j)):
                return None          # something let the program run first
            return (f"it reads the window's pixels ({read}) as soon as the "
                    f"surface exists, which is before the program has drawn "
                    f"its first frame")
    return None


def structural_defect_reason(src: str) -> str | None:
    """Why this contract cannot judge THIS project on THIS machine, or None."""
    escape = root_escape_reason(src)
    if escape:
        return f"it looks outside the project ({escape} on __file__)"
    posix = platform_signal_reason(src)
    if posix:
        return f"it uses {posix}, which this platform rejects"
    desktop = desktop_introspection_reason(src)
    if desktop:
        return (f"it inspects the desktop rather than the program ({desktop}) "
                f"— window enumeration and screen capture depend on a "
                f"logged-in session and on which process owns the window, and "
                f"a Windows venv interpreter launches the real one as a child")
    race = render_race_reason(src)
    if race:
        return (f"{race} — a surface is blank until the program's next "
                f"frame reaches it, so wait for a frame (or retry in a "
                f"loop) before sampling, and keep the assertion as strict")
    return None


_STRUCTURAL_NOTE = """

Your contract CANNOT JUDGE THIS PROJECT ON THIS MACHINE: {reason}.

{platform}

It is saved as `test_acceptance_contract.py` IN THE PROJECT ROOT and run
from there, so the project root is `Path(__file__).resolve().parent`.
Never inspect the desktop (window enumeration, screenshots, GUI
automation): for a graphical program set `SDL_VIDEODRIVER=dummy` and
assert on the program's own state instead. Fix ONLY that defect: keep
every test and every assertion exactly as strict.

The contract you wrote, to revise rather than start over:

```python
{contract}
```"""


def interactive_reason(src: str) -> str | None:
    """Which human-in-the-loop construct makes this contract unattendable."""
    for marker in _INTERACTIVE_MARKERS:
        if marker in src:
            return marker
    return None


_INTERACTIVE_NOTE = """

Your contract CANNOT RUN UNATTENDED: it uses `{marker}`, which asks a
person to look at something, type something, or press a key. This run has
no operator. The contract will launch the application, wait forever for
input nobody will give, and fail — over code that may be perfect — while
leaving a window open that looks like a hung program.

Rewrite it to assert the same behaviours PROGRAMMATICALLY. Hard rules:

  - No `tkinter`, no `input()`, no `getpass`, no reading stdin, no
    screenshot-and-ask. Nothing that waits for a human.
  - Do NOT weaken what you check to achieve this. Drive the program's own
    public API, read its state back, and assert on the values.
  - If a behaviour is genuinely only observable on screen, assert on the
    state that produces it (positions, scores, phase, returned values)
    rather than asking someone to look.
  - The test must terminate on its own, every time, with no keypress.
"""


def _substantive_count(src: str) -> int:
    """How many discriminating assertions a candidate has, for ranking."""
    import ast

    from .seed_strength import _substantive_assertions
    try:
        return _substantive_assertions(ast.parse(src))
    except SyntaxError:
        return 0


def _generate(llm_client, task: str, extra: str = "") -> str | None:
    """One generation round: prompt, extract the fence, sanity-check it."""
    prompt = _PROMPT.format(task=task.strip())
    if extra:
        prompt += "\n\n" + extra
    try:
        raw = llm_client.generate_response(prompt)
    except Exception as exc:
        log.warning("[AcceptanceSeed] generation failed: %s", exc)
        return None
    match = _FENCE_RE.search(raw or "")
    src = (match.group(1) if match else (raw or "")).strip()
    if not _looks_like_a_suite(src):
        log.warning("[AcceptanceSeed] response was not a usable test module "
                    "— no independent check seeded")
        return None
    return src


def _looks_like_a_suite(src: str) -> bool:
    """Cheap sanity gate — it must at least be a runnable test module."""
    if not src or "import unittest" not in src:
        return False
    if "class " not in src or "def test" not in src:
        return False
    try:
        compile(src, "<seed>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def existing_test_files(root: str) -> list[str]:
    """Test files already on disk, using the pipeline's own definition."""
    from .evidence import _SKIP_DIRS
    from .pipeline import _is_test_file

    found: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                if _is_test_file(rel.replace(os.sep, "/")):
                    found.append(rel)
    except OSError:
        pass
    return found


def _should_seed(task: str, root: str, path: str) -> bool:
    """Whether to write a contract now, logging the reason either way.

    Four answers, and the order matters. Someone else's suite wins over
    everything — including a stale seed of ours sitting beside it —
    because the whole point of the skip is that independent evidence we
    did not author already exists.
    """
    others = [f for f in existing_test_files(root)
              if os.path.normpath(f) != os.path.normpath(SEED_BASENAME)]
    if others:
        log.info("[AcceptanceSeed] skipped: %d test file(s) already predate "
                 "the run and are stronger evidence (%s)",
                 len(others), ", ".join(sorted(others)[:3]))
        return False

    if not os.path.exists(path):
        return True

    state = seed_state(path)
    if state is None:
        # No header: written by a user, or by a build of this module
        # from before the header existed. Either way it is not ours to
        # replace, and it is still real pre-existing evidence.
        log.info("[AcceptanceSeed] skipped: %s predates the run and this "
                 "module did not write it", SEED_BASENAME)
        return False

    task_hash, body_hash, body = state
    if _fingerprint(body) != body_hash:
        log.info("[AcceptanceSeed] skipped: %s was edited since it was "
                 "seeded — whoever changed it owns it now", SEED_BASENAME)
        return False
    if task_hash == _fingerprint(task):
        # Same task, unedited, but structurally unable to judge this project:
        # reusing it guarantees the same false verdict on every rerun of the
        # prompt. Ours, untouched, and provably broken — so replace it.
        _defect = structural_defect_reason(body)
        if _defect:
            log.info("[AcceptanceSeed] re-seeding: %s was seeded from this "
                     "task but %s, so it can never judge it",
                     SEED_BASENAME, _defect)
            return True
        log.info("[AcceptanceSeed] skipped: %s was seeded from this same "
                 "task and still applies", SEED_BASENAME)
        return False

    log.info("[AcceptanceSeed] re-seeding: %s was written for a DIFFERENT "
             "task, so it is not evidence about this one", SEED_BASENAME)
    return True


def seed_acceptance_tests(task: str, root: str, llm_client,
                          language: str | None = None,
                          identity_task: str | None = None) -> str | None:
    """Write a task-derived suite and return its path, or None.

    Returns None — silently, and without writing — whenever the honest
    answer is "no independent check was established": a non-Python
    project, a task with nothing checkable in it, an unusable response, or
    a suite already present that is better evidence than this one.

    *task* is what the suite is WRITTEN from — the caller passes the
    enriched requirement, which is the fuller statement. *identity_task*
    is what decides whether a contract on disk belongs to this same task,
    and must be the user's raw text: the enriched form is LLM output and
    differs between runs of an identical prompt, so fingerprinting it
    would re-seed every run and discard a contract that was fine.
    Defaults to *task* for callers that have only one of them.
    """
    if language and language.lower() not in ("python", "py"):
        log.debug("[AcceptanceSeed] skipped: language is %s", language)
        return None
    if not task or not task.strip():
        return None

    identity = identity_task if (identity_task or "").strip() else task
    path = os.path.join(root, SEED_BASENAME)
    if not _should_seed(identity, root, path):
        return None

    src = _generate(llm_client, task)
    if src is None:
        return None
    _mock = mocking_reason(src)
    if _mock:
        log.warning("[AcceptanceSeed] response mocks the system under test "
                    "(%s) — refusing it. A contract that stubs the code "
                    "cannot be evidence about the code, and this run has "
                    "no seeded independent check", _mock)
        return None

    # Repaired rather than refused outright: a contract that asks a human
    # usually checks the RIGHT behaviours and only reads them the wrong
    # way, so the strong draft is worth recovering. Refused if the retry is
    # still interactive — keeping it would guarantee a failed run.
    _human = interactive_reason(src)
    if _human:
        log.info("[AcceptanceSeed] the contract needs a human (%s) — it can "
                 "never pass an unattended run, asking again for one that "
                 "asserts programmatically", _human)
        for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
            retry = _generate(llm_client, task,
                              extra=_INTERACTIVE_NOTE.format(marker=_human))
            if retry is None or mocking_reason(retry):
                continue
            _still = interactive_reason(retry)
            if _still is None:
                log.info("[AcceptanceSeed] the repaired contract runs "
                         "unattended — using it")
                src, _human = retry, None
                break
            _human = _still
        if _human:
            log.warning(
                "[AcceptanceSeed] the contract still needs a human (%s) after "
                "%d attempt(s) — refusing it. It would launch the app, wait "
                "for input nobody will give, and fail the run over code that "
                "may be perfect; no file is the honest outcome",
                _human, _REPAIR_ATTEMPTS)
            return None

    # Structurally unable to judge this project on this machine: a root that
    # climbs out of the project, or a signal the platform rejects. Refused
    # if every retry keeps it, for the interactive screen's reason — the
    # affected tests fail over any code, so keeping it guarantees a false
    # verdict.
    _defect = structural_defect_reason(src)
    if _defect:
        log.info("[AcceptanceSeed] the contract cannot judge this project on "
                 "this machine (%s) — asking again", _defect)
        for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
            retry = _generate(llm_client, task,
                              extra=_STRUCTURAL_NOTE.format(
                                  reason=_defect, platform=_platform_note(),
                                  contract=src.strip()))
            if retry is None or mocking_reason(retry) \
                    or interactive_reason(retry):
                continue
            _still = structural_defect_reason(retry)
            if _still is None:
                log.info("[AcceptanceSeed] the repaired contract can judge "
                         "this project — using it")
                src, _defect = retry, None
                break
            _defect = _still
        if _defect:
            log.warning(
                "[AcceptanceSeed] the contract still cannot judge this project "
                "(%s) after %d attempt(s) — refusing it. The affected tests "
                "would fail over any code; no file is the honest outcome",
                _defect, _REPAIR_ATTEMPTS)
            return None

    # Documentation wording, repaired before strength is judged — a README
    # test's assertions no longer count as substantive, so judging strength
    # first would send a second, differently-worded complaint about the same
    # test. Measured 2026-09-13: the behaviour test passed, the README test
    # failed on Markdown bold between "Python" and "3.10", and a working
    # snake game exited 1.
    #
    # Kept rather than refused if every retry still reads the docs: most
    # wording checks pass by luck, the behaviour tests beside them are real
    # evidence, and refusing would leave nothing. Said out loud instead.
    from .seed_strength import DOCUMENTATION_NOTE, documentation_grep_reason
    _docs = documentation_grep_reason(src)
    if _docs:
        log.info("[AcceptanceSeed] the contract asserts on documentation "
                 "wording (%s) — asking again for one that checks behaviour",
                 _docs)
        for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
            retry = _generate(llm_client, task,
                              extra=DOCUMENTATION_NOTE.format(
                                  reason=_docs, contract=src.strip()))
            if retry is None or mocking_reason(retry) \
                    or interactive_reason(retry):
                continue
            _still = documentation_grep_reason(retry)
            if _still is None:
                log.info("[AcceptanceSeed] the repaired contract leaves the "
                         "documentation alone — using it")
                src, _docs = retry, None
                break
            _docs = _still
        if _docs:
            log.warning("[AcceptanceSeed] the contract still asserts on "
                        "documentation wording after %d attempt(s) (%s) — "
                        "keeping it, but it can fail a correct README over "
                        "phrasing or markup", _REPAIR_ATTEMPTS, _docs)

    # Strength, judged once and repaired once. Measured across three runs
    # of one prompt: 2 substantive tests, then 1 that asserted only that
    # the process had not exited — which passes over any program that
    # starts. Accurately measuring a weak instrument still reports a weak
    # measurement as a strong claim, so ask again with the specific
    # complaint, the same shape as `repair_verify_commands`.
    from .seed_strength import REPAIR_NOTE, weak_contract_reason
    _weak = weak_contract_reason(src)
    if _weak:
        log.info("[AcceptanceSeed] first contract is too weak (%s) — asking "
                 "again for one that can fail", _weak)
        # Two attempts, because an unusable RESPONSE is not the same as a
        # weak contract and must not be read as one. Measured 2026-08-18
        # 09:04: the repair came back unparseable, the single attempt was
        # spent, and the run silently kept a contract with zero
        # discriminating assertions — the strongest draft the model could
        # have written was never asked for a second time.
        for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
            retry = _generate(llm_client, task,
                              extra=REPAIR_NOTE.format(reason=_weak))
            if retry is None:
                log.info("[AcceptanceSeed] repair attempt %d/%d came back "
                         "unusable — that is a bad response, not a weak "
                         "contract, so asking again", _attempt,
                         _REPAIR_ATTEMPTS)
                continue
            if mocking_reason(retry):
                log.info("[AcceptanceSeed] repair attempt %d/%d mocks the "
                         "system under test — asking again", _attempt,
                         _REPAIR_ATTEMPTS)
                continue
            _weak_after = weak_contract_reason(retry)
            if _weak_after is None:
                log.info("[AcceptanceSeed] the repaired contract asserts real "
                         "behaviour — using it")
                src, _weak = retry, None
                break
            if _substantive_count(retry) > _substantive_count(src):
                # Not strong, but stronger; keeping the better of the two
                # is never worse than keeping the first.
                src, _weak = retry, _weak_after
    if _weak:
        # Kept rather than refused: a weak check that runs still catches a
        # crashing artifact, and refusing would trade a shallow instrument
        # for none at all. Said out loud, because "independent" will be
        # reported about it.
        log.warning("[AcceptanceSeed] the seeded contract remains SHALLOW "
                    "(%s) — this run's independent evidence can catch a "
                    "broken build but not wrong behaviour", _weak)

    if not src.endswith("\n"):
        src += "\n"
    # The body is hashed as it will sit on disk — everything after the
    # header line — so a later run can tell "unchanged since we wrote it"
    # from "someone has taken this over".
    body = "\n" + src
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_header(identity, body) + body)
    except OSError as exc:
        log.warning("[AcceptanceSeed] could not write %s: %s", path, exc)
        return None

    log.info("[AcceptanceSeed] wrote %s from the task text, before any step "
             "ran — it counts as independent evidence for exactly as long "
             "as the run leaves it byte-identical", SEED_BASENAME)
    return path


# ── Does the contract actually RUN? ──────────────────────────────────
#
# Everything above validates the contract STATICALLY: `_looks_like_a_suite`
# compiles it, `mocking_reason` reads its imports, `weak_contract_reason`
# walks its AST. Not one of those executes it, and every measured failure
# of a seeded contract has been a RUNTIME failure:
#
#   `ambient_lights[0].getColor()[:3]`   TypeError — LVecBase4f has no slice
#   `self.win.requestProperties(...)`    AttributeError — it is a
#                                        GraphicsBuffer, not a window
#
# Both compile perfectly. The reason they cluster in framework
# introspection is structural: the contract is written before the code
# exists, so it cannot name the artifact's own API, and the only
# vocabulary it can predict is the framework's — which is precisely the
# most environment- and version-fragile code there is to write blind.
#
# The contract cannot be executed when it is written; there is nothing to
# import yet. The first moment it CAN be is as soon as the code exists,
# and that is where this runs — instead of at the very end, which is
# where a broken contract used to surface, after the whole budget was
# spent. Measured 2026-09-08: one such contract failed a step's own gate
# and cost 30 turns, an escalation and 402k tokens over a game that
# scored 18/18 against external probes.

_UNRUNNABLE_NOTE = """

IGNORE the instruction above to write a NEW contract. This is a REPAIR of
one you have already written, and the code it tests now EXISTS.

Here is your contract exactly as it stands:

--- BEGIN CONTRACT ---
{contract}
--- END CONTRACT ---

Executed against the project as it now stands, it CRASHED instead of
judging it:

{output}

Return the COMPLETE corrected file. Hard constraints:

  - Keep the same behaviours under test, and keep them just as strict.
    Do NOT delete, weaken or skip an assertion to make the error go
    away, and do NOT wrap anything in try/except or unittest.skip.
  - Do NOT replace the contract with a test that calls `self.fail(...)`,
    `self.skipTest(...)` or otherwise refuses to check anything. The
    project exists and your contract already imports it successfully;
    a contract that gives up is worse than the crashing one.
  - The defect is in HOW the check inspects the program, not in WHAT it
    checks.

This kind of crash almost always comes from reaching into the
FRAMEWORK's internals — objects whose API you had to guess, or that only
exist when there is a real window or display. Prefer asserting on the
program's OWN public API and observable state (its classes, methods and
returned values) over the framework's internal objects. If a behaviour
can only be observed through the framework, assert on something the
program itself exposes instead.
"""

# unittest reports a crash as an ERROR and a judged disagreement as a
# FAILURE. Only the first says the instrument is broken; a failure may
# simply be code that is not finished yet, which at this point in the run
# is the normal state of affairs.
_FAILED_TAIL_RE = re.compile(r"FAILED \(([^)]*)\)")
_ERRORS_RE = re.compile(r"errors=(\d+)")

# The module under test does not exist yet — the contract is fine, the
# code has just not been written. Not a defect, and not this function's
# business; try again after the next wave.
_NOT_READY_RE = re.compile(
    r"ModuleNotFoundError|No module named|ImportError|cannot import name")

# An import error is only how a contract that IMPORTS the project reports a
# missing project. One that reaches it through a SUBPROCESS reports the same
# fact as an ordinary assertion about a return code, and the import
# signature never fires. Measured 2026-09-12: a contract whose harness did
# `runpy.run_path("main.py")` inside `subprocess.run(...)` was executed after
# wave 1 — a CMD step, before any code-producing step had run — and reported
# `FAILED (failures=1)` over a `FileNotFoundError` for a `main.py` no step
# had written yet. Scoped to a missing PYTHON SOURCE file, because a
# contract complaining that a module is absent is "not built yet" while one
# complaining that a save-file or an export is absent has genuinely judged.
_NOT_BUILT_RE = re.compile(
    r"(?:FileNotFoundError|No such file or directory|can't open file)"
    r"[^\n]*\.py")


def _not_ready_reason(output: str):
    """Short reason the contract could not reach the project yet, or None."""
    found = _NOT_READY_RE.search(output or "")
    if found:
        return found.group(0)
    found = _NOT_BUILT_RE.search(output or "")
    return "a project source file does not exist yet" if found else None


# The `.py` rule above is a guess at "not built yet"; the plan is the fact.
# Measured 2026-09-15, a 12-step Snake run on 0.8.1: after wave 6 the
# contract reported `FAILED (failures=3, errors=1)`, and the one error was
#
#     Path("README.md").stat().st_size
#     FileNotFoundError: [WinError 2] ... cannot find the file specified: 'README.md'
#
# README.md was the declared target of step 5.2, which ran in wave 7. The
# error was read as a broken instrument, two repair attempts spent 26k
# tokens and blocked the run for three minutes, both were rejected — and
# the unmodified contract passed 5/5 at the end. A missing file that a step
# still to run declares as its target is code not written yet, whatever its
# extension.
_MISSING_FILE_LINE_RE = re.compile(
    r"FileNotFoundError|No such file or directory|cannot find the "
    r"(?:file|path) specified|can't open file", re.IGNORECASE)


def _names_pending_target(output: str, pending_targets) -> str | None:
    """A pending step's target named by a missing-file error, or None.

    Judged per line, so a pending target mentioned in some unrelated
    assertion message does not excuse a genuine crash elsewhere.
    """
    if not output or not pending_targets:
        return None
    from ..paths import strip_dot_slash
    wanted = {strip_dot_slash(t.replace("\\", "/")).lower()
              for t in pending_targets if t and t.strip()}
    wanted.discard("")
    for line in output.splitlines():
        if not _MISSING_FILE_LINE_RE.search(line):
            continue
        norm = line.replace("\\\\", "/").replace("\\", "/").lower()
        for target in wanted:
            # A path is named when it ends a quoted or separated token:
            # 'README.md', C:/proj/tests/test_game.py, tests/test_game.py
            if re.search(r"(?:^|[\s'\"/:])" + re.escape(target) + r"(?=['\"\s]|$)",
                         norm):
                return target
    return None


# One path -> repair attempts already spent. A contract that still cannot
# run afterwards is left alone: the end-of-run path reports it as
# inconclusive, which is the honest outcome and already implemented.
_REPAIR_STATE: dict = {}

# Paths that have already RUN and judged, so the per-wave check does not
# spend a subprocess (and, for a graphical contract, a window) every wave
# re-learning the same thing. Deliberately separate from `_REPAIR_STATE`:
# one is a repair budget, the other is "no new information this wave", and
# the measured defect was exactly a single variable answering both.
_JUDGED: set = set()

# The most recent `(ok, output)` this module observed for a contract, so the
# end-of-run verdict does not pay for a second identical subprocess. Only
# the FINAL check records here. A per-wave run measures an unfinished tree
# and has no business reaching the verdict — the saving is sound only
# because the final check runs immediately before `run_pre_existing_tests`
# with nothing in between, and a contract that plays a real session is not
# cheap (4.4s in the measured run).
_LAST_RUN: dict = {}


def reset_contract_repairs() -> None:
    """Forget repair bookkeeping — for a second run in one process."""
    _REPAIR_STATE.clear()
    _JUDGED.clear()
    _LAST_RUN.clear()


def last_contract_run(root: str):
    """``(ok, output)`` from the most recent contract execution, or None.

    Only valid for the file as it currently sits on disk — every path that
    rewrites or restores the contract invalidates it.
    """
    return _LAST_RUN.get(os.path.join(root, SEED_BASENAME))


def _errors_reported(output: str) -> int:
    """How many tests CRASHED, as opposed to failing an assertion."""
    tail = _FAILED_TAIL_RE.search(output or "")
    if not tail:
        return 0
    found = _ERRORS_RE.search(tail.group(1))
    return int(found.group(1)) if found else 0


def _run_contract(executor, root: str, record: bool = False):
    cmd = "python -m unittest " + SEED_BASENAME
    try:
        result = executor.run_command(cmd, timeout=180, cwd=root)
    except TypeError:
        # Executors that take no cwd — the caller is already rooted there.
        result = executor.run_command(cmd, timeout=180)
    if record:
        _LAST_RUN[os.path.join(root, SEED_BASENAME)] = result
    return result


def verify_contract_runs(executor, root: str, llm_client, task: str,
                         identity_task: str = None,
                         max_repairs: int = 1,
                         final: bool = False,
                         pending_targets=None):
    """Execute the seeded contract; repair it once if it cannot run.

    *pending_targets* are the files declared by plan steps that have not run
    yet. A crash on one of them missing is deferred like an import error —
    the file is simply not written yet — rather than repaired.

    Returns ``(relative path, new sha256)`` when the file was rewritten —
    the caller MUST update its pre-existing-test snapshot with it, or the
    repair would read as "the agent edited the contract" and forfeit the
    very independence this is protecting.

    Returns ``None`` whenever nothing was changed, including every case
    where the honest answer is "not yet": no contract, not ours, the code
    it imports does not exist, or the contract ran and merely disagreed
    with the code. An assertion FAILURE is deliberately never repaired —
    at this point in the run the code is usually incomplete, and rewriting
    the instrument because it says so is exactly the cheat this module
    exists to prevent.
    """
    if executor is None or llm_client is None:
        return None
    path = os.path.join(root, SEED_BASENAME)
    state = seed_state(path)
    if state is None:
        return None                       # absent, or not ours to touch
    repairs_left = _REPAIR_STATE.get(path, 0) < max_repairs
    if not final and (not repairs_left or path in _JUDGED):
        return None
    # The FINAL check always runs it, whatever the mid-run bookkeeping says.
    # That costs nothing net: `run_pre_existing_tests` needs a fresh result
    # and would otherwise pay for this very subprocess itself. Measured
    # 2026-09-12, this is the whole defect — a wave-1 run (before any
    # code-producing step) retired the check, so the final one never
    # executed, and its two-minute-old result was handed to the verdict as
    # if it described the finished tree.

    ok, out = _run_contract(executor, root, record=final)
    if ok:
        # It runs. Nothing more to check this wave, and re-running it every
        # wave would cost a subprocess (and, for a graphical project, a
        # window) for no new information.
        _JUDGED.add(path)
        return None
    if not final:
        _pending = _names_pending_target(out, pending_targets)
        if _pending:
            log.debug("[AcceptanceSeed] contract needs %s, which a later step "
                      "of the plan will write — deferring the runnability "
                      "check", _pending)
            return None
    if _not_ready_reason(out):
        if not final:
            log.debug("[AcceptanceSeed] contract cannot reach the project "
                      "yet (%s) — deferring the runnability check",
                      _not_ready_reason(out))
            return None
        # There is no later wave to defer to. Measured 2026-09-12: the
        # contract became importable only during the post-wave phases, so
        # all seven per-wave checks deferred, the check never ran once, and
        # the first execution was the verdict itself — with no chance to
        # repair. "Not built yet" is a true statement mid-run and a
        # meaningless one here, so say what is actually wrong instead.
        log.warning("[AcceptanceSeed] the contract still cannot reach the "
                    "project at the end of the run — it can never have "
                    "judged anything:\n%s",
                    "\n".join((out or "").strip().splitlines()[-10:]))
        _JUDGED.add(path)
        return None
    if _errors_reported(out) == 0:
        # It ran and judged. Whether it is RIGHT is the end-of-run
        # question, and a seeded contract may not convict the code anyway.
        # This must NOT spend the repair budget: a mid-run disagreement is
        # the least conclusive state there is, because the code is
        # incomplete by construction, and retiring the check on it is what
        # stopped the final one from ever looking.
        _JUDGED.add(path)
        return None

    if not repairs_left:
        return None                       # ran, crashed, nothing left to try
    _REPAIR_STATE[path] = _REPAIR_STATE.get(path, 0) + 1
    tail = "\n".join((out or "").strip().splitlines()[-25:])
    log.warning("[AcceptanceSeed] the seeded contract CRASHES rather than "
                "judging the code — it is the instrument that is broken, "
                "not necessarily the artifact. Repairing it now rather "
                "than discovering this at the end of the run:\n%s", tail)

    try:
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
    except OSError as exc:
        log.warning("[AcceptanceSeed] could not read the contract: %s", exc)
        return None
    original_strength = _substantive_count(state[2])

    # Asked more than once, for the reason the weakness repair above
    # already documents: an unusable RESPONSE is not the same as a repair
    # that was judged and rejected, and must not spend the budget as if
    # it were. Measured live 2026-09-09: the single attempt came back
    # having burned all 16,384 output tokens on reasoning ("Response hit
    # the output-token limit"), the contract stayed broken, and the run
    # ended exactly as it would have without any of this.
    identity = identity_task if (identity_task or "").strip() else task
    latest = tail
    # Why the PREVIOUS attempt was rejected. Sending an identical prompt
    # twice asks a blind question: measured live 2026-09-09, both attempts
    # came back weaker than the original (0 then 1 substantive assertion
    # against 2) because nothing told the model that the first had been
    # refused, or why. The still-crashing branch below already feeds its
    # error back; every other rejection now does the same.
    rejected = ""
    for _attempt in range(1, _REPAIR_ATTEMPTS + 1):
        # The contract itself goes in the prompt. Without it the model was
        # being asked to fix a file it could not see, under a base prompt
        # whose framing is "before any code exists" — and it did the only
        # thing that framing allows: it regenerated from scratch, decided
        # the task named no importable module, and returned a contract
        # whose single test was `self.fail(...)`. Its own prior output is
        # safe to show; the ARTIFACT is not, because a contract shaped by
        # the code it judges is no longer independent of it.
        candidate = _generate(
            llm_client, task,
            extra=_UNRUNNABLE_NOTE.format(output=latest,
                                          contract=state[2].strip())
                  + rejected)
        if candidate is None:
            log.info("[AcceptanceSeed] runnability repair %d/%d came back "
                     "unusable — that is a bad response, not a verdict on "
                     "the repair, so asking again", _attempt,
                     _REPAIR_ATTEMPTS)
            continue
        _mock = mocking_reason(candidate)
        if _mock:
            log.info("[AcceptanceSeed] runnability repair %d/%d mocks the "
                     "system under test (%s) — asking again", _attempt,
                     _REPAIR_ATTEMPTS, _mock)
            rejected = (
                "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it mocked or "
                "stubbed the system under test (%s). A contract that stubs "
                "the code cannot be evidence about the code — drive the "
                "real objects." % _mock)
            continue
        # A repair is free to introduce this defect even when the original
        # did not, so the runnability path screens for it too.
        _human = interactive_reason(candidate)
        if _human:
            log.info("[AcceptanceSeed] runnability repair %d/%d needs a "
                     "human (%s) — asking again", _attempt,
                     _REPAIR_ATTEMPTS, _human)
            rejected = _INTERACTIVE_NOTE.format(marker=_human)
            continue
        # Same reasoning for documentation wording — but only a repair that
        # INTRODUCES it is refused. A contract kept with README checks after
        # its seeding retries would otherwise become unrepairable, and a
        # crash is the more urgent of the two defects.
        from .seed_strength import documentation_grep_reason
        _docs = documentation_grep_reason(candidate)
        if _docs and not documentation_grep_reason(state[2]):
            log.info("[AcceptanceSeed] runnability repair %d/%d started "
                     "asserting on documentation wording (%s) — asking again",
                     _attempt, _REPAIR_ATTEMPTS, _docs)
            rejected = (
                "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it added assertions "
                "on documentation wording (%s). Do not read README or any "
                ".md file — fix the crash, keep the behaviour checks." % _docs)
            continue
        # Refused whether introduced or inherited: unlike wording checks, a
        # root outside the project or a signal this platform rejects is never
        # right, and fixing it alongside the crash costs nothing extra.
        _defect = structural_defect_reason(candidate)
        if _defect:
            log.info("[AcceptanceSeed] runnability repair %d/%d cannot judge "
                     "this project (%s) — asking again", _attempt,
                     _REPAIR_ATTEMPTS, _defect)
            rejected = _STRUCTURAL_NOTE.format(reason=_defect,
                                               platform=_platform_note(),
                                               contract=candidate.strip())
            continue
        # A crash is trivially "fixed" by asserting less. Rank the repair
        # the same way the weakness repair does, and refuse a trade of
        # correctness for coverage.
        if _substantive_count(candidate) < original_strength:
            log.info("[AcceptanceSeed] runnability repair %d/%d is weaker "
                     "than the original (%d substantive assertion(s) vs "
                     "%d) — that trades a broken check for a shallow one, "
                     "so asking again", _attempt, _REPAIR_ATTEMPTS,
                     _substantive_count(candidate), original_strength)
            rejected = (
                "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it kept only %d "
                "meaningful assertion(s) where the contract above has %d. "
                "You removed checks instead of fixing the crash. Reproduce "
                "EVERY assertion from the contract above, changing only the "
                "expression that raised."
                % (_substantive_count(candidate), original_strength))
            continue

        # Static checks passed, so prove it the only way that counts: run
        # it. The write has to happen first — the contract is executed by
        # its own name, from the project root, exactly as the end-of-run
        # check will execute it.
        if not candidate.endswith("\n"):
            candidate += "\n"
        body = "\n" + candidate
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_header(identity, body) + body)
        except OSError as exc:
            log.warning("[AcceptanceSeed] could not write the repair: %s",
                        exc)
            return None

        ok_after, out_after = _run_contract(executor, root, record=final)
        if ok_after or _errors_reported(out_after) == 0:
            from .evidence import _digest
            digest = _digest(path)
            if digest is None:
                return None
            log.info("[AcceptanceSeed] the contract now runs (%s) — "
                     "repaired before the run spent its budget proving a "
                     "broken instrument right",
                     "and passes" if ok_after else "and judges the code")
            return SEED_BASENAME, digest

        # Still crashing. Restore the original — two broken contracts are
        # not better than one, and the original at least had whatever
        # strength the seeding checks approved — then feed the NEW error
        # back. Measured live 2026-09-09: a candidate that passed every
        # static check and then crashed ended the repair outright, which
        # threw away the most useful signal available (a fresh, specific
        # error about the attempt just made) while attempts remained.
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(original)
        except OSError:
            pass
        # `_LAST_RUN` now describes the REPAIRED file, which is no longer the
        # one on disk. Handing that to the verdict would report a result for
        # bytes nobody can read.
        _LAST_RUN.pop(path, None)
        latest = "\n".join((out_after or "").strip().splitlines()[-25:])
        rejected = ("\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: it still "
                    "crashed, with the error shown above. Fix that one, and "
                    "keep every assertion.")
        log.info("[AcceptanceSeed] runnability repair %d/%d still crashes — "
                 "restored the original and asking again with the new "
                 "error:\n%s", _attempt, _REPAIR_ATTEMPTS, latest)

    log.warning("[AcceptanceSeed] no usable repair after %d attempt(s) — "
                "keeping the original contract; this run's seeded evidence "
                "will be reported as inconclusive", _REPAIR_ATTEMPTS)
    return None

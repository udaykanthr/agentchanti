"""Benchmark task definitions for the agent-loop A/B harness.

Each task is a dict:
  id           — short slug used in output and workdir names
  task         — the plain-English task given to the pipeline
  files        — seed files written into the workdir before the run
  success_cmds — ground-truth shell commands; ALL must exit 0 for the
                 task to count as succeeded (independent of what the
                 pipeline claims)
  language     — expected project language (informational)

Tasks name their target files explicitly so ground-truth checks don't
have to guess what the model called things.
"""

from pathlib import Path

BUGGY_CALC = '''\
"""Tiny calculator module."""


def add(a, b):
    return a - b


def multiply(a, b):
    return a * b
'''

CALC_TESTS = '''\
from calc import add, multiply


def test_add():
    assert add(2, 3) == 5


def test_multiply():
    assert multiply(2, 3) == 6
'''

STRINGUTILS = '''\
"""String helpers."""


def slugify(text):
    return "-".join(text.lower().split())
'''

SHAPES = '''\
"""Area helpers."""
import math


def circle_area(radius):
    if radius < 0:
        raise ValueError("radius must be >= 0")
    return math.pi * radius ** 2


def rect_area(width, height):
    if width < 0 or height < 0:
        raise ValueError("dimensions must be >= 0")
    return width * height
'''

MESSY = '''\
import os, sys
import json


def greet(name):
    unused = 42
    message = "hello " + name
    return message
'''


def _dj(args: str) -> str:
    """manage.py ground-truth command for the django-webapp task.

    Prefers the project venv's python when the pipeline created one
    (Django is installed there, not in the harness env), falling back
    to the ambient python. Windows cmd syntax — the harness runs
    success_cmds with shell=True on the host.
    """
    return (
        "cd spacious_site && "
        "(if exist venv\\Scripts\\python.exe "
        f"(venv\\Scripts\\python.exe manage.py {args}) "
        f"else (python manage.py {args}))"
    )


PACMAN_STRICT = """\
Build a Pac-Man clone in Python with Pygame.

Public API - exact names. An external harness drives these, so do not rename
anything below and do not require any argument that is not listed:
  game.Game(seed: int = 0)             constructs a playable game, no window
  Game.advance(dt: float) -> None      advance the simulation by dt SECONDS
  Game.state -> str                    "start" | "playing" | "win" | "game_over"
  Game.start() -> None                 leave the start screen
  Game.press(direction: str) -> None   "up" | "down" | "left" | "right"
  Game.entities() -> list              each entity exposes .tile -> (x, y) ints
  Game.pellets_remaining() -> int
  Game.map.is_wall(x: int, y: int) -> bool
  `python main.py --headless --frames 300` must run and exit 0

Behaviour checked externally, not by your tests:
1. No entity's .tile is ever a wall, at any dt drawn from 0.001..0.5.
2. dt must genuinely scale motion: simulating 10 seconds must not leave every
   entity on the same tile as simulating 0.1 seconds.
3. press("left") then advancing one second must change the player's tile,
   unless a wall blocks that direction. The same for the other three.
4. pellets_remaining() starts above zero, never increases, and comes down as
   the player moves across pellet tiles. state is "win" only at zero.
5. Game.state is always one of the four documented strings.

Constraints:
- The player and all four ghosts spawn on walkable tiles, and no walkable
  region of the maze is cut off from the player's.
- Ghosts re-evaluate direction at tile centres, never by floating-point
  equality with a boundary.
- Four ghosts with distinct behaviour (chase, random, patrol), pellets and
  power pellets, mouth animation while eating, and a start screen.
- No external assets - draw with rectangles and circles.
- Your tests must not reposition entities, assign to state, or stub any
  method in order to pass. Drive the game only through the API above.
- No parameter may be accepted and then ignored by its method body.

Deliver working code and a passing `python -m unittest`.
"""


def _venv_py(args: str) -> str:
    """Run the project's own interpreter when the pipeline made one.

    Same reasoning as `_dj`: a plan step that creates a venv installs
    pygame into it, not into the harness env, and a ground-truth check
    run against the wrong interpreter reports a failure that is really a
    missing import. Windows cmd syntax - success_cmds run with
    shell=True on the host.
    """
    return ("(if exist venv\\Scripts\\python.exe "
            f"(venv\\Scripts\\python.exe {args}) "
            f"else (python {args}))")


def _probe(check: str) -> str:
    """A ground-truth probe, by absolute path out of the repo.

    Deliberately NOT seeded into the task's `files`: the agent must not
    be able to read, narrowly satisfy, or "repair" the thing judging it.
    Every failure in this family so far came from acceptance the agent
    authored itself.
    """
    probe = Path(__file__).resolve().parent / "probes" / "pacman_strict.py"
    return _venv_py(f'"{probe}" {check}')


TASKS = [
    {
        "id": "func-noloop",
        "task": ("Create peak.py with a function print_pattern(n) that "
                 "prints the numbers 1 up to n, then n+1, then back down "
                 "from n to 1, comma-separated on one line, WITHOUT using "
                 "any for or while loops. Add pytest tests in test_peak.py."),
        "files": {},
        "success_cmds": [
            "python -m pytest -q",
            "python -c \"from peak import print_pattern; print_pattern(3)\"",
        ],
        "language": "python",
    },
    {
        "id": "bugfix",
        "task": ("The tests in this project fail. Find the bug, fix it, "
                 "and make sure all tests pass."),
        "files": {"calc.py": BUGGY_CALC, "test_calc.py": CALC_TESTS},
        "success_cmds": ["python -m pytest -q"],
        "language": "python",
    },
    {
        "id": "feature",
        "task": ("Add a function reverse_words(text) to stringutils.py that "
                 "returns the words of the input in reverse order joined by "
                 "single spaces. Add pytest tests for it in "
                 "test_stringutils.py."),
        "files": {"stringutils.py": STRINGUTILS},
        "success_cmds": [
            "python -m pytest -q",
            "python -c \"from stringutils import reverse_words; "
            "assert reverse_words('a b c') == 'c b a'\"",
        ],
        "language": "python",
    },
    {
        "id": "cmd-recovery",
        "task": ("Run the ruff linter on messy.py and fix every issue it "
                 "reports until ruff passes cleanly."),
        "files": {"messy.py": MESSY},
        "success_cmds": ["python -m ruff check messy.py"],
        "language": "python",
    },
    {
        "id": "tests-for-existing",
        "task": ("Write pytest tests for shapes.py in test_shapes.py "
                 "covering circle_area and rect_area, including the "
                 "negative-input ValueError cases. Run them."),
        "files": {"shapes.py": SHAPES},
        "success_cmds": ["python -m pytest -q"],
        "language": "python",
    },
    {
        # The framework-wiring class where content-mode runs repeatedly
        # died on cross-file defaults (URL namespaces, {% load static %},
        # LOGIN_URL) — the case the plan_mode A/B actually decides.
        "id": "django-webapp",
        "task": ("create a django application in a new folder named "
                 "spacious_site with a responsive spacious homepage at / "
                 "(header, large herobanner, price list component, large "
                 "footer), login, signup and forgot password screens, and "
                 "by default logged in users should auto redirect to a "
                 "dashboard page at /dashboard/. Add Django tests covering "
                 "the pages and the redirect behaviour."),
        "files": {},
        "success_cmds": [
            _dj("check"),
            _dj("test --noinput"),
            # Behaviour probes independent of the generated tests:
            # anonymous homepage renders; anonymous dashboard redirects.
            # ALLOWED_HOSTS override: outside the test runner Django does
            # not auto-allow the client's 'testserver' host.
            _dj('shell -c "from django.conf import settings; '
                "settings.ALLOWED_HOSTS = ['*']; "
                "from django.test import Client; import sys; "
                "r = Client().get('/'); "
                'sys.exit(0 if r.status_code == 200 else 1)"'),
            _dj('shell -c "from django.conf import settings; '
                "settings.ALLOWED_HOSTS = ['*']; "
                "from django.test import Client; import sys; "
                "r = Client().get('/dashboard/'); "
                'sys.exit(0 if r.status_code in (301, 302) else 1)"'),
        ],
        "language": "python",
    },
    {
        # The strict variant of the Pac-Man family. The loose wording
        # ("assert these invariants in tests") kept passing while the
        # artifact was broken, because it handed acceptance to the agent:
        # one run's suite relocated four ghosts in setUp to hide a
        # spawn-inside-a-wall crash, another ran 700 iterations of which
        # 17 simulated anything, a third accepted `dt` and never read it.
        # So this task pins the public API by name and judges it only
        # from _PROBE, which lives in the repo and is never seeded into
        # the workdir.
        "id": "pacman-strict",
        "task": PACMAN_STRICT,
        "files": {},
        "success_cmds": [
            # The agent's own claim, kept alongside ground truth rather
            # than standing in for it.
            _venv_py("-m unittest -v"),
            # It has to run, not just import. `[SmokeTest] No runnable
            # Python entry point — skipping` fired on every earlier run
            # in this family, so nothing ever launched the artifact.
            _venv_py("main.py --headless --frames 300"),
            _probe("spawns"),
            _probe("walls"),
            _probe("dt"),
            _probe("input"),
            _probe("pellets"),
        ],
        "language": "python",
    },
    {
        "id": "multi-file",
        "task": ("Create a package mathx: mathx/core.py defining add(a, b) "
                 "and mul(a, b), and mathx/__init__.py re-exporting both. "
                 "Add pytest tests in test_mathx.py that import from mathx "
                 "directly."),
        "files": {},
        "success_cmds": [
            "python -m pytest -q",
            "python -c \"from mathx import add, mul; "
            "assert add(2, 2) == 4 and mul(3, 3) == 9\"",
        ],
        "language": "python",
    },
]

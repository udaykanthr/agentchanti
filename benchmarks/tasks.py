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

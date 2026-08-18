"""Can this seeded contract tell a working artifact from a stub?

`shallow_gate_reason` asks the question of a plan's ``verify:``. This
asks it of the acceptance contract, which had no equivalent — so its
strength was luck of the draw, and the draw is wide. Three runs of the
identical prompt on two models produced:

* 23 tests that all mocked the system under test (refused since),
* 2 tests asserting rendered node colours, camera position, and that the
  snake's cell changed after an arrow event,
* **1 test asserting only that the process stayed alive for a moment.**

The third earned `Evidence: independent (pre-existing-tests)` honestly —
it really did run and really did pass. It would also pass over any
Panda3D script that starts, including one with none of the behaviour the
task describes. Measuring a weak instrument accurately still reports a
weak measurement as a strong claim.

WHAT COUNTS AS STRENGTH
-----------------------
An assertion is substantive when it can distinguish two different
implementations. That rules out three shapes seen in the wild:

* **Liveness** — start a subprocess, assert it has not exited. True of
  anything that starts.
* **Existence** — ``hasattr``, ``assertIsNotNone`` on something just
  constructed, ``assertTrue(module)``. True as long as the file parses.
* **Tautology** — ``assertTrue(True)``, or membership in a set that
  contains every possible value.

And it requires at least one of:

* a comparison against a **concrete literal** the task states, or
* a **strict relation between two observed values** (the rule the seed
  prompt already states for "something must CHANGE"), or
* an assertion about a **collection's contents or size**.

The check is deliberately blunt and easy to satisfy, like the gate one:
a contract that asserts one real thing passes. The cost of being wrong
in the strict direction is one extra generation call.
"""

from __future__ import annotations

import ast
from typing import Optional

# Assertions that say nothing about behaviour on their own.
_WEAK_ASSERTS = {
    "assertTrue", "assertFalse", "assertIsNotNone", "assertIsNone",
    "assertIsInstance", "assertNotIsInstance", "assertIn", "assertNotIn",
    "assertHasAttr",
}

# Assertions that compare two things, which is where discrimination lives.
_COMPARING_ASSERTS = {
    "assertEqual", "assertNotEqual", "assertLess", "assertLessEqual",
    "assertGreater", "assertGreaterEqual", "assertAlmostEqual",
    "assertNotAlmostEqual", "assertCountEqual", "assertListEqual",
    "assertDictEqual", "assertSetEqual", "assertTupleEqual",
    "assertSequenceEqual", "assertRegex",
}

# Names whose presence marks an assertion as being about process liveness
# rather than behaviour.
_LIVENESS_NAMES = {"poll", "returncode", "pid", "is_alive", "wait"}


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_literal(node.operand)
    return False


def _mentions_liveness(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _LIVENESS_NAMES:
            return True
        if isinstance(sub, ast.Name) and sub.id in _LIVENESS_NAMES:
            return True
    return False


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


# Reading the module's own text instead of running it. A test that greps
# source is weak and brittle at once: it passes a stub that merely
# MENTIONS the right words, and it fails correct code that names things
# differently.
#
# Measured 2026-08-18 09:26. A contract asserted
#
#     self.assertTrue("reset" in values or "restart" in values ...,
#                     "Boundary or self collisions must reset the game")
#
# over a token set extracted from the module's AST. The artifact defines
# `reset_game` and calls it on every collision — 16 resets under an
# external probe — but the token is `reset_game`, membership is exact,
# and the run was failed by a substring that was not a whole token. The
# same contract also "proved" scoring with
# `any(isinstance(n, ast.AugAssign) ...)`, which any `x += 1` anywhere
# satisfies.
#
# This is the lesson `verify_dt_invariance` already encodes by reserving
# an exit code for could-not-verify: generated projects share no
# vocabulary, so asserting on their vocabulary is not verification.
_SOURCE_INSPECTION_CALLS = {
    "getsource", "getsourcefile", "getsourcelines", "getfile",
    "parse", "walk", "dump", "unparse", "getlines",
}


def _inspects_source(node: ast.AST) -> bool:
    """Does this subtree read the program's text rather than run it?

    Only the unambiguous readers count. A bare ``__file__`` deliberately
    does not: iteration 5's contract used it to LOCATE a sibling script
    to launch as a subprocess, which is running the program, not reading
    it — flagging that would mislabel a legitimate test and spend a
    repair cycle explaining the wrong thing.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in _SOURCE_INSPECTION_CALLS:
            # `ast.parse`, `inspect.getsource`, `linecache.getlines`.
            value = sub.value
            if isinstance(value, ast.Name) and value.id in (
                    "ast", "inspect", "linecache"):
                return True
    return False


def source_inspecting_tests(tree: ast.AST) -> list[str]:
    """Names of test functions that read source instead of driving it."""
    out: list[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
                and _inspects_source(node)):
            out.append(node.name)
    return out


def _excluded_node_ids(tree: ast.AST) -> set[int]:
    """Every node inside a source-inspecting test.

    Excluded at FUNCTION granularity rather than per assertion: a test
    that parses the module is a source-inspection test, and its
    assertions are about text whether or not each one names the parse.
    """
    excluded: set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
                and _inspects_source(node)):
            for sub in ast.walk(node):
                excluded.add(id(sub))
    return excluded


def _substantive_assertions(tree: ast.AST) -> int:
    """How many assertions could tell two implementations apart."""
    count = 0
    skip = _excluded_node_ids(tree)
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Assert):
            # A bare `assert` on anything but a constant is a real claim.
            if not _is_literal(node.test) and not _mentions_liveness(node.test):
                count += 1
            continue
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in _COMPARING_ASSERTS and name not in _WEAK_ASSERTS:
            continue
        if _mentions_liveness(node):
            continue                      # "it did not exit" is not behaviour
        if name in _COMPARING_ASSERTS:
            args = [a for a in node.args]
            if len(args) >= 2:
                # Two literals compared to each other assert nothing about
                # the code; everything else compares something observed.
                if _is_literal(args[0]) and _is_literal(args[1]):
                    continue
                count += 1
            elif args:
                count += 1
            continue
        # A weak assertion earns its keep only when it inspects a
        # collection — `assertIn(cell, game.occupied())` is a real claim,
        # `assertTrue(game)` is not.
        if name in ("assertIn", "assertNotIn") and len(node.args) >= 2:
            if not _is_literal(node.args[1]):
                count += 1
    return count


def weak_contract_reason(src: str, min_substantive: int = 2) -> Optional[str]:
    """Why this contract cannot distinguish a real artifact from a stub.

    Returns None when it is strong enough. *min_substantive* is the
    number of discriminating assertions required; two, because one is
    within reach of a stub written to satisfy exactly one check, and
    demanding many would reject the honest short contract the prompt asks
    for on a small task.
    """
    if not src or not src.strip():
        return "empty"
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None                       # a different check's problem

    tests = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("test")]
    if not tests:
        return None                       # `_looks_like_a_suite` covers this

    # Judged BEFORE the count, and independently of it. A source-grep
    # test is not merely worthless as evidence — it is a liability,
    # because it can fail correct code that names things differently.
    # Iteration 6's contract carried nine substantive assertions across
    # two honest tests AND one grep test, so counting alone accepted it,
    # and the grep test then failed a run whose artifact scored 20/20
    # externally. One such test is enough to send the contract back.
    grepping = source_inspecting_tests(tree)
    if grepping:
        return (f"{', '.join(grepping[:2])} inspects the program's source "
                f"text instead of running it, which passes any stub that "
                f"mentions the right words and fails correct code that "
                f"names things differently")

    substantive = _substantive_assertions(tree)
    if substantive >= min_substantive:
        return None

    if substantive == 0:
        return ("it asserts nothing that could distinguish a working "
                "implementation from a stub — every assertion is about "
                "existence, liveness or a constant")
    return (f"it makes only {substantive} assertion(s) that could fail on "
            f"wrong behaviour, so nearly any program that starts would "
            f"satisfy it")


REPAIR_NOTE = """\
The contract you produced is too weak to be evidence: {reason}.

Rewrite it. Every test must drive the system and compare an OBSERVED
value against something concrete — a literal the task states, or another
observed value that must differ from it. Asserting that a process is
still running, that an attribute exists, or that an object is not None
says nothing about whether the behaviour the task describes works.

Do NOT read the program's source text. No `inspect.getsource`, no
`ast.parse` of the module, no opening `__file__` and searching it for
identifiers. Checking that the code MENTIONS "reset" passes a stub that
mentions it and fails correct code that calls the method `restart` —
you are verifying vocabulary, not behaviour. Import the module, build
the objects, call the methods, and assert on what they return and how
the state changes.

Keep the same rules as before (no mocks, no stubs, no assigning to
state, import inside each test) and output ONLY the Python file in one
``` fenced block."""

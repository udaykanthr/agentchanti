"""Acceptance tests written from the task, before any code exists.

`evidence.py` only counts two things as independent: user-supplied
`acceptance_cmds`, or a pre-existing test file the run left byte-identical.
A greenfield build has neither, so it is judged by a suite it wrote itself
— and three measured runs shipped exit 0 over artifacts that failed every
external probe while their own tests were green.

The point is *when*, not who. A test written after game.py exists is
written by an agent that has just read game.py. Seeding one from the task
text before the first step runs is the only moment a check can be written
that the code cannot have shaped, and it then flows through the existing
snapshot/independence machinery unchanged.
"""

import os

from unittest.mock import MagicMock

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, seed_acceptance_tests,
)

GOOD = '''```python
import unittest


class AcceptanceContract(unittest.TestCase):
    def test_player_moves(self):
        from game import Game
        g = Game(seed=0)
        g.start()
        before = g.entities()[0].tile
        g.press("left")
        for _ in range(500):
            g.advance(0.001)
        self.assertNotEqual(g.entities()[0].tile, before)
```'''


def _client(response):
    c = MagicMock()
    c.generate_response.return_value = response
    return c


def test_a_usable_suite_is_written(tmp_path):
    path = seed_acceptance_tests("Build a game.", str(tmp_path),
                                 _client(GOOD), language="python")
    assert path is not None
    written = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")
    assert "import unittest" in written
    assert "assertNotEqual" in written


def test_an_existing_suite_is_never_overwritten(tmp_path):
    """A test that already predates the run is stronger evidence than
    anything generated here, and clobbering it would destroy the very
    independence this exists to create."""
    (tmp_path / "test_mine.py").write_text("import unittest\n",
                                           encoding="utf-8")
    path = seed_acceptance_tests("Build a game.", str(tmp_path),
                                 _client(GOOD), language="python")
    assert path is None
    assert not (tmp_path / SEED_BASENAME).exists()


def test_an_unusable_response_writes_nothing(tmp_path):
    """No file is an honest "nothing independent verified this". A broken
    or empty suite would be worse than none — it would fail the run for a
    reason that has nothing to do with the code."""
    for junk in ("I could not determine any testable behaviour.",
                 "```python\nthis is not valid python(\n```",
                 "```python\nimport unittest\n```",     # no test method
                 ""):
        assert seed_acceptance_tests("Build a game.", str(tmp_path),
                                     _client(junk), language="python") is None
        assert not (tmp_path / SEED_BASENAME).exists()


def test_a_non_python_project_is_skipped(tmp_path):
    assert seed_acceptance_tests("Build an app.", str(tmp_path),
                                 _client(GOOD), language="javascript") is None


def test_a_generation_failure_never_raises(tmp_path):
    client = MagicMock()
    client.generate_response.side_effect = RuntimeError("provider down")
    assert seed_acceptance_tests("Build a game.", str(tmp_path),
                                 client, language="python") is None


def test_the_seed_is_a_test_file_the_snapshot_will_record(tmp_path):
    """The whole mechanism depends on `_is_test_file` recognising it and
    `snapshot_test_files` hashing it a moment later."""
    from agentchanti.orchestrator.evidence import snapshot_test_files
    from agentchanti.orchestrator.pipeline import _is_test_file

    assert _is_test_file(SEED_BASENAME)
    seed_acceptance_tests("Build a game.", str(tmp_path),
                          _client(GOOD), language="python")
    snap = snapshot_test_files(str(tmp_path))
    assert SEED_BASENAME in snap


# ── re-seeding when the task changes ─────────────────────────────────
#
# Measured 2026-08-17 in one directory: a contract seeded from a
# "Panda3D cube collector" prompt survived the prompt being rewritten
# into a Snake game, and three later runs reported
# `Evidence: independent (pre-existing-tests)` over a file whose only
# assertion was that main.py does not exit within five seconds.

CUBE = "Build a Panda3D game where a cube collects red spheres."
SNAKE = "Build a Panda3D Snake game with a grid and food."


def _seed(task, tmp_path, response=GOOD):
    return seed_acceptance_tests(task, str(tmp_path), _client(response),
                                 language="python")


def test_a_changed_task_re_seeds(tmp_path):
    assert _seed(CUBE, tmp_path) is not None
    first = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")

    second_suite = GOOD.replace("test_player_moves", "test_snake_grows")
    assert _seed(SNAKE, tmp_path, second_suite) is not None
    second = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")

    assert "test_snake_grows" in second
    assert second != first


def test_the_same_task_does_not_re_seed(tmp_path):
    assert _seed(CUBE, tmp_path) is not None
    before = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")

    client = _client(GOOD)
    assert seed_acceptance_tests(CUBE, str(tmp_path), client,
                                 language="python") is None
    # Not merely unchanged on disk — no generation was even attempted.
    client.generate_response.assert_not_called()
    assert (tmp_path / SEED_BASENAME).read_text(encoding="utf-8") == before


def test_whitespace_only_task_edits_do_not_re_seed(tmp_path):
    assert _seed(CUBE, tmp_path) is not None
    reflowed = CUBE.replace(" ", "\n  ")
    assert _seed(reflowed, tmp_path) is None


def test_a_hand_edited_seed_is_never_clobbered(tmp_path):
    """Whoever edited it owns it — the same refusal ghost_heal makes."""
    assert _seed(CUBE, tmp_path) is not None
    path = tmp_path / SEED_BASENAME
    edited = path.read_text(encoding="utf-8") + "\n# my own extra check\n"
    path.write_text(edited, encoding="utf-8")

    assert _seed(SNAKE, tmp_path) is None
    assert path.read_text(encoding="utf-8") == edited


def test_a_suite_without_our_header_is_left_alone(tmp_path):
    """A user's own file at the same path is not ours to replace."""
    path = tmp_path / SEED_BASENAME
    path.write_text("import unittest\n# hand written\n", encoding="utf-8")
    assert _seed(SNAKE, tmp_path) is None
    assert "hand written" in path.read_text(encoding="utf-8")


def test_someone_elses_suite_beats_a_stale_seed(tmp_path):
    """Another test file means real independent evidence already exists,
    so there is nothing for a re-seed to add."""
    assert _seed(CUBE, tmp_path) is not None
    (tmp_path / "test_mine.py").write_text("import unittest\n",
                                           encoding="utf-8")
    before = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")
    assert _seed(SNAKE, tmp_path) is None
    assert (tmp_path / SEED_BASENAME).read_text(encoding="utf-8") == before


def test_the_header_does_not_break_the_suite(tmp_path):
    """It rides in the file the runner collects, so it must be inert."""
    _seed(CUBE, tmp_path)
    written = (tmp_path / SEED_BASENAME).read_text(encoding="utf-8")
    assert written.startswith("# agentchanti:acceptance-seed task=")
    compile(written, SEED_BASENAME, "exec")
    assert "import unittest" in written


def test_identity_is_judged_on_the_raw_task_not_the_enriched_one(tmp_path):
    """The enriched task is LLM output and differs between runs.

    Measured 2026-08-18: the first live seed stamped the fingerprint of
    the IntentAgent's REQUIREMENTS_SPEC, not the prompt, so the very
    next run of the identical prompt would have seen a different hash
    and re-seeded — trading the old never-regenerates bug for an
    always-regenerates one.
    """
    enriched_a = CUBE + "\n\nREQUIREMENTS_SPEC: goal one, phrased this way."
    enriched_b = CUBE + "\n\nREQUIREMENTS_SPEC: goal one, worded differently."

    assert seed_acceptance_tests(enriched_a, str(tmp_path), _client(GOOD),
                                 language="python",
                                 identity_task=CUBE) is not None
    client = _client(GOOD)
    assert seed_acceptance_tests(enriched_b, str(tmp_path), client,
                                 language="python",
                                 identity_task=CUBE) is None
    client.generate_response.assert_not_called()


def test_the_suite_is_still_written_from_the_enriched_task(tmp_path):
    """Identity and content come from different strings on purpose."""
    enriched = CUBE + "\n\nREQUIREMENTS_SPEC: the fuller statement."
    client = _client(GOOD)
    seed_acceptance_tests(enriched, str(tmp_path), client, language="python",
                          identity_task=CUBE)
    prompt = client.generate_response.call_args[0][0]
    assert "the fuller statement" in prompt


def test_identity_task_defaults_to_the_task(tmp_path):
    assert _seed(CUBE, tmp_path) is not None
    assert _seed(CUBE, tmp_path) is None          # same task, no re-seed
    assert _seed(SNAKE, tmp_path) is not None     # different task, re-seed


def test_a_re_seed_is_still_independent_evidence(tmp_path):
    """Re-seeding happens before any step runs, so the snapshot taken a
    moment later still records it as pre-existing."""
    from agentchanti.orchestrator.evidence import snapshot_test_files

    _seed(CUBE, tmp_path)
    _seed(SNAKE, tmp_path, GOOD.replace("test_player_moves", "test_snake"))
    assert SEED_BASENAME in snapshot_test_files(str(tmp_path))


def test_rewriting_the_seed_forfeits_independence(tmp_path):
    """The rule that makes seeding safe: if a later step edits it, the
    hash changes and the run says so rather than claiming verification."""
    from agentchanti.orchestrator.evidence import (
        snapshot_test_files, surviving_pre_existing_tests,
    )
    seed_acceptance_tests("Build a game.", str(tmp_path),
                          _client(GOOD), language="python")
    snap = snapshot_test_files(str(tmp_path))
    assert surviving_pre_existing_tests(str(tmp_path), snap) == [SEED_BASENAME]

    (tmp_path / SEED_BASENAME).write_text(
        "import unittest\n\n\nclass T(unittest.TestCase):\n"
        "    def test_ok(self):\n        pass\n", encoding="utf-8")
    assert surviving_pre_existing_tests(str(tmp_path), snap) == []


def test_the_seed_file_is_protected_from_overwrite():
    """Executor refuses to clobber it the way it refuses requirements.txt."""
    from agentchanti.executor import Executor
    assert SEED_BASENAME in Executor._PROTECTED_FILENAMES


def test_it_is_written_before_any_step_could_have_shaped_it(tmp_path):
    """Regression guard on the prompt itself: it must ask for assertions
    from the TASK, never from files on disk."""
    client = _client(GOOD)
    seed_acceptance_tests("Build a Pac-Man clone.", str(tmp_path),
                          client, language="python")
    prompt = client.generate_response.call_args.args[0]
    assert "BEFORE any code exists" in prompt
    assert "Build a Pac-Man clone." in prompt
    # The rules that came out of the measured failures.
    assert "assertLessEqual" in prompt        # monotonicity is not progress
    assert "hundreds of" in prompt            # a range needs exercising
    assert "RANGE" in prompt


def test_the_cli_call_site_uses_names_that_exist():
    """The isolated tests above all passed while the live call raised
    NameError: 'task' is not defined — the call site was never exercised.
    This checks the three names it passes are actually bound in the
    enclosing function."""
    import ast
    import inspect

    from agentchanti.orchestrator import cli

    tree = ast.parse(inspect.getsource(cli))
    call = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "seed_acceptance_tests"):
            call = node
            break
    assert call is not None, "cli.py no longer seeds acceptance tests"

    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef) and f.name == "_main_impl")
    bound = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
             for t in ast.walk(n) if isinstance(t, ast.Name)}
    bound |= {n.arg for n in ast.walk(fn) if isinstance(n, ast.arg)}

    used = [a for a in call.args if isinstance(a, ast.Name)]
    used += [k.value for k in call.keywords if isinstance(k.value, ast.Name)]
    for name in used:
        assert name.id in bound, (
            f"seed_acceptance_tests is passed `{name.id}`, which is not "
            f"bound in _main_impl")
    # The task text must reach it via args, not a bare local.
    assert any(isinstance(a, ast.Attribute) and a.attr == "task"
               for a in call.args), "the task text is not passed through"

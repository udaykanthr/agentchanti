"""The seeded contract must be RUN before it is trusted.

Every check the seeder had was static — `compile()`, an import scan, an
AST walk — and every measured failure of a seeded contract has been a
runtime crash in framework introspection it wrote blind:

    ambient_lights[0].getColor()[:3]  -> TypeError (LVecBase4f has no slice)
    self.win.requestProperties(...)   -> AttributeError (GraphicsBuffer)

Both compile. Both were trusted. Both surfaced only at the end of the
run — one of them after 30 turns, an escalation and 402k tokens spent on
a game that scored 18/18 against external probes.

These tests drive `verify_contract_runs` against a real executor and a
real file on disk, because the seam being tested is "does anything
execute it", which a mocked runner cannot demonstrate.
"""
import os
import subprocess
import sys
import textwrap

import pytest

from agentchanti.orchestrator.acceptance_seed import (
    SEED_BASENAME, _header, reset_contract_repairs, seed_state,
    verify_contract_runs,
)


class RealExecutor:
    """Runs the command for real, in *root*. No pipeline dependencies."""

    def __init__(self, root):
        self.root = str(root)
        self.commands = []

    def run_command(self, cmd, timeout=120, cwd=None, **kw):
        self.commands.append(cmd)
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", SEED_BASENAME],
            cwd=cwd or self.root, capture_output=True, text=True,
            timeout=timeout,
        )
        return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


class ScriptedClient:
    """LLM stand-in that hands back a fixed sequence of contracts."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("asked for more repairs than expected")
        return "```python\n" + self.responses.pop(0) + "\n```"


def _write_contract(root, body, task="a snake game"):
    """Write *body* as a seeded contract, header and all."""
    path = os.path.join(str(root), SEED_BASENAME)
    body = "\n" + textwrap.dedent(body).strip() + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_header(task, body) + body)
    return path


# A contract that CRASHES: it slices a value that does not support it.
# The shape of the real defect, reduced to something with no Panda3D.
CRASHING = """
    import unittest

    import project


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            game = project.Game()
            self.assertEqual(game.advance(), "moved")
            self.assertEqual(game.score, 0)
            # The instrument's own bug: colour is an int, not a sequence.
            self.assertEqual(sum(game.colour[:3]), 3)
"""

# The same behaviours, asserted through the program's own API.
REPAIRED = """
    import unittest

    import project


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            game = project.Game()
            self.assertEqual(game.advance(), "moved")
            self.assertEqual(game.score, 0)
            self.assertEqual(game.colour, 1)
"""

# Passes every static check — compiles, imports unittest, defines a test,
# and asserts just as much as the original — and still crashes when run.
STILL_CRASHING = """
    import unittest

    import project


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            game = project.Game()
            self.assertEqual(game.advance(), "moved")
            self.assertEqual(game.score, 0)
            self.assertEqual(len(game.palette), 3)
"""

# A "repair" that fixes the crash by asserting almost nothing.
GUTTED = """
    import unittest

    import project


    class Contract(unittest.TestCase):
        def test_behaviour(self):
            self.assertTrue(hasattr(project, "Game"))
"""

PROJECT = """
    class Game:
        def __init__(self):
            self.score = 0
            self.colour = 1

        def advance(self):
            return "moved"
"""


@pytest.fixture(autouse=True)
def _clean_state():
    reset_contract_repairs()
    yield
    reset_contract_repairs()


@pytest.fixture
def project(tmp_path):
    (tmp_path / "project.py").write_text(textwrap.dedent(PROJECT),
                                         encoding="utf-8")
    return tmp_path


def test_a_crashing_contract_is_repaired_before_the_run_ends(project):
    """The measured incident: a contract that compiles and cannot run."""
    path = _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(REPAIRED).strip())

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is not None, "a crashing contract was left in place"
    rel, digest = result
    assert rel == SEED_BASENAME
    assert len(digest) == 64

    # It now runs against the real project.
    proc = subprocess.run([sys.executable, "-m", "unittest", SEED_BASENAME],
                          cwd=str(project), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # And it is still ours, so provenance survives the repair.
    assert seed_state(path) is not None


def test_the_repair_prompt_carries_the_crash_and_the_contract(project):
    """A repair needs both halves, and the second one was missing.

    Measured live 2026-09-09: without the contract in the prompt, the
    model was being asked to fix a file it could not see, under a base
    prompt whose framing is "before any code exists". It did the only
    thing that allows — regenerated from scratch, decided the task named
    no importable module, and returned a contract whose single test was
    `self.fail(...)`.
    """
    _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(REPAIRED).strip())

    verify_contract_runs(RealExecutor(project), str(project), client,
                         "a snake game")

    assert client.prompts, "no repair was requested"
    prompt = client.prompts[-1]
    assert "TypeError" in prompt, "the crash itself never reached the model"
    assert "game.colour[:3]" in prompt, \
        "the contract being repaired was not in the prompt"
    assert "self.fail" in prompt, \
        "nothing stopped the model from giving up instead of repairing"


def test_the_repair_is_never_shown_the_artifact(project):
    """Independence is the whole point, and it is easy to lose here.

    Showing the model its OWN prior contract is safe. Showing it the code
    under test would make the repaired contract a function of the artifact
    it is supposed to judge independently.
    """
    _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(REPAIRED).strip())

    verify_contract_runs(RealExecutor(project), str(project), client,
                         "a snake game")

    source = (project / "project.py").read_text(encoding="utf-8")
    body = source.split("class Game:", 1)[1]
    assert body.strip() not in client.prompts[-1], \
        "the artifact's source leaked into the repair prompt"


def test_a_weaker_repair_is_refused(project):
    """A crash is trivially 'fixed' by checking less. That is not a fix."""
    path = _write_contract(project, CRASHING)
    before = open(path, encoding="utf-8").read()
    gutted = textwrap.dedent(GUTTED).strip()
    client = ScriptedClient(gutted, gutted)      # weak on every attempt

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is None, "a gutted contract was accepted"
    assert open(path, encoding="utf-8").read() == before, \
        "the original contract was not restored"


def test_an_unusable_response_does_not_spend_the_repair(project):
    """The defect the first live run exposed.

    The repair call came back having burned all 16,384 output tokens on
    reasoning — "Response hit the output-token limit" — so it was not a
    repair that got judged and rejected, it was no repair at all. Spending
    the whole budget on it left the contract broken and the run ended
    exactly as it would have without any of this. The weakness repair
    directly above this one already loops for the same reason.
    """
    _write_contract(project, CRASHING)
    client = ScriptedClient("sorry, I cannot help with that",
                            textwrap.dedent(REPAIRED).strip())

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is not None, \
        "one unusable response spent the entire repair budget"
    assert len(client.prompts) == 2, "the model was never asked again"
    proc = subprocess.run([sys.executable, "-m", "unittest", SEED_BASENAME],
                          cwd=str(project), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_failing_but_working_contract_is_left_alone(project):
    """An assertion failure is the instrument WORKING.

    At this point in a run the code is usually incomplete, so a contract
    that disagrees with it may be entirely right. Rewriting it here is
    the exact cheat the seeder exists to prevent.
    """
    disagrees = CRASHING.replace(
        'self.assertEqual(sum(game.colour[:3]), 3)',
        'self.assertEqual(game.score, 99)')
    path = _write_contract(project, disagrees)
    before = open(path, encoding="utf-8").read()
    client = ScriptedClient()          # any call would raise

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is None
    assert open(path, encoding="utf-8").read() == before
    assert not client.prompts, "an honest disagreement triggered a rewrite"


def test_a_repair_that_still_crashes_is_retried_with_the_new_error(project):
    """The defect the classic end-to-end run exposed.

    A candidate that passes every static check and then crashes used to
    end the repair outright — throwing away the most useful signal there
    is (a fresh, specific error about the attempt just made) while
    attempts remained. Measured live 2026-09-09: the run restored the
    original, logged "the repaired contract still crashes" without
    saying how, and reported inconclusive evidence.
    """
    path = _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(STILL_CRASHING).strip(),
                            textwrap.dedent(REPAIRED).strip())

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is not None, "a crashing candidate ended the repair"
    assert len(client.prompts) == 2, "the model was never asked again"
    # The second ask must carry the SECOND crash, not the first one again.
    assert "palette" in client.prompts[-1], \
        "the new error never reached the model"
    assert seed_state(path) is not None
    proc = subprocess.run([sys.executable, "-m", "unittest", SEED_BASENAME],
                          cwd=str(project), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_original_survives_every_failed_repair(project):
    """Two broken contracts are not better than one."""
    path = _write_contract(project, CRASHING)
    before = open(path, encoding="utf-8").read()
    crashing = textwrap.dedent(STILL_CRASHING).strip()
    client = ScriptedClient(crashing, crashing)   # never recovers

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is None
    assert open(path, encoding="utf-8").read() == before, \
        "a failed repair was left on disk"


def test_a_contract_whose_project_does_not_exist_yet_is_deferred(tmp_path):
    """Not-yet-written code is not a broken instrument."""
    _write_contract(tmp_path, CRASHING)          # no project.py at all
    client = ScriptedClient()

    result = verify_contract_runs(RealExecutor(tmp_path), str(tmp_path),
                                  client, "a snake game")

    assert result is None
    assert not client.prompts, "an ImportError was read as a contract defect"


def test_a_users_own_suite_is_never_touched(project):
    """No header means it is not ours, whatever it does."""
    path = os.path.join(str(project), SEED_BASENAME)
    body = textwrap.dedent(CRASHING).strip() + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)                            # deliberately unstamped
    client = ScriptedClient()

    result = verify_contract_runs(RealExecutor(project), str(project),
                                  client, "a snake game")

    assert result is None
    assert open(path, encoding="utf-8").read() == body
    assert not client.prompts


def test_the_check_does_not_run_twice_per_run(project):
    """Re-running it every wave costs a subprocess for no new information."""
    _write_contract(project, CRASHING)
    client = ScriptedClient(textwrap.dedent(REPAIRED).strip())
    executor = RealExecutor(project)

    first = verify_contract_runs(executor, str(project), client,
                                 "a snake game")
    runs_after_first = len(executor.commands)
    second = verify_contract_runs(executor, str(project), client,
                                  "a snake game")

    assert first is not None
    assert second is None
    assert len(executor.commands) == runs_after_first, \
        "the contract was re-executed after it had already been settled"

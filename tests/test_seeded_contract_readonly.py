"""The run may not rewrite the check that judges it.

A user's `acceptance_cmds` script has been read-only to the agent since
2026-08-19. The seeded contract - the only independent evidence a
greenfield run has - was merely *detected* when modified, never prevented.

Measured 2026-09-24, gpt-6-astra on "implement a 2d snake game". The run
built a packaged project under `src/snake_game/`; the contract, written
before any code existed, did not match that layout, so the agent improved
the CONTRACT instead: 52 lines to 394 (`342 insertions(+), 52 deletions(-)`).
Ghost was 64 hold / 0 violated / 0 disagreements, and the run failed on the
last line with "no test survived that the agent did not write or rewrite".
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.agent_tools import AgentTools
from agentchanti.executor import Executor
from agentchanti.paths import seeded_contract_reason

HEADER = ("# agentchanti:acceptance-seed task=86b9d889e10e2d56 "
          "body=c4653a598fc70710\n")
CONTRACT = HEADER + '"""Black-box acceptance checks."""\nimport unittest\n'


class SeededContractTestCase(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="seedcontract_")
        with open(os.path.join(self.root, "test_acceptance_contract.py"),
                  "w", encoding="utf-8") as fh:
            fh.write(CONTRACT)
        executor = MagicMock()
        executor.run_command.return_value = (True, "ok")
        self.tools = AgentTools(project_root=self.root, executor=executor)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestDetection(SeededContractTestCase):

    def test_the_stamped_contract_is_recognised(self):
        why = seeded_contract_reason(self.root, "test_acceptance_contract.py")
        self.assertIsNotNone(why)
        self.assertIn("did not author", why)

    def test_a_users_own_suite_is_not_protected(self):
        """No seed stamp - never ours to refuse."""
        with open(os.path.join(self.root, "test_user.py"), "w") as fh:
            fh.write("import unittest\n")
        self.assertIsNone(seeded_contract_reason(self.root, "test_user.py"))

    def test_a_file_that_does_not_exist_is_not_protected(self):
        self.assertIsNone(
            seeded_contract_reason(self.root, "test_acceptance_contract.py"
                                   .replace("contract", "absent")))

    def test_only_at_the_project_root(self):
        """The seeder writes at the root; a same-named file in a
        subdirectory is an ordinary file."""
        sub = os.path.join(self.root, "tests")
        os.makedirs(sub)
        with open(os.path.join(sub, "test_acceptance_contract.py"), "w") as fh:
            fh.write(CONTRACT)
        self.assertIsNone(seeded_contract_reason(
            self.root, "tests/test_acceptance_contract.py"))


class TestTheRefusal(SeededContractTestCase):

    def test_write_is_refused_and_the_bytes_survive(self):
        result = self.tools._tool_write_file(
            "test_acceptance_contract.py", "import unittest\n# rewritten\n")
        self.assertIn("ERROR", result)
        self.assertIn("do not rewrite the check", result)
        with open(os.path.join(self.root, "test_acceptance_contract.py"),
                  encoding="utf-8") as fh:
            self.assertEqual(fh.read(), CONTRACT)

    def test_edit_is_refused(self):
        result = self.tools._tool_edit_file(
            "test_acceptance_contract.py", "import unittest", "import pytest")
        self.assertIn("ERROR", result)

    def test_reading_stays_allowed(self):
        """The model must be able to see what it has to satisfy."""
        result = self.tools._tool_read_file("test_acceptance_contract.py")
        self.assertNotIn("ERROR", result)
        self.assertIn("acceptance-seed", result)

    def test_the_project_is_still_writable(self):
        result = self.tools._tool_write_file("snake.py", "SIZE = 20\n")
        self.assertNotIn("ERROR", result)

    def test_a_plan_declared_target_is_refused_too(self):
        """The route that re-created the package-shadow collision after the
        healer was guarded: a plan naming the contract as a step target."""
        Executor().write_files(
            {"test_acceptance_contract.py": "# plan body\n"},
            base_dir=self.root)
        with open(os.path.join(self.root, "test_acceptance_contract.py"),
                  encoding="utf-8") as fh:
            self.assertEqual(fh.read(), CONTRACT)


class TestTheSeederItself(SeededContractTestCase):
    """The seeder writes with plain open(), so seeding and its own repair
    path are unaffected - only the agent's writers are refused."""

    def test_direct_write_still_works(self):
        path = os.path.join(self.root, "test_acceptance_contract.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(HEADER + "# repaired by the seeder\n")
        with open(path, encoding="utf-8") as fh:
            self.assertIn("repaired by the seeder", fh.read())

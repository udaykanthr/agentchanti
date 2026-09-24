"""A write guard cannot see `run_command`, so the bytes are restored.

Measured 2026-09-24, gpt-6-astra, the run after the write guard landed:

    run_command: python -m ruff check test_acceptance_contract.py --fix
                 && python -m ruff format ...

`--fix` removed two unused imports and a blank line - five deletions in a
194-line file - and the run forfeited its own evidence and failed at 24:34
and 542k tokens, over an artifact whose own suite was green.

Nothing about what the contract asserts changed, which is beside the point:
"byte-identical" is the only rule that cannot be gamed, because any softer
rule needs a model to judge which edits are harmless.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from agentchanti.agent_tools import AgentTools, contract_mutating_command
from agentchanti.orchestrator import acceptance_seed as seed


class RestoreTestCase(unittest.TestCase):

    def setUp(self):
        seed.reset_contract_repairs()
        self.root = tempfile.mkdtemp(prefix="contract_restore_")
        self.path = os.path.join(self.root, seed.SEED_BASENAME)
        self.original = ("# agentchanti:acceptance-seed task=abc body=def\n"
                         "\nimport unittest\nimport snake_game.app\n")
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(self.original)
        seed._remember_seed_bytes(self.path, self.original)

    def tearDown(self):
        seed.reset_contract_repairs()
        shutil.rmtree(self.root, ignore_errors=True)

    def _read(self):
        with open(self.path, encoding="utf-8") as fh:
            return fh.read()


class TestRestore(RestoreTestCase):

    def test_the_measured_formatter_edit_is_undone(self):
        """ruff --fix drops the unused import; the bytes come back."""
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("# agentchanti:acceptance-seed task=abc body=def\n"
                     "\nimport unittest\n")

        restored = seed.restore_seeded_contract(self.root)

        self.assertEqual(restored, seed.SEED_BASENAME)
        self.assertEqual(self._read(), self.original)

    def test_a_deleted_contract_comes_back(self):
        os.remove(self.path)
        self.assertEqual(seed.restore_seeded_contract(self.root),
                         seed.SEED_BASENAME)
        self.assertEqual(self._read(), self.original)

    def test_an_untouched_contract_is_left_alone(self):
        self.assertIsNone(seed.restore_seeded_contract(self.root))

    def test_a_file_this_run_did_not_seed_is_never_restored(self):
        """No remembered bytes - not ours to put back."""
        seed.reset_contract_repairs()
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("# someone else's file\n")
        self.assertIsNone(seed.restore_seeded_contract(self.root))
        self.assertIn("someone else", self._read())

    def test_a_stale_run_result_is_discarded(self):
        """The cached verdict described bytes that are no longer on disk."""
        seed._LAST_RUN[self.path] = (True, "OK")
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("# changed\n")
        seed.restore_seeded_contract(self.root)
        self.assertIsNone(seed.last_contract_run(self.root))


class TestTheCommandRefusal(RestoreTestCase):

    def setUp(self):
        super().setUp()
        executor = MagicMock()
        executor.run_command.return_value = (True, "ok")
        self.tools = AgentTools(project_root=self.root, executor=executor)

    def test_the_measured_command_is_refused(self):
        why = contract_mutating_command(
            self.root,
            "python -m ruff check test_acceptance_contract.py --fix")
        self.assertIsNotNone(why)
        self.assertIn("extend-exclude", why)

    def test_the_tool_refuses_it_and_never_runs_it(self):
        result = self.tools._tool_run_command(
            "python -m ruff check test_acceptance_contract.py --fix")
        self.assertIn("ERROR", result)
        self.tools._executor.run_command.assert_not_called()

    def test_running_or_reading_the_contract_is_fine(self):
        for cmd in ("python -m unittest test_acceptance_contract.py",
                    "python -m pytest -q test_acceptance_contract.py",
                    "type test_acceptance_contract.py"):
            self.assertIsNone(contract_mutating_command(self.root, cmd), cmd)

    def test_formatting_the_project_is_untouched(self):
        self.assertIsNone(contract_mutating_command(
            self.root, "python -m ruff format snake_game/model.py"))

    def test_nothing_is_refused_when_no_contract_exists(self):
        os.remove(self.path)
        self.assertIsNone(contract_mutating_command(
            self.root, "ruff check test_acceptance_contract.py --fix"))

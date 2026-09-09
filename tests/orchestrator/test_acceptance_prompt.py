"""Ask for an independent instrument BEFORE the run, not after it.

`require_independent_evidence` is satisfiable three ways and they are not
equally strong. User `acceptance_cmds` and a user's own pre-existing suite
are outside the run. A contract the pipeline seeds is not — it is a model
output, written before the code exists, and measured across four runs of
one prompt it produced three distinct false verdicts over artifacts that
scored 17-18/18 against external probes.

The pre-flight already refuses to let an UNSATISFIABLE configuration run
silently. This is the same argument one notch weaker: satisfiable, but
only by an instrument nobody outside the run wrote.
"""
import builtins
import os

import pytest

from agentchanti.orchestrator.acceptance_seed import SEED_BASENAME, _header
from agentchanti.orchestrator.cli import _prompt_for_acceptance_cmds
from agentchanti.orchestrator.evidence import _was_seeded


class Cfg:
    def __init__(self, cmds=None):
        self.ACCEPTANCE_CMDS = list(cmds or [])


def _answers(monkeypatch, *lines):
    """Feed *lines* to input(), then behave like a closed stdin."""
    supplied = list(lines)

    def fake_input(_prompt=""):
        if not supplied:
            raise EOFError
        return supplied.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)


def test_auto_mode_never_blocks_but_says_what_would_help(monkeypatch, caplog):
    """An unattended run must not stop at a prompt nobody can answer."""
    def explode(_prompt=""):
        raise AssertionError("--auto must never prompt")

    monkeypatch.setattr(builtins, "input", explode)
    cfg = Cfg()

    with caplog.at_level("WARNING"):
        assert _prompt_for_acceptance_cmds(cfg, auto=True) == []

    assert cfg.ACCEPTANCE_CMDS == []
    assert "acceptance_cmds" in caplog.text
    # The advice has to name the remedy, not just the problem.
    assert "can neither write nor edit" in caplog.text


def test_supplied_commands_are_adopted(monkeypatch):
    _answers(monkeypatch, "python acceptance_check.py", "npm run e2e", "")
    cfg = Cfg()

    supplied = _prompt_for_acceptance_cmds(cfg, auto=False)

    assert supplied == ["python acceptance_check.py", "npm run e2e"]
    assert cfg.ACCEPTANCE_CMDS == supplied, \
        "the answer never reached the config the run reads"


def test_existing_commands_are_kept(monkeypatch):
    """Answering must add to the config, never replace what was there."""
    _answers(monkeypatch, "pytest -q acceptance/", "")
    cfg = Cfg(["node check.cjs"])

    _prompt_for_acceptance_cmds(cfg, auto=False)

    assert cfg.ACCEPTANCE_CMDS == ["node check.cjs", "pytest -q acceptance/"]


def test_declining_continues_the_run(monkeypatch, caplog):
    """The artifacts are still worth having — this warns, never refuses."""
    _answers(monkeypatch, "")
    cfg = Cfg()

    with caplog.at_level("WARNING"):
        assert _prompt_for_acceptance_cmds(cfg, auto=False) == []

    assert cfg.ACCEPTANCE_CMDS == []
    assert "continuing without an independent check" in caplog.text


def test_a_closed_stdin_is_not_an_error(monkeypatch):
    """Piped or non-tty invocations hit EOF immediately; that is not a crash."""
    _answers(monkeypatch)                      # every input() raises EOFError
    cfg = Cfg()

    assert _prompt_for_acceptance_cmds(cfg, auto=False) == []
    assert cfg.ACCEPTANCE_CMDS == []


def test_a_seeded_contract_does_not_count_as_a_users_suite(tmp_path):
    """The discriminator the call site depends on.

    Counting a seeded contract here would report the strong case ("tests
    predate this run") for the weak one ("we wrote our own check"), and
    the prompt would never fire in exactly the case it exists for.
    """
    seeded = tmp_path / SEED_BASENAME
    body = "\nimport unittest\n"
    seeded.write_text(_header("a task", body) + body, encoding="utf-8")

    mine = tmp_path / "test_mine.py"
    mine.write_text("import unittest\n", encoding="utf-8")

    assert _was_seeded(str(tmp_path), SEED_BASENAME) is True
    assert _was_seeded(str(tmp_path), "test_mine.py") is False

    snapshot = {SEED_BASENAME: "x", "test_mine.py": "y"}
    user_tests = [rel for rel in snapshot
                  if not _was_seeded(str(tmp_path), rel)]
    assert user_tests == ["test_mine.py"]


def test_a_seeded_contract_alone_leaves_no_user_suite(tmp_path):
    """The case that must trigger the prompt: seeded contract and nothing else."""
    seeded = tmp_path / SEED_BASENAME
    body = "\nimport unittest\n"
    seeded.write_text(_header("a task", body) + body, encoding="utf-8")

    snapshot = {SEED_BASENAME: "x"}
    user_tests = [rel for rel in snapshot
                  if not _was_seeded(str(tmp_path), rel)]
    assert user_tests == [], \
        "a generated contract was counted as evidence the user supplied"

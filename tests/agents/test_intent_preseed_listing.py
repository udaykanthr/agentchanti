"""An empty KB should not cost a round trip for `ls`.

Measured 2026-09-30 across five languages — Go, Rust, Java, and the two
earlier CLI cases — identical every time on a greenfield task:

    [IntentAnalysis] Pre-seeding context with KB_SEARCH '...'
    [IntentAnalysis] Pre-seed returned no results.
    [IntentAnalysis] Iteration 1: RUN_CMD 'ls'
    [IntentAnalysis] Successfully generated REQUIREMENTS_SPEC.

One whole LLM call whose entire output was `RUN_CMD: ls`, for information
the pipeline already had: `Project scan: 2 files detected, 1 source files
collected` was logged seconds earlier. The `ls` output then rode into the
next prompt as evidence, so the round trip cost twice.

Handing the listing over when the KB comes back empty removes the round
trip without touching the investigation loop — on an existing codebase
the KB is not empty and the branch never runs.
"""
import os

import pytest

from agentchanti.agents.intent import IntentAgent


def _agent():
    return object.__new__(IntentAgent)


class TestTheListing:

    def test_it_lists_the_project_files(self, tmp_path, monkeypatch):
        (tmp_path / "main.go").write_text("package main", encoding="utf-8")
        (tmp_path / "go.mod").write_text("module todo", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        listing = _agent()._directory_listing()
        assert "main.go" in listing
        assert "go.mod" in listing

    def test_directories_are_marked(self, tmp_path, monkeypatch):
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        assert "src/" in _agent()._directory_listing()

    @pytest.mark.parametrize("junk", ["node_modules", "target", ".git",
                                      "__pycache__", "venv", ".next"])
    def test_build_output_and_dependencies_are_skipped(self, tmp_path,
                                                       monkeypatch, junk):
        """Not the project, for the same reason the scanners skip them."""
        (tmp_path / junk).mkdir()
        (tmp_path / "real.py").write_text("x = 1", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        listing = _agent()._directory_listing()
        assert "real.py" in listing
        assert junk not in listing

    def test_it_is_bounded(self, tmp_path, monkeypatch):
        """This exists to save ONE call, so it must never cost more than
        the call it replaces."""
        for i in range(200):
            (tmp_path / f"f{i:03d}.txt").write_text("x", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        listing = _agent()._directory_listing(limit=60)
        assert listing.count("\n") <= 60
        assert "truncated" in listing

    def test_an_unreadable_directory_yields_nothing(self, monkeypatch):
        """Never raise into the run: no listing is an acceptable outcome."""
        def boom(_path):
            raise OSError("nope")
        monkeypatch.setattr(os, "listdir", boom)
        assert _agent()._directory_listing() == ""

    def test_an_empty_directory_yields_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _agent()._directory_listing() == ""


class TestItIsWiredToTheEmptyKbBranch:
    """A helper nothing calls is the mistake `protect_acceptance_files`
    already made once."""

    def test_the_preseed_uses_it_when_the_kb_is_empty(self):
        import inspect

        from agentchanti.agents import intent
        src = inspect.getsource(intent)
        # Called on the branch that logs the empty pre-seed, not elsewhere.
        empty_branch = src.index("Pre-seed returned no results")
        called_at = src.index("self._directory_listing()")
        assert called_at > empty_branch
        # And it tells the model not to bother with `ls`.
        assert "no need to run `ls`" in src

    def test_a_successful_preseed_does_not_add_a_listing(self):
        """When the KB has content the loop is doing real work; the listing
        would be noise on top of it."""
        import inspect

        from agentchanti.agents import intent
        src = inspect.getsource(intent)
        success = src.index("Pre-seed successful")
        empty = src.index("Pre-seed returned no results")
        listing = src.index("self._directory_listing()")
        assert success < empty < listing, (
            "the listing must sit in the empty-KB branch only")

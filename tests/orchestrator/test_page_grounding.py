"""Tests for page grounding + acceptance checks (orchestrator/page_grounding.py).

The mechanism exists because a run can finish green while the task's
user-visible requirement is unmet (observed: signup help_text still
rendered on page load after a "hide validation text on load" task). The
briefing emits machine-checkable ``Acceptance:`` assertions, the Django
probe executes them, and pre-analysis grounds the task's quoted screen
lines against a live render.
"""

import ast
import json
import os

import pytest
from unittest.mock import MagicMock

from agentchanti.orchestrator.page_grounding import (
    _MAX_ACCEPTANCE_CHECKS,
    _PAGES_PROBE,
    _find_django_root,
    extract_task_page_lines,
    parse_acceptance_checks,
    pinned_urls_from_task,
)
from agentchanti.orchestrator.smoke_test import (
    _DJANGO_PROBE,
    _run_django_verification,
)


class TestParseAcceptanceChecks:

    def test_parses_both_kinds(self):
        briefing = (
            "TASK BRIEFING:\n"
            "Goal: x\n"
            "Acceptance:\n"
            '- GET /accounts/signup/ MUST_NOT_CONTAIN "150 characters or fewer"\n'
            '- GET /accounts/signup/ MUST_CONTAIN "Create an account"\n'
        )
        checks = parse_acceptance_checks(briefing)
        assert checks == [
            {"url": "/accounts/signup/", "kind": "must_not_contain",
             "needle": "150 characters or fewer"},
            {"url": "/accounts/signup/", "kind": "must_contain",
             "needle": "Create an account"},
        ]

    def test_none_and_prose_ignored(self):
        assert parse_acceptance_checks("Acceptance: NONE") == []
        assert parse_acceptance_checks("") == []
        assert parse_acceptance_checks(
            "Expected output: page must not contain errors") == []

    def test_relative_urls_rejected(self):
        checks = parse_acceptance_checks(
            '- GET signup.html MUST_CONTAIN "x"')
        assert checks == []

    def test_duplicates_removed_and_capped(self):
        line = '- GET /a/ MUST_CONTAIN "x"\n'
        assert len(parse_acceptance_checks(line * 5)) == 1
        many = "\n".join(
            f'- GET /p{i}/ MUST_CONTAIN "x"' for i in range(20))
        assert len(parse_acceptance_checks(many)) == _MAX_ACCEPTANCE_CHECKS


class TestPinnedUrlsFromTask:
    """URLs the task names verbatim become must_resolve acceptance
    checks — the A/B run shipped the dashboard at a route other than
    the pinned /dashboard/ and stayed green."""

    def test_django_benchmark_task(self):
        task = ("create a django application with a responsive spacious "
                "homepage at / (header, large herobanner), and by default "
                "logged in users should auto redirect to a dashboard page "
                "at /dashboard/.")
        assert pinned_urls_from_task(task) == ["/", "/dashboard/"]

    def test_nested_and_unslashed_paths(self):
        task = "serve the API at /api/v1/users and docs at /docs"
        assert pinned_urls_from_task(task) == ["/api/v1/users", "/docs"]

    def test_prose_slashes_ignored(self):
        task = ("build a signup and/or login flow, run A/B tests, "
                "see https://example.com/docs for reference")
        assert pinned_urls_from_task(task) == []

    def test_windows_paths_ignored(self):
        task = r"read the config from C:\apps\config and fix src\main.py"
        assert pinned_urls_from_task(task) == []

    def test_dedupe_and_cap(self):
        task = " ".join(f"page at /p{i}/ and again /p{i}/" for i in range(12))
        urls = pinned_urls_from_task(task)
        assert len(urls) == _MAX_ACCEPTANCE_CHECKS
        assert len(set(urls)) == len(urls)

    def test_non_string_is_empty(self):
        assert pinned_urls_from_task(None) == []


class TestExtractTaskPageLines:

    def test_extracts_pasted_screen_lines(self):
        task = (
            "Fix the signup page\n"
            "example of current screen:\n"
            "\n"
            "Required. 150 characters or fewer. Letters, digits only.\n"
            "ok\n"  # too short — dropped
            "Your password must contain at least 8 characters.\n"
        )
        lines = extract_task_page_lines(task)
        assert "Required. 150 characters or fewer. Letters, digits only." in lines
        assert "Your password must contain at least 8 characters." in lines
        assert "ok" not in lines

    def test_dedupes(self):
        task = "Enter the same password as before.\n" * 3
        assert extract_task_page_lines(task) == [
            "Enter the same password as before."]


class TestFindDjangoRoot:

    def test_subproject_and_child_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        site = tmp_path / "mysite"
        site.mkdir()
        (site / "manage.py").write_text("#\n")
        # found as a first-level child without a hint
        assert _find_django_root() == "mysite"
        # explicit hint wins
        assert _find_django_root("mysite") == "mysite"

    def test_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _find_django_root() is None


class TestProbeSources:
    """The probe scripts are written to disk and executed by an arbitrary
    project interpreter — they must be valid, self-contained, ASCII-only
    Python."""

    @pytest.mark.parametrize("src", [_PAGES_PROBE, _DJANGO_PROBE],
                             ids=["pages_probe", "django_probe"])
    def test_compiles_and_ascii(self, src):
        ast.parse(src)
        src.encode("ascii")

    def test_django_probe_covers_acceptance_and_discovery(self):
        assert "ACCEPTANCE_FAILED" in _DJANGO_PROBE
        assert "get_resolver" in _DJANGO_PROBE  # renders all no-arg routes
        assert "sys.argv[4]" in _DJANGO_PROBE

    def test_django_probe_handles_must_resolve(self):
        # Task-pinned URLs: exists-at-this-path check, redirects allowed
        assert "must_resolve" in _DJANGO_PROBE
        assert "404" in _DJANGO_PROBE


class TestDjangoVerificationPlumbing:
    """_run_django_verification hands the briefing's acceptance checks to
    the probe as its 4th argument."""

    def _run(self, tmp_path, briefing):
        site = tmp_path / "site"
        (site / "config").mkdir(parents=True)
        (site / "manage.py").write_text("#\n")
        (site / "config" / "settings.py").write_text("DEBUG = True\n")

        captured = {}

        def fake_run(cmd, **kwargs):
            # The acceptance file is the 4th quoted argument of the probe
            # command; read it before the finally-block deletes it.
            parts = cmd.split('"')
            acc_path = parts[-2]
            with open(acc_path, encoding="utf-8") as f:
                captured["acceptance"] = json.load(f)
            captured["cmd"] = cmd
            return True, "DJANGO_PROBE_DONE"

        executor = MagicMock()
        executor.run_command.side_effect = fake_run
        memory = MagicMock()
        memory.all_files.return_value = {}
        memory._task_briefing = briefing
        ok, err = _run_django_verification(
            memory, executor, MagicMock(), MagicMock(),
            "task", "python", None, str(site))
        return ok, err, captured

    def test_acceptance_checks_reach_the_probe(self, tmp_path):
        ok, err, captured = self._run(
            tmp_path,
            'Acceptance:\n- GET /signup/ MUST_NOT_CONTAIN "help text"\n')
        assert ok and err == ""
        assert captured["acceptance"] == [
            {"url": "/signup/", "kind": "must_not_contain",
             "needle": "help text"}]
        assert "acceptance.json" in captured["cmd"]

    def test_no_briefing_means_empty_checks(self, tmp_path):
        ok, _, captured = self._run(tmp_path, "")
        assert ok
        assert captured["acceptance"] == []

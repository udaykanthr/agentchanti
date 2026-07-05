"""Reviewer + diagnosis API grounding.

Observed live: after a clean probe, the reviewer LLM FAILed correct
arcade 3.x code as "misspelled" and recommended the removed 2.x APIs
back — the coder obeyed and un-fixed the file.  A clean probe must now
arm the reviewer with the verified-API list, and the diagnosis fix path
must reject fixes that use missing APIs.

Tests run the real probe against the current interpreter: ``json.loads``
always exists, ``json.laods`` never does.
"""

from agentchanti.executor import Executor
from agentchanti.orchestrator.diagnosis import _apply_fix
from agentchanti.orchestrator.step_handlers import (_api_grounding_context,
                                                    _review_verdict)


class _Mem:
    def __init__(self, files):
        self._files = dict(files)

    def all_files(self):
        return dict(self._files)

    def get(self, p):
        return self._files.get(p)

    def update(self, files):
        self._files.update(files)

    def related_context(self, *_args, **_kwargs):
        return ""

    def summary(self):
        return ", ".join(self._files)


class _Display:
    def step_info(self, *_args, **_kwargs):
        pass

    def step_tokens(self, *_args, **_kwargs):
        pass

    def add_llm_log(self, *_args, **_kwargs):
        pass


class TestReviewVerdict:
    def test_approval_phrases_accepted(self):
        assert _review_verdict("Code looks good.", [])
        assert _review_verdict("LGTM", None)
        assert not _review_verdict("FAIL: broken imports", [])

    def test_probe_errors_override_approval(self):
        # The live failure: the reviewer approved code the probe had
        # flagged, shipping dead APIs to disk. Probe errors are ground
        # truth — approval can never waive them.
        errs = ["`arcade.draw_rectangle_filled` does not exist in the "
                "installed arcade 3.3.3"]
        assert not _review_verdict("Code looks good. No issues found.", errs)

    def test_empty_review_not_approved(self):
        assert not _review_verdict("", [])
        assert not _review_verdict(None, [])


class TestApiGroundingContext:
    def test_clean_probe_returns_verified_ctx(self):
        files = {"main.py": "import json\nprint(json.loads('{}'))\n"}
        errs, ctx = _api_grounding_context(files, _Mem(files), Executor())
        assert errs == []
        assert "VERIFIED APIs" in ctx
        assert "`json.loads`" in ctx
        assert "Do NOT flag" in ctx

    def test_versions_included_when_known(self):
        files = {"main.py": "import pytest\npytest.skip\n"}

        class Ctx:
            installed_versions = {"pytest": "9.9.9"}

        errs, ctx = _api_grounding_context(
            files, _Mem(files), Executor(), Ctx())
        assert errs == []
        assert "pytest==9.9.9" in ctx

    def test_bad_api_returns_errors_and_no_ctx(self):
        files = {"main.py": "import json\nprint(json.laods('{}'))\n"}
        errs, ctx = _api_grounding_context(files, _Mem(files), Executor())
        assert len(errs) == 1
        assert "json.laods" in errs[0]
        assert ctx == ""

    def test_non_python_files_skip(self):
        errs, ctx = _api_grounding_context(
            {"a.md": "# doc"}, _Mem({}), Executor())
        assert errs == [] and ctx == ""

    def test_load_failure_feeds_gate(self, tmp_path, monkeypatch):
        # The per-step execution check: a file that cannot even be
        # imported in the project environment must fail the gate —
        # regardless of what the reviewer or the AST probe think.
        monkeypatch.chdir(tmp_path)
        content = "import json\nraise RuntimeError('boom at import')\n"
        (tmp_path / "mod.py").write_text(content)
        files = {"mod.py": content}
        errs, ctx = _api_grounding_context(
            files, _Mem(files), Executor(), language="python")
        assert any("fails to load" in e for e in errs)
        assert ctx == ""

    def test_loadable_file_still_verified(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = "import json\nX = json.loads('{}')\n"
        (tmp_path / "mod.py").write_text(content)
        files = {"mod.py": content}
        errs, ctx = _api_grounding_context(
            files, _Mem(files), Executor(), language="python")
        assert errs == []
        assert "VERIFIED APIs" in ctx


ORIG = "import json\nA = 1\nB = 2\nprint(json.loads('{}'))\n"


def _diag(call):
    return (
        "ROOT CAUSE: bad call\n"
        "#### [FILE]: main.py\n"
        "```python\n"
        "import json\n"
        "A = 1\n"
        "B = 2\n"
        f"print({call})\n"
        "```\n"
    )


class TestDiagnosisApiGuard:
    def test_ungrounded_fix_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text(ORIG)
        mem = _Mem({"main.py": ORIG})
        applied = _apply_fix(
            _diag("json.lodas('{}')"), Executor(), mem, _Display(), 0)[0]
        assert not applied
        assert mem.all_files()["main.py"] == ORIG
        assert (tmp_path / "main.py").read_text() == ORIG

    def test_grounded_fix_applied(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        broken = ORIG.replace("loads", "laods")
        (tmp_path / "main.py").write_text(broken)
        mem = _Mem({"main.py": broken})
        applied = _apply_fix(
            _diag("json.loads('{}')"), Executor(), mem, _Display(), 0)[0]
        assert applied
        assert "loads" in mem.all_files()["main.py"]

    def test_try_guarded_fallback_fix_applied(self, tmp_path, monkeypatch):
        # The live failure: a diagnosis fix that guards a possibly-missing
        # API in try/except with a working fallback was rejected by the
        # probe gate. It is runtime-safe and must now apply.
        monkeypatch.chdir(tmp_path)
        broken = ORIG.replace("loads", "laods")
        (tmp_path / "main.py").write_text(broken)
        mem = _Mem({"main.py": broken})
        guarded = (
            "ROOT CAUSE: bad call\n"
            "#### [FILE]: main.py\n"
            "```python\n"
            "import json\n"
            "A = 1\n"
            "B = 2\n"
            "try:\n"
            "    print(json.lodas('{}'))\n"
            "except Exception:\n"
            "    print(json.loads('{}'))\n"
            "```\n"
        )
        applied = _apply_fix(guarded, Executor(), mem, _Display(), 0)[0]
        assert applied
        assert "except Exception" in mem.all_files()["main.py"]


class TestDiagnosisRejectionFeedback:
    def test_rejection_reason_recorded_and_reaches_next_attempt(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text(ORIG)
        mem = _Mem({"main.py": ORIG})
        _apply_fix(_diag("json.lodas('{}')"), Executor(), mem, _Display(), 0)
        assert "json.lodas" in getattr(mem, "_last_fix_rejection", "")

        from agentchanti.orchestrator.diagnosis import _diagnose_failure

        class _LLM:
            prompt = ""

            @staticmethod
            def generate_response(p):
                _LLM.prompt = p
                return "no fix"

        _diagnose_failure(
            "fix the game", "CODE", "AttributeError: boom", mem, _LLM,
            _Display(), 0, previous_diagnosis="old output")
        assert "REJECTED by a runtime API check" in _LLM.prompt
        assert "json.lodas" in _LLM.prompt

    def test_accepted_fix_clears_stale_rejection(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        broken = ORIG.replace("loads", "laods")
        (tmp_path / "main.py").write_text(broken)
        mem = _Mem({"main.py": broken})
        mem._last_fix_rejection = "stale reason from an earlier step"
        _apply_fix(_diag("json.loads('{}')"), Executor(), mem, _Display(), 0)
        assert mem._last_fix_rejection == ""

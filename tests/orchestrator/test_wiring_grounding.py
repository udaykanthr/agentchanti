"""WiringVerification must not apply rewrites that use missing APIs.

The wiring stage regenerates whole files from an LLM pass; without a probe
gate it can reintroduce removed APIs that earlier stages already fixed
(observed live: a correct arcade 3.x fix overwritten with the removed
``draw_rectangle_filled``).  These tests run the real probe against the
current interpreter — ``json.laods`` is always missing, ``json.loads``
always present.
"""

import re
from agentchanti.executor import Executor
from agentchanti.orchestrator.pipeline import (
    _resolve_fix_scope_files,
    run_wiring_verification,
)


class _FakeMemory:
    def __init__(self, files):
        self._files = dict(files)

    def all_files(self):
        return dict(self._files)

    def update(self, files):
        self._files.update(files)

    def get(self, path):
        return self._files.get(path)


class _FakeDisplay:
    def show_status(self, *_args, **_kwargs):
        pass


class _Coder:
    def __init__(self, response):
        self.prompts = []
        self._response = response
        outer = self

        class _LLM:
            @staticmethod
            def generate_response(prompt):
                outer.prompts.append(prompt)
                return outer._response

        self.llm_client = _LLM()


SRC = "import json\nprint(json.loads('{}'))\n"


def _run(coder, memory, tmp_path):
    return run_wiring_verification(
        memory=memory, executor=Executor(), coder=coder,
        display=_FakeDisplay(), task="build a thing",
        language="python", project_root=str(tmp_path))


class TestWiringGrounding:
    def test_ungrounded_rewrite_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        memory = _FakeMemory({"main.py": SRC})
        bad = ("#### [FILE]: main.py\n```python\nimport json\n"
               "print(json.laods('{}'))\n```")
        ok, err = _run(_Coder(bad), memory, tmp_path)
        assert ok and err == ""  # rejection is non-fatal
        # neither memory nor disk was touched
        assert memory.all_files()["main.py"] == SRC
        assert not (tmp_path / "main.py").exists()

    def test_grounded_rewrite_applied(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        memory = _FakeMemory({"main.py": SRC})
        good = ("#### [FILE]: main.py\n```python\nimport json\n"
                "print(json.load)\n```")
        ok, err = _run(_Coder(good), memory, tmp_path)
        assert ok and err == ""
        assert "json.load" in memory.all_files()["main.py"]
        assert (tmp_path / "main.py").exists()

    def test_prompt_contains_installed_versions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        memory = _FakeMemory({"main.py": SRC})
        coder = _Coder("NO_ISSUES_FOUND")
        ok, _ = _run(coder, memory, tmp_path)
        assert ok
        prompt = coder.prompts[0]
        # Populated, not a specific unrelated package — see the note in
        # tests/orchestrator/test_smoke_test.py::test_fix_prompt_is_grounded.
        assert re.search(r"\w+==\d", prompt), prompt[:400]
        assert "EXACT versions" in prompt


class TestWiringScopeGuard:
    """Fixes touching files outside the verification context are rejected
    whole — the LLM otherwise invents entire files it never saw (observed:
    core/urls.py rewritten blind with a new app_name namespace)."""

    def test_out_of_scope_rewrite_rejects_whole_set(self, tmp_path,
                                                    monkeypatch):
        monkeypatch.chdir(tmp_path)
        memory = _FakeMemory({"main.py": SRC})
        stray = (
            "#### [FILE]: main.py\n```python\nimport json\n"
            "print(json.load)\n```\n\n"
            "#### [FILE]: core/urls.py\n```python\napp_name = 'core'\n```"
        )
        ok, err = _run(_Coder(stray), memory, tmp_path)
        assert ok and err == ""  # rejection is non-fatal
        # nothing was written — not even the in-scope rewrite, which may
        # depend on the invented file
        assert memory.all_files()["main.py"] == SRC
        assert not (tmp_path / "main.py").exists()
        assert not (tmp_path / "core" / "urls.py").exists()

    def test_prompt_forbids_inventing_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        coder = _Coder("NO_ISSUES_FOUND")
        ok, _ = _run(coder, _FakeMemory({"main.py": SRC}), tmp_path)
        assert ok
        assert "NEVER invent new" in coder.prompts[0]


class TestResolveFixScopeFiles:
    """Suffix matches must not be shadowed by a basename hit on an
    earlier same-named file (observed: core/urls.py and config/urls.py
    both resolving to accounts/urls.py, written first)."""

    _MEMORY = {
        "spacious_site/accounts/urls.py": "accounts urlconf",
        "spacious_site/core/urls.py": "core urlconf",
        "spacious_site/config/urls.py": "config urlconf",
    }

    def test_suffix_match_beats_basename_shadow(self):
        result = _resolve_fix_scope_files(
            ["core/urls.py", "config/urls.py"], [],
            _FakeMemory(self._MEMORY))
        assert result.get("spacious_site/core/urls.py") == "core urlconf"
        assert result.get("spacious_site/config/urls.py") == "config urlconf"
        assert "spacious_site/accounts/urls.py" not in result

    def test_bare_basename_collects_all_matches(self):
        result = _resolve_fix_scope_files(
            ["urls.py"], [], _FakeMemory(self._MEMORY))
        assert set(result) == set(self._MEMORY)

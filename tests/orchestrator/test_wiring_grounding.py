"""WiringVerification must not apply rewrites that use missing APIs.

The wiring stage regenerates whole files from an LLM pass; without a probe
gate it can reintroduce removed APIs that earlier stages already fixed
(observed live: a correct arcade 3.x fix overwritten with the removed
``draw_rectangle_filled``).  These tests run the real probe against the
current interpreter — ``json.laods`` is always missing, ``json.loads``
always present.
"""

from agentchanti.executor import Executor
from agentchanti.orchestrator.pipeline import run_wiring_verification


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
        assert "pytest==" in coder.prompts[0]
        assert "EXACT versions" in coder.prompts[0]

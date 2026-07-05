"""Tests for the runtime smoke verification stage."""

import os
import textwrap

from agentchanti.executor import Executor
from agentchanti.orchestrator.smoke_test import (
    _attempt_fix,
    _files_from_traceback,
    _files_mentioning_error_symbols,
    _is_headless_failure,
    _launch,
    _same_crash,
    build_run_command,
    find_python_entrypoint,
    run_smoke_verification,
)


MAIN_GUARD = 'if __name__ == "__main__":\n    main()\n'


class TestFindEntrypoint:
    def test_finds_main_guard_file(self):
        files = {
            "src/app.py": f"def main():\n    pass\n\n{MAIN_GUARD}",
            "src/lib.py": "def helper():\n    pass\n",
        }
        assert find_python_entrypoint(files) == "src/app.py"

    def test_prefers_main_py(self):
        files = {
            "src/game.py": f"def main():\n    pass\n{MAIN_GUARD}",
            "src/main.py": f"def main():\n    pass\n{MAIN_GUARD}",
        }
        assert find_python_entrypoint(files) == "src/main.py"

    def test_skips_test_files(self):
        files = {
            "tests/test_app.py": f"def main():\n    pass\n{MAIN_GUARD}",
            "conftest.py": f"def main():\n    pass\n{MAIN_GUARD}",
        }
        assert find_python_entrypoint(files) is None

    def test_no_entrypoint(self):
        files = {"src/lib.py": "def helper():\n    pass\n"}
        assert find_python_entrypoint(files) is None


class TestBuildRunCommand:
    def test_package_module_form(self, tmp_path, monkeypatch):
        (tmp_path / "src" / "pkg").mkdir(parents=True)
        (tmp_path / "src" / "__init__.py").write_text("")
        (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
        (tmp_path / "src" / "pkg" / "main.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert build_run_command("src/pkg/main.py") == "python -m src.pkg.main"

    def test_dunder_main_runs_package(self, tmp_path, monkeypatch):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        (tmp_path / "pkg" / "__main__.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert build_run_command("pkg/__main__.py") == "python -m pkg"

    def test_script_form_without_package(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert build_run_command("run.py") == 'python "run.py"'

    def test_script_form_when_no_init_chain(self, tmp_path, monkeypatch):
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "main.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert build_run_command("scripts/main.py") == 'python "scripts/main.py"'

    def test_namespace_parent_still_uses_module_form(self, tmp_path, monkeypatch):
        # src/ has no __init__.py (namespace pkg) but src/pkg/ does —
        # `python -m` is required or relative imports break
        (tmp_path / "src" / "pkg").mkdir(parents=True)
        (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
        (tmp_path / "src" / "pkg" / "main.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert build_run_command("src/pkg/main.py") == "python -m src.pkg.main"


class TestHeadlessDetection:
    def test_no_display_is_environmental(self):
        assert _is_headless_failure(
            "pyglet.canvas.xlib.NoSuchDisplayException: "
            "Cannot connect to None")

    def test_code_crash_is_not_environmental(self):
        assert not _is_headless_failure(
            "RuntimeError: start_render() can only be called once")


class TestFilesFromTraceback:
    def test_maps_traceback_paths_to_memory(self):
        out = (
            'Traceback (most recent call last):\n'
            '  File "C:\\proj\\src\\game.py", line 4, in on_draw\n'
            '    arcade.start_render()\n'
            'RuntimeError: boom\n'
        )
        files = {"src/game.py": "...", "src/other.py": "..."}
        assert _files_from_traceback(out, files) == ["src/game.py"]

    def test_ignores_stdlib_paths(self):
        out = '  File "C:\\Python313\\lib\\runpy.py", line 88, in run\n'
        assert _files_from_traceback(out, {"src/game.py": "..."}) == []


class TestLaunch:
    def test_clean_quick_exit_is_success(self, tmp_path):
        script = tmp_path / "ok.py"
        script.write_text('print("hello")\n')
        ok, out = _launch(Executor(), f'python "{script}"')
        assert ok
        assert "hello" in out

    def test_crash_is_captured(self, tmp_path):
        script = tmp_path / "boom.py"
        script.write_text('raise RuntimeError("smoke boom")\n')
        ok, out = _launch(Executor(), f'python "{script}"')
        assert not ok
        assert "smoke boom" in out

    def test_long_running_process_is_success_and_killed(self, tmp_path):
        script = tmp_path / "server.py"
        script.write_text("import time\ntime.sleep(60)\n")
        executor = Executor()
        ok, _ = _launch(executor, f'python "{script}"')
        assert ok
        # the process tree must have been cleaned up
        assert executor._background_processes == []


class _FakeMemory:
    def __init__(self, files):
        self._files = dict(files)

    def all_files(self):
        return dict(self._files)

    def update(self, files):
        self._files.update(files)


class _FakeDisplay:
    def show_status(self, *_args, **_kwargs):
        pass


class TestRunSmokeVerification:
    def test_skips_without_entrypoint(self):
        memory = _FakeMemory({"src/lib.py": "def f():\n    pass\n"})
        ok, err = run_smoke_verification(
            memory, Executor(), coder=None, display=_FakeDisplay(),
            task="t", language="python")
        assert ok and err == ""

    def test_skips_non_python(self):
        memory = _FakeMemory({"src/app.js": "console.log(1)"})
        ok, _ = run_smoke_verification(
            memory, Executor(), coder=None, display=_FakeDisplay(),
            task="t", language="javascript")
        assert ok

    def test_disabled_via_cfg(self):
        class Cfg:
            SMOKE_TEST_ENABLED = False

        memory = _FakeMemory({})
        ok, _ = run_smoke_verification(
            memory, Executor(), coder=None, display=_FakeDisplay(),
            task="t", language="python", cfg=Cfg())
        assert ok

    def test_crashing_app_fails_pipeline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = textwrap.dedent("""\
            def main():
                raise RuntimeError("intentional smoke crash")

            if __name__ == "__main__":
                main()
        """)
        (tmp_path / "main.py").write_text(content)

        class _NoFixCoder:
            class llm_client:
                @staticmethod
                def generate_response(_prompt):
                    return "no code blocks here"

        memory = _FakeMemory({"main.py": content})
        ok, err = run_smoke_verification(
            memory, Executor(), coder=_NoFixCoder(), display=_FakeDisplay(),
            task="t", language="python", max_fix_attempts=1)
        assert not ok
        assert "intentional smoke crash" in err

    def test_working_app_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = 'def main():\n    print("ok")\n\nif __name__ == "__main__":\n    main()\n'
        (tmp_path / "main.py").write_text(content)
        memory = _FakeMemory({"main.py": content})
        ok, err = run_smoke_verification(
            memory, Executor(), coder=None, display=_FakeDisplay(),
            task="t", language="python")
        assert ok and err == ""


class _ScriptedCoder:
    """Coder stub returning canned responses in order; records prompts."""

    def __init__(self, responses):
        self.prompts = []
        self._responses = list(responses)
        outer = self

        class _LLM:
            @staticmethod
            def generate_response(prompt):
                outer.prompts.append(prompt)
                return outer._responses.pop(0)

        self.llm_client = _LLM()


CRASH = (
    "Traceback (most recent call last):\n"
    '  File "C:\\proj\\main.py", line 2, in <module>\n'
    "AttributeError: module 'json' has no attribute 'laods'\n"
)

# `json.laods` does not exist — the probe must reject this fix
BAD_FIX = (
    "#### [FILE]: main.py\n```python\nimport json\n"
    'print(json.laods("{}"))\n```'
)
GOOD_FIX = (
    "#### [FILE]: main.py\n```python\nimport json\n"
    'print(json.loads("{}"))\n```'
)


class TestGroundedFixLoop:
    def test_ungrounded_fix_reasked_with_probe_errors(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original = 'import json\nprint(json.laods("{}"))\n'
        memory = _FakeMemory({"main.py": original})
        coder = _ScriptedCoder([BAD_FIX, GOOD_FIX])
        ok = _attempt_fix(
            CRASH, 'python "main.py"', memory, Executor(), coder, "main.py")
        assert ok
        assert len(coder.prompts) == 2
        assert "PREVIOUS ATTEMPT REJECTED" in coder.prompts[1]
        assert "json.laods" in coder.prompts[1]
        assert "laods" not in memory.all_files()["main.py"]

    def test_fix_rejected_when_still_ungrounded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        original = 'import json\nprint(json.laods("{}"))\n'
        memory = _FakeMemory({"main.py": original})
        coder = _ScriptedCoder([BAD_FIX, BAD_FIX])
        ok = _attempt_fix(
            CRASH, 'python "main.py"', memory, Executor(), coder, "main.py")
        assert not ok
        assert memory.all_files()["main.py"] == original
        assert not (tmp_path / "main.py").exists()

    def test_fix_prompt_is_grounded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        memory = _FakeMemory({"main.py": "import json\n"})
        coder = _ScriptedCoder([GOOD_FIX])
        _attempt_fix(
            CRASH, 'python "main.py"', memory, Executor(), coder, "main.py")
        assert "INSTALLED PACKAGES" in coder.prompts[0]
        assert "pytest==" in coder.prompts[0]
        assert "#### [FILE]:" in coder.prompts[0]


# ── Swallowed-traceback targeting ──────────────────────────────────
# The live failure: main.py's try/except printed the error instead of
# raising, so the crash output had no File lines — the fix loop could
# only edit the entry point while the failing call sat in game.py.

GAME_SRC = (
    "class W:\n"
    "    def on_draw(self):\n"
    "        start_render()\n"
)
MAIN_SRC = (
    "def main():\n"
    "    try:\n"
    "        W()\n"
    "    except Exception as e:\n"
    "        print('Failed to start the game:', e)\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)
SWALLOWED_CRASH = (
    "Failed to start the game: start_render() can only be called once "
    "during the application's lifetime... you likely intended to call "
    "clear() instead."
)


class TestErrorSymbolTargeting:
    def test_implicated_file_found_by_symbol(self):
        files = {
            "game.py": GAME_SRC,
            "main.py": MAIN_SRC,
            "tests/test_game.py": "start_render(",
            "snake.py": "def move(self):\n    pass\n",
        }
        hits = _files_mentioning_error_symbols(SWALLOWED_CRASH, files)
        assert "game.py" in hits
        assert "tests/test_game.py" not in hits  # test files excluded
        assert "snake.py" not in hits

    def test_stopword_symbols_ignored(self):
        files = {"a.py": "print('x')\n"}
        assert _files_mentioning_error_symbols(
            "print() failed somehow", files) == []

    def test_no_call_syntax_no_matches(self):
        files = {"a.py": "start_render\n"}
        assert _files_mentioning_error_symbols(
            "something went wrong", files) == []

    def test_attempt_fix_reaches_implicated_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        memory = _FakeMemory({"main.py": MAIN_SRC, "game.py": GAME_SRC})
        fix = (
            "#### [FILE]: game.py\n```python\n"
            "class W:\n"
            "    def on_draw(self):\n"
            "        self.clear()\n"
            "```"
        )
        coder = _ScriptedCoder([fix])
        fixed = _attempt_fix(
            SWALLOWED_CRASH, 'python "main.py"', memory, Executor(),
            coder, "main.py")
        assert fixed == ["game.py"]
        assert "start_render" not in memory.all_files()["game.py"]
        assert "game.py" in coder.prompts[0]  # source was shown to the LLM

    def test_stuck_note_reaches_prompt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        memory = _FakeMemory({"main.py": MAIN_SRC, "game.py": GAME_SRC})
        fix = "#### [FILE]: game.py\n```python\nX = 1\n```"
        coder = _ScriptedCoder([fix])
        _attempt_fix(
            SWALLOWED_CRASH, 'python "main.py"', memory, Executor(),
            coder, "main.py",
            stuck_note="\n\nIMPORTANT: the crash output is UNCHANGED")
        assert "crash output is UNCHANGED" in coder.prompts[0]


class TestSameCrash:
    def test_identical_crashes_match(self):
        assert _same_crash("boom\nline", "boom\nline")

    def test_different_crashes_do_not(self):
        assert not _same_crash("error A", "error B")

    def test_empty_never_matches(self):
        assert not _same_crash("", "x")
        assert not _same_crash("x", "")

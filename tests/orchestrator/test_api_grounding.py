"""Tests for API grounding — installed-version context and attribute probing.

The probe runs against THIS interpreter's environment (pytest is installed),
which lets us assert real behavior without mocks: a fabricated attribute on
an installed package must be flagged; real attributes must not.
"""

from agentchanti.executor import Executor
from agentchanti.orchestrator.api_grounding import (
    _collect_checks,
    format_packages_with_versions,
    get_installed_package_versions,
    local_top_levels_from_files,
    probe_api_usage,
)


class TestCollectChecks:
    def test_module_attribute_usage(self):
        files = {"a.py": "import arcade\narcade.draw_rectangle_filled(1)\n"}
        checks = _collect_checks(files)
        assert ("arcade", "draw_rectangle_filled") in checks

    def test_aliased_import(self):
        files = {"a.py": "import numpy as np\nnp.zeros(3)\n"}
        checks = _collect_checks(files)
        assert ("numpy", "zeros") in checks

    def test_from_import(self):
        files = {"a.py": "from arcade import Window\n"}
        checks = _collect_checks(files)
        assert ("arcade", "Window") in checks

    def test_relative_import_skipped(self):
        files = {"a.py": "from .constants import GRID\n"}
        assert _collect_checks(files) == set()

    def test_local_module_skipped(self):
        files = {"a.py": "import src.snake\nsrc.foo()\n"}
        checks = _collect_checks(files, local_top_levels={"src"})
        assert checks == set()

    def test_try_block_imports_skipped(self):
        # try/except import fallbacks are deliberate — not probe targets
        files = {"a.py": (
            "try:\n"
            "    import fancy_lib\n"
            "except ImportError:\n"
            "    fancy_lib = None\n"
        )}
        assert _collect_checks(files) == set()

    def test_try_block_attribute_usage_skipped(self):
        # try/except API fallbacks (version shims) are runtime-safe —
        # flagging them rejects functionally correct fixes
        files = {"a.py": (
            "import arcade\n"
            "try:\n"
            "    arcade.new_api()\n"
            "except Exception:\n"
            "    arcade.old_api()\n"
        )}
        checks = _collect_checks(files)
        assert ("arcade", "new_api") not in checks
        assert ("arcade", "old_api") not in checks

    def test_dunder_attrs_skipped(self):
        files = {"a.py": "import os\nprint(os.__file__)\n"}
        checks = _collect_checks(files)
        assert ("os", "__file__") not in checks

    def test_syntax_error_ignored(self):
        files = {"a.py": "def broken(:\n"}
        assert _collect_checks(files) == set()


class TestCollectApiUsages:
    def test_lists_module_attr_usages(self):
        from agentchanti.orchestrator.api_grounding import collect_api_usages
        files = {"a.py": "import json\njson.loads('{}')\n"}
        assert "json.loads" in collect_api_usages(files)

    def test_local_modules_excluded(self):
        from agentchanti.orchestrator.api_grounding import collect_api_usages
        files = {"a.py": "import src.snake\nsrc.foo()\n"}
        assert collect_api_usages(files, {"src"}) == []


class TestLocalTopLevels:
    def test_derives_from_paths(self):
        tops = local_top_levels_from_files(
            ["src/snake_game/main.py", "utils.py", "tests\\test_a.py"])
        assert tops == {"src", "utils", "tests"}


class TestFormatPackagesWithVersions:
    def test_appends_known_versions(self):
        out = format_packages_with_versions(
            ["arcade", "pytest"], {"arcade": "3.3.3"})
        assert out == ["arcade==3.3.3", "pytest"]

    def test_no_versions(self):
        assert format_packages_with_versions(["arcade"], {}) == ["arcade"]


class TestProbeRealEnvironment:
    """Probe against the current interpreter — pytest is always installed."""

    def test_missing_attribute_flagged(self):
        files = {"a.py": "import pytest\npytest.this_api_does_not_exist()\n"}
        errors = probe_api_usage(files, Executor())
        assert len(errors) == 1
        assert "pytest.this_api_does_not_exist" in errors[0]

    def test_real_attribute_passes(self):
        files = {"a.py": "import pytest\npytest.skip\npytest.mark.foo\n"}
        assert probe_api_usage(files, Executor()) == []

    def test_missing_module_flagged(self):
        files = {"a.py": "import surely_not_a_real_module_xyz\n"}
        errors = probe_api_usage(files, Executor())
        assert len(errors) == 1
        assert "surely_not_a_real_module_xyz" in errors[0]

    def test_try_guarded_fallback_not_flagged(self):
        # The exact pattern a diagnosis LLM produced live: a guarded call
        # to a missing API with a working fallback — runtime-safe (the
        # AttributeError is caught), so the probe must not reject it
        files = {"a.py": (
            "import json\n"
            "try:\n"
            "    json.laods('{}')\n"
            "except Exception:\n"
            "    json.loads('{}')\n"
        )}
        assert probe_api_usage(files, Executor()) == []

    def test_submodule_from_import_not_false_positive(self):
        # `from os import path` — path is a module attr; must not be flagged
        files = {"a.py": "from os import path\n"}
        assert probe_api_usage(files, Executor()) == []


class TestGetInstalledVersions:
    def test_returns_versions_from_current_env(self):
        versions = get_installed_package_versions()
        assert isinstance(versions, dict)
        assert "pytest" in versions
        assert versions["pytest"][0].isdigit()


class TestCloseMatchSuggestions:
    def test_suggests_similar_api_name(self):
        # json.load exists; "laods" is a typo close to "loads"/"load"
        files = {"a.py": "import json\njson.laods('{}')\n"}
        errors = probe_api_usage(files, Executor())
        assert len(errors) == 1
        assert "Did you mean" in errors[0]
        assert "json.loads" in errors[0] or "json.load" in errors[0]

    def test_no_hint_when_nothing_close(self):
        files = {"a.py": "import json\njson.zzqqxxwwyy()\n"}
        errors = probe_api_usage(files, Executor())
        assert len(errors) == 1
        assert "Did you mean" not in errors[0]


class TestIsInstallCommand:
    def test_pip_install(self):
        from agentchanti.orchestrator.api_grounding import is_install_command
        assert is_install_command("pip install arcade")
        assert is_install_command(
            "call venv\\Scripts\\activate && pip install arcade")
        assert is_install_command("python -m pip install -r requirements.txt")
        assert is_install_command("npm install express")

    def test_non_install(self):
        from agentchanti.orchestrator.api_grounding import is_install_command
        assert not is_install_command("python -m pytest")
        assert not is_install_command("pip list")
        assert not is_install_command("mkdir src")


class TestRefreshInstalledVersions:
    def test_refresh_populates_context(self):
        from agentchanti.orchestrator.api_grounding import (
            refresh_installed_versions)

        class Ctx:
            installed_packages = []
            installed_versions = {}

        ctx = Ctx()
        refresh_installed_versions(ctx, executor=Executor())
        assert ctx.installed_versions.get("pytest")
        assert "pytest" in ctx.installed_packages
        assert "pip" not in ctx.installed_packages  # noise filtered

    def test_none_context_is_noop(self):
        from agentchanti.orchestrator.api_grounding import (
            refresh_installed_versions)
        refresh_installed_versions(None)  # must not raise

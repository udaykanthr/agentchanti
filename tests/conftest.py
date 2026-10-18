"""Shared test isolation.

Anything the library persists outside the repo is redirected to a
per-test temporary location here. A unit test must never read or write
the developer's real home directory: the reasoning-effort floor store
(``~/.agentchanti/effort_floors.json``) was written by the suite itself
on first run, which both leaked state between tests in the same session
and pinned the developer's *real* configured models to a reduced
reasoning effort for the store's whole TTL.
"""

import os

import pytest

from agentchanti.llm import openai_client


@pytest.fixture(autouse=True)
def _isolate_effort_floor_store(tmp_path, monkeypatch):
    """Give every test its own empty effort-floor store."""
    store = os.path.join(str(tmp_path), "effort_floors.json")
    monkeypatch.setattr(openai_client, "_effort_floor_store", lambda: store)
    return store


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Point `~` at a per-test directory, for the same reason as above.

    `_find_config_file` searches CWD and then the HOME directory, so any
    test that calls `Config.load()` has always been able to read the
    developer's real `.agentchanti.yaml` — a provider, a review mode or an
    MCP server on one machine and not another, deciding a test's outcome.

    `mcp add --scope user` made it a WRITE, and did so immediately:
    measured 2026-10-12, a test asserting that scope was refused ran after
    the refusal became support, and left a real server in the developer's
    own `~/.agentchanti.yaml`. The test was wrong about the behaviour; the
    damage was possible because the home directory was real.

    A test that genuinely needs a home config sets one up inside this
    directory, which is also what makes such a test reproducible; request
    the fixture by name to get the path.

    Made by `tmp_path_factory` rather than inside `tmp_path`, because
    `tmp_path` is a test's own working directory in dozens of tests and one
    of them asserts that an empty directory lists nothing — a fake home
    sitting in it is a file the test did not put there.
    """
    home = tmp_path_factory.mktemp("home")
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    # Windows' `expanduser` consults HOMEDRIVE/HOMEPATH before USERPROFILE
    # in some paths, so leaving them pointing at the real profile would
    # leave the hole open on the one platform this project runs on most.
    drive, rest = os.path.splitdrive(str(home))
    monkeypatch.setenv("HOMEDRIVE", drive)
    monkeypatch.setenv("HOMEPATH", rest)
    return home

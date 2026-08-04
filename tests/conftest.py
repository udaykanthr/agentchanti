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

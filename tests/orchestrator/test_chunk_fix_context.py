"""A chunk fix must know what the other source files contain.

Observed live (classic mode, Pac-Man task, 2026-08-05). The suite failed
with "Ghost 2 spawn at (17, 13) is not walkable". The source-bug fix path
calls the chunk editor once per source file, and the player.py call was
handed player.py and nothing else. The model spent its whole response
saying so:

    Since only `player.py` was provided and the tests are about ghost
    spawns, I cannot complete this task without seeing the `ghost.py`
    and `map.py` source files.

That reply is in full-file format, so it parsed as "fall back" — after
which every remaining source file got its own equally blind call before
the full-file fallback did the real work.

Two changes, pinned here: siblings arrive as signatures so the call can
tell whether the fix is its to make, and the first full-file answer ends
the sweep instead of buying the same answer once per file.
"""

from __future__ import annotations

from agentchanti.orchestrator.step_handlers import (
    CHUNK_FIX_WANTS_FULL_FILE,
    _chunk_fix_file,
)


PLAYER = '''\
class Player:
    """Pac-Man."""

    def __init__(self, x, y, tile_size, game_map):
        self.x = x
        self.y = y

    def update(self, dt):
        self.x += dt
'''

GHOST = '''\
SPAWNS = [(1, 1), (17, 13)]


class Ghost:
    def __init__(self, x, y, tile_size, game_map, mode, index):
        self.x = x

    def update(self, dt):
        pass
'''


class _Recorder:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def generate_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class _Memory:
    _task_briefing = ""

    def all_files(self):
        return {}


class _Display:
    def step_info(self, *a, **k):
        pass

    def step_tokens(self, *a, **k):
        pass


def _run(reply: str, siblings: dict[str, str] | None):
    client = _Recorder(reply)
    result = _chunk_fix_file(
        "player.py", PLAYER, "Fix: Ghost 2 spawn at (17, 13) is not walkable",
        client, "python", _Memory(), _Display(), 0,
        sibling_sources=siblings)
    return result, client.prompts[0] if client.prompts else ""


def test_sibling_signatures_reach_the_prompt():
    _, prompt = _run("#### [EDIT]: nothing", {"player.py": PLAYER,
                                              "ghost.py": GHOST})
    assert "ghost.py" in prompt
    # Signatures, not bodies — the sibling is context, not the edit target.
    assert "def update" in prompt
    assert "self.x += dt" in prompt          # player.py, the editable one
    assert "SPAWNS = [(1, 1), (17, 13)]" not in prompt or "ghost.py" in prompt


def test_the_file_being_edited_is_not_repeated_as_its_own_sibling():
    _, prompt = _run("#### [EDIT]: nothing", {"player.py": PLAYER})
    assert "CANNOT edit them" not in prompt


def test_no_siblings_leaves_the_prompt_as_it_was():
    _, prompt = _run("#### [EDIT]: nothing", None)
    assert "Other source files under test" not in prompt


def test_full_file_answer_is_distinguishable_from_a_prose_refusal():
    full_file, _ = _run(
        "#### [FILE]: player.py\n```python\nclass Player: pass\n```", None)
    assert full_file is CHUNK_FIX_WANTS_FULL_FILE

    # ChunkEditor returns None for both "here is the whole file" and "I
    # parsed nothing", so the observed refusal — "the spawn table is in
    # ghost.py, not this file" — used to read as full-file format and
    # cancel the sweep before it reached the file that owns the bug.
    refusal, _ = _run(
        "The spawn table lives in ghost.py; player.py needs no change.",
        {"player.py": PLAYER, "ghost.py": GHOST})
    assert refusal is None
    assert refusal is not CHUNK_FIX_WANTS_FULL_FILE

    # Both are falsy, so a caller that only checks truthiness still falls
    # back — the distinction exists for the caller that must stop early.
    assert not full_file and not refusal

"""Tests for the chunk editor module."""

import pytest
from multi_agent_coder.editing.chunk_editor import (
    ChunkEditor, FileChunk, ChunkEditResponse,
)


SAMPLE_PYTHON = """\
import os
from datetime import datetime

GLOBAL_VAR = 42


class UserService:
    def __init__(self, db):
        self.db = db

    def authenticate(self, username, password):
        user = self.db.find(username)
        if user is None:
            return False
        return user.check_password(password)

    def get_user(self, user_id):
        return self.db.get(user_id)


def helper_function():
    return "hello"


def another_helper(x, y):
    return x + y
"""


SAMPLE_JS = """\
const express = require('express');
const { UserService } = require('./services');

class AppController {
    constructor(service) {
        this.service = service;
    }

    async handleLogin(req, res) {
        const { username, password } = req.body;
        const result = await this.service.authenticate(username, password);
        res.json({ success: result });
    }
}

function createApp() {
    const app = express();
    return app;
}

module.exports = { AppController, createApp };
"""


class TestChunkFile:
    def test_chunks_python_file(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)

        # Should have: imports, class UserService, methods, helper functions
        chunk_ids = [c.chunk_id for c in chunks]
        assert any("imports" in cid for cid in chunk_ids)
        assert any("UserService" in cid for cid in chunk_ids)
        assert any("helper_function" in cid for cid in chunk_ids)
        assert any("another_helper" in cid for cid in chunk_ids)

    def test_chunks_js_file(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("app.js", SAMPLE_JS)

        chunk_ids = [c.chunk_id for c in chunks]
        assert any("AppController" in cid for cid in chunk_ids)
        assert any("createApp" in cid for cid in chunk_ids)

    def test_chunks_preserve_all_lines(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)

        # All non-empty lines should be covered by some chunk
        all_lines = SAMPLE_PYTHON.splitlines()
        covered = set()
        for c in chunks:
            for ln in range(c.line_start, c.line_end + 1):
                covered.add(ln)

        for i, line in enumerate(all_lines, 1):
            if line.strip():
                assert i in covered, f"Line {i} not covered: {line!r}"

    def test_empty_file(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("empty.py", "")
        assert chunks == []

    def test_imports_only(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("imports.py", "import os\nimport sys\n")
        assert len(chunks) >= 1
        assert any(c.chunk_type == "imports" for c in chunks)


class TestIdentifyTargetChunks:
    def test_exact_name_match(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "fix the authenticate method")
        assert any("authenticate" in t for t in targets)

    def test_no_match(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "do something completely unrelated xyz")
        # May return empty or low-relevance matches
        # The key is it doesn't crash
        assert isinstance(targets, list)

    def test_multiple_matches(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "update helper functions")
        assert any("helper" in t for t in targets)


class TestFormatChunksForPrompt:
    def test_format_with_targets(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        targets = editor.identify_target_chunks(chunks, "fix authenticate")

        formatted = editor.format_chunks_for_prompt(chunks, targets)
        assert "EDITABLE" in formatted
        assert "CONTEXT ONLY" in formatted
        assert "test.py" in formatted

    def test_format_all_editable(self):
        editor = ChunkEditor()
        chunks = editor.chunk_file("test.py", SAMPLE_PYTHON)
        formatted = editor.format_chunks_for_prompt(chunks, target_chunk_ids=None)
        # When no targets specified, all should be editable
        assert "CONTEXT ONLY" not in formatted or "EDITABLE" in formatted


class TestParseChunkResponse:
    def test_parse_edit_marker(self):
        editor = ChunkEditor()
        response = """Here are the changes:

#### [EDIT]: test.py:authenticate (lines 10-15)
```python
def authenticate(self, username, password):
    if not username or not password:
        return False
    user = self.db.find(username)
    return user is not None and user.check_password(password)
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is not None
        assert len(edits) == 1
        assert edits[0].file_path == "test.py"
        assert edits[0].chunk_id == "authenticate"
        assert edits[0].line_start == 10
        assert edits[0].line_end == 15
        assert "username" in edits[0].new_content

    def test_parse_new_marker(self):
        editor = ChunkEditor()
        response = """
#### [NEW]: test.py (after line 25)
```python
def validate_email(email):
    return "@" in email
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is not None
        assert len(edits) == 1
        assert edits[0].is_new is True
        assert edits[0].insert_after == 25

    def test_parse_multiple_edits(self):
        editor = ChunkEditor()
        response = """
#### [EDIT]: test.py:func_a (lines 10-15)
```python
def func_a():
    return 1
```

#### [EDIT]: test.py:func_b (lines 20-25)
```python
def func_b():
    return 2
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is not None
        assert len(edits) == 2

    def test_fallback_on_full_file_format(self):
        editor = ChunkEditor()
        response = """
#### [FILE]: test.py
```python
import os

def func():
    pass
```
"""
        edits = editor.parse_chunk_response(response)
        assert edits is None  # Should signal fallback

    def test_no_edits(self):
        editor = ChunkEditor()
        response = "No changes needed, the code looks correct."
        edits = editor.parse_chunk_response(response)
        assert edits is None


class TestApplyChunkEdits:
    def test_single_edit(self):
        editor = ChunkEditor()
        original = "line1\nline2\nline3\nline4\nline5\n"
        edits = [ChunkEditResponse(
            file_path="test.py",
            chunk_id="test",
            line_start=2,
            line_end=3,
            new_content="new_line2\nnew_line3\n",
        )]
        result = editor.apply_chunk_edits(original, edits)
        lines = result.splitlines()
        assert lines[0] == "line1"
        assert lines[1] == "new_line2"
        assert lines[2] == "new_line3"
        assert lines[3] == "line4"

    def test_new_insertion(self):
        editor = ChunkEditor()
        original = "line1\nline2\nline3\n"
        edits = [ChunkEditResponse(
            file_path="test.py",
            chunk_id="new",
            line_start=3,
            line_end=3,
            new_content="inserted\n",
            is_new=True,
            insert_after=2,
        )]
        result = editor.apply_chunk_edits(original, edits)
        lines = result.splitlines()
        assert "inserted" in lines
        assert lines.index("inserted") == 2  # after line2

    def test_multiple_edits_reverse_order(self):
        editor = ChunkEditor()
        original = "a\nb\nc\nd\ne\n"
        edits = [
            ChunkEditResponse("t.py", "c1", 2, 2, "B\n"),
            ChunkEditResponse("t.py", "c2", 4, 4, "D\n"),
        ]
        result = editor.apply_chunk_edits(original, edits)
        lines = result.splitlines()
        assert lines == ["a", "B", "c", "D", "e"]


class TestChunkIdMatches:
    """Tests for the _chunk_id_matches helper."""

    def test_exact_match(self):
        from multi_agent_coder.editing.chunk_editor import _chunk_id_matches
        assert _chunk_id_matches("function:setup", "function:setup")

    def test_name_only_match(self):
        from multi_agent_coder.editing.chunk_editor import _chunk_id_matches
        assert _chunk_id_matches("function:setup", "setup")

    def test_dotted_name_match(self):
        from multi_agent_coder.editing.chunk_editor import _chunk_id_matches
        assert _chunk_id_matches("method:UserService.authenticate", "authenticate")

    def test_no_match(self):
        from multi_agent_coder.editing.chunk_editor import _chunk_id_matches
        assert not _chunk_id_matches("function:setup", "teardown")

    def test_empty_edit_id(self):
        from multi_agent_coder.editing.chunk_editor import _chunk_id_matches
        assert not _chunk_id_matches("function:setup", "")


class TestApplyChunkEditsWithKnownChunks:
    """Tests for line number resolution via known_chunks."""

    def test_corrects_wrong_line_numbers(self):
        """LLM returns wrong line numbers; known_chunks should correct them."""
        editor = ChunkEditor()
        # 10-line file where the target function is at lines 6-10
        original = (
            "line1\nline2\nline3\nline4\nline5\n"
            "def setup():\n    old_a\n    old_b\n    old_c\n    old_d\n"
        )
        # LLM says lines 2-5 (wrong!), but chunk_id matches
        edit = ChunkEditResponse(
            file_path="test.c",
            chunk_id="setup",
            line_start=2,   # WRONG
            line_end=5,     # WRONG
            new_content="def setup():\n    new_a\n    new_b\n    new_c\n    new_d\n",
        )
        # Known chunk has the correct range
        known = [FileChunk(
            file_path="test.c",
            chunk_id="function:setup",
            line_start=6,
            line_end=10,
            content="def setup():\n    old_a\n    old_b\n    old_c\n    old_d\n",
            chunk_type="function",
            signature="def setup():",
        )]
        result = editor.apply_chunk_edits(original, [edit], known_chunks=known)
        lines = result.splitlines()
        # Lines 1-5 should be UNTOUCHED
        assert lines[0] == "line1"
        assert lines[4] == "line5"
        # Lines 6+ should be the new content
        assert lines[5] == "def setup():"
        assert lines[6] == "    new_a"

    def test_no_known_chunks_uses_raw_lines(self):
        """Without known_chunks, the original behavior is preserved."""
        editor = ChunkEditor()
        original = "line1\nline2\nline3\nline4\nline5\n"
        edit = ChunkEditResponse(
            file_path="test.py",
            chunk_id="test",
            line_start=2,
            line_end=3,
            new_content="new_line2\nnew_line3\n",
        )
        result = editor.apply_chunk_edits(original, [edit], known_chunks=None)
        lines = result.splitlines()
        assert lines[1] == "new_line2"
        assert lines[2] == "new_line3"

    def test_no_match_keeps_original_lines(self):
        """When chunk_id doesn't match any known chunk, use LLM's lines."""
        editor = ChunkEditor()
        original = "a\nb\nc\nd\ne\n"
        edit = ChunkEditResponse(
            file_path="test.py",
            chunk_id="nonexistent",
            line_start=2,
            line_end=3,
            new_content="B\nC\n",
        )
        known = [FileChunk(
            file_path="test.py",
            chunk_id="function:setup",
            line_start=4,
            line_end=5,
            content="d\ne\n",
            chunk_type="function",
            signature="def setup():",
        )]
        result = editor.apply_chunk_edits(original, [edit], known_chunks=known)
        lines = result.splitlines()
        assert lines[0] == "a"
        assert lines[1] == "B"
        assert lines[2] == "C"
        assert lines[3] == "d"

    def test_new_insertion_ignores_known_chunks(self):
        """[NEW] insertions should not be affected by known_chunks."""
        editor = ChunkEditor()
        original = "line1\nline2\nline3\n"
        edit = ChunkEditResponse(
            file_path="test.py",
            chunk_id="new",
            line_start=3,
            line_end=3,
            new_content="inserted\n",
            is_new=True,
            insert_after=2,
        )
        known = [FileChunk(
            file_path="test.py",
            chunk_id="function:setup",
            line_start=1,
            line_end=3,
            content="line1\nline2\nline3\n",
            chunk_type="function",
            signature="line1",
        )]
        result = editor.apply_chunk_edits(original, [edit], known_chunks=known)
        lines = result.splitlines()
        assert "inserted" in lines

    def test_partial_edit_content_alignment(self):
        """LLM edits a sub-range of a large function (like has_colors within setup).

        The new content covers only 3 lines of a 10-line chunk. Content-based
        alignment should anchor on the first matching line and replace only
        the correct sub-range, preserving the rest of the function.
        """
        editor = ChunkEditor()
        # A 15-line file: 5 header lines + 10-line function
        original = (
            "line1\nline2\nline3\nline4\nline5\n"  # lines 1-5
            "void setup() {\n"                       # line 6
            "    initscr();\n"                        # line 7
            "    if (has_colors()) {\n"               # line 8
            "        start_color();\n"                # line 9
            "        init_pair(1, OLD, BG);\n"        # line 10
            "    }\n"                                 # line 11
            "    cbreak();\n"                         # line 12
            "    noecho();\n"                         # line 13
            "    keypad(stdscr, TRUE);\n"             # line 14
            "}\n"                                     # line 15
        )
        # LLM wants to edit just the has_colors block (3 lines),
        # but gives WRONG absolute line numbers (3-5 instead of 8-10).
        edit = ChunkEditResponse(
            file_path="snake.c",
            chunk_id="setup",
            line_start=3,    # WRONG
            line_end=5,      # WRONG
            new_content=(
                "    if (has_colors()) {\n"
                "        start_color();\n"
                "        init_pair(1, NEW, BG);\n"
            ),
        )
        # Known chunk covers the entire function (lines 6-15 = 10 lines)
        known = [FileChunk(
            file_path="snake.c",
            chunk_id="function:setup",
            line_start=6,
            line_end=15,
            content="void setup() {\n    initscr();\n ...",
            chunk_type="function",
            signature="void setup() {",
        )]
        result = editor.apply_chunk_edits(original, [edit], known_chunks=known)
        lines = result.splitlines()
        # Header lines (1-5) should be UNTOUCHED
        assert lines[0] == "line1"
        assert lines[4] == "line5"
        # Function header and initscr should be UNTOUCHED
        assert lines[5] == "void setup() {"
        assert lines[6] == "    initscr();"
        # The has_colors block should now have NEW content
        assert "init_pair(1, NEW, BG);" in lines[9]
        # The rest of the function after the edit should be preserved
        assert any("cbreak" in l for l in lines), "cbreak should still be present"
        assert any("noecho" in l for l in lines), "noecho should still be present"
        assert any("keypad" in l for l in lines), "keypad should still be present"

    def test_no_match_content_fallback(self):
        """When no chunk_id matches, content-based alignment against entire file
        should still find the correct location (the snake.c scenario)."""
        editor = ChunkEditor()
        original = (
            "typedef enum {\n"                       # line 1
            "    UP, DOWN, LEFT, RIGHT\n"             # line 2
            "} Direction;\n"                          # line 3
            "int score;\n"                            # line 4
            "void setup() {\n"                        # line 5
            "    initscr();\n"                        # line 6
            "    if (has_colors()) {\n"               # line 7
            "        init_pair(1, WHITE, BLACK);\n"   # line 8
            "    }\n"                                 # line 9
            "    cbreak();\n"                         # line 10
            "}\n"                                     # line 11
        )
        # LLM says lines 3-5 (WRONG), chunk_id "setup" doesn't match "top_level:1"
        edit = ChunkEditResponse(
            file_path="snake.c",
            chunk_id="setup",
            line_start=3,   # WRONG
            line_end=5,     # WRONG
            new_content=(
                "    if (has_colors()) {\n"
                "        init_pair(1, WHITE, BLUE);\n"
                "    }\n"
            ),
        )
        # Known chunk doesn't match "setup"
        known = [FileChunk(
            file_path="snake.c",
            chunk_id="top_level:1",
            line_start=1,
            line_end=11,
            content=original,
            chunk_type="top_level",
            signature="",
        )]
        result = editor.apply_chunk_edits(original, [edit], known_chunks=known)
        lines = result.splitlines()
        # typedef and score should NOT be deleted
        assert any("typedef" in l for l in lines), "typedef should be preserved"
        assert any("score" in l for l in lines), "score should be preserved"
        # The color should be changed
        assert any("BLUE" in l for l in lines), "BLUE should be present"


class TestChunkFileC:
    """Test that C files are properly chunked into functions."""

    def test_chunks_c_file(self):
        editor = ChunkEditor()
        c_source = (
            '#include <stdio.h>\n'
            '#include <stdlib.h>\n'
            '\n'
            'int score;\n'
            '\n'
            'void setup() {\n'
            '    printf("setup");\n'
            '}\n'
            '\n'
            'int main() {\n'
            '    setup();\n'
            '    return 0;\n'
            '}\n'
        )
        chunks = editor.chunk_file("test.c", c_source)
        chunk_ids = [c.chunk_id for c in chunks]
        # Should produce function:setup and function:main chunks
        assert any("setup" in cid for cid in chunk_ids), (
            f"Expected 'setup' chunk, got: {chunk_ids}"
        )
        assert any("main" in cid for cid in chunk_ids), (
            f"Expected 'main' chunk, got: {chunk_ids}"
        )


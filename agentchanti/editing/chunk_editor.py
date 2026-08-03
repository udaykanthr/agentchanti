"""
Chunk editor — lightweight regex-based file chunking and chunk-level edits.

Provides a middle ground between full-file rewrites and the full diff-aware
editing pipeline (which requires KB code graph + tree-sitter).  Works with
any language using simple indent + keyword heuristics.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language-specific chunk boundary patterns
# ---------------------------------------------------------------------------

_PY_PATTERNS = [
    re.compile(r"^(class\s+\w+)", re.MULTILINE),
    re.compile(r"^(def\s+\w+)", re.MULTILINE),
    re.compile(r"^(    def\s+\w+)", re.MULTILINE),
]

_JS_PATTERNS = [
    re.compile(r"^((?:export\s+)?(?:default\s+)?class\s+\w+)", re.MULTILINE),
    re.compile(r"^((?:export\s+)?(?:async\s+)?function\s+\w+)", re.MULTILINE),
    re.compile(
        r"^((?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:function|\())",
        re.MULTILINE,
    ),
    # React component metadata: Foo.propTypes = / Foo.defaultProps = / Foo.displayName =
    # These must be separate chunk boundaries so the LLM never silently drops them
    # when asked to edit only the function body above.
    re.compile(r"^(\w[\w.]*\.(propTypes|defaultProps|displayName|contextTypes)\s*=)", re.MULTILINE),
    # Bare `export default <identifier>` (not class/function — those are already above)
    re.compile(r"^(export\s+default\s+\w+\s*$)", re.MULTILINE),
]

_GO_PATTERNS = [
    re.compile(r"^(func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+)", re.MULTILINE),
    re.compile(r"^(type\s+\w+\s+(?:struct|interface))", re.MULTILINE),
]

_JAVA_PATTERNS = [
    re.compile(
        r"^(\s*(?:public|private|protected)\s+(?:static\s+)?class\s+\w+)",
        re.MULTILINE,
    ),
    re.compile(
        r"^(\s*(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\()",
        re.MULTILINE,
    ),
]

_RUST_PATTERNS = [
    re.compile(r"^((?:pub\s+)?fn\s+\w+)", re.MULTILINE),
    re.compile(r"^((?:pub\s+)?struct\s+\w+)", re.MULTILINE),
    re.compile(r"^((?:pub\s+)?impl\s+)", re.MULTILINE),
]

_C_PATTERNS = [
    # C/C++ function definitions at column 0:
    # return_type [*] function_name(
    re.compile(
        r"^((?:static\s+|inline\s+|extern\s+)*"
        r"\w+(?:\s*\*+)?\s+\w+\s*\()",
        re.MULTILINE,
    ),
    # struct / enum / union / typedef at column 0
    re.compile(r"^(typedef\s+(?:struct|enum|union)\b)", re.MULTILINE),
    re.compile(r"^((?:struct|enum|union)\s+\w+\s*\{)", re.MULTILINE),
]

_LANG_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": _PY_PATTERNS,
    "javascript": _JS_PATTERNS,
    "typescript": _JS_PATTERNS,
    "go": _GO_PATTERNS,
    "java": _JAVA_PATTERNS,
    "rust": _RUST_PATTERNS,
    "c": _C_PATTERNS,
    "cpp": _C_PATTERNS,
}

_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".java": "java", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
}

_IMPORT_PATTERNS = [
    re.compile(r"^\s*(import\s|from\s\S+\s+import)"),
    re.compile(r"^\s*(const|let|var)\s+.*=\s*require\("),
    re.compile(r"^\s*import\s+.+\s+from\s+"),
    re.compile(r"^\s*import\s+['\"]"),
    re.compile(r"^\s*using\s+"),
    re.compile(r"^\s*#include\s+"),
    re.compile(r"^\s*use\s+"),
    re.compile(r"^\s*require\s+"),
]

# Response parsing patterns
# Note: chunk names can be multi-word (e.g. "function HeroBanner", "class MyApp")
# so we use [^(]+? instead of \S+ to capture everything up to the opening paren.

# Matches EDIT markers with an explicit numeric line range (most precise form).
# The "lines?" keyword is optional — LLMs often omit it.
# Examples:
#   #### [EDIT]: foo.py (1-200)
#   #### [EDIT]: foo.py:python (lines 5-30)
#   #### [EDIT]: foo.py:python:my_func (lines 5-30)
_EDIT_MARKER = re.compile(
    r"####\s*\[EDIT\]:\s*(\S+?)(?::([^(]+?))?\s*\((?:lines?\s*)?(\d+)\s*-\s*(\d+)\)",
)

# Matches EDIT/NEW markers that signal a full-file replacement with no numeric range.
# LLMs use many variants for "replace the whole file":
#   #### [EDIT]: foo.py:python              (no parens at all)
#   #### [EDIT]: foo.py (new file)
#   #### [EDIT]: foo.py:python (full-file replacement)
#   #### [EDIT]: foo.py:python (full file)
#   #### [EDIT]: foo.py:python (replace)
# We match these ONLY when there is no numeric range (to avoid shadowing _EDIT_MARKER).
_EDIT_FULL_FILE_MARKER = re.compile(
    r"####\s*\[EDIT\]:\s*(\S+?)(?::([^(\n]+?))?"
    r"(?:\s*\((?:new\s+file|full[- ]?file[^)]*|replace[^)]*)\))?"
    r"\s*$",
    re.IGNORECASE,
)

# Matches "#### [EDIT]: file.py:symbol" with an optional non-numeric
# "(lines ...)" range — LLMs often echo the prompt template's placeholder
# verbatim ("(lines start-end)") or drop the range entirely.  The chunk
# must then be resolved by symbol name against known_chunks at apply time
# (sentinel line range 0-0).  Without the parens, the suffix is only a
# symbol when it is not a language tag ("file.py:python" stays full-file).
# The parenthetical is a HINT, not a range: "(lines 5-30)" is caught by
# _EDIT_MARKER above, so anything reaching here is prose the model chose
# to describe WHERE inside the symbol it is editing. Restricting it to
# the literal word "lines" meant a diagnosis that correctly wrote
#   #### [EDIT]: config.py:MAZE (row 16)
# matched none of the three markers, was silently dropped, and the run
# halted with "Diagnosis produced no actionable fix" — twice, on a
# correct diagnosis. Any parenthetical is accepted now; the chunk is
# resolved by symbol name regardless of what it says.
_EDIT_SYMBOL_MARKER = re.compile(
    r"####\s*\[EDIT\]:\s*(\S+?):([\w.]+)\s*(?:(\([^)]*\))\s*)?$",
    re.IGNORECASE,
)

# Suffixes after "file:" that denote a language tag (full-file marker),
# never a symbol name to resolve.
_LANGUAGE_TAGS = frozenset({
    "py", "python", "js", "javascript", "jsx", "ts", "typescript", "tsx",
    "java", "go", "golang", "rust", "rb", "ruby", "c", "cpp", "cs",
    "csharp", "php", "swift", "kotlin", "scala", "sh", "bash", "shell",
    "html", "css", "scss", "json", "yaml", "yml", "toml", "xml", "md",
    "markdown", "sql", "text", "txt", "code",
})

_NEW_MARKER = re.compile(
    r"####\s*\[NEW\]:\s*(\S+)\s*\(after\s+line\s+(\d+)\)",
)
_FULL_FILE_MARKER = re.compile(r"####\s*\[FILE\]:")
_CODE_BLOCK = re.compile(r"```\w*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileChunk:
    """A logical chunk of a file (function, class, or top-level block)."""
    file_path: str
    chunk_id: str          # e.g. "func:authenticate_user" or "class:UserService"
    line_start: int        # 1-indexed
    line_end: int
    content: str           # actual source lines
    chunk_type: str        # "function" | "class" | "method" | "imports" | "top_level"
    signature: str         # first meaningful line (def/class declaration)
    parent: str | None = None  # parent class name if method


@dataclass
class ChunkEditResponse:
    """Parsed chunk edit from LLM response."""
    file_path: str
    chunk_id: str
    line_start: int
    line_end: int
    new_content: str
    is_new: bool = False       # True for [NEW] insertions
    insert_after: int = 0      # line number to insert after (for new chunks)


def _reopens_declaration(new_content: str, chunk: FileChunk) -> bool:
    """True when *new_content* restates the chunk's own declaration.

    ``class Map:`` replacing a ``class Map:`` chunk is a whole-symbol
    rewrite, even at a third of the original length — models shorten code
    when they fix it. A fragment (one row of a maze literal, one entry of
    a dict) never opens with the declaration, which is what makes this a
    reliable discriminator where line count is not.
    """
    sig = (chunk.signature or "").strip().rstrip(":").rstrip()
    if not sig:
        return False
    for line in new_content.strip().splitlines():
        first = line.strip()
        if not first:
            continue
        # Compare declaration heads: `def f(self, a=1)` vs `def f(self)`
        # is still the same symbol being reopened.
        head = first.rstrip(":").rstrip()
        if head == sig:
            return True
        for kw in ("class ", "def ", "async def "):
            if sig.startswith(kw) and head.startswith(kw):
                name_a = sig[len(kw):].split("(")[0].strip()
                name_b = head[len(kw):].split("(")[0].strip()
                return bool(name_a) and name_a == name_b
        # A module-level constant chunk: `MAZE = [` reopened as `MAZE = [`.
        if "=" in sig and "=" in head:
            return sig.split("=")[0].strip() == head.split("=")[0].strip()
        return False
    return False


def _chunk_id_matches(chunk_id: str, edit_id: str) -> bool:
    """Check if a chunk_id matches an edit's chunk_id.

    The LLM may return just the function/class name (e.g. ``setup``)
    while the canonical chunk_id is ``function:setup``.
    """
    if not edit_id:
        return False
    # Exact match
    if chunk_id == edit_id:
        return True
    # chunk_id ends with the edit_id after the colon
    # e.g. chunk_id="function:setup" matches edit_id="setup"
    if ":" in chunk_id:
        _, name_part = chunk_id.rsplit(":", 1)
        if name_part == edit_id:
            return True
        # Handle dotted names: "method:UserService.authenticate" vs "authenticate"
        if "." in name_part and name_part.rsplit(".", 1)[-1] == edit_id:
            return True
    return False


# ---------------------------------------------------------------------------
# ChunkEditor
# ---------------------------------------------------------------------------

class ChunkEditor:
    """Regex-based file chunking and chunk-level edit application."""

    #: Set by :meth:`apply_chunk_edits` when a splice was rejected and the
    #: original content returned unchanged.  Reset at the start of every
    #: call, so it always describes the most recent one.
    last_apply_rejected: bool = False

    def chunk_file(self, file_path: str, content: str) -> list[FileChunk]:
        """Split a file into logical chunks using regex patterns.

        Returns a list of FileChunk objects covering the entire file.
        """
        lines = content.splitlines(True)
        total = len(lines)
        if total == 0:
            return []

        ext = os.path.splitext(file_path)[1].lower()
        lang = _EXT_TO_LANG.get(ext, "python")
        patterns = _LANG_PATTERNS.get(lang, _PY_PATTERNS)

        # Find all definition boundaries
        boundaries: list[tuple[int, str, str, int]] = []  # (line_idx, name, type, indent)

        for pattern in patterns:
            for m in pattern.finditer(content):
                line_idx = content[:m.start()].count("\n")
                sig_text = m.group(1).strip()

                # Determine type and name
                indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
                chunk_type, name = self._classify_signature(sig_text, indent)
                boundaries.append((line_idx, name, chunk_type, indent))

        # Sort by position
        boundaries.sort(key=lambda b: b[0])

        # Remove duplicates (overlapping patterns)
        seen_lines: set[int] = set()
        unique: list[tuple[int, str, str, int]] = []
        for b in boundaries:
            if b[0] not in seen_lines:
                unique.append(b)
                seen_lines.add(b[0])
        boundaries = unique

        # Build chunks
        chunks: list[FileChunk] = []

        # Imports chunk (from start to first definition or end of imports)
        imports_end = self._find_imports_end(lines)
        if imports_end > 0:
            chunks.append(FileChunk(
                file_path=file_path,
                chunk_id="imports",
                line_start=1,
                line_end=imports_end,
                content="".join(lines[:imports_end]),
                chunk_type="imports",
                signature="(imports)",
            ))

        # Definition chunks
        for i, (line_idx, name, chunk_type, indent) in enumerate(boundaries):
            # Skip definitions inside the imports block
            if line_idx < imports_end:
                continue

            # Find end: next boundary at same or lower indent, or EOF
            end_idx = total - 1
            for j in range(i + 1, len(boundaries)):
                next_idx, _, _, next_indent = boundaries[j]
                if next_indent <= indent:
                    # End just before the next definition
                    end_idx = next_idx - 1
                    # Trim trailing blank lines
                    while end_idx > line_idx and not lines[end_idx].strip():
                        end_idx -= 1
                    break

            # Detect parent class for methods
            parent = None
            if chunk_type == "method":
                parent = self._find_parent_class(boundaries, i, indent)

            chunk_content = "".join(lines[line_idx:end_idx + 1])
            sig = lines[line_idx].rstrip() if line_idx < total else ""

            chunk_id = f"{chunk_type}:{name}"
            if parent:
                chunk_id = f"method:{parent}.{name}"

            chunks.append(FileChunk(
                file_path=file_path,
                chunk_id=chunk_id,
                line_start=line_idx + 1,  # 1-indexed
                line_end=end_idx + 1,
                content=chunk_content,
                chunk_type=chunk_type,
                signature=sig,
                parent=parent,
            ))

        # Module-level constants are edit targets in their own right — a
        # maze layout, a config dict, a lookup table.  Without a named
        # chunk they land in an anonymous top_level gap, so an
        # "[EDIT]: f.py:DEFAULT_MAZE" cannot resolve by symbol and falls
        # back to the LLM's line arithmetic, which is routinely a line or
        # two short and leaves an orphan tail behind the splice.
        if lang == "python":
            chunks.extend(
                self._python_const_chunks(file_path, content, lines, chunks))

        # Fill gaps: any lines not covered by chunks become "top_level" chunks
        chunks = self._fill_gaps(chunks, lines, file_path, imports_end, total)

        # Sort by line_start
        chunks.sort(key=lambda c: c.line_start)
        return chunks

    def format_chunks_for_prompt(
        self,
        chunks: list[FileChunk],
        target_chunk_ids: list[str] | None = None,
    ) -> str:
        """Format chunks for LLM consumption.

        Target chunks get full content with EDITABLE markers.
        Non-target chunks get signature-only with CONTEXT markers.
        """
        if not chunks:
            return ""

        # Group by file
        by_file: dict[str, list[FileChunk]] = {}
        for c in chunks:
            by_file.setdefault(c.file_path, []).append(c)

        parts: list[str] = []
        all_target = target_chunk_ids is None

        for fpath, file_chunks in by_file.items():
            file_chunks.sort(key=lambda c: c.line_start)
            total = max(c.line_end for c in file_chunks) if file_chunks else 0
            parts.append(f"=== FILE: {fpath} ({total} lines total) ===")
            parts.append("")

            prev_end = 0
            for chunk in file_chunks:
                # Show gap marker
                gap = chunk.line_start - prev_end - 1
                if gap > 3:
                    parts.append(f"# ... [{gap} lines omitted] ...")
                    parts.append("")

                is_target = all_target or chunk.chunk_id in (target_chunk_ids or [])

                if chunk.chunk_type == "imports":
                    parts.append(f"# ─── IMPORTS (lines {chunk.line_start}-{chunk.line_end}) ───")
                    parts.append(chunk.content.rstrip())
                elif is_target:
                    parts.append(
                        f"# ═══ EDITABLE: {chunk.chunk_id} "
                        f"(lines {chunk.line_start}-{chunk.line_end}) ═══"
                    )
                    parts.append(chunk.content.rstrip())
                else:
                    parts.append(
                        f"# ─── CONTEXT ONLY: {chunk.chunk_id} "
                        f"(lines {chunk.line_start}-{chunk.line_end}) ───"
                    )
                    parts.append(chunk.signature)

                parts.append("")
                prev_end = chunk.line_end

            parts.append("=== END FILE ===")
            parts.append("")

        return "\n".join(parts)

    def identify_target_chunks(
        self,
        chunks: list[FileChunk],
        step_text: str,
    ) -> list[str]:
        """Identify which chunks are likely to be edited based on step text.

        Returns list of chunk_ids sorted by relevance.
        """
        step_lower = step_text.lower()
        scored: list[tuple[float, str]] = []

        for chunk in chunks:
            if chunk.chunk_type == "imports":
                continue  # imports are always included as context

            score = 0.0
            name_parts = chunk.chunk_id.split(":")[-1].lower()
            # Split camelCase and snake_case
            words = re.split(r"[_.\s]|(?<=[a-z])(?=[A-Z])", name_parts)
            words = [w.lower() for w in words if len(w) > 2]

            for word in words:
                if word in step_lower:
                    score += 50.0

            # Direct name mention
            raw_name = chunk.chunk_id.split(":")[-1]
            if raw_name.lower() in step_lower:
                score += 100.0

            # Signature keyword matching
            sig_words = re.findall(r"\w{3,}", chunk.signature.lower())
            for sw in sig_words:
                if sw in step_lower:
                    score += 10.0

            if score > 0:
                scored.append((score, chunk.chunk_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored]

    def parse_chunk_response(
        self,
        llm_response: str,
    ) -> list[ChunkEditResponse] | None:
        """Parse LLM response for chunk edits.

        Returns list of ChunkEditResponse, or None if the LLM used
        full-file format (signals fallback to full-file parsing).
        """
        # Detect full-file format — signal fallback
        if _FULL_FILE_MARKER.search(llm_response):
            logger.debug("[ChunkEditor] LLM used full-file format, signaling fallback")
            return None

        edits: list[ChunkEditResponse] = []

        # Split response by markers
        lines = llm_response.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()

            # Check for [EDIT] marker with explicit numeric line range (most precise).
            edit_match = _EDIT_MARKER.match(line_stripped)
            if edit_match:
                fpath = edit_match.group(1)
                chunk_name = (edit_match.group(2) or "").strip()
                line_start = int(edit_match.group(3))
                line_end = int(edit_match.group(4))

                # Extract code block
                code, end_i = self._extract_code_block(lines, i + 1)
                if code is not None:
                    edits.append(ChunkEditResponse(
                        file_path=fpath,
                        chunk_id=chunk_name,
                        line_start=line_start,
                        line_end=line_end,
                        new_content=code,
                    ))
                    i = end_i
                    continue

            # Full-file EDIT variants — no numeric range present.
            # Matches: "#### [EDIT]: file:lang", "#### [EDIT]: file (new file)",
            # "#### [EDIT]: file:lang (full-file replacement)", etc.
            # Guard: only fires when _EDIT_MARKER didn't already match this line,
            # and only when the line actually starts with #### [EDIT]: to avoid
            # false positives on prose lines.
            if (not edit_match
                    and line_stripped.startswith("####")
                    and "[EDIT]:" in line_stripped):
                # Symbol-named edit — "#### [EDIT]: game.py:on_draw" with a
                # non-numeric "(lines start-end)" placeholder or no range at
                # all.  Emit a sentinel range; apply resolves it by chunk_id.
                # Without parens, a language-tag suffix ("file.py:python")
                # keeps its full-file-replacement behavior.
                sym_match = _EDIT_SYMBOL_MARKER.match(line_stripped)
                if sym_match and (
                    sym_match.group(3)
                    or sym_match.group(2).strip().lower() not in _LANGUAGE_TAGS
                ):
                    fpath = sym_match.group(1)
                    chunk_name = sym_match.group(2).strip()
                    code, end_i = self._extract_code_block(lines, i + 1)
                    if code is not None and chunk_name:
                        logger.info(
                            "[ChunkEditor] Symbol-resolved edit for %s:%s "
                            "(no numeric line range)", fpath, chunk_name)
                        edits.append(ChunkEditResponse(
                            file_path=fpath,
                            chunk_id=chunk_name,
                            line_start=0,
                            line_end=0,
                            new_content=code,
                        ))
                        i = end_i
                        continue
                full_match = _EDIT_FULL_FILE_MARKER.match(line_stripped)
                if full_match:
                    fpath = full_match.group(1)
                    code, end_i = self._extract_code_block(lines, i + 1)
                    if code is not None:
                        logger.debug(
                            "[ChunkEditor] Full-file replacement for %s", fpath)
                        edits.append(ChunkEditResponse(
                            file_path=fpath,
                            chunk_id="top_level",
                            line_start=1,
                            line_end=999999,
                            new_content=code,
                        ))
                        i = end_i
                        continue

            # Check for [NEW] marker
            new_match = _NEW_MARKER.match(line_stripped)
            if new_match:
                fpath = new_match.group(1)
                after_line = int(new_match.group(2))

                code, end_i = self._extract_code_block(lines, i + 1)
                if code is not None:
                    edits.append(ChunkEditResponse(
                        file_path=fpath,
                        chunk_id="new",
                        line_start=after_line + 1,
                        line_end=after_line + 1,
                        new_content=code,
                        is_new=True,
                        insert_after=after_line,
                    ))
                    i = end_i
                    continue

            i += 1

        return edits if edits else None

    def apply_chunk_edits(
        self,
        original_content: str,
        edits: list[ChunkEditResponse],
        known_chunks: list[FileChunk] | None = None,
    ) -> str:
        """Splice edited chunks back into the original file content.

        Applies edits in reverse line order to preserve line numbering.

        When *known_chunks* is provided, each edit's line range is resolved
        against the known chunks first.  This corrects hallucinated line
        numbers from the LLM that would otherwise corrupt the file.
        """
        self.last_apply_rejected = False
        lines = original_content.splitlines(True)
        total_lines = len(lines)

        # Resolve line ranges before sorting
        resolved_edits: list[tuple[int, int, ChunkEditResponse]] = []
        for edit in edits:
            start, end = self._resolve_edit_lines(
                edit, known_chunks, total_lines, lines,
            )
            resolved_edits.append((start, end, edit))

        # Sort by resolved line_start descending (apply from bottom up)
        resolved_edits.sort(key=lambda t: t[0], reverse=True)

        for start, end, edit in resolved_edits:
            new_lines = edit.new_content.splitlines(True)
            # Strip leading unified-diff hunk markers (@@…@@) that the LLM
            # sometimes emits when it confuses chunk format with patch format.
            # A file starting with "@@ " is invalid Python/JS/HTML.
            if new_lines and re.match(r'^@@\s', new_lines[0]):
                logger.warning(
                    "[ChunkEditor] Stripping diff hunk marker from start of '%s'",
                    edit.file_path,
                )
                new_lines = [
                    l for l in new_lines
                    if not re.match(r'^@@\s', l) and not re.match(r'^[-+]{3}\s', l)
                ]
            # Ensure last line has newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            # ── Indentation repair ──
            # When the LLM returns a chunk with wrong indentation (e.g.
            # 3-space instead of 4-space), detect the mismatch against
            # the original code and re-indent the new lines to match.
            if not edit.is_new and new_lines:
                s_idx = max(0, start - 1)
                e_idx = min(len(lines), end)
                orig_slice = lines[s_idx:e_idx]

                def _indent_info(line_list: list[str]) -> tuple[int | None, int | None]:
                    """Return (min_indent, body_indent) for a chunk.

                    min_indent: the smallest indent (declaration lines
                    like ``class`` or ``def`` at the chunk boundary).
                    body_indent: the first indent level above min
                    (actual code body).  If all lines have the same
                    indent, body_indent == min_indent.
                    """
                    widths = []
                    for ln in line_list:
                        stripped = ln.lstrip()
                        if stripped and not stripped.startswith('#'):
                            widths.append(len(ln) - len(stripped))
                    if not widths:
                        return None, None
                    min_w = min(widths)
                    body_w = min_w
                    for w in widths:
                        if w > min_w:
                            body_w = w
                            break
                    return min_w, body_w

                orig_min, orig_body = _indent_info(orig_slice)
                new_min, new_body = _indent_info(new_lines)

                if (orig_body is not None
                        and new_body is not None
                        and orig_body != new_body
                        # Skip re-indentation when the replacement has
                        # broader scope than the original (e.g. LLM
                        # included a class header at indent 0 but the
                        # original range was a method at indent 4).
                        # The reference frame is mismatched — any
                        # correction would be wrong.
                        and (new_min is None or orig_min is None
                             or new_min >= orig_min)):
                    # Determine whether declaration lines (at min
                    # indent) should also be shifted.  When both chunks
                    # have the same min indent (e.g. both start with
                    # ``class`` at column 0), only body lines need
                    # shifting.  When min indents also differ, shift
                    # everything.
                    shift_all = (orig_min != new_min)

                    # Use proportional re-indentation to correctly
                    # handle multi-level nesting.  A flat delta only
                    # fixes the first indent level; deeper levels
                    # diverge (e.g. 3-space body at depth-2 = 6, but
                    # 4-space target at depth-2 = 8; flat +1 → 7 ✗).
                    _new_step = (new_body - (new_min or 0))
                    _orig_step = (orig_body - (orig_min or 0))
                    _use_ratio = (_new_step > 0 and _orig_step > 0
                                  and _new_step != _orig_step)
                    _ratio = _orig_step / _new_step if _use_ratio else 1.0
                    _base_delta = (orig_min or 0) - (new_min or 0)
                    # Flat delta fallback (for same-step-size cases)
                    delta = orig_body - new_body

                    fixed: list[str] = []
                    for ln in new_lines:
                        if ln.strip() == "":
                            fixed.append(ln)
                        else:
                            cur_indent = len(ln) - len(ln.lstrip())
                            # Skip declaration-level lines when min
                            # indents already match (they're correct)
                            if (not shift_all
                                    and new_min is not None
                                    and cur_indent <= new_min):
                                fixed.append(ln)
                            elif _use_ratio:
                                # Proportional: map indent relative to
                                # new_min onto orig_min's scale.
                                # Only apply ratio for clean nesting
                                # levels (multiples of new_step).
                                # Continuation lines (e.g. wrapped
                                # function args) have arbitrary indent
                                # and should use flat delta instead.
                                _rel = cur_indent - (new_min or 0)
                                if _new_step and _rel % _new_step == 0:
                                    target = (orig_min or 0) + round(
                                        _rel * _ratio)
                                    target = max(0, target)
                                    fixed.append(
                                        " " * target + ln.lstrip())
                                elif delta > 0:
                                    fixed.append(" " * delta + ln)
                                else:
                                    removable = min(-delta, cur_indent)
                                    fixed.append(ln[removable:])
                            elif delta > 0:
                                fixed.append(" " * delta + ln)
                            else:
                                removable = min(-delta, cur_indent)
                                fixed.append(ln[removable:])
                    new_lines = fixed
                    logger.debug(
                        "[ChunkEditor] Re-indented chunk for '%s' "
                        "(original=%d, new=%d, delta=%+d, shift_all=%s)",
                        edit.file_path, orig_body, new_body, delta,
                        shift_all,
                    )

            # ── Duplicate decorator guard ──
            # When the LLM includes a decorator (e.g. @dataclass) that
            # already exists on the line just before the replacement
            # range, the splice creates a duplicate.  Detect and strip.
            # IMPORTANT: only strip when the decorator is NOT part of the
            # original replaced range — if the original range started with
            # the same decorator, the LLM is legitimately including it
            # and stripping it would remove a required decorator.
            if not edit.is_new and new_lines:
                s_check = max(0, start - 1)
                if s_check > 0 and new_lines:
                    line_before = lines[s_check - 1].strip()
                    first_new = new_lines[0].strip()
                    # Check if the original replaced range also started
                    # with this decorator — if so, it's not a duplicate.
                    _orig_first = ""
                    _orig_s = max(0, start - 1)
                    _orig_e = min(len(lines), end)
                    if _orig_s < _orig_e:
                        for _ol in lines[_orig_s:_orig_e]:
                            if _ol.strip():
                                _orig_first = _ol.strip()
                                break
                    _orig_had_decorator = (
                        _orig_first.startswith('@')
                        and _orig_first == first_new
                    )
                    if (line_before == first_new
                            and line_before.startswith('@')
                            and len(new_lines) > 1
                            and not _orig_had_decorator):
                        new_lines = new_lines[1:]
                        logger.debug(
                            "[ChunkEditor] Stripped duplicate decorator "
                            "'%s' from replacement", line_before,
                        )

            if edit.is_new:
                if edit.insert_after == 0 and lines:
                    # "after line 0" on an existing file means full replacement,
                    # not prepend — avoids duplicate content when the LLM uses
                    # [NEW] to rewrite a file that already has content.
                    lines[:] = new_lines
                else:
                    # Insert after the specified line
                    insert_pos = min(edit.insert_after, len(lines))
                    lines[insert_pos:insert_pos] = new_lines
            else:
                # Replace line range (1-indexed to 0-indexed)
                s = max(0, start - 1)
                e = min(len(lines), end)
                lines[s:e] = new_lines

        # ── Post-splice syntax sanity check (Python only) ──
        # If the splice produced broken Python, warn and return the
        # original content unchanged — downstream syntax guards will
        # also catch this, but catching it here avoids writing broken
        # files when the caller trusts apply_chunk_edits.
        result = "".join(lines)
        if any(e.file_path.endswith(('.py', '.pyw')) for _, _, e in resolved_edits):
            import ast as _ast
            try:
                _ast.parse(result, filename=edits[0].file_path if edits else "<chunk>")
            except SyntaxError as _se:
                logger.warning(
                    "[ChunkEditor] Post-splice syntax error in %s: %s "
                    "(line %s) — returning original content",
                    edits[0].file_path if edits else "?",
                    _se.msg, _se.lineno,
                )
                # Returning the original silently is indistinguishable from
                # a successful edit: the caller writes it back, reports
                # "applied", and the next diagnosis attempt re-derives the
                # same fix against an unchanged file.  Record the rejection
                # so the caller can fall through to its own fallback now
                # instead of spending another round trip to learn nothing.
                self.last_apply_rejected = True
                return original_content
        return result

    @staticmethod
    def _resolve_edit_lines(
        edit: ChunkEditResponse,
        known_chunks: list[FileChunk] | None,
        total_lines: int,
        original_lines: list[str] | None = None,
    ) -> tuple[int, int]:
        """Resolve correct line range for an edit using known chunks.

        Returns ``(line_start, line_end)`` — possibly corrected.

        Handles two scenarios:
        1. **Full chunk replacement** — the LLM replaced the whole chunk
           but used wrong absolute line numbers.  Fix: use the chunk's
           actual line range.
        2. **Partial chunk edit** — the LLM only edited a sub-range
           within the chunk (e.g. one block inside a large function).
           Fix: use content-based alignment to find the correct
           sub-range within the chunk, preserving the rest.
        """
        # Sentinel range (0-0): the marker had no usable line numbers — the
        # chunk MUST be resolved by symbol name or the edit cannot be
        # applied safely.
        if not edit.is_new and (edit.line_start <= 0
                                or edit.line_end < edit.line_start):
            for chunk in (known_chunks or []):
                if (chunk.file_path == edit.file_path
                        and _chunk_id_matches(chunk.chunk_id, edit.chunk_id)):
                    new_span = len(edit.new_content.strip().splitlines())
                    chunk_span = chunk.line_end - chunk.line_start + 1
                    # A symbol edit carrying far less content than the
                    # chunk holds is editing PART of it — one row of a
                    # maze, one entry of a dict. Replacing the whole chunk
                    # with it would splice a single row over a 21-row
                    # constant: silent corruption, strictly worse than the
                    # silent no-op this branch used to produce.
                    #
                    # SHAPE decides that, not size. A rewrite that reopens
                    # the chunk's own declaration is a whole-symbol
                    # replacement however much shorter it is — a 101-line
                    # `class Map:` replacing a 179-line one is a rewrite,
                    # not a fragment, and treating it as partial rejected a
                    # correct fix and halted the run. A genuine fragment
                    # (one maze row) never opens with the declaration.
                    if (chunk_span > 2 and new_span < chunk_span * 0.7
                            and not _reopens_declaration(edit.new_content,
                                                         chunk)):
                        sub = ChunkEditor._align_within_chunk(
                            edit, chunk, original_lines)
                        if sub:
                            logger.info(
                                "[ChunkEditor] Resolved %s:%s to a partial "
                                "edit → lines %d-%d (of chunk %d-%d)",
                                edit.file_path, edit.chunk_id, sub[0], sub[1],
                                chunk.line_start, chunk.line_end)
                            return sub
                        raise ValueError(
                            f"Cannot place partial edit for "
                            f"{edit.file_path}:{edit.chunk_id} — {new_span} "
                            f"line(s) into a {chunk_span}-line chunk with no "
                            f"unambiguous match; refusing to overwrite it")
                    logger.info(
                        "[ChunkEditor] Resolved %s:%s by symbol → lines %d-%d",
                        edit.file_path, edit.chunk_id,
                        chunk.line_start, chunk.line_end)
                    return chunk.line_start, chunk.line_end
            raise ValueError(
                f"Cannot resolve edit for {edit.file_path}:{edit.chunk_id} — "
                f"no numeric line range and no matching chunk")

        if not known_chunks or edit.is_new:
            return edit.line_start, edit.line_end

        # Try matching by chunk_id
        for chunk in known_chunks:
            if (chunk.file_path == edit.file_path
                    and _chunk_id_matches(chunk.chunk_id, edit.chunk_id)):

                edit_span = edit.line_end - edit.line_start + 1
                chunk_span = chunk.line_end - chunk.line_start + 1

                # --- Full chunk replacement ---
                # If the edit covers most of the chunk, use the chunk range.
                if edit_span >= chunk_span * 0.7:
                    if (edit.line_start != chunk.line_start
                            or edit.line_end != chunk.line_end):
                        logger.info(
                            "[ChunkEditor] Corrected line range for %s:%s: "
                            "%d-%d → %d-%d (matched chunk %s)",
                            edit.file_path, edit.chunk_id,
                            edit.line_start, edit.line_end,
                            chunk.line_start, chunk.line_end,
                            chunk.chunk_id,
                        )
                    return chunk.line_start, chunk.line_end

                # --- Partial chunk edit ---
                # The LLM edited a sub-range within the chunk.
                # Try content-based alignment: find the first non-blank
                # line of the new content in the original within the chunk.
                if original_lines is not None:
                    new_content_lines = edit.new_content.splitlines()
                    # Find first non-blank line for anchoring
                    anchor = ""
                    for ncl in new_content_lines:
                        stripped = ncl.strip()
                        if stripped:
                            anchor = stripped
                            break

                    if anchor:
                        search_start = chunk.line_start - 1  # 0-indexed
                        search_end = min(chunk.line_end, len(original_lines))
                        for i in range(search_start, search_end):
                            if original_lines[i].strip() == anchor:
                                resolved_start = i + 1  # back to 1-indexed
                                resolved_end = resolved_start + edit_span - 1
                                # Clamp to chunk boundary
                                resolved_end = min(resolved_end, chunk.line_end)
                                logger.info(
                                    "[ChunkEditor] Content-aligned partial "
                                    "edit for %s:%s: %d-%d → %d-%d "
                                    "(anchor: %.40s)",
                                    edit.file_path, edit.chunk_id,
                                    edit.line_start, edit.line_end,
                                    resolved_start, resolved_end,
                                    anchor,
                                )
                                return resolved_start, resolved_end

                # Fallback: keep the LLM's span size but shift it to
                # be within the chunk's range using proportional offset.
                if chunk_span > edit_span:
                    # Clamp: place the edit proportionally within chunk
                    max_offset = chunk_span - edit_span
                    llm_offset = edit.line_start - 1  # rough offset
                    offset = min(max(0, llm_offset), max_offset)
                    resolved_start = chunk.line_start + offset
                    resolved_end = resolved_start + edit_span - 1
                    logger.info(
                        "[ChunkEditor] Offset-adjusted partial edit "
                        "for %s:%s: %d-%d → %d-%d",
                        edit.file_path, edit.chunk_id,
                        edit.line_start, edit.line_end,
                        resolved_start, resolved_end,
                    )
                    return resolved_start, resolved_end

                # Last resort: use chunk range
                return chunk.line_start, chunk.line_end

        # No chunk_id match — try content-based alignment against the
        # entire file.  This handles cases where the file is chunked as
        # a single top_level block (e.g. C files without language patterns).
        if original_lines is not None:
            new_content_lines = edit.new_content.splitlines()
            anchor = ""
            for ncl in new_content_lines:
                stripped = ncl.strip()
                if stripped:
                    anchor = stripped
                    break

            if anchor:
                edit_span = edit.line_end - edit.line_start + 1
                for i in range(len(original_lines)):
                    if original_lines[i].strip() == anchor:
                        resolved_start = i + 1  # 1-indexed
                        resolved_end = resolved_start + edit_span - 1
                        resolved_end = min(resolved_end, total_lines)
                        # If the original edit was meant to reach the end of
                        # the file (line_end >= total_lines - 2) and our
                        # content-aligned end falls short, extend it to cover
                        # all remaining lines.  Without this, lines beyond
                        # resolved_end are left as orphan duplicates.
                        if edit.line_end >= total_lines - 2:
                            resolved_end = total_lines
                        logger.info(
                            "[ChunkEditor] Content-aligned edit (no chunk "
                            "match) for %s:%s: %d-%d → %d-%d "
                            "(anchor: %.40s)",
                            edit.file_path, edit.chunk_id,
                            edit.line_start, edit.line_end,
                            resolved_start, resolved_end,
                            anchor,
                        )
                        return resolved_start, resolved_end

        # Sanity check: warn if line numbers exceed file length
        if edit.line_start > total_lines or edit.line_end > total_lines:
            logger.warning(
                "[ChunkEditor] Edit line range %d-%d exceeds file "
                "length %d for %s:%s",
                edit.line_start, edit.line_end, total_lines,
                edit.file_path, edit.chunk_id,
            )

        return edit.line_start, edit.line_end

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_signature(sig: str, indent: int) -> tuple[str, str]:
        """Classify a signature into (type, name)."""
        sig_stripped = sig.strip()

        if sig_stripped.startswith("class "):
            name = re.match(r"class\s+(\w+)", sig_stripped)
            return "class", name.group(1) if name else "unknown"

        if "def " in sig_stripped or "function " in sig_stripped:
            name_match = re.search(r"(?:def|function)\s+(\w+)", sig_stripped)
            name = name_match.group(1) if name_match else "unknown"
            if indent > 0:
                return "method", name
            return "function", name

        if sig_stripped.startswith(("func ", "fn ")):
            name_match = re.search(r"(?:func|fn)\s+(?:\([^)]*\)\s+)?(\w+)", sig_stripped)
            return "function", name_match.group(1) if name_match else "unknown"

        if sig_stripped.startswith("type "):
            name_match = re.search(r"type\s+(\w+)", sig_stripped)
            return "class", name_match.group(1) if name_match else "unknown"

        if sig_stripped.startswith(("pub fn", "pub struct", "pub impl", "impl ")):
            if "fn " in sig_stripped:
                name_match = re.search(r"fn\s+(\w+)", sig_stripped)
                return "function", name_match.group(1) if name_match else "unknown"
            if "struct " in sig_stripped:
                name_match = re.search(r"struct\s+(\w+)", sig_stripped)
                return "class", name_match.group(1) if name_match else "unknown"
            if "impl " in sig_stripped:
                name_match = re.search(r"impl\s+(\w+)", sig_stripped)
                return "class", name_match.group(1) if name_match else "unknown"

        # Arrow functions / const assignments
        const_match = re.match(r"(?:export\s+)?(?:const|let|var)\s+(\w+)", sig_stripped)
        if const_match:
            return "function", const_match.group(1)

        # Java/C# methods
        method_match = re.search(r"\b(\w+)\s*\(", sig_stripped)
        if method_match:
            name = method_match.group(1)
            if name not in ("if", "for", "while", "switch", "catch"):
                return "method" if indent > 0 else "function", name

        return "top_level", "unknown"

    @staticmethod
    def _find_parent_class(
        boundaries: list[tuple[int, str, str, int]],
        method_idx: int,
        method_indent: int,
    ) -> str | None:
        """Find the parent class for a method by looking at preceding boundaries."""
        for j in range(method_idx - 1, -1, -1):
            _, name, chunk_type, indent = boundaries[j]
            if indent < method_indent and chunk_type == "class":
                return name
        return None

    @staticmethod
    def _find_imports_end(lines: list[str]) -> int:
        """Find the line index (0-based) where imports end."""
        last_import = 0
        in_docstring = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    in_docstring = False
                    continue
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_docstring = True
                continue
            if in_docstring:
                continue
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            if any(p.match(line) for p in _IMPORT_PATTERNS):
                last_import = i + 1
            elif last_import > 0:
                break

        return last_import

    @staticmethod
    def _align_within_chunk(
        edit: ChunkEditResponse,
        chunk: FileChunk,
        original_lines: list[str] | None,
    ) -> tuple[int, int] | None:
        """Locate the sub-range of *chunk* that *edit* replaces.

        The replacement text is the EDITED version, so it will not appear
        in the original — exact anchoring cannot work here. Similarity
        can: a rewritten maze row still resembles the row it replaces far
        more than it resembles any other row.

        Returns None unless one line is both a strong match and clearly
        better than the runner-up. Maze rows look alike, and a confident
        guess between two near-equal candidates silently edits the wrong
        line, which is exactly the failure this guard exists to prevent.
        """
        if not original_lines:
            return None
        new_lines = [l for l in edit.new_content.strip().splitlines()
                     if l.strip()]
        if len(new_lines) != 1:
            # Multi-line partial edits need real anchors, not similarity.
            return None

        import difflib
        target = new_lines[0].strip()
        scored: list[tuple[float, int]] = []
        for i in range(chunk.line_start - 1,
                       min(chunk.line_end, len(original_lines))):
            cand = original_lines[i].strip()
            if not cand:
                continue
            ratio = difflib.SequenceMatcher(None, target, cand).ratio()
            scored.append((ratio, i + 1))
        if not scored:
            return None
        scored.sort(reverse=True)
        best, line_no = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best < 0.6 or (best - runner_up) < 0.05:
            logger.warning(
                "[ChunkEditor] Ambiguous partial edit for %s:%s — best match "
                "%.2f vs runner-up %.2f; refusing to guess which line",
                edit.file_path, edit.chunk_id, best, runner_up)
            return None
        return line_no, line_no

    @staticmethod
    def _python_const_chunks(
        file_path: str,
        content: str,
        lines: list[str],
        existing: list[FileChunk],
    ) -> list[FileChunk]:
        """Named chunks for module-level assignments (``const:NAME``).

        The span comes from ``ast``, not from counting brackets, so a
        multi-line literal ends where it actually ends.  That precision is
        the whole point: the regex chunker would run a constant up to the
        next ``def``, and a span that is too wide fails the 70% "full chunk
        replacement" test just as surely as one that is too narrow.

        Files that do not parse yield nothing — a syntax error is exactly
        when line numbers are least trustworthy, but guessing spans from
        broken source would be worse than leaving the edit to the existing
        content-alignment fallback.
        """
        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError):
            return []

        covered: set[int] = set()
        for c in existing:
            covered.update(range(c.line_start, c.line_end + 1))

        out: list[FileChunk] = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue

            start, end = node.lineno, (node.end_lineno or node.lineno)
            # Never overlap a def/class chunk: two chunks claiming the same
            # lines make the splice order-dependent.
            if any(ln in covered for ln in range(start, end + 1)):
                continue

            out.append(FileChunk(
                file_path=file_path,
                chunk_id=f"const:{targets[0].id}",
                line_start=start,
                line_end=end,
                content="".join(lines[start - 1:end]),
                chunk_type="const",
                signature=lines[start - 1].rstrip(),
            ))
            covered.update(range(start, end + 1))
        return out

    @staticmethod
    def _fill_gaps(
        chunks: list[FileChunk],
        lines: list[str],
        file_path: str,
        imports_end: int,
        total: int,
    ) -> list[FileChunk]:
        """Fill uncovered line ranges with top_level chunks."""
        covered = set()
        for c in chunks:
            for ln in range(c.line_start, c.line_end + 1):
                covered.add(ln)

        gap_start = None
        result = list(chunks)

        for ln in range(imports_end + 1, total + 1):
            if ln not in covered:
                if gap_start is None:
                    gap_start = ln
            else:
                if gap_start is not None:
                    gap_end = ln - 1
                    gap_content = "".join(lines[gap_start - 1:gap_end])
                    if gap_content.strip():  # Skip pure whitespace gaps
                        result.append(FileChunk(
                            file_path=file_path,
                            chunk_id=f"top_level:{gap_start}",
                            line_start=gap_start,
                            line_end=gap_end,
                            content=gap_content,
                            chunk_type="top_level",
                            signature=lines[gap_start - 1].rstrip() if gap_start <= total else "",
                        ))
                    gap_start = None

        # Handle trailing gap
        if gap_start is not None:
            gap_content = "".join(lines[gap_start - 1:total])
            if gap_content.strip():
                result.append(FileChunk(
                    file_path=file_path,
                    chunk_id=f"top_level:{gap_start}",
                    line_start=gap_start,
                    line_end=total,
                    content=gap_content,
                    chunk_type="top_level",
                    signature=lines[gap_start - 1].rstrip() if gap_start <= total else "",
                ))

        return result

    @staticmethod
    def _strip_diff_markers(code: str) -> str:
        """If *code* looks like a unified diff, apply it and return clean code.

        Detects unified-diff format by requiring **at least one genuine
        diff signal**:
          - a ``@@`` hunk header, OR
          - a ``--- a/...`` / ``+++ b/...`` file header, OR
          - clear ``+``/``-`` add/remove markers (where every non-blank
            line is a + addition or a - removal — pure-context blocks
            do not count).

        Plain indented code (no ``@@``, no ``+/-`` markers) is **not**
        a diff and is returned unchanged.  This guards against the
        pre-2026-04 bug where any indented code block was misclassified
        as a unified diff because most lines started with a space, and
        the function then stripped one leading space from every body
        line — corrupting indentation across fix-loop attempts and
        causing the chunk editor's re-indenter to compound the drift.

        When detected as a real diff, the function applies it:
          - Lines starting with '+' (but not '+++') → keep, strip '+'
          - Lines starting with '-' (but not '---') → discard
          - Context lines starting with ' '         → keep, strip ' '
          - '---' / '+++ ' headers and '\\ No newline' → skip
        """
        lines = code.splitlines()
        non_blank = [l for l in lines if l.strip()]
        if not non_blank:
            return code

        # Strong signal #1: a @@ hunk header.
        has_hunk_header = any(l.startswith("@@") for l in non_blank)

        # Strong signal #2: a --- a/... / +++ b/... file header pair.
        has_file_header = any(
            l.startswith("--- ") or l.startswith("+++ ")
            for l in non_blank
        )

        # Strong signal #3: every non-blank line is a +/- add/remove
        # marker (pure-additions or pure-deletions block).  Pure context
        # is intentionally rejected — indented code is "all context"
        # and that was the false positive being fixed here.
        all_addremove = all(
            (l.startswith("+") and not l.startswith("+++"))
            or (l.startswith("-") and not l.startswith("---"))
            for l in non_blank
        )

        if not (has_hunk_header or has_file_header or all_addremove):
            return code

        result: list[str] = []
        for line in lines:
            if line.startswith("+++") or line.startswith("---"):
                continue  # diff header
            if line.startswith("\\ "):
                continue  # "\ No newline at end of file"
            if line.startswith("@@"):
                continue  # hunk header
            if line.startswith("+"):
                result.append(line[1:])
            elif line.startswith("-"):
                pass  # removed line — discard
            else:
                # Context line: strip exactly one leading space if present
                result.append(line[1:] if line.startswith(" ") else line)

        cleaned = "\n".join(result)
        logger.debug("[ChunkEditor] Stripped unified diff markers from code block")
        return cleaned

    @staticmethod
    def _extract_code_block(
        lines: list[str],
        start_idx: int,
    ) -> tuple[str | None, int]:
        """Extract a fenced code block starting at or after start_idx.

        Returns (code_content, next_line_index) or (None, start_idx).
        """
        i = start_idx
        # Find opening fence
        while i < len(lines):
            if lines[i].strip().startswith("```"):
                break
            i += 1
        else:
            return None, start_idx

        # Collect code lines until closing fence
        code_lines: list[str] = []
        i += 1  # skip opening fence
        while i < len(lines):
            if lines[i].strip() == "```":
                code = "\n".join(code_lines)
                return ChunkEditor._strip_diff_markers(code), i + 1
            code_lines.append(lines[i])
            i += 1

        # No closing fence found — include what we have
        if code_lines:
            code = "\n".join(code_lines)
            return ChunkEditor._strip_diff_markers(code), i
        return None, start_idx

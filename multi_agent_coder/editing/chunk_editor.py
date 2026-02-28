"""
Chunk editor — lightweight regex-based file chunking and chunk-level edits.

Provides a middle ground between full-file rewrites and the full diff-aware
editing pipeline (which requires KB code graph + tree-sitter).  Works with
any language using simple indent + keyword heuristics.
"""

from __future__ import annotations

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
_EDIT_MARKER = re.compile(
    r"####\s*\[EDIT\]:\s*(\S+?)(?::(\S+))?\s*\(lines?\s*(\d+)\s*-\s*(\d+)\)",
)
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

            # Check for [EDIT] marker
            edit_match = _EDIT_MARKER.match(line.strip())
            if edit_match:
                fpath = edit_match.group(1)
                chunk_name = edit_match.group(2) or ""
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

            # Check for [NEW] marker
            new_match = _NEW_MARKER.match(line.strip())
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
            # Ensure last line has newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            if edit.is_new:
                # Insert after the specified line
                insert_pos = min(edit.insert_after, len(lines))
                lines[insert_pos:insert_pos] = new_lines
            else:
                # Replace line range (1-indexed to 0-indexed)
                s = max(0, start - 1)
                e = min(len(lines), end)
                lines[s:e] = new_lines

        return "".join(lines)

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
                return "\n".join(code_lines), i + 1
            code_lines.append(lines[i])
            i += 1

        # No closing fence found — include what we have
        if code_lines:
            return "\n".join(code_lines), i
        return None, start_idx

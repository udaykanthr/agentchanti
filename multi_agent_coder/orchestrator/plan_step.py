"""
PlanStep — structured execution plan data model and parser.

Replaces the old numbered-text plan format with a line-based structured
format that encodes step type, dependencies, target files, and
import/export relationships in a single LLM output.

Format
------
::

    ==PLAN==

    --STEP 1.1 [CMD] depends:none
    Create React project with Vite
    > npm create vite@latest my-app -- --template react-ts
    produces: package.json, vite.config.ts, src/main.tsx, src/App.tsx

    --STEP 2.1 [CODE] depends:1.1
    Create Header component
    target: src/components/Header.tsx
    exports: Header
    imports: none

    ==END==
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """Structured representation of a single pipeline step."""

    id: str                                     # e.g. "1.1", "2.1", "3.2"
    step_type: str                              # CMD, CODE, TEST, IGNORE
    description: str = ""                       # human-readable description
    depends_on: list[str] = field(default_factory=list)
    command: Optional[str] = None               # shell command for CMD steps
    target_files: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    imports_from: dict[str, list[str]] = field(default_factory=dict)  # file -> [symbols]
    status: str = "pending"                     # pending, in_progress, completed, failed, skipped
    actual_exports: list[str] = field(default_factory=list)  # filled after step execution
    inline_code: dict[str, str] = field(default_factory=dict)  # file -> code from plan
    inline_edits: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # file -> [(find, replace), ...]
    kb_docs: list[str] = field(default_factory=list)  # KB doc titles used when writing inline code

    # Which files should import this step's target file (plan-declared or derived)
    imported_by: list[str] = field(default_factory=list)

    # Legacy compat: 0-based integer index assigned after parsing
    index: int = -1

    def to_dict(self) -> dict:
        """Serialize for checkpoint / JSON."""
        d = {
            "id": self.id,
            "step_type": self.step_type,
            "description": self.description,
            "depends_on": self.depends_on,
            "command": self.command,
            "target_files": list(self.target_files),
            "exports": list(self.exports),
            "imports_from": {k: list(v) for k, v in self.imports_from.items()},
            "status": self.status,
            "actual_exports": list(self.actual_exports),
            "index": self.index,
        }
        if self.inline_code:
            d["inline_code"] = dict(self.inline_code)
        if self.inline_edits:
            d["inline_edits"] = {k: list(v) for k, v in self.inline_edits.items()}
        if self.kb_docs:
            d["kb_docs"] = list(self.kb_docs)
        if self.imported_by:
            d["imported_by"] = list(self.imported_by)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PlanStep:
        """Deserialize from checkpoint / JSON."""
        return cls(
            id=d.get("id", "0"),
            step_type=d.get("step_type", "CODE"),
            description=d.get("description", ""),
            depends_on=d.get("depends_on", []),
            command=d.get("command"),
            target_files=d.get("target_files", []),
            exports=d.get("exports", []),
            imports_from={k: list(v) for k, v in d.get("imports_from", {}).items()},
            status=d.get("status", "pending"),
            actual_exports=d.get("actual_exports", []),
            inline_code=d.get("inline_code", {}),
            inline_edits={k: [tuple(p) for p in v] for k, v in d.get("inline_edits", {}).items()},
            kb_docs=d.get("kb_docs", []),
            imported_by=d.get("imported_by", []),
            index=d.get("index", -1),
        )


# ---------------------------------------------------------------------------
# Echo command parser
# ---------------------------------------------------------------------------

# Regex for redirect operators in echo commands
_ECHO_REDIR_RE = re.compile(r'\s+(>{1,2})\s+([^>]+)$')
# Regex for "type nul > file" (Windows) or "touch file" (Unix)
_NUL_RE = re.compile(r'^(?:type\s+nul|touch)\s+>\s+(.+)$')
_TOUCH_RE = re.compile(r'^touch\s+(.+)$')


def _parse_echo_commands(lines: list[str]) -> dict[str, str]:
    """Parse shell-style file creation commands from plan step command lines.

    Extracts file paths and contents from patterns like::

        > cd my-app && echo "import React from 'react'" >> src/App.jsx
        > cd my-app && type nul > src/index.css
        > echo "export default {}" >> config.js

    Handles:
    - ``echo "content" >> file`` (append) and ``echo "content" > file`` (overwrite)
    - ``type nul > file`` / ``touch file`` (empty file creation)
    - ``cd dir && ...`` chains (strips the cd prefix)
    - Escaped quotes inside content (``\\"`` → ``"``)
    - ``\\n`` escape sequences → actual newlines
    - Common LLM typo ``src.App.jsx`` → ``src/App.jsx``

    Parameters
    ----------
    lines : list[str]
        Raw command lines (with ``> `` prefix already stripped).

    Returns
    -------
    dict[str, str]
        Mapping of file path → assembled content.  Empty dict if no
        echo commands were found.
    """
    files: dict[str, str] = {}

    for raw_line in lines:
        # Split on ' && ' to handle chained commands
        parts = raw_line.split(' && ')
        for part in parts:
            part = part.strip()
            # Skip bare 'cd' commands and 'mkdir' commands
            if not part or part.startswith('cd ') or part.startswith('mkdir '):
                continue

            # ── Empty file creation: type nul > file / touch file ──
            m_nul = _NUL_RE.match(part)
            if not m_nul:
                m_nul = _TOUCH_RE.match(part)
            if m_nul:
                fpath = m_nul.group(1).strip().strip('"\'')
                fpath = _fix_dot_paths(fpath)
                if fpath not in files:
                    files[fpath] = ""
                continue

            # ── Echo with redirect ──
            m_redir = _ECHO_REDIR_RE.search(part)
            if m_redir and part.startswith('echo '):
                operator = m_redir.group(1)
                fpath = m_redir.group(2).strip().strip('"\'')
                fpath = _fix_dot_paths(fpath)

                # Extract the content between 'echo ' and the redirect
                content_echo = part[5:m_redir.start()].strip()
                content_echo = _unescape_echo(content_echo)

                if operator == '>':
                    files[fpath] = content_echo + "\n"
                else:  # '>>' append
                    if fpath not in files:
                        files[fpath] = ""
                    files[fpath] += content_echo + "\n"

    return files


def _fix_dot_paths(fpath: str) -> str:
    """Fix common LLM typo: ``src.App.jsx`` → ``src/App.jsx``.

    Only fixes when the first segment looks like a directory name
    (e.g. ``src``, ``lib``, ``app``, ``components``), so legitimate
    dotted filenames like ``postcss.config.mjs`` or ``vite.config.ts``
    are preserved.
    """
    # If path already has slashes, it's fine
    if '/' in fpath or '\\' in fpath:
        return fpath
    # Split on dots; if >2 parts and last part is an extension, fix
    parts = fpath.split('.')
    if len(parts) > 2:
        ext = parts[-1]
        first = parts[0].lower()
        # Common source extensions
        if ext in ('js', 'jsx', 'ts', 'tsx', 'css', 'mjs', 'cjs',
                   'py', 'html', 'json', 'yaml', 'yml', 'toml',
                   'md', 'txt', 'cfg', 'ini', 'xml', 'svg'):
            # Only fix if the first segment looks like a directory
            _dir_prefixes = {
                'src', 'lib', 'app', 'components', 'pages', 'views',
                'utils', 'helpers', 'hooks', 'services', 'api',
                'routes', 'middleware', 'models', 'controllers',
                'public', 'static', 'assets', 'styles', 'tests',
                '__tests__', 'test', 'spec',
            }
            if first in _dir_prefixes:
                return '/'.join(parts[:-1]) + '.' + ext
    return fpath


def _unescape_echo(content: str) -> str:
    """Unescape echo content: strip surrounding quotes and decode escapes."""
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]
        content = content.replace('\\"', '"')
    elif content.startswith("'") and content.endswith("'"):
        content = content[1:-1]
        content = content.replace("\\'", "'")
    # Decode \n escape sequences to actual newlines
    content = content.replace('\\n', '\n')
    return content


def _flush_echo_inline(step: PlanStep, cmd_lines: list[str]) -> None:
    """Parse echo commands from a step's command lines and populate inline_code.

    Only populates ``inline_code`` if it is currently empty (i.e.
    ``---file-content-start---`` was not already used).
    """
    if step.inline_code:
        return  # ---file-content-start--- takes priority
    if not cmd_lines:
        return
    echo_files = _parse_echo_commands(cmd_lines)
    if echo_files:
        step.inline_code.update(echo_files)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(
    r"^--STEP\s+([\d.]+)\s+\[(\w+)\]\s*(?:depends?:(.*))?",
    re.IGNORECASE,
)


def parse_structured_plan(text: str) -> list[PlanStep]:
    """Parse the line-based structured plan format into PlanStep objects.

    Resilient to minor LLM formatting errors — each line is parsed
    independently. Unknown lines are treated as description continuation.

    Inline code between ``---file-content-start---`` and
    ``---file-content-end---`` markers is captured into
    ``PlanStep.inline_code``.  When the step has a single target file,
    the code is mapped to that file.  When multiple targets exist, the
    parser looks for ``// FileName.jsx`` comment headers to split the
    code into per-file blocks.

    When a reasoning model emits multiple draft plans in its thinking
    output, only the LAST ``==PLAN==...==END==`` block is used so that
    intermediate drafts don't bloat the step count.
    """
    # Extract the last ==PLAN== ... ==END== block (handles reasoning models
    # that include multiple draft plans in their thinking output).
    # Use rfind so partial/unclosed ==PLAN== blocks in thinking text don't
    # cause the regex to merge everything into one giant block.
    _upper = text.upper()
    _last_plan = _upper.rfind("==PLAN==")
    if _last_plan >= 0:
        # Use rfind so that small LLMs that emit ==END== after EVERY step
        # (instead of only once at the end) don't cause premature truncation.
        _end_after = _upper.rfind("==END==", _last_plan)
        if _end_after > _last_plan:
            text = text[_last_plan + len("==PLAN=="):_end_after]
        else:
            # No ==END== found after last ==PLAN== — use everything after it
            text = text[_last_plan + len("==PLAN=="):]

    steps: list[PlanStep] = []
    current: Optional[PlanStep] = None
    desc_lines: list[str] = []
    cmd_lines: list[str] = []        # all '> ...' lines for echo parsing
    in_code_block = False
    code_lines: list[str] = []

    in_markdown_fence = False  # track ```...``` inside inline blocks

    # edit: block state
    in_edit_block = False       # between edit: and <<<END>>>
    _edit_section = "find"      # "find" or "replace"
    _edit_find_lines: list[str] = []
    _edit_replace_lines: list[str] = []
    _edit_target: Optional[str] = None  # file path for current edit block

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Edit block handling (find/replace patches) ──
        # Format:
        #   edit:
        #   <<<FIND>>>
        #   old text
        #   <<<REPLACE>>>
        #   new text
        #   <<<END>>>
        if in_edit_block:
            _lnorm = line.upper().replace(" ", "")
            # Structural markers (new step header or plan end) terminate the
            # edit block — fall through to be processed normally.
            if line.upper() in ("==END==",) or _STEP_RE.match(line):
                in_edit_block = False
                _edit_find_lines = []
                _edit_replace_lines = []
                _edit_target = None
                # Fall through to process this line as a step header / ==END==
            elif _lnorm in ("<<<FIND>>>", "<<FIND>>", "<FIND>"):
                _edit_section = "find"
                _edit_find_lines = []
                continue
            elif _lnorm in ("<<<REPLACE>>>", "<<REPLACE>>", "<REPLACE>"):
                _edit_section = "replace"
                _edit_replace_lines = []
                continue
            elif _lnorm in ("<<<END>>>", "<<END>>", "<END>", "<<<EDITEND>>>"):
                # Commit this find/replace pair
                if current is not None and (_edit_find_lines or _edit_replace_lines):
                    _find_str = "\n".join(_edit_find_lines)
                    _repl_str = "\n".join(_edit_replace_lines)
                    _tgt = _edit_target or (current.target_files[0] if current.target_files else "")
                    if _tgt:
                        current.inline_edits.setdefault(_tgt, []).append((_find_str, _repl_str))
                # Stay in edit block — there may be more <<<FIND>>>...<<<END>>> pairs.
                # The block ends naturally when a new --STEP header or ==END== is seen.
                _edit_section = "find"
                _edit_find_lines = []
                _edit_replace_lines = []
                continue
            else:
                # Accumulate lines
                if _edit_section == "find":
                    _edit_find_lines.append(raw_line)
                else:
                    _edit_replace_lines.append(raw_line)
                continue

        # ── Inline code block handling ──
        # Accept multiple LLM output variants:
        #   ---file-content-start---  /  ---file-content-end---
        #   --- content ---  (followed by markdown fence)
        _norm = line.lower().replace(" ", "").rstrip("-")
        if _norm.startswith("---file-content-start") or _norm == "---content":
            in_code_block = True
            in_markdown_fence = False
            code_lines = []
            continue
        if _norm.startswith("---file-content-end") or (
            in_code_block and _norm == "---contentend"
        ):
            in_code_block = False
            in_markdown_fence = False
            if current is not None and code_lines:
                _assign_inline_code(current, code_lines)
            code_lines = []
            continue
        if in_code_block:
            # Structural markers end the code block implicitly
            # (handles LLM omitting ---file-content-end---)
            # Mark as truncated — proper close uses ---file-content-end---
            if line.upper() in ("==END==",) or _STEP_RE.match(line):
                in_code_block = False
                in_markdown_fence = False
                if current is not None and code_lines:
                    _assign_inline_code(current, code_lines)
                    current._inline_truncated = True  # type: ignore[attr-defined]
                code_lines = []
                # Fall through to process this line normally
            else:
                # Strip markdown fences (```js, ```) that wrap inline code
                if line.startswith("```"):
                    if in_markdown_fence:
                        # Closing fence — mark boundary so multi-target steps
                        # can split blocks even without // filename.ext headers.
                        code_lines.append(_FENCE_BOUNDARY)
                    in_markdown_fence = not in_markdown_fence
                    continue
                # Strip "> " prefix from code lines — the LLM sometimes
                # carries over the CMD "> command" format into code blocks
                code_line = raw_line
                if code_line.lstrip().startswith("> "):
                    code_line = code_line.lstrip()[2:]
                code_lines.append(code_line)  # preserve original indentation
                continue

        # Skip plan boundary markers
        if line.upper() in ("==PLAN==", "==END==", ""):
            continue

        # New step header
        m = _STEP_RE.match(line)
        if m:
            # Flush previous step
            if current is not None:
                current.description = " ".join(desc_lines).strip()
                _flush_echo_inline(current, cmd_lines)
                steps.append(current)

            step_id = m.group(1)
            step_type = m.group(2).upper()
            deps_raw = (m.group(3) or "").strip()

            # Parse depends
            depends: list[str] = []
            if deps_raw and deps_raw.lower() != "none":
                depends = [d.strip() for d in deps_raw.split(",") if d.strip()]

            current = PlanStep(id=step_id, step_type=step_type, depends_on=depends)
            desc_lines = []
            cmd_lines = []
            continue

        if current is None:
            continue

        # Command line (for CMD steps)
        if line.startswith("> "):
            cmd_text = line[2:].strip()
            # Strip backtick wrapping added by LLMs (e.g. `> `npm install`` -> `npm install`)
            if len(cmd_text) >= 2 and cmd_text[0] == "`" and cmd_text[-1] == "`":
                cmd_text = cmd_text[1:-1]
            # Skip markdown metadata annotations that the LLM sometimes
            # prefixes with ">" (e.g. "> **produces:** ..." or "> **note:** ...")
            _bare = cmd_text.lstrip("*_ \t")
            _bare_lower = _bare.lower()
            # Handle CODE/TEST metadata that some LLMs incorrectly prefix with "> "
            # (they see CMD steps use "> command" and copy the pattern to metadata)
            if _bare_lower.startswith("target:"):
                raw = _bare[7:].strip()
                if raw:
                    current.target_files = [f.strip() for f in raw.split(",") if f.strip()]
                continue
            elif _bare_lower.startswith("exports:"):
                raw = _bare[8:].strip()
                if raw and raw.lower() != "none":
                    current.exports = [e.strip() for e in raw.split(",") if e.strip()]
                continue
            elif _bare_lower.startswith("imports:"):
                raw = _bare[8:].strip()
                if raw and raw.lower() != "none":
                    for entry in raw.split(","):
                        entry = entry.strip()
                        if ":" in entry:
                            file_path, symbol = entry.rsplit(":", 1)
                            current.imports_from.setdefault(
                                file_path.strip(), []
                            ).append(symbol.strip())
                continue
            elif _bare_lower.startswith("imported_by:"):
                raw = _bare[12:].strip()
                if raw and raw.lower() != "none":
                    current.imported_by = [f.strip() for f in raw.split(",") if f.strip()]
                continue
            elif _bare_lower.startswith("content:"):
                # Inline code block follows — enter code-block mode
                in_code_block = True
                in_markdown_fence = False
                code_lines = []
                rest = _bare[8:].strip()
                # Strip opening markdown fence (```jsx, ```, etc.)
                if rest.startswith("```"):
                    rest = rest[3:].lstrip("abcdefghijklmnopqrstuvwxyz").strip()
                if rest:
                    code_lines.append(rest)
                continue
            elif _bare_lower.startswith("produces:"):
                raw = _bare[9:].strip()
                if raw:
                    current.target_files.extend(
                        f.strip() for f in raw.split(",") if f.strip()
                    )
                continue
            _meta_prefixes = ("note:", "output:", "creates:",
                              "result:", "generates:", "returns:")
            if cmd_text.startswith("**") or any(
                _bare_lower.startswith(p) for p in _meta_prefixes
            ):
                continue  # metadata annotation, not a shell command
            # Check for "content:" appearing mid-line (e.g.
            # "> prop-types:default content: ```" where content: is not
            # at the start).  Only trigger when followed by ``` or nothing.
            _content_pos = _bare_lower.find(" content:")
            if _content_pos >= 0:
                _after = _bare[_content_pos + 9:].strip()
                if not _after or _after.startswith("```"):
                    in_code_block = True
                    in_markdown_fence = False
                    code_lines = []
                    if _after.startswith("```"):
                        _after = _after[3:].lstrip(
                            "abcdefghijklmnopqrstuvwxyz").strip()
                    if _after:
                        code_lines.append(_after)
                    continue
            # Join multiple commands per step with && so all run sequentially
            if current.command:
                current.command = current.command + " && " + cmd_text
            else:
                current.command = cmd_text
            cmd_lines.append(cmd_text)

        # Target files
        elif line.lower().startswith("target:"):
            raw = line[7:].strip()
            if raw:
                current.target_files = [f.strip() for f in raw.split(",") if f.strip()]

        # Exports
        elif line.lower().startswith("exports:"):
            raw = line[8:].strip()
            if raw and raw.lower() != "none":
                current.exports = [e.strip() for e in raw.split(",") if e.strip()]

        # Imports: src/file.py:Symbol, src/other.py:OtherSymbol
        elif line.lower().startswith("imports:"):
            raw = line[8:].strip()
            if raw and raw.lower() != "none":
                for entry in raw.split(","):
                    entry = entry.strip()
                    if ":" in entry:
                        file_path, symbol = entry.rsplit(":", 1)
                        current.imports_from.setdefault(
                            file_path.strip(), []
                        ).append(symbol.strip())

        # Produces (alias for target_files, used by CMD steps)
        elif line.lower().startswith("produces:"):
            raw = line[9:].strip()
            if raw:
                produced = [f.strip() for f in raw.split(",") if f.strip()]
                current.target_files.extend(produced)

        # KB docs declared by planner for reviewer context
        elif line.lower().startswith("kb_docs:"):
            raw = line[8:].strip()
            if raw and raw.lower() != "none":
                current.kb_docs = [t.strip() for t in raw.split(",") if t.strip()]

        # imported_by: which files should import this step's target file
        elif line.lower().startswith("imported_by:"):
            raw = line[12:].strip()
            if raw and raw.lower() != "none":
                current.imported_by = [f.strip() for f in raw.split(",") if f.strip()]

        # edit: block — find/replace patch for an existing file.
        # Supports optional "edit: path/to/file" to specify a different target.
        elif line.lower().startswith("edit:"):
            rest = line[5:].strip()
            # If a file path is given on the same line, use it; otherwise use
            # the step's first target file (resolved when <<<END>>> is hit).
            if rest and not rest.startswith("<<<"):
                _edit_target = rest
            else:
                _edit_target = None
            in_edit_block = True
            _edit_section = "find"
            _edit_find_lines = []
            _edit_replace_lines = []

        # Inline code block via bare "Content:" keyword (no "> " prefix).
        # The "> content:" variant is already handled inside the "> " branch
        # above.  Here we catch the unindented form that the planner sometimes
        # emits when the step has a pre-written code body.
        elif line.lower().startswith("content:"):
            in_code_block = True
            in_markdown_fence = False
            code_lines = []
            rest = line[8:].strip()  # anything after "Content:" on the same line
            # Strip opening markdown fence (```jsx, ```, etc.)
            if rest.startswith("```"):
                rest = rest[3:].lstrip("abcdefghijklmnopqrstuvwxyz").strip()
            if rest:
                code_lines.append(rest)

        # Description line (anything else) — but check for mid-line content:
        elif not line.startswith("=="):
            _lower_line = line.lower()
            _cpos = _lower_line.find(" content:")
            if _cpos >= 0:
                _after_c = line[_cpos + 9:].strip()
                if not _after_c or _after_c.startswith("```"):
                    # Split: text before content: goes to description,
                    # everything after enters code block mode
                    before = line[:_cpos].strip()
                    if before:
                        desc_lines.append(before)
                    in_code_block = True
                    in_markdown_fence = False
                    code_lines = []
                    if _after_c.startswith("```"):
                        _after_c = _after_c[3:].lstrip(
                            "abcdefghijklmnopqrstuvwxyz").strip()
                    if _after_c:
                        code_lines.append(_after_c)
                    continue
            desc_lines.append(line)

    # Flush any open edit block (LLM omitted the final <<<END>>>).
    if in_edit_block and current is not None and (_edit_find_lines or _edit_replace_lines):
        _find_str = "\n".join(_edit_find_lines)
        _repl_str = "\n".join(_edit_replace_lines)
        # Skip if the find string is only file-content markers (e.g.
        # ---file-content-end--- accumulated after the last <<<END>>>).
        _find_meaningful = [
            l for l in _edit_find_lines
            if l.strip() and not l.strip().startswith("---")
        ]
        if _find_meaningful:
            _tgt = _edit_target or (current.target_files[0] if current.target_files else "")
            if _tgt:
                current.inline_edits.setdefault(_tgt, []).append((_find_str, _repl_str))

    # Flush any open inline code block.
    # If the code block was never closed (no ---file-content-end---, ==END==,
    # or next --STEP marker), the LLM output was truncated.  Assign what we
    # have but mark the step so validate_plan() can clear it.
    if in_code_block and current is not None and code_lines:
        _assign_inline_code(current, code_lines)
        current._inline_truncated = True  # type: ignore[attr-defined]

    # Flush last step
    if current is not None:
        current.description = " ".join(desc_lines).strip()
        _flush_echo_inline(current, cmd_lines)
        steps.append(current)

    # Assign 0-based indices
    for idx, step in enumerate(steps):
        step.index = idx

    # Derive imported_by from imports_from relationships across all steps.
    # This is free (no LLM cost) and ensures that when step B declares
    # imports_from step A's target file, step A gets imported_by = step B's target.
    _derive_imported_by(steps)

    # For steps that still have no imported_by after derivation, infer it from
    # the plan structure (entry-point files in later waves).  Works even when the
    # planner forgets to add a wiring step or an explicit imported_by: line.
    _infer_missing_imported_by(steps)

    return steps


def _derive_imported_by(steps: list[PlanStep]) -> None:
    """Populate ``imported_by`` on each step from other steps' ``imports_from``.

    For every consumer step that declares ``imports_from: {file: [symbols]}``,
    find the producer step whose ``target_files`` includes *file* and add the
    consumer's target files to the producer's ``imported_by`` list.

    This is zero-cost (no LLM) and fires automatically after parsing so that
    DepCheck can use ``plan_step.imported_by`` instead of guessing heuristically.
    Explicit ``imported_by:`` lines in the plan take precedence (they are set
    first during parsing; derivation only appends new entries, never clears them).
    """
    import os as _os

    # Build file → step map (normalized paths + basenames for fuzzy lookup)
    file_to_step: dict[str, PlanStep] = {}
    for step in steps:
        for tf in step.target_files:
            norm = tf.replace("\\", "/")
            file_to_step[norm] = step
            file_to_step[_os.path.basename(norm)] = step

    for consumer_step in steps:
        consumer_files = consumer_step.target_files
        if not consumer_files:
            continue
        for src_file in consumer_step.imports_from:
            src_norm = src_file.replace("\\", "/")
            src_basename = _os.path.basename(src_norm)
            producer = file_to_step.get(src_norm) or file_to_step.get(src_basename)
            if producer is None:
                continue
            for cf in consumer_files:
                if cf not in producer.imported_by:
                    producer.imported_by.append(cf)


# Entry-point file basenames that commonly import/mount other components.
# Ordered from most-specific to least-specific.
_ENTRY_POINT_BASENAMES = (
    "main.tsx", "main.ts", "main.jsx", "main.js",
    "App.tsx", "App.ts", "App.jsx", "App.js",
    "index.tsx", "index.ts", "index.jsx", "index.js",
    "router.tsx", "router.ts", "router.jsx", "router.js",
    "__init__.py", "main.py", "app.py", "index.py",
)


def _infer_missing_imported_by(steps: list[PlanStep]) -> None:
    """For CODE steps with exports but no ``imported_by``, infer the consumer
    from other plan steps whose target files look like entry-points or whose
    wave number is later.

    Strategy (in order, first match wins):
    1. Another step in a later wave whose target is an entry-point file and
       that step's description mentions the orphaned step's exported symbol
       or file stem.
    2. Any step in a later wave whose target is an entry-point file.
    3. Any step in a later wave that has no ``imports_from`` declared
       (likely a wiring/mounting step with incomplete metadata).

    Only fires when ``imported_by`` is still empty after ``_derive_imported_by``
    — i.e. the planner neither added a wiring step nor wrote ``imported_by:``.
    Operates purely on plan metadata, zero LLM cost.
    """
    import os as _os

    def _wave(step: PlanStep) -> int:
        """Return the wave number from the step id (e.g. '3.1' → 3)."""
        try:
            return int(step.id.split(".")[0])
        except (ValueError, IndexError):
            return 0

    def _is_entry_point(path: str) -> bool:
        base = _os.path.basename(path).lower()
        return base in {ep.lower() for ep in _ENTRY_POINT_BASENAMES}

    for step in steps:
        # Only care about CODE steps with exports and no consumer yet
        if step.step_type not in ("CODE", "UNCLASSIFIED"):
            continue
        if not step.exports or step.imported_by:
            continue
        if not step.target_files:
            continue

        step_wave = _wave(step)
        exported_lower = {e.lower() for e in step.exports}
        stem_lower = {
            _os.path.splitext(_os.path.basename(tf))[0].lower()
            for tf in step.target_files
        }

        # Collect candidate steps: later wave, has target files
        candidates = [
            s for s in steps
            if _wave(s) > step_wave and s.target_files and s is not step
        ]

        # Strategy 1: entry-point target + description mentions our symbol/stem
        for cand in candidates:
            if not any(_is_entry_point(tf) for tf in cand.target_files):
                continue
            desc_lower = cand.description.lower()
            if exported_lower & set(desc_lower.split()) or stem_lower & set(desc_lower.split()):
                step.imported_by = list(cand.target_files[:1])
                break

        if step.imported_by:
            continue

        # Strategy 2: any entry-point target in a later wave
        for cand in candidates:
            ep_targets = [tf for tf in cand.target_files if _is_entry_point(tf)]
            if ep_targets:
                step.imported_by = [ep_targets[0]]
                break

        if step.imported_by:
            continue

        # Strategy 3: a later step with no imports_from (likely incomplete wiring step)
        for cand in candidates:
            if not cand.imports_from and cand.step_type in ("CODE", "UNCLASSIFIED"):
                step.imported_by = list(cand.target_files[:1])
                break


# File header comment pattern for splitting multi-file inline code blocks
_FILE_COMMENT_RE = re.compile(
    r"^//\s*([\w./-]+\.\w{1,5})\s*$"
)

# Sentinel inserted into code_lines at each closing ``` fence boundary
# so _assign_inline_code can split multiple files without // header comments.
_FENCE_BOUNDARY = "\x00FENCE_BOUNDARY\x00"


def _assign_inline_code(step: PlanStep, code_lines: list[str]) -> None:
    """Assign captured inline code to a step's ``inline_code`` dict.

    For single-target steps, all code goes to that target file.
    For multi-target steps the parser tries three strategies in order:

    1. ``// FileName.ext`` comment headers between blocks.
    2. Fence boundaries (``_FENCE_BOUNDARY`` sentinels inserted at each
       closing `` ``` `` fence) — if the number of non-empty fence blocks
       matches the number of targets, assign block N to target N.
    3. Fallback: everything goes to ``targets[0]``.
    """
    targets = step.target_files

    # ── Strategy 1: // filename.ext comment headers ──
    if len(targets) > 1:
        current_file: Optional[str] = None
        file_lines: dict[str, list[str]] = {}
        for line in code_lines:
            if line == _FENCE_BOUNDARY:
                continue
            m = _FILE_COMMENT_RE.match(line.strip())
            if m:
                fname = m.group(1)
                matched = _match_target(fname, targets)
                current_file = matched or fname
                file_lines.setdefault(current_file, [])
                continue
            if current_file is not None:
                file_lines[current_file] = file_lines.get(current_file, [])
                file_lines[current_file].append(line)
        if file_lines:
            for fpath, lines in file_lines.items():
                content = "\n".join(lines).strip()
                if content:
                    step.inline_code[fpath] = content
            return

    # ── Strategy 2: fence boundary splitting (no // headers) ──
    if len(targets) > 1:
        fence_blocks: list[list[str]] = []
        current_block: list[str] = []
        for line in code_lines:
            if line == _FENCE_BOUNDARY:
                if current_block:
                    fence_blocks.append(current_block)
                current_block = []
            else:
                current_block.append(line)
        if current_block:
            fence_blocks.append(current_block)

        non_empty = [b for b in fence_blocks if any(ln.strip() for ln in b)]
        if len(non_empty) == len(targets):
            for target, block_lines in zip(targets, non_empty):
                content = "\n".join(block_lines).strip()
                if content:
                    step.inline_code[target] = content
            return

    # ── Strategy 3: fallback — all code to first target ──
    clean_lines = [ln for ln in code_lines if ln != _FENCE_BOUNDARY]
    full_code = "\n".join(clean_lines).strip()
    if not full_code:
        return

    if len(targets) >= 1:
        step.inline_code[targets[0]] = full_code


def _match_target(name: str, targets: list[str]) -> Optional[str]:
    """Match a short filename against the full target paths."""
    import os
    for t in targets:
        if os.path.basename(t) == name or t.endswith("/" + name) or t == name:
            return t
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_plan(steps: list[PlanStep], working_dir: Optional[str] = None) -> list[str]:
    """Validate a parsed plan for structural correctness.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []
    all_ids = {s.id for s in steps}

    # Track which files are produced by which steps
    produced_files: dict[str, str] = {}  # file -> step_id

    for step in steps:
        # Check depends_on references exist
        for dep in step.depends_on:
            if dep not in all_ids:
                errors.append(
                    f"Step {step.id} depends on unknown step '{dep}'"
                )

        # Track produced files
        for fpath in step.target_files:
            produced_files[fpath] = step.id

        # Check imports reference files that some step produces or will produce
        for file_path in step.imports_from:
            # It's OK if the file is produced by a later step — the plan
            # declares intent. But if NO step produces it, warn.
            if file_path not in produced_files:
                producers = [
                    s for s in steps
                    if file_path in s.target_files
                ]
                if not producers:
                    # Not an error — could be an existing project file
                    pass

    # Check inline code for truncation: if the parser never saw a closing
    # marker (---file-content-end---, ==END==, or next --STEP), the LLM
    # output was cut off.  Preserve the partial code as a hint for the coder
    # (so it can complete rather than regenerate from scratch), then clear
    # inline_code so the normal coder path handles full generation.
    import os as _os
    for step in steps:
        if not step.inline_code:
            continue
        if getattr(step, "_inline_truncated", False):
            # Preserve partial content as a completion hint before clearing
            step._partial_inline_code = dict(step.inline_code)  # type: ignore[attr-defined]
            step.inline_code.clear()
            errors.append(
                f"Step {step.id}: inline code was truncated (no closing marker) "
                f"— preserved as partial hint, coder will complete it"
            )
            continue

    # Check inline code imports: if inline code contains local imports
    # (e.g. import './Hero.css'), verify that some step produces the file.
    # When a dangling import is found, clear the entire step's inline_code
    # so the regular coder path handles it — the coder can generate both the
    # component AND the missing file with full KB/memory context.
    import re as _re

    # Language-specific local import patterns (group 1 = the import path)
    _IMPORT_PATTERNS: dict[str, list[_re.Pattern]] = {
        # JS/TS: import X from './Y'  |  import './Y'
        "js": [
            _re.compile(r"""(?:import\s+.*?from\s+|import\s+)['"](\.[^'"]+)['"]"""),
            _re.compile(r"""@import\s+['"](\.[^'"]+)['"]"""),   # CSS @import
            _re.compile(r"""require\s*\(\s*['"](\.[^'"]+)['"]\s*\)"""),  # CJS require
        ],
        # Python: from .module import X  |  from . import module
        "py": [
            _re.compile(r"""from\s+(\.[\w.]+)\s+import"""),
        ],
        # Go: import "./pkg"  |  in import block
        "go": [
            _re.compile(r"""(?:import\s+|")(\./[^"]+)"""),
        ],
        # Rust: mod submodule;  (local module declaration)
        "rs": [
            _re.compile(r"""mod\s+(\w+)\s*;"""),
        ],
    }

    # Map file extensions to pattern keys
    _EXT_TO_PATTERN_KEY = {
        ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
        ".ts": "js", ".tsx": "js",
        ".css": "js",   # CSS uses @import
        ".py": "py",
        ".go": "go",
        ".rs": "rs",
    }

    # Extension candidates per language when import has no extension
    _EXT_CANDIDATES = {
        "js": [".js", ".jsx", ".ts", ".tsx", ".css", ".mjs"],
        "py": [".py"],
        "go": [".go"],
        "rs": [".rs"],
    }

    for step in steps:
        if not step.inline_code:
            continue
        has_dangling = False
        for fpath, code in step.inline_code.items():
            if has_dangling:
                break
            ext = _os.path.splitext(fpath)[1].lower()
            pattern_key = _EXT_TO_PATTERN_KEY.get(ext)
            if not pattern_key:
                continue
            patterns = _IMPORT_PATTERNS.get(pattern_key, [])
            file_dir = _os.path.dirname(fpath)
            for pat in patterns:
                if has_dangling:
                    break
                for m in pat.finditer(code):
                    imp_path = m.group(1)
                    # Python relative imports use dots: from .models import X
                    if pattern_key == "py" and imp_path.startswith("."):
                        # Convert .models to ./models for path resolution
                        dotless = imp_path.lstrip(".")
                        depth = len(imp_path) - len(dotless)
                        base = file_dir
                        for _ in range(depth - 1):
                            base = _os.path.dirname(base)
                        imp_path = "./" + dotless.replace(".", "/")
                    # Rust mod X; → ./X.rs or ./X/mod.rs
                    if pattern_key == "rs":
                        imp_path = "./" + imp_path
                    resolved = _os.path.normpath(
                        _os.path.join(file_dir, imp_path)
                    ).replace("\\", "/")
                    candidates = [resolved]
                    if not _os.path.splitext(resolved)[1]:
                        for cand_ext in _EXT_CANDIDATES.get(pattern_key, []):
                            candidates.append(resolved + cand_ext)
                        # Rust: mod X → X/mod.rs
                        if pattern_key == "rs":
                            candidates.append(resolved + "/mod.rs")
                    is_produced = any(
                        c in produced_files or any(
                            c == tf or tf.endswith("/" + _os.path.basename(c))
                            for tf in produced_files
                        )
                        for c in candidates
                    )
                    if is_produced:
                        continue
                    if any(
                        _os.path.basename(c) in _os.path.basename(tf)
                        for c in candidates for tf in produced_files
                    ):
                        continue
                    # Also accept files that already exist on disk
                    _base = working_dir or _os.getcwd()
                    if any(_os.path.exists(_os.path.join(_base, c)) for c in candidates):
                        continue
                    has_dangling = True
                    break
        if has_dangling:
            step.inline_code.clear()
            errors.append(
                f"Step {step.id}: inline code imports a file no step produces "
                f"— cleared inline_code, will use coder LLM call instead"
            )

    # Check for circular dependencies
    if _has_cycle(steps):
        errors.append("Circular dependency detected in plan")

    # Check valid step types
    valid_types = {"CMD", "CODE", "TEST", "IGNORE", "SEARCH"}
    for step in steps:
        if step.step_type not in valid_types:
            errors.append(
                f"Step {step.id} has unknown type '{step.step_type}'"
            )

    return errors


# ---------------------------------------------------------------------------
# Auto-fix: inject missing import-based dependencies
# ---------------------------------------------------------------------------

def fix_import_dependencies(steps: list[PlanStep]) -> list[str]:
    """Auto-inject missing ``depends_on`` entries based on import relationships.

    If step B declares ``imports: src/Foo.jsx:Foo`` and step A has
    ``target: src/Foo.jsx``, then B must depend on A.  When the LLM
    forgets to declare this dependency the wave builder may schedule B
    before A, causing an import failure at runtime.

    This function detects such gaps and patches ``step.depends_on``
    in-place.  Returns a list of human-readable descriptions of fixes
    applied (empty list = nothing changed).

    Must be called **after** ``validate_plan()`` and **before**
    ``build_waves()``.
    """
    fixes: list[str] = []
    all_ids = {s.id for s in steps}

    # Build target-file → producer-step-id map
    file_to_producer: dict[str, str] = {}
    for s in steps:
        for fpath in s.target_files:
            file_to_producer[fpath] = s.id

    for step in steps:
        for file_path in step.imports_from:
            producer_id = file_to_producer.get(file_path)
            if producer_id is None:
                continue  # existing project file, not produced by plan
            if producer_id == step.id:
                continue  # self-reference (step modifies + imports same file)
            if producer_id in step.depends_on:
                continue  # already declared

            # Inject the missing dependency
            step.depends_on.append(producer_id)
            fixes.append(
                f"Step {step.id} imports from {file_path} "
                f"(produced by step {producer_id}) — added depends:{producer_id}"
            )

    # Safety: verify we didn't introduce a cycle
    if fixes and _has_cycle(steps):
        # Roll back all injected deps (unlikely but safe)
        _rollback = {f.split(" — ")[0]: f.split("depends:")[1] for f in fixes}
        for step in steps:
            for desc, dep_id in _rollback.items():
                if dep_id in step.depends_on and f"Step {step.id}" in desc:
                    step.depends_on.remove(dep_id)
        fixes.append("WARNING: rolled back fixes — injecting deps would create a cycle")

    return fixes


def _has_cycle(steps: list[PlanStep]) -> bool:
    """Detect circular dependencies via DFS."""
    all_ids = {s.id for s in steps}
    # Build adjacency: step depends on → dependency
    visited: set[str] = set()
    in_stack: set[str] = set()

    def dfs(sid: str) -> bool:
        if sid in in_stack:
            return True
        if sid in visited:
            return False
        visited.add(sid)
        in_stack.add(sid)
        step = next((s for s in steps if s.id == sid), None)
        if step:
            for dep in step.depends_on:
                if dep in all_ids and dfs(dep):
                    return True
        in_stack.discard(sid)
        return False

    for s in steps:
        if dfs(s.id):
            return True
    return False


# ---------------------------------------------------------------------------
# Wave builder
# ---------------------------------------------------------------------------

def build_waves(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """Topological sort into parallel execution waves, respecting phase order.

    Steps are grouped by their **primary step number** (the integer
    before the dot, e.g. ``1`` in ``1.1``, ``1.2``).  All waves within
    phase 1 complete before phase 2 starts.  Within each phase, steps
    are scheduled based on ``depends_on`` — independent sub-steps
    within the same phase can execute in parallel.

    This ensures that setup phases (CMD scaffolding) finish before
    code-generation phases begin, even when the planner omits explicit
    cross-phase dependencies.
    """
    step_map = {s.id: s for s in steps}

    # ── Group steps by primary number ──
    from collections import OrderedDict
    phase_groups: OrderedDict[str, list[PlanStep]] = OrderedDict()
    for s in steps:
        primary = s.id.split(".")[0] if "." in s.id else s.id
        phase_groups.setdefault(primary, []).append(s)

    # ── Infer implicit deps: within a phase, steps with no declared deps
    #    should wait for any CMD steps that precede them (by step ID).
    #    This handles the common case where the LLM scaffolds a project in
    #    step 1.1 [CMD] and then writes files in 1.4–1.9 [CODE] with
    #    depends:none, causing them to run concurrently with the scaffold.
    for phase_steps in phase_groups.values():
        # Collect CMD step IDs in sorted order within the phase
        cmd_ids_in_phase = sorted(
            s.id for s in phase_steps if s.step_type == "CMD"
        )
        if not cmd_ids_in_phase:
            continue
        for s in phase_steps:
            if s.step_type == "CMD":
                # Chain CMD steps sequentially within the same phase to
                # prevent parallel package-manager operations (npm/pip/yarn)
                # targeting the same directory — causes ENOTEMPTY on Windows.
                # Always enforce this: even if a CMD step already has explicit
                # deps (e.g. depends:1.1), it must also wait for the immediately
                # preceding CMD step so installs don't run concurrently.
                preceding = [cid for cid in cmd_ids_in_phase if cid < s.id]
                if preceding:
                    prev_cmd = preceding[-1]
                    if prev_cmd not in s.depends_on:
                        s.depends_on = list(s.depends_on) + [prev_cmd]
                continue
            if s.depends_on:
                continue  # already has explicit deps — don't override
            # Add any CMD steps whose ID sorts before this step's ID
            implicit = [cid for cid in cmd_ids_in_phase if cid < s.id]
            if implicit:
                s.depends_on = list(implicit)

    waves: list[list[PlanStep]] = []
    completed: set[str] = set()

    for _phase_key, phase_steps in phase_groups.items():
        remaining = {s.id for s in phase_steps}

        while remaining:
            ready = [
                sid for sid in sorted(remaining)
                if all(d in completed for d in step_map[sid].depends_on)
            ]
            if not ready:
                # Circular or missing deps — pick the smallest ID to unblock
                ready = [min(remaining)]
            wave = [step_map[sid] for sid in ready]
            waves.append(wave)
            completed.update(ready)
            remaining -= set(ready)

    return waves


# ---------------------------------------------------------------------------
# Context builder (per-step)
# ---------------------------------------------------------------------------

def build_step_context(
    step: PlanStep,
    all_steps: list[PlanStep],
    memory,
    read_from_disk=None,
) -> dict[str, str]:
    """Build the file context for a step using plan-declared imports.

    Returns a dict of file_path -> content to inject into the LLM prompt.

    Parameters
    ----------
    step:
        The step about to execute.
    all_steps:
        All steps in the plan (for ghost contracts).
    memory:
        FileMemory instance with completed files.
    read_from_disk:
        Optional callable(file_path) -> str|None to read existing files.
    """
    files: dict[str, str] = {}

    # 1. Plan-declared imports (real or ghost)
    for file_path, symbols in step.imports_from.items():
        # Compute the correct Python import path relative to each target file.
        # The first target is used as the reference; if there are no targets,
        # fall back to a simple path→module conversion.
        ref_target = step.target_files[0] if step.target_files else ""
        import_hint = ""
        _JS_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")
        if ref_target and file_path.endswith(".py"):
            import_module = _relative_import_path(ref_target, file_path)
            symbol_str = ", ".join(symbols) if symbols else "..."
            import_hint = f"# Correct Python import: from {import_module} import {symbol_str}\n"
        elif ref_target and any(file_path.endswith(ext) for ext in _JS_EXTS):
            import_module = _relative_import_path(ref_target, file_path)
            # JS/TS relative imports need ./ prefix for same/sub-directory files
            if not import_module.startswith(".."):
                import_module = "./" + import_module
            symbol_str = ", ".join(symbols) if symbols else "..."
            import_hint = f"// Correct import: import {{ {symbol_str} }} from '{import_module}'\n"

        content = memory.get(file_path) if memory else None
        if content:
            files[file_path] = import_hint + content
        elif read_from_disk:
            disk_content = read_from_disk(file_path)
            if disk_content:
                files[file_path] = import_hint + disk_content
        else:
            # Ghost contract: file not yet created, include planned info
            producer = _find_producer(file_path, all_steps)
            if producer and producer.status != "completed":
                ghost = (
                    f"# [PLANNED FILE — will be created by step {producer.id}]\n"
                    f"# Exports: {', '.join(producer.exports) if producer.exports else 'TBD'}\n"
                    + import_hint
                )
                files[file_path] = ghost

    # 2. Target files being modified — read current content + parse imports
    for target in step.target_files:
        if target in files:
            continue
        content = (memory.get(target) if memory else None)
        if content is None and read_from_disk:
            content = read_from_disk(target)
        if content:
            files[target] = content
            # Parse actual imports from the target file to catch undeclared deps
            try:
                from .dependency_check import extract_file_deps
                deps = extract_file_deps(target, content)
                for imp in deps.imports:
                    imp_file = _resolve_import_to_file(imp, memory, read_from_disk)
                    if imp_file and imp_file not in files:
                        imp_content = (memory.get(imp_file) if memory else None)
                        if imp_content is None and read_from_disk:
                            imp_content = read_from_disk(imp_file)
                        if imp_content:
                            files[imp_file] = imp_content
            except Exception:
                pass  # best-effort dependency resolution

    return files


def _relative_import_path(target_file: str, dep_file: str) -> str:
    """
    Compute the correct Python import module name for *dep_file* as seen
    from *target_file*.

    Both paths use forward slashes and are relative to the project root.

    Examples
    --------
    >>> _relative_import_path("src/main.py", "src/cracker_anim.py")
    'cracker_anim'
    >>> _relative_import_path("main.py", "src/cracker_anim.py")
    'src.cracker_anim'
    >>> _relative_import_path("a/b/c.py", "a/b/utils.py")
    'utils'
    >>> _relative_import_path("a/b/c.py", "a/helpers.py")
    'a.helpers'
    """
    import posixpath

    target_dir = posixpath.dirname(target_file.replace("\\", "/"))
    dep_norm = dep_file.replace("\\", "/")
    # Strip .py extension to get module path
    dep_module = dep_norm[:-3] if dep_norm.endswith(".py") else dep_norm

    target_parts = [p for p in target_dir.split("/") if p]
    dep_parts = dep_module.split("/")

    # Find common prefix
    common = 0
    for t, d in zip(target_parts, dep_parts):
        if t == d:
            common += 1
        else:
            break

    # If dep is directly accessible from the same directory, use the remainder
    remaining = dep_parts[common:]
    return ".".join(remaining) if remaining else dep_module.replace("/", ".")


def _find_producer(file_path: str, steps: list[PlanStep]) -> Optional[PlanStep]:
    """Find the step that produces a given file."""
    for s in steps:
        if file_path in s.target_files:
            return s
    return None


def _resolve_import_to_file(
    import_source: str,
    memory,
    read_from_disk=None,
) -> Optional[str]:
    """Best-effort resolution of an import string to a file path in memory."""
    if memory is None:
        return None
    all_files = memory.all_files()

    # Direct match
    if import_source in all_files:
        return import_source

    # Python: dots to path (e.g. "utils.helpers" -> "utils/helpers.py")
    as_path = import_source.replace(".", "/")
    for ext in (".py", ".js", ".ts", ".tsx", ".jsx", ""):
        candidate = as_path + ext
        if candidate in all_files:
            return candidate

    # JS relative: "./utils" -> "utils.js" or "utils/index.js"
    clean = import_source.lstrip("./")
    for ext in (".js", ".ts", ".tsx", ".jsx", "/index.js", "/index.ts", ""):
        candidate = clean + ext
        if candidate in all_files:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Post-step update
# ---------------------------------------------------------------------------

def update_step_after_execution(
    step: PlanStep,
    generated_files: dict[str, str],
) -> None:
    """Update a step's metadata after successful execution.

    Parses real exports from generated code (zero LLM cost).
    """
    step.status = "completed"
    actual_exports: list[str] = []

    for fpath, content in generated_files.items():
        try:
            from .dependency_check import extract_file_deps
            deps = extract_file_deps(fpath, content)
            actual_exports.extend(deps.exports)
        except Exception:
            pass

        # Add to target_files if not already tracked
        if fpath not in step.target_files:
            step.target_files.append(fpath)

    step.actual_exports = actual_exports


# ---------------------------------------------------------------------------
# Legacy compatibility helpers
# ---------------------------------------------------------------------------

def steps_as_text_list(steps: list[PlanStep]) -> list[str]:
    """Convert PlanStep list to legacy list[str] for backward compat."""
    return [s.description for s in steps]


def steps_dependencies_dict(steps: list[PlanStep]) -> dict[int, set[int]]:
    """Convert PlanStep depends_on to legacy dict[int, set[int]] format."""
    id_to_idx = {s.id: s.index for s in steps}
    deps: dict[int, set[int]] = {}
    for s in steps:
        dep_indices = set()
        for dep_id in s.depends_on:
            if dep_id in id_to_idx:
                dep_indices.add(id_to_idx[dep_id])
        deps[s.index] = dep_indices
    return deps


# ---------------------------------------------------------------------------
# Fallback: convert old numbered-list plan to PlanStep objects
# ---------------------------------------------------------------------------

def from_legacy_steps(
    steps: list[str],
    dependencies: dict[int, set[int]],
) -> list[PlanStep]:
    """Convert old-format (list[str] + deps dict) to PlanStep objects.

    Used for checkpoint backward compatibility and gradual migration.
    Step type defaults to 'CODE' (will be classified at runtime).
    """
    result: list[PlanStep] = []
    idx_to_id = {i: str(i + 1) for i in range(len(steps))}

    for idx, text in enumerate(steps):
        dep_ids = [
            idx_to_id[d] for d in dependencies.get(idx, set())
            if d in idx_to_id
        ]
        result.append(PlanStep(
            id=idx_to_id[idx],
            step_type="UNCLASSIFIED",  # needs runtime classification
            description=text,
            depends_on=dep_ids,
            index=idx,
        ))

    return result


# ---------------------------------------------------------------------------
# Heuristic fallback parser for weaker LLMs
# ---------------------------------------------------------------------------

# Matches markdown section headers like: ### Step 1: description
#                                         ## Step 1.1 - description
_HEURISTIC_HEADER_RE = re.compile(
    r'^#{1,4}\s+(?:Step\s+)([\d.]+)[:.)\s-]*\s*(.*)?$',
    re.IGNORECASE,
)

# Matches bold key-value lines: **Type:** CODE  or  **Dependencies:** 1.1
# Handles both `**Key:** value` (markdown bold with colon inside closing **)
# and plain `Key: value` formats. The middle `\*{0,2}:` consumes `**:` or `:`.
_HEURISTIC_KV_RE = re.compile(
    r'^\*{0,2}(step\s*(?:id|#|number)?|type|dependenc(?:y|ies)|depends?(?:\s*on)?'
    r'|target|exports?|imports?|produces?|description)\*{0,2}:\s*(.*)$',
    re.IGNORECASE,
)

# Standalone **Step ID:** 1.1 — can appear as step boundary without a header
_HEURISTIC_STEPID_RE = re.compile(
    r'^\*{0,2}Step\s*(?:ID|#|Number)\*{0,2}:\s*\*{0,2}\s*([\d.]+)',
    re.IGNORECASE,
)

_VALID_STEP_TYPES = {"CMD", "CODE", "TEST", "IGNORE", "SEARCH"}


def parse_heuristic_plan(text: str) -> list[PlanStep]:
    """Fallback parser for non-standard plan formats produced by weaker LLMs.

    Handles markdown-heavy outputs like::

        ### Step 1: Create Pricelist Component
        **Step ID:** 1.1
        **Type:** CODE
        **Dependencies:** None
        **Target:** src/components/Pricelist.jsx
        **Exports:** Pricelist
        **Imports:** None

        ### Step 2: Integrate into App
        **Step ID:** 1.2
        **Type:** CODE
        **Dependencies:** 1.1
        **Target:** src/App.jsx

    Also handles ``> command`` lines and commands inside ```bash fences.

    Returns an empty list if no recognisable steps are found — safe to
    call unconditionally before the legacy numbered-list parser.
    """
    steps: list[PlanStep] = []
    current: Optional[PlanStep] = None
    desc_lines: list[str] = []
    cmd_lines: list[str] = []
    in_code_fence = False

    def _flush() -> None:
        nonlocal current, desc_lines, cmd_lines
        if current is None:
            return
        if not current.description and desc_lines:
            current.description = " ".join(desc_lines).strip()
        _flush_echo_inline(current, cmd_lines)
        steps.append(current)
        current = None
        desc_lines = []
        cmd_lines = []

    def _strip_value(raw: str) -> str:
        """Strip markdown bold markers, backticks, and trailing whitespace."""
        v = raw.strip()
        # Strip leading ** (residual closing bold from **key:** format)
        if v.startswith("**"):
            v = v[2:].strip()
        # Strip backtick wrapping (e.g. `value`)
        v = v.strip("`")
        # Strip trailing markdown bold/whitespace
        v = v.rstrip("* \t").strip()
        return v

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Code fence tracking ──
        if line.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        # ── Markdown section header → new step ──
        m_hdr = _HEURISTIC_HEADER_RE.match(line)
        if m_hdr:
            _flush()
            step_id = m_hdr.group(1)
            description = (m_hdr.group(2) or "").strip()
            current = PlanStep(id=step_id, step_type="UNCLASSIFIED",
                               description=description)
            desc_lines = []
            cmd_lines = []
            continue

        # ── Standalone **Step ID:** as boundary when no header was used ──
        m_sid = _HEURISTIC_STEPID_RE.match(line)
        if m_sid:
            if current is None:
                # No header yet — start a new step
                _flush()
                current = PlanStep(id=m_sid.group(1), step_type="UNCLASSIFIED")
                desc_lines = []
                cmd_lines = []
            else:
                # Inside a step: refine the ID (header may have said "Step 1",
                # **Step ID:** may say "1.1" for more precision)
                current.id = m_sid.group(1)
            continue

        if current is None:
            continue  # skip preamble lines before any step is detected

        # ── Key-value metadata line ──
        m_kv = _HEURISTIC_KV_RE.match(line)
        if m_kv:
            key = m_kv.group(1).lower().replace(" ", "").rstrip("*")
            val = _strip_value(m_kv.group(2))

            if "type" in key:
                t = val.upper()
                if t in _VALID_STEP_TYPES:
                    current.step_type = t

            elif "depend" in key:
                if val.lower() not in ("none", "n/a", ""):
                    current.depends_on = [
                        d.strip() for d in val.split(",") if d.strip()
                    ]

            elif key == "target":
                if val.lower() not in ("none", "n/a", ""):
                    current.target_files = [
                        f.strip() for f in val.split(",") if f.strip()
                    ]

            elif "export" in key:
                if val.lower() not in ("none", "n/a", ""):
                    current.exports = [
                        e.strip() for e in val.split(",") if e.strip()
                    ]

            elif "import" in key:
                if val.lower() not in ("none", "n/a", ""):
                    for entry in val.split(","):
                        entry = entry.strip()
                        if ":" in entry:
                            fp, sym = entry.rsplit(":", 1)
                            current.imports_from.setdefault(
                                fp.strip(), []
                            ).append(sym.strip())

            elif "produce" in key:
                if val.lower() not in ("none", "n/a", ""):
                    for f in val.split(","):
                        f = f.strip()
                        if f and f not in current.target_files:
                            current.target_files.append(f)

            elif "description" in key:
                if val:
                    current.description = val

            elif "stepid" in key or "step#" in key or "stepnumber" in key:
                if re.match(r'^[\d.]+$', val):
                    current.id = val

            continue  # metadata line — don't fall through to description

        # ── Command line (> cmd) ── works both inside and outside fences
        if line.startswith("> "):
            cmd_text = line[2:].strip()
            # Strip backtick wrapping added by LLMs (e.g. `> `npm install`` -> `npm install`)
            if len(cmd_text) >= 2 and cmd_text[0] == "`" and cmd_text[-1] == "`":
                cmd_text = cmd_text[1:-1]
            _bare = cmd_text.lstrip("*_ \t")
            _meta_prefixes = ("produces:", "note:", "output:", "creates:",
                              "result:", "generates:", "returns:")
            if not cmd_text.startswith("**") and not any(
                _bare.lower().startswith(p) for p in _meta_prefixes
            ):
                if current.command:
                    current.command = current.command + " && " + cmd_text
                else:
                    current.command = cmd_text
                cmd_lines.append(cmd_text)
            continue

        # ── Ignore horizontal rules and plan boundary markers ──
        if line.startswith("---") or line.startswith("===") or line.startswith("***"):
            continue

        # ── Description continuation (plain text, not a metadata key) ──
        if line and not line.startswith("==") and not line.startswith("**"):
            desc_lines.append(line)

    _flush()

    # ── Post-process: infer type from content when still UNCLASSIFIED ──
    for idx, step in enumerate(steps):
        step.index = idx
        if step.step_type == "UNCLASSIFIED":
            if step.command and not step.target_files:
                step.step_type = "CMD"
            elif step.target_files:
                step.step_type = "CODE"

    # Only return steps that have at least a description or target/command
    return [s for s in steps if s.description or s.target_files or s.command]


def is_structured_plan(text: str) -> bool:
    """Check if LLM output is in the new structured format."""
    # Check each line independently (handles leading whitespace in raw text)
    for line in text.splitlines():
        if _STEP_RE.match(line.strip()):
            return True
    return False


# ---------------------------------------------------------------------------
# Manifest reclassification — CODE → CMD for protected manifest files
# ---------------------------------------------------------------------------

# Manifest basenames that are protected from direct LLM overwrite.
# Keep in sync with Executor._PROTECTED_FILENAMES.
_MANIFEST_BASENAMES: frozenset = frozenset({
    'package.json', 'requirements.txt', 'Cargo.toml', 'go.mod',
    'Gemfile', 'composer.json', 'Pipfile', 'pyproject.toml', 'setup.py',
})

# Manifest basename → package-manager install prefix
_MANIFEST_PM_PREFIX: dict = {
    'package.json':    'npm install',
    'requirements.txt': 'pip install',
    'Cargo.toml':      'cargo add',
    'go.mod':          'go get',
    'Gemfile':         'bundle add',
    'composer.json':   'composer require',
    'Pipfile':         'pipenv install',
    'pyproject.toml':  'pip install',
    'setup.py':        'pip install',
}

# package.json fields that are metadata, not dependency names
_PKG_JSON_NON_DEP_FIELDS = frozenset({
    'name', 'version', 'description', 'main', 'module', 'browser',
    'type', 'author', 'license', 'homepage', 'repository', 'bugs',
    'private', 'engines', 'os', 'cpu',
})


def _extract_packages_from_inline_edits(step: 'PlanStep') -> list:
    """Return package names added by the step's inline find/replace edits."""
    import os as _os
    packages: list = []
    for fpath, pairs in (step.inline_edits or {}).items():
        basename = _os.path.basename(fpath)
        for find_str, replace_str in pairs:
            find_lines = {l.strip().rstrip(',') for l in find_str.splitlines() if l.strip()}
            for raw_line in replace_str.splitlines():
                stripped = raw_line.strip().rstrip(',')
                if not stripped or stripped in find_lines:
                    continue
                if basename == 'package.json':
                    # Match: "animejs": "^4.2.0"
                    m = re.match(r'^"([^"]+)"\s*:\s*"[^"]*"$', stripped)
                    if m and m.group(1) not in _PKG_JSON_NON_DEP_FIELDS:
                        packages.append(m.group(1))
                elif basename in ('requirements.txt', 'Pipfile', 'pyproject.toml', 'setup.py'):
                    # Match: animejs==4.2.0  or  animejs>=3
                    m = re.match(r'^([A-Za-z0-9_.-]+)', stripped)
                    if m and not stripped.startswith('#') and not stripped.startswith('['):
                        packages.append(m.group(1))
                elif basename == 'Cargo.toml':
                    # Match: animejs = "4.2.0"  (skip section headers)
                    if not stripped.startswith('[') and '=' in stripped:
                        pkg_name = stripped.split('=')[0].strip()
                        if re.match(r'^[a-zA-Z0-9_-]+$', pkg_name):
                            packages.append(pkg_name)
                elif basename == 'go.mod':
                    # Match: require github.com/foo/bar v1.2.3
                    m = re.match(r'^(?:require\s+)?(\S+/\S+)\s+v', stripped)
                    if m:
                        packages.append(m.group(1))
    return packages


def reclassify_manifest_steps(plan_steps: list) -> list:
    """Convert CODE steps whose targets are *only* dependency manifest files into
    CMD package-manager install steps.

    Rationale: The executor's protected-file guard blocks direct overwrites of
    ``package.json``, ``requirements.txt``, etc.  Converting to ``npm install``
    (or the ecosystem equivalent) lets the package manager do the safe, atomic
    update and actually installs the package into ``node_modules`` /
    ``site-packages``.  The guard still exists as a safety net for any stray
    LLM-generated writes that slip through.

    Steps where no packages can be detected from the inline edits are left
    unchanged so they fall through to the normal CODE path (which will log a
    warning about the protected write).
    """
    import os as _os
    import logging as _logging
    _log = _logging.getLogger(__name__)

    for step in plan_steps:
        if step.step_type != "CODE":
            continue
        if not step.target_files:
            continue

        basenames = [_os.path.basename(f) for f in step.target_files]
        if not all(b in _MANIFEST_BASENAMES for b in basenames):
            continue  # at least one non-manifest target — leave as CODE

        packages = _extract_packages_from_inline_edits(step)

        if not packages:
            _log.debug(
                "[PlanStep] Step %s targets manifest(s) only but no packages "
                "detected from inline edits — leaving as CODE", step.id,
            )
            continue

        primary = basenames[0]
        prefix = _MANIFEST_PM_PREFIX.get(primary, 'pip install')
        install_cmd = f"{prefix} {' '.join(packages)}"

        _log.info(
            "[PlanStep] Step %s reclassified CODE→CMD "
            "(manifest-only target): %s", step.id, install_cmd,
        )
        step.step_type = "CMD"
        step.command = install_cmd
        step.inline_edits = {}
        step.inline_code = {}

    return plan_steps

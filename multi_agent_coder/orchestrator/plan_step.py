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
            index=d.get("index", -1),
        )


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
    """
    steps: list[PlanStep] = []
    current: Optional[PlanStep] = None
    desc_lines: list[str] = []
    in_code_block = False
    code_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Inline code block handling ──
        if line.lower().replace(" ", "").startswith("---file-content-start"):
            in_code_block = True
            code_lines = []
            continue
        if line.lower().replace(" ", "").startswith("---file-content-end"):
            in_code_block = False
            if current is not None and code_lines:
                _assign_inline_code(current, code_lines)
            code_lines = []
            continue
        if in_code_block:
            code_lines.append(raw_line)  # preserve original indentation
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
            continue

        if current is None:
            continue

        # Command line (for CMD steps)
        if line.startswith("> "):
            current.command = line[2:].strip()

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

        # Description line (anything else)
        elif not line.startswith("=="):
            desc_lines.append(line)

    # Flush last step
    if current is not None:
        current.description = " ".join(desc_lines).strip()
        steps.append(current)

    # Assign 0-based indices
    for idx, step in enumerate(steps):
        step.index = idx

    return steps


# File header comment pattern for splitting multi-file inline code blocks
_FILE_COMMENT_RE = re.compile(
    r"^//\s*([\w./-]+\.\w{1,5})\s*$"
)


def _assign_inline_code(step: PlanStep, code_lines: list[str]) -> None:
    """Assign captured inline code to a step's ``inline_code`` dict.

    For single-target steps, all code goes to that target file.
    For multi-target steps, the parser splits on ``// FileName.ext``
    comment headers to map code to each file.
    """
    full_code = "\n".join(code_lines).strip()
    if not full_code:
        return

    targets = step.target_files

    if len(targets) == 1:
        step.inline_code[targets[0]] = full_code
        return

    # Multi-target: try splitting on // filename.ext comment headers
    current_file: Optional[str] = None
    file_lines: dict[str, list[str]] = {}

    for line in code_lines:
        m = _FILE_COMMENT_RE.match(line.strip())
        if m:
            fname = m.group(1)
            # Match against known targets or use as-is
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
    elif len(targets) > 0:
        # No file headers found — assign all code to first target
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

def validate_plan(steps: list[PlanStep]) -> list[str]:
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
    """Topological sort into parallel execution waves.

    Each wave is a list of steps whose dependencies are all satisfied.
    Steps within a wave can execute in parallel.
    """
    completed: set[str] = set()
    remaining = {s.id for s in steps}
    step_map = {s.id: s for s in steps}
    waves: list[list[PlanStep]] = []

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
        content = memory.get(file_path) if memory else None
        if content:
            files[file_path] = content
        elif read_from_disk:
            disk_content = read_from_disk(file_path)
            if disk_content:
                files[file_path] = disk_content
        else:
            # Ghost contract: file not yet created, include planned info
            producer = _find_producer(file_path, all_steps)
            if producer and producer.status != "completed":
                ghost = (
                    f"# [PLANNED FILE — will be created by step {producer.id}]\n"
                    f"# Exports: {', '.join(producer.exports) if producer.exports else 'TBD'}\n"
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


def is_structured_plan(text: str) -> bool:
    """Check if LLM output is in the new structured format."""
    # Check each line independently (handles leading whitespace in raw text)
    for line in text.splitlines():
        if _STEP_RE.match(line.strip()):
            return True
    return False

"""
AnalyseAgent — analyses the task, plan, and project state to produce a
structured ProjectContext that flows through the entire pipeline.

Runs AFTER the Planner (we need the plan steps) but BEFORE step execution.
Gives every downstream agent (Coder, Reviewer, Tester) awareness of:
  - The end-to-end goal and what success looks like
  - Dependency manifest (installed vs needed)
  - Import patterns and module system
  - Test strategy (what to test, correct assertion patterns)
  - File structure and naming conventions
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from .base import Agent

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProjectContext — the structured output
# ---------------------------------------------------------------------------

@dataclass
class ProjectContext:
    """Structured analysis that flows through the entire pipeline.

    Every field is optional so partial analysis still works.
    """

    # End-to-end goal
    goal_summary: str = ""
    success_criteria: list[str] = field(default_factory=list)

    # Language & framework
    language: str = ""
    framework: str = ""
    module_system: str = ""  # "esm", "commonjs", or ""

    # Dependencies
    installed_packages: list[str] = field(default_factory=list)
    required_packages: list[str] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)

    # Import patterns
    import_style: str = ""  # "import/export", "require/module.exports", "mixed"
    import_examples: list[str] = field(default_factory=list)

    # Test strategy
    test_framework: str = ""  # "jest", "vitest", "pytest", etc.
    test_patterns: list[str] = field(default_factory=list)
    testable_units: list[str] = field(default_factory=list)
    assertion_hints: list[str] = field(default_factory=list)

    # File structure
    source_root: str = ""
    test_root: str = ""
    naming_convention: str = ""  # "camelCase", "snake_case", etc.
    file_extension: str = ""  # ".ts", ".js", ".py", etc.
    entry_points: list[str] = field(default_factory=list)

    def format_for_coder(self) -> str:
        """Compact context block for the Coder agent."""
        parts = []
        if self.goal_summary:
            parts.append(f"END-TO-END GOAL: {self.goal_summary}")
        if self.language:
            parts.append(f"Language: {self.language}" +
                         (f" ({self.framework})" if self.framework else ""))
        if self.module_system:
            parts.append(f"Module system: {self.module_system}")
        if self.import_style:
            parts.append(f"Import style: {self.import_style}")
        if self.import_examples:
            parts.append("Import examples from this project:")
            for ex in self.import_examples[:5]:
                parts.append(f"  {ex}")
        if self.installed_packages:
            parts.append(f"Installed packages: {', '.join(self.installed_packages[:30])}")
        if self.missing_packages:
            parts.append(f"NEED TO INSTALL: {', '.join(self.missing_packages)}")
        if self.source_root:
            parts.append(f"Source root: {self.source_root}")
        return "\n".join(parts)

    def format_for_tester(self) -> str:
        """Detailed context block for the Tester agent — this is the key
        improvement.  Gives the tester everything it needs to generate
        tests that actually pass on the first try."""
        parts = []

        if self.goal_summary:
            parts.append(f"=== WHAT THIS CODE SHOULD DO ===")
            parts.append(self.goal_summary)

        if self.success_criteria:
            parts.append("\n=== SUCCESS CRITERIA (what tests should verify) ===")
            for i, crit in enumerate(self.success_criteria, 1):
                parts.append(f"{i}. {crit}")

        if self.language:
            lang_line = f"Language: {self.language}"
            if self.framework:
                lang_line += f" | Framework: {self.framework}"
            if self.module_system:
                lang_line += f" | Module: {self.module_system}"
            parts.append(f"\n{lang_line}")

        if self.test_framework:
            parts.append(f"Test framework: {self.test_framework}")

        if self.import_style:
            parts.append(f"\n=== IMPORT RULES ===")
            parts.append(f"Style: {self.import_style}")
            if self.module_system == "esm":
                parts.append("USE: import {{ x }} from './path.js';")
                parts.append("DO NOT USE: require()")
            elif self.module_system == "commonjs":
                parts.append("USE: const {{ x }} = require('./path');")
                parts.append("DO NOT USE: import/export")
            if self.import_examples:
                parts.append("Working import examples from this project:")
                for ex in self.import_examples[:5]:
                    parts.append(f"  {ex}")

        if self.installed_packages:
            parts.append(f"\n=== AVAILABLE PACKAGES (already installed) ===")
            parts.append(", ".join(self.installed_packages[:40]))
            parts.append("ONLY import from packages listed above or from relative project files.")
            parts.append("Do NOT import from packages that are not installed.")

        if self.testable_units:
            parts.append(f"\n=== WHAT TO TEST ===")
            for unit in self.testable_units:
                parts.append(f"- {unit}")

        if self.assertion_hints:
            parts.append(f"\n=== ASSERTION GUIDANCE ===")
            for hint in self.assertion_hints:
                parts.append(f"- {hint}")

        if self.test_patterns:
            parts.append(f"\n=== TEST PATTERNS FOR THIS PROJECT ===")
            for pat in self.test_patterns:
                parts.append(f"- {pat}")

        if self.source_root:
            parts.append(f"\nSource root: {self.source_root}")
        if self.test_root:
            parts.append(f"Test root: {self.test_root}")
        if self.file_extension:
            parts.append(f"File extension: {self.file_extension}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Static analysis helpers (no LLM needed)
# ---------------------------------------------------------------------------

def _read_package_json(root: str = ".") -> dict:
    """Read and parse package.json if it exists."""
    path = os.path.join(root, "package.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _read_requirements(root: str = ".") -> list[str]:
    """Read requirements.txt packages."""
    path = os.path.join(root, "requirements.txt")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip().split("==")[0].split(">=")[0].split("<=")[0]
                    for line in f if line.strip() and not line.startswith("#")]
    except OSError:
        return []


def _extract_import_examples(source_files: dict[str, str],
                              language: str, max_examples: int = 8) -> list[str]:
    """Extract real import statements from existing source files."""
    examples = []
    for fpath, content in source_files.items():
        # Skip test files, node_modules, configs
        if any(skip in fpath for skip in ["node_modules", "__pycache__", ".test.", ".spec."]):
            continue
        for line in content.splitlines()[:30]:  # Only scan first 30 lines
            stripped = line.strip()
            if language in ("javascript", "typescript"):
                if stripped.startswith(("import ", "const ")) and ("from " in stripped or "require(" in stripped):
                    examples.append(stripped)
            elif language == "python":
                if stripped.startswith(("import ", "from ")):
                    examples.append(stripped)
            if len(examples) >= max_examples:
                return examples
    return examples


def _detect_module_system(pkg: dict, source_files: dict[str, str]) -> str:
    """Detect ESM vs CommonJS for JS/TS projects."""
    if pkg.get("type") == "module":
        return "esm"

    # Check for ESM indicators in source files
    esm_count = 0
    cjs_count = 0
    for content in list(source_files.values())[:20]:
        if re.search(r'\bexport\s+(default|const|function|class)\b', content):
            esm_count += 1
        if re.search(r'\bmodule\.exports\b|\brequire\s*\(', content):
            cjs_count += 1

    if esm_count > cjs_count:
        return "esm"
    elif cjs_count > 0:
        return "commonjs"
    return ""


def _detect_test_framework_from_project(pkg: dict, root: str = ".") -> str:
    """Detect test framework from package.json and config files."""
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

    if "vitest" in deps:
        return "vitest"
    if "jest" in deps:
        return "jest"
    if "@jest/globals" in deps:
        return "jest"
    if "mocha" in deps:
        return "mocha"

    # Check for config files
    for name in ("vitest.config.ts", "vitest.config.js", "vite.config.ts", "vite.config.js"):
        if os.path.isfile(os.path.join(root, name)):
            return "vitest"
    for name in ("jest.config.js", "jest.config.ts", "jest.config.mjs", "jest.config.cjs"):
        if os.path.isfile(os.path.join(root, name)):
            return "jest"

    # Python
    if os.path.isfile(os.path.join(root, "pytest.ini")) or os.path.isfile(os.path.join(root, "setup.cfg")):
        return "pytest"
    if os.path.isfile(os.path.join(root, "requirements.txt")):
        reqs = _read_requirements(root)
        if "pytest" in reqs:
            return "pytest"

    return ""


def _get_installed_packages(pkg: dict, root: str = ".") -> list[str]:
    """Get list of all installed packages."""
    packages = []
    if pkg:
        packages.extend(pkg.get("dependencies", {}).keys())
        packages.extend(pkg.get("devDependencies", {}).keys())
    reqs = _read_requirements(root)
    if reqs:
        packages.extend(reqs)
    return sorted(set(packages))


def _detect_source_root(source_files: dict[str, str]) -> str:
    """Detect the primary source root from file paths."""
    roots = {}
    for fpath in source_files:
        parts = fpath.replace("\\", "/").split("/")
        if len(parts) > 1 and parts[0] not in ("node_modules", "__pycache__", ".git", "test", "tests", "__tests__"):
            roots[parts[0]] = roots.get(parts[0], 0) + 1
    if roots:
        return max(roots, key=roots.get)
    return "."


def _detect_test_root(source_files: dict[str, str]) -> str:
    """Detect the test directory from file paths.

    Checks both root-level (``__tests__/``) and nested (``src/__tests__/``)
    test directories.  Prefers more-specific (nested) matches so that the
    LLM writes tests to the correct location.
    """
    candidates = ("__tests__", "tests", "test", "spec")
    # First pass: look for nested test dirs (e.g. src/__tests__)
    for fpath in source_files:
        norm = fpath.replace("\\", "/")
        for candidate in candidates:
            needle = "/" + candidate + "/"
            idx = norm.find(needle)
            if idx != -1:
                # Return path up to and including the test dir
                return norm[: idx + len(needle) - 1]  # e.g. "src/__tests__"
    # Second pass: root-level test dirs
    for candidate in candidates:
        for fpath in source_files:
            if fpath.replace("\\", "/").startswith(candidate + "/"):
                return candidate
    return ""


# ---------------------------------------------------------------------------
# Static-only analysis (the default, no LLM cost)
# ---------------------------------------------------------------------------

def build_project_context(
    task: str,
    plan_steps: list[str],
    source_files: dict[str, str],
    language: str | None = None,
    project_profile=None,
    subproject_root: str | None = None,
) -> ProjectContext:
    """Build a ProjectContext from static analysis only (zero LLM cost).

    This is the primary entry point. For most projects, static analysis
    provides enough information. LLM enrichment is optional and additive.
    """
    root = subproject_root or "."
    pkg = _read_package_json(root)
    ctx = ProjectContext()

    # Language
    if language:
        ctx.language = language
    elif project_profile:
        ctx.language = getattr(project_profile, "language", "") or ""

    # Framework
    if project_profile:
        ctx.framework = getattr(project_profile, "framework", "") or ""

    # Module system (JS/TS only)
    if ctx.language in ("javascript", "typescript"):
        ctx.module_system = _detect_module_system(pkg, source_files)
        ctx.import_style = "import/export" if ctx.module_system == "esm" else (
            "require/module.exports" if ctx.module_system == "commonjs" else "")

    # File extension
    ext_map = {
        "javascript": ".js", "typescript": ".ts",
        "python": ".py", "go": ".go", "rust": ".rs", "java": ".java",
    }
    ctx.file_extension = ext_map.get(ctx.language, "")

    # Packages
    ctx.installed_packages = _get_installed_packages(pkg, root)

    # Analyze plan steps for required packages
    _pkg_patterns = re.findall(
        r'(?:npm\s+install|pip\s+install|yarn\s+add)\s+([\w@/\-. ]+)',
        " ".join(plan_steps), re.I)
    for match in _pkg_patterns:
        for p in match.split():
            p = p.strip().rstrip(",")
            if p and p not in ("--save-dev", "--save", "-D", "-g"):
                ctx.required_packages.append(p)
    ctx.missing_packages = [p for p in ctx.required_packages
                            if p not in ctx.installed_packages]

    # Test framework
    ctx.test_framework = _detect_test_framework_from_project(pkg, root)
    if not ctx.test_framework and project_profile:
        fws = getattr(project_profile, "test_frameworks", [])
        if fws:
            ctx.test_framework = fws[0]

    # Import examples from existing code
    ctx.import_examples = _extract_import_examples(
        source_files, ctx.language)

    # Source and test roots
    ctx.source_root = _detect_source_root(source_files)
    ctx.test_root = _detect_test_root(source_files)
    if project_profile:
        if getattr(project_profile, "source_root", ""):
            ctx.source_root = project_profile.source_root
        if getattr(project_profile, "test_root", ""):
            ctx.test_root = project_profile.test_root

    # Entry points
    if project_profile:
        ctx.entry_points = getattr(project_profile, "entry_points", []) or []

    # Goal summary from task
    ctx.goal_summary = task

    # Extract testable units from plan steps
    code_steps = [s for s in plan_steps
                  if re.search(r'\b(create|write|add|implement|build|update)\b', s, re.I)
                  and not re.search(r'\b(test|spec)\b', s, re.I)]
    for step in code_steps:
        # Extract file paths mentioned in the step
        files = re.findall(r'`([^`]+\.\w+)`', step)
        if files:
            ctx.testable_units.append(
                f"{step.strip()} (files: {', '.join(files)})")
        else:
            ctx.testable_units.append(step.strip())

    # Success criteria from plan — infer what the code should do
    for step in plan_steps:
        if re.search(r'\b(endpoint|route|api|function|component|class|module)\b', step, re.I):
            ctx.success_criteria.append(step.strip())

    # Assertion hints based on what we know
    if ctx.language in ("javascript", "typescript"):
        if ctx.test_framework == "vitest":
            ctx.assertion_hints.append(
                "Import test utilities: import { describe, it, expect } from 'vitest'")
        elif ctx.test_framework == "jest" and ctx.module_system == "esm":
            ctx.assertion_hints.append(
                "ESM project: import { describe, it, expect } from '@jest/globals'")
        elif ctx.test_framework == "jest":
            ctx.assertion_hints.append(
                "Jest globals (describe, it, expect) are available without import")
    elif ctx.language == "python":
        ctx.assertion_hints.append(
            "Use pytest: assert statements, pytest.raises() for exceptions")

    # Test patterns
    if ctx.test_framework:
        ctx.test_patterns.append(
            f"Use {ctx.test_framework} as the test runner")
    if ctx.module_system == "esm":
        ctx.test_patterns.append(
            "ES Module project — use import/export, include file extensions in paths")
    if ctx.import_examples:
        ctx.test_patterns.append(
            f"Follow the same import style as existing code")

    return ctx


# ---------------------------------------------------------------------------
# AnalyseAgent — optional LLM enrichment
# ---------------------------------------------------------------------------

class AnalyseAgent(Agent):
    """Analyses the task + plan + project to produce enriched ProjectContext.

    The static analysis in build_project_context() handles most cases.
    This agent adds LLM-powered enrichment for:
      - Deeper success criteria extraction
      - Assertion hint generation
      - Testable unit identification from complex plans
    """

    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """
You are analysing a coding task BEFORE execution begins.
Your job is to produce a structured analysis that will help the Coder and
Tester agents succeed on the first attempt.

Respond with a JSON object (no markdown, no explanation) with these keys:

{
  "goal_summary": "One paragraph explaining what the code should do end-to-end",
  "success_criteria": ["criterion 1", "criterion 2", ...],
  "testable_units": ["function/module 1 — what it should do", ...],
  "assertion_hints": ["what to assert and how", ...],
  "required_packages": ["package1", "package2", ...],
  "test_patterns": ["pattern 1", ...]
}

RULES:
- success_criteria: What would a QA engineer verify? Be specific.
- testable_units: What functions/classes/endpoints will exist? What should each do?
- assertion_hints: What specific values, types, or behaviors should tests check?
  Include expected return types, edge cases, error conditions.
- required_packages: Only packages explicitly needed that aren't standard library.
- test_patterns: Patterns specific to this project (e.g. "mock database calls",
  "test HTTP endpoints with supertest", "use React Testing Library for components").

Output ONLY the JSON object.
"""
        return self.llm_client.generate_response(prompt)

    def enrich_context(self, ctx: ProjectContext, task: str,
                       plan_steps: list[str],
                       source_files: dict[str, str]) -> ProjectContext:
        """Use LLM to enrich an existing ProjectContext with deeper analysis."""
        plan_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_steps))

        # Build compact source summary (imports + signatures only)
        source_summary = ""
        for fpath, content in list(source_files.items())[:15]:
            if any(skip in fpath for skip in ["node_modules", "__pycache__", ".git"]):
                continue
            lines = content.splitlines()[:20]
            source_summary += f"\n--- {fpath} ---\n" + "\n".join(lines) + "\n"

        context = (
            f"Task: {task}\n\n"
            f"Plan steps:\n{plan_text}\n\n"
            f"Language: {ctx.language}\n"
            f"Framework: {ctx.framework}\n"
            f"Module system: {ctx.module_system}\n"
            f"Test framework: {ctx.test_framework}\n"
            f"Installed packages: {', '.join(ctx.installed_packages[:30])}\n\n"
            f"Source files (first 20 lines each):\n{source_summary}"
        )

        try:
            response = self.process(task, context=context)
            data = _parse_json_response(response)
            if not data:
                return ctx

            # Merge LLM insights into existing context (additive, not replacing)
            if data.get("goal_summary"):
                ctx.goal_summary = data["goal_summary"]
            if data.get("success_criteria"):
                ctx.success_criteria = data["success_criteria"]
            if data.get("testable_units"):
                ctx.testable_units = data["testable_units"]
            if data.get("assertion_hints"):
                # Prepend static hints, then add LLM hints
                existing = ctx.assertion_hints[:]
                ctx.assertion_hints = existing + [
                    h for h in data["assertion_hints"] if h not in existing]
            if data.get("required_packages"):
                for pkg in data["required_packages"]:
                    if pkg not in ctx.required_packages:
                        ctx.required_packages.append(pkg)
                ctx.missing_packages = [
                    p for p in ctx.required_packages
                    if p not in ctx.installed_packages]
            if data.get("test_patterns"):
                existing = ctx.test_patterns[:]
                ctx.test_patterns = existing + [
                    p for p in data["test_patterns"] if p not in existing]

        except Exception as e:
            _logger.warning("[AnalyseAgent] LLM enrichment failed: %s", e)

        return ctx


def _parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    response = response.strip()
    # Strip markdown code fences
    if response.startswith("```"):
        lines = response.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        response = "\n".join(lines)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON within the response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}

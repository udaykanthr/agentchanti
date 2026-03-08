import os
import platform
import re
import logging

from .base import Agent, uniquify_context

_logger = logging.getLogger(__name__)


def _shell_example() -> str:
    """Return an OS-appropriate file-listing command example."""
    if os.name == 'nt':
        return "  1. List all project files with `dir /s /b`"
    return "  1. List all project files with `find . -type f`"


def _os_context_for_prompt() -> str:
    """Return OS/shell context string for LLM prompts."""
    if os.name == 'nt':
        return (
            f"HOST OS: Windows ({platform.version()})\n"
            "SHELL: cmd.exe (NOT bash, NOT PowerShell)\n"
            "All shell commands in backticks MUST use Windows cmd.exe syntax.\n"
            "  - Use `mkdir` instead of `mkdir -p`\n"
            "  - Use `call venv\\Scripts\\activate` instead of `source venv/bin/activate`\n"
            "  - Use `set VAR=value` instead of `export VAR=value`\n"
            "  - Use `del` instead of `rm`, `copy` instead of `cp`\n"
            "  - Use `dir /s /b` instead of `find . -type f` or `ls`\n"
            "  - Use `type <file>` instead of `cat <file>`\n"
            "  - Do NOT use PowerShell cmdlets (Get-ChildItem, etc.)\n"
        )
    sysname = platform.system()
    os_label = "macOS" if sysname == "Darwin" else sysname
    shell = os.environ.get('SHELL', '/bin/bash').rsplit('/', 1)[-1]
    return (
        f"HOST OS: {os_label} ({platform.release()})\n"
        f"SHELL: {shell}\n"
        "All shell commands in backticks should use standard Unix shell syntax.\n"
    )


# ── Task intent classification (regex-based, no LLM) ────────────

_INTENT_PATTERNS: dict[str, list[re.Pattern]] = {
    "bug_fix": [
        re.compile(r'\b(fix|bug|error|crash|broken|fail|issue|wrong|incorrect|not working)\b', re.I),
        re.compile(r'\b(debug|traceback|exception|stack\s*trace|segfault)\b', re.I),
    ],
    "refactor": [
        re.compile(r'\b(refactor|restructure|reorganize|clean\s*up|simplify|optimize|improve)\b', re.I),
        re.compile(r'\b(rename|extract|move|split|merge|consolidate)\b', re.I),
    ],
    "test": [
        re.compile(r'\b(test|spec|coverage|unittest|pytest|jest|vitest)\b', re.I),
        re.compile(r'\b(write\s+\w*\s*tests?|add\s+\w*\s*tests?|create\s+\w*\s*tests?|unit\s+tests?|integration\s+tests?)\b', re.I),
    ],
    "feature": [
        re.compile(r'\b(add|create|implement|build|develop|new|introduce)\b', re.I),
        re.compile(r'\b(feature|endpoint|page|component|module|api|route)\b', re.I),
    ],
}


def _classify_task_intent(task: str) -> str:
    """Classify task intent without LLM. Returns one of:
    bug_fix, refactor, test, feature, general.
    """
    scores: dict[str, int] = {k: 0 for k in _INTENT_PATTERNS}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if pat.search(task):
                scores[intent] += 1

    if not any(scores.values()):
        return "general"

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _build_file_skeleton(content: str, max_lines: int = 30) -> str:
    """Extract a compact skeleton from file content: imports + signatures."""
    lines = content.splitlines()
    skeleton: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Keep imports
        if stripped.startswith(("import ", "from ", "require(", "const ", "let ", "var ")):
            if "import" in stripped or "require" in stripped:
                skeleton.append(line)
                continue
        # Keep class/function/method definitions
        if re.match(r'^\s*(def |class |async def |function |export |const \w+ = |async function )', line):
            skeleton.append(line)
            continue
        # Keep decorators
        if stripped.startswith("@"):
            skeleton.append(line)
            continue

    if len(skeleton) > max_lines:
        skeleton = skeleton[:max_lines]

    return "\n".join(skeleton) if skeleton else content[:500]


def _find_relevant_files(task: str, source_files: dict[str, str] | None,
                         kb_context_builder=None, max_files: int = 5
                         ) -> list[tuple[str, str, str]]:
    """Find files relevant to the task.

    Returns list of (path, reason, skeleton) tuples.
    """
    results: list[tuple[str, str, str]] = []

    # Strategy 1: KB semantic search (best quality)
    if kb_context_builder is not None:
        try:
            kb_results = kb_context_builder.get_relevant_files(
                task_description=task, changed_files=[], max_files=max_files)
            if kb_results and source_files:
                for fpath in kb_results[:max_files]:
                    if fpath in source_files:
                        skeleton = _build_file_skeleton(source_files[fpath])
                        results.append((fpath, "KB semantic match", skeleton))
            if results:
                return results
        except Exception as e:
            _logger.debug(f"[PreAnalysis] KB search failed: {e}")

    # Strategy 2: Keyword matching against file paths and content
    if source_files:
        # Extract meaningful keywords from task (skip common words)
        stop_words = {
            "the", "a", "an", "to", "in", "for", "of", "and", "or", "is",
            "it", "on", "at", "by", "with", "from", "as", "be", "this",
            "that", "all", "are", "was", "were", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "can", "not", "but", "if", "then", "so",
            "add", "create", "update", "fix", "implement", "make", "use",
            "new", "file", "code", "project",
        }
        words = set(re.findall(r'\b[a-zA-Z_]\w{2,}\b', task.lower())) - stop_words

        if words:
            scored: list[tuple[str, int]] = []
            for fpath, content in source_files.items():
                score = 0
                fpath_lower = fpath.lower()
                content_lower = content.lower()[:2000]  # limit scan

                for word in words:
                    if word in fpath_lower:
                        score += 3  # filename match is strong signal
                    if word in content_lower:
                        score += 1

                if score > 0:
                    scored.append((fpath, score))

            scored.sort(key=lambda x: -x[1])
            for fpath, score in scored[:max_files]:
                skeleton = _build_file_skeleton(source_files[fpath])
                results.append((fpath, f"keyword match (score={score})", skeleton))

    return results


class PlannerAgent(Agent):

    def _build_prompt(self, task: str, context: str, language: str | None = None) -> str:
        """Override base to skip the platform line — the planner's hard-coded
        template already includes a detailed HOST ENVIRONMENT section via
        _os_context_for_prompt(), so the base class one-liner is redundant."""
        prompt = f"Role: {self.role}\nGoal: {self.goal}\n\n"
        if language:
            from ..language import get_language_name
            prompt += f"Language: {get_language_name(language)}\n\n"
        # NOTE: platform line intentionally omitted — provided later
        if self.prompt_suffix:
            prompt += f"Instructions: {self.prompt_suffix}\n\n"
        if context:
            context = uniquify_context(context)
            prompt += f"Context: {context}\n\n"
        prompt += f"Task: {task}\n\nResponse:"
        return prompt

    def pre_analyze(self, task: str, *,
                    source_files: dict[str, str] | None = None,
                    kb_context_builder=None,
                    knowledge_base=None) -> str:
        """Analyze the task and project to build enriched planner context.

        Runs BEFORE process(). Returns a context string to prepend to
        the existing planner context. Returns empty string if nothing useful.
        """
        parts: list[str] = []

        # 1. Task intent classification
        intent = _classify_task_intent(task)
        if intent != "general":
            intent_hints = {
                "bug_fix": "This is a BUG FIX task. Focus on identifying the root cause and fixing it. Avoid unnecessary refactoring.",
                "refactor": "This is a REFACTORING task. Focus on restructuring without changing behavior. Preserve all existing tests.",
                "test": "This is a TESTING task. Focus on writing comprehensive tests for existing code.",
                "feature": "This is a NEW FEATURE task. Plan for implementation, integration with existing code, and proper error handling.",
            }
            parts.append(f"[Task Analysis] {intent_hints.get(intent, '')}")

        # 2. Find and annotate relevant files
        relevant = _find_relevant_files(
            task, source_files, kb_context_builder, max_files=5)

        if relevant:
            parts.append("\n[Relevant Files Analysis]")
            parts.append("The following existing files are most relevant to this task:")
            for fpath, reason, skeleton in relevant:
                parts.append(f"\n--- {fpath} ({reason}) ---")
                parts.append(skeleton)
                if intent == "bug_fix":
                    parts.append(f"  ^ Check this file for the bug described in the task")
                elif intent == "feature":
                    parts.append(f"  ^ This file may need modification for the new feature")

        # 3. Knowledge base context — SKIPPED here to avoid duplication.
        # knowledge_base.format_for_planner() is already injected by api.py
        # into the planner context. Adding it again here would duplicate
        # installed packages, tech stack, and patterns.

        # 4. Global KB documentation (framework guides, installation docs)
        if kb_context_builder is not None:
            try:
                kb_context_builder._ensure_global()
                if kb_context_builder._global_store is not None:
                    docs = kb_context_builder._global_store.search(
                        query=task,
                        categories=["doc", "pattern"],
                        top_k=15,
                        api_client=kb_context_builder._api_client,
                    )
                    if docs:
                        # Filter out docs about conflicting frameworks
                        # (e.g. Angular docs for a React task)
                        from ..orchestrator.plan_optimizer import (
                            _TECH_KEYWORDS as _TK,
                            has_framework_conflict,
                        )
                        task_techs = set(
                            w.lower() for w in _TK.findall(task)
                        )
                        doc_hints: list[str] = []
                        for doc in docs:
                            if task_techs:
                                doc_text = (
                                    (doc.title or "") + " "
                                    + " ".join(doc.tags or [])
                                )
                                doc_techs = set(
                                    w.lower() for w in _TK.findall(doc_text)
                                )
                                if has_framework_conflict(task_techs, doc_techs):
                                    _logger.debug(
                                        "[PreAnalysis] Skipping '%s' — "
                                        "framework conflict",
                                        doc.title,
                                    )
                                    continue
                            content = doc.content or doc.title
                            if content:
                                doc_hints.append(f"### {doc.title}\n{content}")
                        if doc_hints:
                            parts.append("\n[Framework/Library Documentation]")
                            parts.append(
                                "IMPORTANT: Follow these guides when generating steps:")
                            parts.extend(doc_hints)
            except Exception as e:
                _logger.debug(f"[PreAnalysis] Global KB doc search failed: {e}")

        return "\n".join(parts) if parts else ""

    def process(self, task: str, context: str = "") -> str:
        prompt = self._build_prompt(task, context)
        prompt += """

You are a SENIOR SOFTWARE ARCHITECT creating an execution plan that will be
carried out by an automated pipeline. Each step is executed by one of four
agents: a CODER (writes files), a CMD runner (executes shell commands), a
TESTER (generates and runs unit tests), or a SEARCHER (searches the web for
documentation and latest info). Your plan MUST be precise enough for
these agents to succeed on the first attempt.

═══════ HOST ENVIRONMENT ═══════
""" + _os_context_for_prompt() + """
═══════ STEP FORMAT ═══════
Write a numbered list. Each step MUST be a single, concrete action:

  1. Install dependencies with `npm install express cors` (depends: none)
  2. Create the Express server in `src/server.js` with GET /api/health endpoint (depends: 1)
  3. Add input validation utility in `src/utils/validate.js` (depends: 1)
  4. Search for the latest Express.js middleware best practices (depends: none)
  5. Update `src/server.js` to use validation from `src/utils/validate.js` (depends: 2, 3, 4)

═══════ STEP RULES (CRITICAL) ═══════

1. **Reference EXACT file paths**: Every CODE step must name the specific
   file(s) to create or modify. Use the paths from the project context above.
   Say "Update `src/index.js`" NOT "Update the main file".

2. **One action per step**: Each step should do ONE thing. Don't combine
   "create file AND install package" in one step. Split them.
   Exception: ALL modifications to the SAME file MUST be combined into a single CODE step.

3. **CMD steps for shell commands**: Installing packages, running scripts,
   creating directories — put the exact command in backticks.
   Examples: `npm install express`, `pip install flask`, `mkdir -p src/utils`

4. **CODE steps for file changes**: Creating or modifying source files.
   Always specify the file path. For modifications, say "Update `path/to/file`"
   and describe WHAT to change.
   CRITICAL: You MUST combine ALL changes for a single file into exactly ONE CODE step.
   DO NOT create multiple steps for the same file under any circumstances. If a file needs
   multiple changes (e.g., updating a function and adding imports), describe ALL of them in the same step's context.

5. **SEARCH steps for web lookups**: When you need up-to-date documentation,
   API references, error solutions, or latest framework CLI flags, add a
   SEARCH step. The result will be available to subsequent steps.
   Example: "Search for the latest Next.js 15 app router migration guide"

6. **Existing files = MODIFY, not recreate**: When files already exist in the
   project (shown in context above), plan to UPDATE them. Reference their
   exact paths. Do NOT plan to create a file that already exists.

7. **Dependencies between steps**: Add `(depends: N)` or `(depends: N, M)` at
   the end of steps that need prior steps completed first. Steps without
   dependencies can run in parallel.

8. **Logical ordering**: Dependencies first, then dependents:
   - Install packages → Write code that uses them
   - Create utility files → Write code that imports them
   - Write source code → Write tests for it

9. **NO meta-steps**: Do NOT include steps like "Analyze the project",
   "Review the code", "Identify the bug", or "Plan the implementation".
   Jump straight to actionable steps.

10. **ALWAYS include a test step**: Every plan MUST include at least one TEST
    step that writes and runs tests for the code created or modified in prior
    CODE steps. Place the TEST step AFTER all CODE steps it validates.
    Test execution is the primary quality gate — it catches real bugs that
    static review cannot.

11. **Shell commands are non-interactive**: Always include --yes, -y, or
    --defaults flags for tools that prompt for input:
    - `npx create-next-app . --yes`
    - `npm init -y`
    - `ng new myapp --defaults`

12. **Sub-project paths and naming**: When a step creates a new project in a
    subdirectory, you MUST derive the project folder name from the user's task
    description. Do NOT invent a creative name — use what the user asked for.
    If the task says "create a SPA" or "build a React app" without a specific
    name, use a simple descriptive slug (e.g. `spa-app` or `react-app`).
    CRITICAL: Use the EXACT SAME folder name in ALL steps. Never change it.
    - If step 1 creates `my-app/`, ALL subsequent steps use `my-app/`.
    - CORRECT: "Create dashboard page in `my-app/src/pages/Dashboard.js`"
    - WRONG: "Create dashboard page in `src/pages/Dashboard.js`"
    - CMD steps that operate on the sub-project must include `cd my-app &&`
      before the command (e.g. `cd my-app && npm install axios`)
    - If KB documentation uses placeholder names like `<project-name>`,
      replace them with the consistent project name you chose above.

13. **SKIP already-installed packages**: If the project knowledge above lists
    packages as already installed, do NOT add install steps for them.
    Only install NEW packages that are not yet present.

═══════ QUALITY CHECKLIST (verify before outputting) ═══════
- [ ] Every file path in the plan matches an existing project file OR is
      clearly marked as a new file to create
- [ ] No two steps create/modify the same file (consolidate into one step)
- [ ] Import dependencies: if step N creates a module that step M imports,
      M depends on N
- [ ] Package dependencies: if step N installs a package that step M uses,
      M depends on N
- [ ] No vague steps ("improve the code", "fix issues", "update as needed")
- [ ] Each step is specific enough that a developer could execute it without
      asking questions
- [ ] Total steps are between 2-15 (break large tasks down, but don't over-split)
- [ ] No install steps for packages already listed in project knowledge
"""
        return self.llm_client.generate_response(prompt)

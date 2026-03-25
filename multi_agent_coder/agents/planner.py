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
                    knowledge_base=None,
                    test_analysis: str | None = None,
                    language: str | None = None) -> str:
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
                        top_k=10,
                        api_client=kb_context_builder._api_client,
                    )
                    _logger.info(
                        "[PreAnalysis] Global KB search returned %d docs",
                        len(docs) if docs else 0,
                    )
                    if docs:
                        # Filter docs to only include relevant ones.
                        # Four filters applied in order:
                        # 1. Framework conflict: reject docs about a
                        #    conflicting framework (e.g. Angular for React)
                        # 2. Tech mismatch: skip docs whose tech keywords
                        #    have zero overlap with the task's tech keywords
                        # 3. Title relevance: skip docs where fewer than
                        #    _MIN_TITLE_SCORE of their title words appear in
                        #    the task — catches docs that share one secondary
                        #    tech (e.g. Vitest) but are primarily about
                        #    something unrelated (e.g. Three.js, OAuth)
                        # 4. Generic cap: allow a few topic-relevant generic
                        #    docs (no tech keywords) through as a backstop
                        from ..orchestrator.plan_optimizer import (
                            _TECH_KEYWORDS as _TK,
                            has_framework_conflict,
                            normalize_tech_keywords,
                        )
                        task_techs = normalize_tech_keywords(set(
                            w.lower() for w in _TK.findall(task)
                        ))
                        # Precompute task word set (≥4 chars) for title
                        # relevance scoring (Filter 3).
                        # Include the detected project language so that
                        # language-specific KB docs (e.g. "Python Stdlib
                        # Reference") are not filtered out when the task
                        # description doesn't mention the language by name.
                        _task_words_set = set(
                            re.findall(r'[a-zA-Z]{4,}', task.lower())
                        )
                        if language and len(language) >= 4:
                            _task_words_set.add(language.lower())
                        _MIN_TITLE_SCORE = 0.20  # ≥20% of title words in task
                        doc_hints: list[str] = []
                        _preloaded: list = []  # filtered GlobalKBResult objects
                        _MAX_GENERIC_DOCS = 2  # cap for topic-relevant generics
                        _generic_count = 0
                        for doc in docs:
                            doc_text = (
                                (doc.title or "") + " "
                                + " ".join(doc.tags or [])
                                + " " + ((doc.content or "")[:500])
                            )
                            doc_techs = normalize_tech_keywords(set(
                                w.lower() for w in _TK.findall(doc_text)
                            ))

                            # Filter 1: framework conflict
                            if task_techs and has_framework_conflict(task_techs, doc_techs):
                                _logger.debug(
                                    "[PreAnalysis] Skipping '%s' — "
                                    "framework conflict",
                                    doc.title,
                                )
                                continue

                            # Filter 2: tech mismatch
                            if task_techs and doc_techs and not (task_techs & doc_techs):
                                _logger.debug(
                                    "[PreAnalysis] Skipping '%s' — "
                                    "tech mismatch (%s vs task %s)",
                                    doc.title, doc_techs, task_techs,
                                )
                                continue

                            # Filter 3: title relevance
                            # Compute what fraction of the doc's title words
                            # (≥4 chars) appear in the task text.  Docs that
                            # only match the task on one secondary tech but
                            # whose title focuses on something unrelated score
                            # below the threshold and are skipped.
                            title_words = set(
                                re.findall(r'[a-zA-Z]{4,}',
                                           (doc.title or "").lower())
                            )
                            if title_words:
                                title_score = (
                                    len(title_words & _task_words_set)
                                    / len(title_words)
                                )
                                if title_score < _MIN_TITLE_SCORE:
                                    if doc_techs:
                                        # Tech-matched but off-topic title
                                        _logger.debug(
                                            "[PreAnalysis] Skipping '%s' — "
                                            "low title relevance "
                                            "(%.0f%% overlap, needs ≥%.0f%%)",
                                            doc.title,
                                            title_score * 100,
                                            _MIN_TITLE_SCORE * 100,
                                        )
                                        continue
                                    else:
                                        # Generic doc with irrelevant title
                                        _logger.debug(
                                            "[PreAnalysis] Skipping '%s' — "
                                            "off-topic generic "
                                            "(%.0f%% title overlap)",
                                            doc.title, title_score * 100,
                                        )
                                        continue
                                elif not doc_techs:
                                    # Filter 4: relevant generic doc — cap
                                    _generic_count += 1
                                    if _generic_count > _MAX_GENERIC_DOCS:
                                        _logger.debug(
                                            "[PreAnalysis] Skipping '%s' — "
                                            "generic cap reached (%d/%d)",
                                            doc.title, _generic_count,
                                            _MAX_GENERIC_DOCS,
                                        )
                                        continue
                                    _logger.debug(
                                        "[PreAnalysis] Including generic "
                                        "doc '%s' (%d/%d)",
                                        doc.title, _generic_count,
                                        _MAX_GENERIC_DOCS,
                                    )
                            elif not doc_techs:
                                # Generic doc with no parseable title words
                                # — apply cap as fallback
                                _generic_count += 1
                                if _generic_count > _MAX_GENERIC_DOCS:
                                    _logger.debug(
                                        "[PreAnalysis] Skipping '%s' — "
                                        "generic cap reached (%d/%d)",
                                        doc.title, _generic_count,
                                        _MAX_GENERIC_DOCS,
                                    )
                                    continue
                                _logger.debug(
                                    "[PreAnalysis] Including generic "
                                    "doc '%s' (%d/%d)",
                                    doc.title, _generic_count,
                                    _MAX_GENERIC_DOCS,
                                )

                            content = doc.content or doc.title
                            if content:
                                doc_hints.append(f"### {doc.title}\n{content}")
                                _preloaded.append(doc)
                                _logger.info(
                                    "[PreAnalysis] Loaded doc: '%s'",
                                    doc.title,
                                )
                        # Persist filtered docs so build_context merges
                        # them into every step's KB context automatically.
                        if _preloaded and kb_context_builder is not None:
                            kb_context_builder._preloaded_docs = _preloaded
                        if doc_hints:
                            parts.append("\n[Framework/Library Documentation]")
                            parts.append(
                                "CRITICAL: These KB docs are curated and up-to-date. "
                                "You MUST follow them exactly:\n"
                                "- Use the EXACT install commands from these docs "
                                "(including all peer dependencies like postcss, jsdom, etc.)\n"
                                "- If a doc says a command is deprecated/removed, "
                                "do NOT use that command\n"
                                "- If a doc shows specific packages to install together, "
                                "install ALL of them in one step — do not omit any\n"
                                "- Your training data may be outdated — these docs override it")
                            parts.extend(doc_hints)
                        _logger.info(
                            "[PreAnalysis] %d doc(s) injected into planner context",
                            len(doc_hints),
                        )
            except Exception as e:
                _logger.debug(f"[PreAnalysis] Global KB doc search failed: {e}")

        # 5. Baseline test analysis results
        if test_analysis:
            parts.append(f"\n[Baseline Test Analysis]\n{test_analysis}")
            parts.append(
                "\nIMPORTANT: The analysis above was produced by ACTUALLY RUNNING "
                "the test suite. It is authoritative. Your plan MUST respect it:\n"
                "- If files are listed as HEALTHY/PASSING, do NOT include steps that modify them.\n"
                "- If only specific files are BROKEN, plan ONLY code changes to those files.\n"
                "- If the error is a test assertion failure (not a setup/config error), "
                "fix the test code — do NOT recreate config or setup files."
            )

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
═══════ OUTPUT FORMAT ═══════
Output your plan using this EXACT line-based format. Each step starts with
a --STEP header line followed by metadata lines and a description.

==PLAN==

--STEP 1.1 [CMD] depends:none
Install Express and CORS
> npm install express cors
produces: package.json

--STEP 2.1 [CODE] depends:1.1
Create the Express server with GET /api/health endpoint
target: src/server.js
exports: app, startServer
imports: none
content:
```js
const express = require('express');
const app = express();
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));
function startServer(port = 3000) { return app.listen(port); }
module.exports = { app, startServer };
```
---file-content-end---

--STEP 2.2 [CODE] depends:1.1
Create input validation utility
target: src/utils/validate.js
exports: validateInput, sanitize
imports: none
content:
```js
function validateInput(input) { return input != null && input !== ''; }
function sanitize(input) { return String(input).trim().replace(/[<>]/g, ''); }
module.exports = { validateInput, sanitize };
```
---file-content-end---

--STEP 3.1 [CODE] depends:2.1, 2.2
Update server to use validation middleware
target: src/server.js
imports: src/utils/validate.js:validateInput, src/utils/validate.js:sanitize
content:
```js
const express = require('express');
const { validateInput, sanitize } = require('./utils/validate');
const app = express();
app.use(express.json());
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));
app.post('/api/data', (req, res) => {
  if (!validateInput(req.body.value)) return res.status(400).json({ error: 'invalid' });
  res.json({ value: sanitize(req.body.value) });
});
function startServer(port = 3000) { return app.listen(port); }
module.exports = { app, startServer };
```
---file-content-end---

--STEP 4.1 [TEST] depends:3.1
Write and run tests for the server and validation
target: src/__tests__/server.test.js
imports: src/server.js:app, src/utils/validate.js:validateInput

==END==

═══════ LINE REFERENCE ═══════
--STEP <id> [<TYPE>] depends:<deps>   ← REQUIRED. id=wave.seq, TYPE=CMD|CODE|TEST, deps=comma-separated step ids or "none"
<description text>                     ← REQUIRED. One-line description of what this step does.
> <shell command>                      ← CMD steps only. The exact command to run.
target: <file1>, <file2>               ← CODE/TEST steps. Files to create or modify.
exports: <Symbol1>, <Symbol2>          ← CODE steps. Symbols this file will export.
imports: <file>:<Symbol>, ...          ← CODE/TEST steps. File:Symbol pairs this step needs. "none" if no imports.
produces: <file1>, <file2>             ← CMD steps. Files created by the command.
content:                               ← CODE/TEST steps. ALWAYS include complete file source here.
```<lang>                              ←   Fenced code block immediately after content:
<complete file source>                 ←   Full file — not a snippet. Every line.
```                                    ←   Close the fence.
---file-content-end---                 ←   REQUIRED closing marker after every content: block.

═══════ STEP ID FORMAT ═══════
Use wave.sequence numbering:
  - Wave 1 (no deps): 1.1, 1.2, 1.3
  - Wave 2 (depends on wave 1): 2.1, 2.2
  - Wave 3 (depends on wave 2): 3.1, 3.2, 3.3
Steps in the same wave can run in parallel. Each wave runs after the previous.

═══════ STEP RULES (CRITICAL) ═══════

1. **Reference EXACT file paths**: Every CODE step must name the specific
   file(s) in the target: line. Use the paths from the project context above.
   Say "target: src/index.js" NOT "target: the main file".

2. **One action per step**: Each step should do ONE thing. Don't combine
   "create file AND install package" in one step. Split them.
   Exception: ALL modifications to the SAME file MUST be combined into a single CODE step.

3. **CMD steps for shell commands**: Installing packages, running scripts,
   creating directories. Put the exact command on a line starting with "> ".

4. **CODE steps for file changes**: Creating or modifying source files.
   Always specify target: with the file path.
   CRITICAL: You MUST combine ALL changes for a single file into exactly ONE CODE step.

5. **SEARCH steps for web lookups**: When you need up-to-date documentation.
   Use [SEARCH] type. No target: line needed.

6. **Existing files = MODIFY, not recreate**: When files already exist in the
   project, plan to UPDATE them. Reference their exact paths.

7. **Declare imports explicitly**: For every CODE step, list all file:symbol
   pairs it will import on the imports: line. This ensures correct dependency
   tracking and context injection.

8. **Declare exports explicitly**: For every CODE step, list the main symbols
   (functions, classes, constants) the file will export on the exports: line.

9. **NO meta-steps**: Do NOT include steps like "Analyze the project",
   "Review the code". Jump straight to actionable steps.

10. **TEST steps only when explicitly requested**: Do NOT include TEST steps
    unless the user's task explicitly asks for tests. Tests consume significant
    tokens and time. When tests ARE requested, place them AFTER all CODE steps.
    CRITICAL: If the [Baseline Test Analysis] shows tests are PASSING,
    do NOT include any steps for "Fixing tests" or "Ensuring test setup".
    NEVER include steps to modify or "fix" test files that are explicitly
    marked as PASSING or Healthy in the analysis.
    Only include global setup/config modification steps if the analysis
    indicates a global failure (e.g., all tests fail or execution fails to start).

11. **Shell commands are non-interactive**: Always include --yes, -y, or
    --defaults flags for tools that prompt for input.

12. **Sub-project paths**: Use the EXACT SAME folder name in ALL steps.
    CMD steps must include `cd <name> &&` before the command.

13. **SKIP already-installed packages**: If the project knowledge lists
    packages as already installed, do NOT add install steps for them.

14. **KB documentation overrides your training data**: Use exact commands
    from KB docs — including ALL packages and peer dependencies.

15. **Configuration BEFORE code**: All setup/config steps MUST come before
    code that relies on them (test framework, CSS config, build tools).

16. **Scaffold blank projects FIRST**: When project is BLANK/EMPTY, scaffold
    first using the appropriate tool for the detected language
    (Python: `python -m venv venv && pip install ...`;
     JavaScript/TypeScript: `npm create vite@latest`, `npm install`;
     Go: `go mod init`; Rust: `cargo init`)
    before any code steps. Use the language-specific examples from the
    PROJECT STATE context above — do NOT default to npm for non-JS projects.

17. **Leaf components BEFORE parents**: Create child components first, then
    parents that import them. Declare imports: to enforce correct ordering.

18. **ALWAYS include inline code for CODE and TEST steps**: Every CODE and
    TEST step MUST include a `content:` block with the complete file source.
    This is MANDATORY — it eliminates a separate Coder LLM call per step.
    Use the exact format:
      content:
      ```<lang>
      <full file content>
      ```
      ---file-content-end---
    The code must be COMPLETE — every import, every function, every line.
    Do NOT write a stub or partial implementation.

═══════ QUALITY CHECKLIST ═══════
- [ ] Every CODE step has a target: line with exact file paths
- [ ] Every CODE step has exports: and imports: lines
- [ ] Every CODE and TEST step has a content: block with complete file source
- [ ] No two steps have the same target: file (consolidate into one step)
- [ ] If step B imports from step A's target file, B depends on A
- [ ] No vague steps — each step is specific and actionable
- [ ] Total steps between 2-15
- [ ] No install steps for already-installed packages
- [ ] Config/tooling steps come BEFORE code that depends on them
- [ ] Leaf components created BEFORE parent components
"""
        return self.llm_client.generate_response(prompt)

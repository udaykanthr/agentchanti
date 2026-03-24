from .base import Agent
from ..language import get_language_name


class ReviewerAgent(Agent):
    def process(self, task: str, context: str = "", language: str | None = None,
                review_mode: str = "full") -> str:
        prompt = self._build_prompt(task, context, language=language)
        lang_name = get_language_name(language) if language else "the project's language"

        if review_mode == "diff":
            prompt += self._diff_review_prompt(lang_name)
        else:
            prompt += self._full_review_prompt(lang_name, context)

        return self.llm_client.generate_response(prompt)

    @staticmethod
    def _diff_review_prompt(lang_name: str) -> str:
        """Compact review prompt for diff/chunk-based reviews."""
        return f"""

You are reviewing a CODE DIFF for correctness. You see only the changed lines
and their surrounding context. Language: {lang_name}

──────── UNIFIED DIFF FORMAT ────────
Lines starting with `-` were REMOVED from the old file.
Lines starting with `+` were ADDED to the new file.
Lines with no prefix are unchanged context lines.
CRITICAL: If you see `-foo` and `+foo` with identical content, it means only
whitespace/newline changed — there is NOT a duplicate. Do NOT flag a
duplicate unless you see two `+` lines with the same content.

──────── FOCUS ON ────────
1. Will the changes work correctly at runtime?
2. Are imports/references still valid after these changes?
3. Any type mismatches, missing error handling, or undefined variables
   in the CHANGED code?
4. Do the changes match the stated task?
5. If IMPORTER context is provided: does this change duplicate something
   already present in the importer (e.g. Router, Provider, HOC wrapper)?

──────── DO NOT FLAG ────────
- Code style, naming, or formatting
- Unchanged code visible in context
- Performance suggestions (unless it causes a bug)
- Missing features not related to the task
- Removed content when the step explicitly specifies the replacement
  (e.g. step says "override with X" or shows exact target content —
  the removal is intentional; only check that the replacement is correct)
- A `-line` / `+line` pair with identical content — this is a
  whitespace/newline-only change, NOT a duplicate statement

──────── STEP INTENT RULE (CRITICAL) ────────
If the step description explicitly states what the file should contain
(e.g. shows exact code, says "replace with", "override", "set to"),
then judge the change against that stated intent — NOT against what
was removed. Removing previous content to produce the exact output
the step requested is CORRECT. Do NOT fail for intentional replacement.

──────── IMPORTER CONTEXT (if provided) ────────
If [IMPORTER — read-only context] sections are present, use them to
detect duplicates introduced by this change — e.g. wrapping a component
in <BrowserRouter> when the importer already wraps it. Flag these as
FAIL since they cause runtime errors. Do NOT flag anything else in the
importer; it is read-only context, not under review.

──────── OUTPUT FORMAT ────────
If changes are correct: Respond with EXACTLY: "Code looks good."
If any issue: **FAIL**: <file> — <description>
"""

    @staticmethod
    def _full_review_prompt(lang_name: str, context: str) -> str:
        """Full review prompt (existing behavior)."""
        import re
        known_files = re.findall(r"####\s*\[FILE\]:\s*(\S+)", context)
        file_listing = "\n".join(f"  - {f}" for f in known_files) if known_files else "(no file listing available)"

        return f"""

You are a STRICT code reviewer. Your job is to catch issues that WILL cause
runtime failures, test failures, or import errors. Language: {lang_name}

KNOWN PROJECT FILES:
{file_listing}

──────── MANDATORY CHECKS (you MUST verify each one) ────────

1. **IMPORT PATHS** (most common failure cause):
   - Every `import` / `require()` / `from ... import` MUST resolve to a real
     file listed above.
   - For Python: `from src.calculator import add` requires `src/calculator.py`
     to exist. Flag if the module path doesn't match any known file.
   - For JS/TS: `require('../src/app')` must match a real relative path.
     Flag if file doesn't exist in the listing.
   - Flag ANY stripped or shortened import prefixes
     (e.g. `from calculator import` when file is at `src/calculator.py`).

2. **EXPORT / FUNCTION EXISTENCE**:
   - Every imported name (`add`, `Calculator`, etc.) must actually be
     defined or exported in the source file. Cross-check against the source
     code in context.
   - For JS: if source uses `module.exports = {{ ... }}`, test must `require()`
     the correct keys.
   - Flag importing a name that doesn't exist in the source.

3. **LANGUAGE-SPECIFIC ISSUES**:
   - Python: missing `__init__.py`? Wrong relative vs absolute import?
   - JS/TS: `require()` vs `import` mismatch with module type?
     (CommonJS vs ES Modules)
   - Mismatched test framework calls (e.g. using `jest.fn()` in a mocha project).

4. **RUNTIME ERRORS**:
   - Undefined variables, wrong argument counts, type mismatches.
   - Calling methods that don't exist on the object.
   - Missing `async/await` on Promise-returning functions.

5. **TEST CORRECTNESS**:
   - Tests that assert wrong expected values (e.g. `expect(add(2,3)).toBe(6)`).
   - Tests that test nothing (empty test bodies, no assertions).
   - Tests that will always pass regardless of implementation.

──────── SCOPE RESTRICTION (CRITICAL) ────────
You MUST restrict your review ONLY to the code changes relevant to the
current step.  Do NOT review, comment on, or suggest changes to code that
was NOT modified by this step.  If unrelated code has pre-existing issues,
IGNORE them — they are out of scope.

──────── DO NOT FLAG ────────
- Code style, naming conventions, or missing docstrings
- Minor refactoring opportunities
- Performance suggestions (unless it causes a bug)
- Missing edge-case tests (focus on whether EXISTING tests are correct)
- Pre-existing issues in code NOT touched by this step
- Formatting, whitespace, or comment style in unmodified code
- Removed content when the step explicitly specifies the replacement

──────── STEP INTENT RULE (CRITICAL) ────────
If the step description explicitly states what the file should contain
(e.g. shows exact code, says "replace with", "override", "set to"),
then judge the change against that stated intent — NOT against what
was removed. Removing previous content to produce the exact output
the step requested is CORRECT. Do NOT fail for intentional replacement.

──────── OUTPUT FORMAT ────────
If ALL mandatory checks pass and the code will work correctly:
  Respond with EXACTLY: "Code looks good."

If ANY check fails, list EACH issue as:
  **FAIL [check_number]**: <file> — <description of the problem>

Be SPECIFIC. Say which import is wrong and what it should be.
Do NOT say "Code looks good" if you found any issue above.
"""

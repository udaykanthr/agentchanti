"""
Step classification and command extraction utilities.
"""

import re

from ..cli_display import CLIDisplay, token_tracker


# ── Deterministic classification: test runner commands ──────────
# If the step text mentions a known test runner, skip the LLM entirely
# and classify as TEST.  This is faster and more reliable than relying
# on prompt hints.
_TEST_CMD_RE = re.compile(
    # Match test runner names only when they are standalone commands,
    # NOT when part of a hyphenated package name (e.g. jest-dom,
    # jest-environment-jsdom, vitest-dev).  Uses negative lookbehind
    # and lookahead for hyphens.
    r'(?<![-/])(?:'
    r'pytest|jest|vitest|mocha|rspec|phpunit'
    r'|npm\s+test|yarn\s+test|pnpm\s+test'
    r'|go\s+test|cargo\s+test'
    r'|python\s+-m\s+pytest'
    r'|npx\s+vitest|npx\s+jest'
    r'|run\s+(?:the\s+)?test(?:s|\s+suite)\b'  # "run tests", "run the test suite"
    r'|execute\s+(?:the\s+)?tests?\b'           # "execute tests", "execute the tests"
    r'|validate\s+(?:the\s+)?(?:test|component)'# "validate the components and test setup"
    r')(?![-\w]*[-])',          # reject jest-dom, vitest-dev, etc.
    re.IGNORECASE,
)

# Steps that mention test runners but are NOT test execution steps.
# These are configuration/setup steps (adding scripts, config, etc.)
# and should be classified by the LLM, not fast-tracked as TEST.
_TEST_CONFIG_RE = re.compile(
    r'('
    # Script/package.json edits
    r'add\s+.*scripts?\s+to\b'
    r'|update\s+.*scripts?\s+in\b'
    r'|modify\s+.*package\.json'
    r'|add\s+.*to\s+.*package\.json'
    r'|edit\s+.*package\.json'
    # Setup / config language
    r'|configure\s+.*test'
    r'|(?:setup|set\s+up)\s+.*test'
    r'|install\s+.*test'
    r'|ensure\s+.*(?:configured|enabled|set\s*up|installed)'
    r'|import\s+.*(?:jest-dom|@testing-library)'
    # File creation/update targeting config or source files (not test execution)
    r'|create\s+.*(?:config|file|\.js|\.ts|\.jsx|\.tsx)'
    r'|(?:update|modify|edit|change)\s+.*\.config\.'
    r'|(?:update|modify|edit|change)\s+.*(?:vite|vitest|jest|webpack|babel|tsconfig|eslint)\.'
    r')',
    re.IGNORECASE,
)


def _classify_step(step_text: str, llm_client, display: CLIDisplay, step_idx: int) -> str:
    display.step_info(step_idx, "Classifying step...")

    # Fast path: deterministic classification for test commands.
    # Skip the fast path if the step looks like test configuration/setup
    # (e.g. "Add test scripts to package.json") — those should go to the LLM.
    if _TEST_CMD_RE.search(step_text) and not _TEST_CONFIG_RE.search(step_text):
        display.step_tokens(step_idx, 0, 0)
        return "TEST"

    prompt = (
        "Classify the following task step into exactly one category.\n"
        "Reply with ONLY one word: CMD, CODE, TEST, SEARCH, or IGNORE\n\n"
        "  CMD    = anything that can be done by running shell commands, including:\n"
        "           - scanning or listing files/directories (ls, tree, find, dir)\n"
        "           - reading or inspecting file contents (cat, type, head)\n"
        "           - checking project structure or dependencies\n"
        "           - installing packages (pip install, npm install)\n"
        "           - running scripts, builds, or any CLI tool\n"
        "           - navigating or exploring a codebase\n"
        "           (Do NOT use CMD for commands that run tests. Use TEST instead.)\n\n"
        "  CODE   = create or modify source code files (writing new code or editing existing files)\n\n"
        "  TEST   = write or run unit/integration tests (e.g. running pytest, jest, npm test)\n\n"
        "  SEARCH = search the web for documentation, API references, error solutions,\n"
        "           or latest framework / library information\n\n"
        "  IGNORE = not actionable by a program (e.g. open an IDE, review code visually,\n"
        "           think about architecture, make a decision)\n\n"
        f"Step: {step_text}\n\n"
        "Category:"
    )
    sent_before, recv_before = token_tracker.snapshot()
    display.step_info(step_idx, "Classifying step via LLM...")

    response = llm_client.generate_response(prompt).strip().upper()

    sent_after, recv_after = token_tracker.snapshot()
    sent_delta = sent_after - sent_before
    recv_delta = recv_after - recv_before
    display.step_tokens(step_idx, sent_delta, recv_delta)

    for keyword in ("IGNORE", "CMD", "CODE", "TEST", "SEARCH"):
        if keyword in response:
            return keyword
    return "CODE"


def _is_file_path(text: str) -> bool:
    """Return True if *text* looks like a bare file/directory path, not a command."""
    text = text.strip()
    # No spaces usually means it's a path, not a command with arguments
    # Exception: single-word commands like "pytest" are handled by _looks_like_command
    if ' ' in text:
        return False
    # Looks like a file path: contains slashes and/or has a file extension
    has_sep = '/' in text or '\\' in text
    has_ext = bool(re.search(r'\.\w{1,5}$', text))
    return has_sep or has_ext


def _looks_like_command(text: str) -> bool:
    """Return True if *text* looks like an executable shell command."""
    text = text.strip()
    if not text:
        return False
    # Reject bare file paths
    if _is_file_path(text):
        return False

    # Reject lines that look like Ruby/Gemfile DSL rather than shell commands.
    # e.g. "source 'https://rubygems.org'", "gem 'rspec'", "gem 'rails', '~> 7.0'"
    if re.match(r"^(source|gem|group|require)\s+['\"]", text):
        return False

    # Reject JavaScript export statements — "export function ...", "export default ...",
    # "export class ...", "export const/let/var ..." are code, not shell commands.
    # Shell export is "export VAR=value" or "export -n VAR".
    if re.match(r'^export\s+(function|default|class|const|let|var|async)\b', text):
        return False

    # Extract the first token, splitting on whitespace AND shell operators
    # so that "echo.>file" splits to "echo." and "type nul > file" splits to "type"
    first_token = re.split(r'[\s>|&;<]', text)[0].lower()
    # Strip trailing .exe suffix (not rstrip which eats individual chars)
    if first_token.endswith('.exe'):
        first_token = first_token[:-4]
    # Strip trailing dots (CMD echo. syntax)
    first_token = first_token.rstrip('.')

    known_commands = {
        'pip', 'pip3', 'python', 'python3', 'py',
        'npm', 'npx', 'node', 'yarn', 'pnpm',
        'go', 'cargo', 'rustc', 'mvn', 'gradle', 'javac', 'java',
        'ruby', 'bundle', 'gem', 'rspec',
        'git', 'docker', 'make', 'cmake',
        'mkdir', 'rmdir', 'del', 'copy', 'move', 'ren', 'type', 'dir',
        'ls', 'cat', 'cp', 'mv', 'rm', 'find', 'grep', 'chmod', 'chown',
        'cd', 'echo', 'set', 'export', 'source', 'touch',
        'curl', 'wget', 'ssh', 'scp',
        'apt', 'apt-get', 'brew', 'choco', 'yum', 'dnf', 'pacman',
        'powershell', 'pwsh', 'cmd',
        'pytest', 'jest', 'tox', 'mypy', 'flake8', 'black', 'ruff',
    }
    return first_token in known_commands


def _cleanup_shell_command(cmd: str) -> str:
    """Clean up dangling operators and whitespace from a shell command."""
    cmd = cmd.strip()
    # Remove dangling bash-style continuation or logic operators
    # e.g. "npm install && \" or "npm install \"
    cmd = re.sub(r'(&&|\|\||\\)\s*$', '', cmd).strip()
    # Second pass in case there were multiple (e.g. "&& \")
    cmd = re.sub(r'(&&|\|\||\\)\s*$', '', cmd).strip()
    return cmd


def _extract_commands_from_text(text: str, *, code_blocks_only: bool = False) -> list[str]:
    """Extract shell commands from *text*, handling both triple- and single-backtick blocks.

    Prefers triple-backtick code blocks (```cmd, ```bash, ```shell, ```)
    over single-backtick inline code.  Filters out file paths and non-commands.
    """
    commands: list[str] = []
    seen: set[str] = set()

    def process_block(block: str) -> list[str]:
        """Process a code block, merging bash line continuations."""
        block_cmds = []
        # Merge lines ending with '\'
        merged_lines = []
        current_merged = ""
        for line in block.splitlines():
            line_strip = line.rstrip()
            if line_strip.endswith("\\"):
                current_merged += line_strip[:-1].strip() + " "
            else:
                current_merged += line.strip()
                if current_merged.strip():
                    merged_lines.append(current_merged.strip())
                current_merged = ""
        if current_merged.strip():
            merged_lines.append(current_merged.strip())

        for line in merged_lines:
            if _looks_like_command(line):
                cleaned = _cleanup_shell_command(line)
                if cleaned and cleaned not in seen:
                    block_cmds.append(cleaned)
                    seen.add(cleaned)
        return block_cmds

    # 1. Triple-backtick code blocks (```lang\n...\n```)
    for m in re.finditer(r"```(?:\w*)\n(.*?)```", text, re.DOTALL):
        block = m.group(1).strip()
        # Heredoc blocks (cat << 'EOF' ... EOF) must stay as a single command;
        # splitting them line-by-line destroys the file content.
        if re.search(r'<<-?\s*[\'"]?(\w+)[\'"]?', block):
            cleaned = _cleanup_shell_command(block)
            if cleaned and cleaned not in seen:
                commands.append(cleaned)
                seen.add(cleaned)
        # Blocks with shell control flow (if/for/while/case) must also run as
        # a single unit — splitting them loses variable assignments and breaks
        # elif/else/fi/done/esac structure.
        elif re.search(
            r'^\s*(if\s|for\s|while\s|case\s|until\s|\bfi\b|\bdone\b|\besac\b)',
            block,
            re.MULTILINE,
        ):
            cleaned = _cleanup_shell_command(block)
            if cleaned and cleaned not in seen:
                commands.append(cleaned)
                seen.add(cleaned)
        else:
            commands.extend(process_block(block))

    # 2. Single-backtick inline commands (`...`) — skip when code_blocks_only
    if not code_blocks_only:
        for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", text):
            cmd = m.group(1).strip()
            if cmd and _looks_like_command(cmd):
                cleaned = _cleanup_shell_command(cmd)
                if cleaned and cleaned not in seen:
                    commands.append(cleaned)
                    seen.add(cleaned)

    return commands


def _extract_command_from_step(step_text: str) -> str | None:
    """Extract an inline command from a step description.

    Only matches backtick content that looks like a real command,
    skipping bare file paths like ``tests/test_main.py``.
    """
    for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", step_text):
        candidate = m.group(1).strip()
        if _looks_like_command(candidate):
            return candidate
    return None

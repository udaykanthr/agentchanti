import ast
import os
import re
import shutil
import subprocess
from typing import Dict, List, Tuple
from .cli_display import log

# Substituted for a failing command's empty output. A silent failure says
# nothing the caller can act on, so consumers key off this to react to
# "undiagnosable" rather than re-reading an error that is not there.
NO_OUTPUT_MARKER = "but produced no output"

# Appended when a command DIED — twice — instead of reporting a verdict.
# "Failed" and "crashed" are different facts: the first is evidence about
# the code, the second is evidence about nothing. Consumers that would
# otherwise record a regression key off this to stay undecided.
CRASHED_MARKER = "[process terminated abnormally — no verdict]"


def _crash_helpers():
    """Lazily fetch the abnormal-exit helpers from ``wave_snapshots``.

    That module is stdlib-only, but *reaching* it executes
    ``orchestrator/__init__``, which imports this one — so a module-level
    import is circular. Deferring to call time (the same thing ``pipeline``
    does at its own crash guards) breaks the cycle. If the import fails for
    any reason the predicate degrades to "never abnormal", which preserves
    today's behaviour rather than silently swallowing real failures.
    """
    try:
        from .orchestrator.wave_snapshots import (  # noqa: PLC0415
            describe_abnormal_exit, is_abnormal_exit, log_crash_diagnostics)
        return is_abnormal_exit, describe_abnormal_exit, log_crash_diagnostics
    except Exception:                                  # pragma: no cover
        return (lambda _code: False), (lambda _code: ""), (lambda *_a: None)


class Executor:
    def __init__(self):
        self._background_processes: List[subprocess.Popen] = []
        # Exit status of the most recent run_command. The (success, output)
        # return loses it, and the code is the only way to tell a genuine
        # test failure from a process that died abnormally — a native
        # fast-fail prints ordinary-looking output and then exits
        # 0xC0000409, which read as "the tests regressed".
        self.last_exit_code: int | None = None

    # Phrases that indicate the model is producing generic filler instead
    # of an actual plan.  Matched case-insensitively against each step.
    _VAGUE_STEP_PATTERNS = [
        r"^implement\b.*\b(core|main|basic|provided|the)\b.*\b(functionality|solution|feature|logic)\b",
        r"^(begin|start)\b.*\b(simple|basic|clear)\b.*\b(abstraction|understanding|overview)\b",
        r"^(review|analyze|understand|study)\b.*\b(problem|statement|requirements?|codebase)\b",
        r"^(set up|setup|configure)\b.*\b(environment|workspace|tooling)\b",
        r"^(ensure|verify|validate)\b.*\b(everything|all|code)\b.*\b(works?|correct|proper)\b",
        r"^(finalize|complete|finish)\b.*\b(implementation|solution|project)\b",
        r"^(test|debug)\b.*\b(thoroughly|completely|everything)\b",
        r"^(deploy|deliver|submit)\b.*\b(final|completed?|finished)\b",
        r"^(read|gather|collect)\b.*\b(information|data|input)\b",
        r"^(write|create)\b.*\b(documentation|docs|readme)\b.*\b(for|about)\b",
    ]

    @classmethod
    def validate_plan_quality(cls, steps: List[str]) -> tuple[bool, str]:
        """Check if parsed plan steps are actionable or generic filler.

        Returns ``(is_valid, reason)``.  A plan is invalid when:
        - Too few steps (< 1) or too many (> 25)
        - Majority of steps match vague/generic patterns
        - Steps are extremely short (avg < 8 chars) suggesting fragments
        """
        if not steps:
            return False, "no steps parsed"
        if len(steps) > 25:
            return False, f"too many steps ({len(steps)})"

        avg_len = sum(len(s) for s in steps) / len(steps)
        if avg_len < 8:
            return False, "steps are too short / fragmented"

        vague_count = 0
        for step in steps:
            for pat in cls._VAGUE_STEP_PATTERNS:
                if re.search(pat, step, re.IGNORECASE):
                    vague_count += 1
                    break

        if len(steps) <= 3 and vague_count >= len(steps):
            return False, "all steps are generic filler"
        if vague_count > len(steps) * 0.5:
            return False, f"{vague_count}/{len(steps)} steps are vague/generic"

        return True, ""

    # Section headers that signal the end of the actual plan.
    # LLMs often append "Related Test Cases", "Notes", etc. sections
    # with their own numbered items that should NOT be parsed as steps.
    _PLAN_SECTION_BOUNDARY = re.compile(
        r'^\s*(?:'
        r'#{1,4}\s+(?:Related|Test\s+Case|Note|Example|Appendix|Reference|Detail|Explanation)'
        r'|\*{2}Test\s+Case'
        r'|---+\s*$'            # horizontal rule
        r'|===+\s*$'            # double horizontal rule
        r')',
        re.IGNORECASE,
    )

    @staticmethod
    def parse_plan_steps(plan_text: str) -> List[str]:
        """
        Splits a numbered plan into individual step strings using
        simple string manipulation.

        Stops parsing when a section boundary is detected (e.g.
        ``### Related Test Cases``) to avoid capturing LLM-generated
        pseudo-code sub-items as real plan steps.

        Input:
            1. Check python env
            2. Create calculator.py
            3. Write tests
        Output: ["Check python env", "Create calculator.py", "Write tests"]
        """
        steps = []
        plan_ended = False
        # Match lines starting with a number followed by a dot
        pattern = r"^\s*\d+\.\s*(.*)"
        for line in plan_text.splitlines():
            # Check for section boundary — stop parsing numbered items
            if len(steps) > 0 and Executor._PLAN_SECTION_BOUNDARY.match(line):
                plan_ended = True

            if plan_ended:
                continue

            match = re.match(pattern, line)
            if match:
                step_text = match.group(1).strip()
                # Append even if empty so we have an entry to append continuation lines to
                steps.append(step_text)
            elif steps:
                # Append continuation lines to the current step
                line_stripped = line.strip()
                if line_stripped:
                    steps[-1] += " " + line_stripped
                    
        # Remove any completely empty steps
        return [s for s in steps if s.strip()]

    # Generic placeholder path segments that local models hallucinate
    _PLACEHOLDER_SEGMENTS = {
        'path', 'to', 'your', 'my', 'the', 'project', 'folder',
        'directory', 'example', 'sample', 'some', 'filename',
        'yourproject', 'myproject', 'your_project', 'my_project',
    }

    @staticmethod
    def _sanitize_filename(raw: str) -> str:
        """Clean up LLM-generated filenames that may contain junk.

        Also blocks path traversal (``../``) so that LLM output can
        never write files outside the project directory.
        Returns empty string for clearly invalid/placeholder paths.
        """
        # Normalise Unicode look-alike characters before any other processing
        raw = Executor._sanitize_unicode(raw)
        # Reject anything with newlines (multi-line capture mistake)
        name = raw.split('\n')[0].strip()
        # Strip trailing parenthetical descriptions: "file.py (main file)"
        name = re.sub(r'\s*\(.*?\)\s*$', '', name)
        # Strip trailing comments: "file.py # main module"
        name = re.sub(r'\s*#.*$', '', name)
        # Remove backticks
        name = name.strip('`').strip()
        # Remove template-style brackets: [path/to]/[filename].[ext]
        name = re.sub(r'\[([^\]]*)\]', r'\1', name)
        # Normalize backslashes to forward slashes
        name = name.replace('\\', '/')
        # Remove leading ./ if present
        name = re.sub(r'^\./', '', name)
        # Block path traversal: remove all ".." segments
        parts = [p for p in name.split('/') if p and p != '..']
        name = '/'.join(parts)
        # Remove leading slashes (absolute paths → relative)
        name = name.lstrip('/')
        name = name.strip()

        # Strip duplicate directory prefixes: my-app/my-app/src → my-app/src
        if '/' in name:
            segments = name.split('/')
            if len(segments) >= 2 and segments[0] == segments[1]:
                name = '/'.join(segments[1:])

        # Reject if too long (real filenames rarely exceed 200 chars)
        if len(name) > 200:
            return ""
        # Reject if it contains spaces (almost never valid in code paths)
        if ' ' in name:
            return ""
        # Reject placeholder paths like "path/to/file.py", "your/project/app.js"
        if parts:
            dir_parts = {p.lower() for p in parts[:-1]}  # all except filename
            if dir_parts & Executor._PLACEHOLDER_SEGMENTS:
                return ""

        return name

    # Binary/media extensions that should never be written by a CODE step.
    # LLMs sometimes hallucinate image/font placeholders (e.g. SVG inside a
    # .png path) — block them so they don't corrupt the asset directory.
    _BINARY_EXTENSIONS: frozenset = frozenset({
        # Raster images
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff', '.tif',
        '.avif', '.ico', '.heic', '.heif',
        # Vector / raw — SVG is text but belongs in static assets, not code gen
        '.svg',
        # Fonts
        '.woff', '.woff2', '.ttf', '.eot', '.otf',
        # Audio / video
        '.mp3', '.mp4', '.wav', '.ogg', '.flac', '.aac',
        '.avi', '.mov', '.mkv', '.webm',
        # Archives / binaries
        '.zip', '.tar', '.gz', '.7z', '.rar',
        '.exe', '.dll', '.so', '.dylib', '.bin', '.wasm',
        # Documents
        '.pdf', '.docx', '.xlsx', '.pptx',
    })

    @staticmethod
    def _is_binary_path(file_path: str) -> bool:
        """Return True if *file_path* has a binary/media extension that a CODE
        step should never generate."""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in Executor._BINARY_EXTENSIONS

    @staticmethod
    def _looks_like_code(content: str) -> bool:
        """Return True if *content* looks like actual code rather than prose.

        Local models sometimes return natural-language paragraphs instead of
        code.  This heuristic catches the most obvious cases so we don't
        write garbage files to disk.
        """
        lines = content.strip().splitlines()
        if not lines:
            return False
        # Prose indicator: average line length > 120 chars (code is usually shorter)
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len > 120:
            return False
        # Prose indicator: majority of lines read as sentences rather than
        # code. "Starts with a capital" alone is NOT that test: an
        # idiomatic constants module is nothing but capital-initial lines
        # (`TILE_SIZE = 24`, `BLACK = (0, 0, 0)`), so the old check
        # rejected every constants.py a coder ever produced — the step
        # then failed with "No files parsed from coder response" and no
        # retry could win, because the code was correct all along.
        if len(lines) >= 3:
            prose_starts = sum(1 for l in lines
                               if Executor._looks_like_sentence(l))
            if prose_starts > len(lines) * 0.5:
                return False
        return True

    # Punctuation that is everywhere in code and essentially absent from an
    # English sentence. One occurrence is enough to settle a line as code.
    # The trailing alternative is a `Key: value` mapping (YAML, and the
    # capitalised keys a workflow file uses): a colon with content after it
    # is structure, whereas a prose colon ends its line ("Here is why:").
    _CODE_PUNCT_RE = re.compile(r"[=(){}\[\];]|->|::|^\S+:\s+\S")

    # Openers strong enough to mark a line as prose on their own.
    _PROSE_OPENERS = ('The ', 'This ', 'It ', 'Please ', 'Here ',
                      'A ', 'An ', 'I am ', 'I can ')

    @staticmethod
    def _looks_like_sentence(line: str) -> bool:
        """True when *line* reads as English prose rather than code."""
        s = line.strip()
        if not s:
            return False
        # Code punctuation outranks any capitalisation signal.
        if Executor._CODE_PUNCT_RE.search(s):
            return False
        if s.startswith(Executor._PROSE_OPENERS):
            return True
        # Bare capital-initial line with no code punctuation at all.
        return s[0].isupper() and not s.startswith(('I', 'If', 'In'))

    # A file whose own content legitimately contains ``` fences. Only these
    # get the outermost-fence rule below; for a source file a second fence in
    # the same region is far more likely to be a follow-up example than
    # nesting, so those keep the first-closer behaviour.
    _FENCE_BEARING_EXTS = frozenset({
        '.md', '.markdown', '.mdx', '.rst', '.adoc', '.txt',
    })
    _FENCE_BEARING_STEMS = frozenset({
        'readme', 'changelog', 'contributing', 'license', 'authors', 'notice',
    })

    # Fence info strings that say the same thing the extension does, for the
    # fuzzy parser's blocks where no filename is known until the block is read.
    _FENCE_BEARING_LANGS = frozenset({
        'markdown', 'md', 'mdx', 'rst', 'rest', 'restructuredtext',
        'adoc', 'asciidoc', 'text', 'txt',
    })

    _FILE_MARKER_RE = re.compile(r"####\s*\[FILE\]:\s*(.*)")
    # Anchored to the start of a line, which matters on its own: an unanchored
    # ``` search ends a block on a diff's "+```" or on a fence quoted inside a
    # sentence.
    _FENCE_LINE_RE = re.compile(r"^([`~]{3,})[ \t]*([^\r\n]*?)[ \t]*\r?$",
                                re.M)
    # A line naming a file, immediately followed by a fence line. Anchored
    # with re.M rather than "(?:^|\n)": the scan resumes at the character
    # after a consumed block, so the newline that would satisfy the
    # alternation is already behind the search position and the file
    # following a taken block would never be seen.
    _NAME_BEFORE_FENCE_RE = re.compile(
        r"^[^\n]*?(?:`([^`\n]+\.\w{1,5})`|(\b\S+\.\w{1,5}))"
        r"[ \t]*:?[ \t]*\r?\n(?=[`~]{3,})", re.M)

    @staticmethod
    def _is_fence_bearing(filename: str) -> bool:
        """True when *filename* is a format whose content contains fences."""
        stem, ext = os.path.splitext(os.path.basename(filename))
        if ext.lower() in Executor._FENCE_BEARING_EXTS:
            return True
        return not ext and stem.lower() in Executor._FENCE_BEARING_STEMS

    @staticmethod
    def _fence_bearing(filename: "str | None" = None, info: str = "") -> bool:
        """Fence-bearing by filename if known, else by the fence's info string."""
        if filename and Executor._is_fence_bearing(filename):
            return True
        return (info or "").lower() in Executor._FENCE_BEARING_LANGS

    @staticmethod
    def _fenced_block_span(text: str, search_from: int = 0,
                           filename: "str | None" = None,
                           info_hint: "str | None" = None):
        """Locate one fenced block; return ``(content, end_offset)`` or None.

        ``end_offset`` is past the closing fence's line, so a caller scanning
        left to right can resume there and never re-read a block's interior.
        That is what stops a Markdown file's own inner fences being mistaken
        for further files.
        """
        m = Executor._FENCE_LINE_RE.search(text, search_from)
        if m is None:
            return None
        fence_char, fence_len = m.group(1)[0], len(m.group(1))
        info = m.group(2).strip() if info_hint is None else info_hint
        nl = text.find("\n", m.end())
        if nl == -1:
            return None
        body_start = nl + 1

        # A closing fence is fence characters alone on their line, at least as
        # long as the opener — so an inner ``` cannot close a ```` block.
        closers = [c for c in Executor._FENCE_LINE_RE.finditer(text, body_start)
                   if c.group(1)[0] == fence_char
                   and len(c.group(1)) >= fence_len
                   and not c.group(2).strip()]
        if not closers:
            return None
        chosen = (closers[-1]
                  if fence_len == 3 and Executor._fence_bearing(filename, info)
                  else closers[0])

        body = text[body_start:chosen.start()]
        if body.endswith("\n"):
            body = body[:-1]
        if body.endswith("\r"):
            body = body[:-1]
        end = text.find("\n", chosen.end())
        return body, (len(text) if end == -1 else end + 1)

    @staticmethod
    def _iter_fenced_blocks(text: str):
        """Yield ``(info, content, start, end)`` per TOP-LEVEL fenced block.

        Blocks nested inside another block are never yielded: the outer block
        consumes them. Without this a README's inner ``` snippets were each
        read as a separate file to write — one response produced phantom
        ``requirements.txt`` and ``main.py`` entries from its own examples.
        """
        pos = 0
        while True:
            m = Executor._FENCE_LINE_RE.search(text, pos)
            if m is None:
                return
            found = Executor._fenced_block_span(text, m.start())
            if found is None:
                return
            content, end = found
            yield m.group(2).strip(), content, m.start(), end
            pos = end

    @staticmethod
    def _extract_fenced_content(region: str, filename: str) -> "str | None":
        """Content of the fenced block in *region*, honouring nested fences.

        The old one-shot regex used a non-greedy body (``(.*?)\\n```py``),
        which ends at the FIRST fence line inside the block. That is right
        for source files and catastrophic for Markdown, whose own content is
        full of fences: an 808-token README came back as 15 lines, cut off
        mid-sentence at "install the required dependency:" — every command
        the step's verify gate looked for lived inside a fence. Worse, the
        truncation is deterministic, so all three diagnosis attempts
        regenerated the same document, got the same 15 lines, logged
        "previous fix changed nothing", and the pipeline halted.

        Two rules replace it:

        * A closing fence must be at least as long as the opening one, so a
          model that correctly wraps ``` content in a ```` fence is parsed
          exactly, with no guessing at all.
        * Where both fences are three characters the nesting is genuinely
          ambiguous, and the file's own format breaks the tie — see
          ``_is_fence_bearing``.
        """
        found = Executor._fenced_block_span(region, 0, filename)
        return None if found is None else found[0]

    @staticmethod
    def parse_code_blocks(text: str) -> Dict[str, str]:
        """
        Parses Markdown code blocks with file markers.
        Expected: #### [FILE]: path/to/file.py followed by ```lang ... ```
        """
        files = {}
        markers = list(Executor._FILE_MARKER_RE.finditer(text))
        for idx, match in enumerate(markers):
            # Each marker owns the text up to the next marker, so a nested
            # fence can never swallow the file that follows it.
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            region = text[match.end():end]
            raw_filename = match.group(1).strip()
            filename = Executor._sanitize_filename(raw_filename)
            # Skip if filename still looks invalid
            if not filename:
                continue
            content = Executor._extract_fenced_content(region, filename)
            if content is None:
                continue
            if '/' not in filename and '.' not in filename:
                if filename.lower() not in {"makefile", "dockerfile", "license", "readme", "procfile", "justfile"}:
                    continue
            # Skip binary/media files — LLMs hallucinate text placeholders for images
            if Executor._is_binary_path(filename):
                log.warning(f"[Executor] Skipping '{filename}': binary/media extension not writable by CODE step")
                continue
            # Skip if content looks like prose rather than code
            if not Executor._looks_like_code(content):
                log.warning(f"[Executor] Skipping '{filename}': content looks like prose, not code")
                continue
            files[filename] = content
        return files

    @staticmethod
    def _try_add_file(files: Dict[str, str], filename: str, content: str):
        """Add file to *files* dict only if the content looks like real code."""
        if Executor._is_binary_path(filename):
            log.warning(f"[Executor] Skipping '{filename}': binary/media extension not writable by CODE step")
            return
        if not Executor._looks_like_code(content):
            log.warning(f"[Executor] Skipping '{filename}': content looks like prose, not code")
            return
        files[filename] = content

    @staticmethod
    def _is_standalone_module(content: str, lang: str = "") -> bool:
        """True when *content* could plausibly BE a whole file.

        Pattern 5 attributes an unlabelled block to a file and writes it as
        that file's entire content. That is right for a model which emits a
        complete module without naming it, and catastrophically wrong for a
        method-level chunk: a class body written as a module is
        ``unexpected indent (line 1)``.

        Observed on a chunk-edit whose splice was rejected — the same
        response was then re-parsed as full files, and Pattern 5 assigned
        three indented method bodies to three real modules, each on a
        single symbol match. All three were reverted by the syntax guard,
        so the diagnosis round produced nothing at all.

        A genuinely complete file passes both checks trivially, so this
        does not narrow what Pattern 5 accepts today — it only rejects
        fragments, which were never writable as files in the first place.
        """
        stripped = content.strip()
        if not stripped:
            return False
        # A file's first real line starts at column 0 in every language
        # this runs on. A fragment lifted from inside a class or function
        # does not.
        for raw in content.splitlines():
            if not raw.strip():
                continue
            if raw[:1] in (" ", "\t"):
                return False
            break
        if lang == "python":
            try:
                ast.parse(content)
            except SyntaxError:
                return False
        return True

    # `cat <<'EOF' > path` / `cat > path <<EOF`, closed by the delimiter on
    # its own line. Anchored at the start of the block: a heredoc appearing
    # later is content being discussed, not a wrapper around the whole file.
    _HEREDOC_WRITE_RE = re.compile(
        r'\A\s*(?:cat|tee)\b[^\n]*?<<-?\s*[\'"]?(?P<delim>[A-Za-z_]\w*)[\'"]?'
        r'[^\n]*\n(?P<body>.*?)^[ \t]*(?P=delim)[ \t]*$',
        re.DOTALL | re.MULTILINE)

    # The opener alone. A response cut at the output-token cap loses its
    # closing delimiter, and dropping just the recipe line still keeps the
    # `cat <<` out of the source; whatever remains is then judged by the
    # ordinary syntax check, which is what catches the truncation itself.
    _HEREDOC_OPENER_RE = re.compile(
        r'\A\s*(?:cat|tee)\b[^\n]*?<<-?\s*[\'"]?[A-Za-z_]\w*[\'"]?[^\n]*\n')

    # Shell targets may legitimately CONTAIN a heredoc, so never unwrap them.
    _SHELL_SUFFIXES = (".sh", ".bash", ".zsh", ".ksh", ".fish")

    @staticmethod
    def _unwrap_shell_file_write(block: str, target: str) -> str:
        """Return the heredoc body when *block* is a shell recipe writing a file.

        A model asked for code sometimes answers with the *command* that
        creates it::

            cat <<'EOF' > hello_world.py
            print("Hello World")
            EOF

        Attributed to the step's only target, those wrapper lines are written
        into the .py file verbatim. The syntax check above cannot catch it,
        because `cat <<'EOF' > hello_world.py` is VALID Python — a left-shift
        followed by a comparison — so it parses cleanly and only fails at
        import with `NameError: name 'cat' is not defined`. Observed on a
        local model: three diagnosis attempts and a halted run over a
        one-line hello world that was correct inside the heredoc all along.

        Unwrapping keeps the body the model actually meant. Shell targets are
        left alone: a .sh file may legitimately contain a heredoc.
        """
        if not block or target.lower().endswith(Executor._SHELL_SUFFIXES):
            return block
        match = Executor._HEREDOC_WRITE_RE.search(block)
        if match:
            body = match.group("body")
        else:
            opener = Executor._HEREDOC_OPENER_RE.match(block)
            if not opener:
                return block
            body = block[opener.end():]
        if not body.strip():
            return block          # empty heredoc — nothing better to offer
        log.info("[Executor] Unwrapped a shell heredoc that wrapped '%s' — "
                 "writing the file body, not the `cat <<` recipe", target)
        return body

    @staticmethod
    def parse_blocks_for_single_target(text: str, target: str) -> Dict[str, str]:
        """Attribute an unlabelled code block to the step's only target.

        Every other extractor needs the model to name the file — a
        ``#### [FILE]:`` marker, a path after the fence language, or a
        ``# path`` first line. Pattern 5 needs an existing KB symbol
        index, which a blank project does not have at step 2.

        So a model that answers with prose and a bare ``` fence produces
        nothing at all. Observed on Gemini: correct, complete code, no
        filename anywhere, "No files parsed from coder response" twice,
        two diagnosis rounds, then the pipeline halted after 12 minutes
        and 129k tokens having written nothing.

        When the step declares exactly ONE target there is nothing to
        guess: the code belongs to that file. Deterministic, no LLM call.
        The largest block wins, because explanatory snippets are short and
        the implementation is not.
        """
        if not text or not target:
            return {}
        blocks = [m.group(1) for m in
                  re.finditer(r"```(?:[a-zA-Z0-9_+\-]*)\n(.*?)```", text,
                              re.DOTALL)]
        # A response cut at the output-token cap ends mid-block, so its
        # final fence never closes and the pattern above misses it
        # entirely. Observed on Gemini: truncated at 16,384 tokens with
        # one unterminated fence, and the step produced nothing at all.
        if text.count("```") % 2 == 1:
            tail = re.split(r"```(?:[a-zA-Z0-9_+\-]*)\n", text)[-1]
            if tail.strip():
                blocks.append(tail)
        # Before sizing them up: a block may be a shell recipe wrapping the
        # real file body, and the wrapper is not part of the file.
        unwrapped = [(Executor._unwrap_shell_file_write(b, target), b)
                     for b in blocks]
        candidates = [(body, body != original)
                      for body, original in unwrapped if body.strip()]
        if not candidates:
            return {}
        best, was_unwrapped = max(candidates, key=lambda c: len(c[0]))
        # A couple of lines is a fragment being discussed, not a file — but
        # only when we are guessing. A heredoc named the file explicitly, so
        # its body is a file however short it is; applying the fragment rule
        # there would reject the very content the unwrap recovered (a
        # two-line hello world, in the run that prompted this).
        if not was_unwrapped and len(best.strip().splitlines()) < 3:
            return {}
        # Never write source that cannot parse. A truncated block is the
        # common case here, and half a module is worse than none: it
        # fails at import with a SyntaxError that reads like a code bug
        # rather than a truncated response.
        if target.endswith(".py"):
            from .py_syntax import check_python_syntax
            if check_python_syntax(best, target):
                log.warning(
                    "[Executor] Unlabelled block for '%s' does not parse "
                    "(likely a truncated response) — not writing it", target)
                return {}
        files: Dict[str, str] = {}
        Executor._try_add_file(files, target, best.rstrip("\n"))
        return files

    @staticmethod
    def parse_code_blocks_fuzzy(text: str) -> Dict[str, str]:
        """Fallback parser for LLM responses that don't follow the strict format.

        Handles common patterns:
        1. ``#### [FILE]:`` on the first line inside ANY code block (python, diff, etc.)
        2. Diff blocks with ``+`` prefixed ``#### [FILE]:`` lines
        3. Code blocks preceded by a line mentioning a file path
        4. Code blocks whose first line is a ``# filepath`` comment

        Every pattern walks TOP-LEVEL blocks only (``_iter_fenced_blocks``).
        The old ``(.*?)```` searches were both non-greedy and unanchored, so a
        block ended at the first ``` anywhere — including one inside a
        Markdown document's own body, or a diff's ``+``` `` line. Run against
        the README that halted a pipeline, this parser returned the same
        truncated 5 lines the strict parser did AND invented two extra files,
        ``requirements.txt`` and ``main.py``, out of that README's own usage
        examples.
        """
        files: Dict[str, str] = {}

        # ── Pattern 1: #### [FILE]: as first line inside any code block ──
        # The LLM sometimes wraps everything in ```python ... ``` but puts
        # the marker inside.  The content may be plain code or diff-style.
        for _info, block, _bs, _be in Executor._iter_fenced_blocks(text):
            first_line = block.split("\n", 1)[0].strip()
            fmatch = re.match(r"^(?:\+\s*)?####\s*\[FILE\]:\s*(.+)", first_line)
            if not fmatch:
                continue
            raw = fmatch.group(1).strip()
            filename = Executor._sanitize_filename(raw)
            if not filename:
                continue
            if '/' not in filename and '.' not in filename:
                if filename.lower() not in {"makefile", "dockerfile", "license", "readme", "procfile", "justfile"}:
                    continue
            rest = block.split("\n", 1)[1] if "\n" in block else ""
            # Check if the content uses diff markers (+/-/@@)
            has_diff = any(
                l.startswith(('+', '-', '@@'))
                for l in rest.splitlines()[:10] if l.strip()
            )
            if has_diff:
                content_lines = []
                for line in rest.splitlines():
                    if line.startswith('@@'):
                        continue
                    elif line.startswith('-'):
                        continue
                    elif line.startswith('+'):
                        content_lines.append(line[1:])
                    else:
                        content_lines.append(line)
                if content_lines:
                    Executor._try_add_file(files, filename, "\n".join(content_lines))
            else:
                Executor._try_add_file(files, filename, rest.rstrip("\n"))

        if files:
            return files

        # ── Pattern 2: diff blocks with +#### [FILE]: or +# filepath ──
        for _info, block, _bs, _be in Executor._iter_fenced_blocks(text):
            if _info.lower() != "diff":
                continue
            fname_match = (
                re.search(r"^\+\s*####\s*\[FILE\]:\s*(.+)", block, re.MULTILINE)
                or re.search(r"^\+\s*#\s*(\S+\.\w{1,5})\s*$", block, re.MULTILINE)
            )
            if not fname_match:
                continue
            raw = fname_match.group(1).strip()
            filename = Executor._sanitize_filename(raw)
            if not filename:
                continue
            if '/' not in filename and '.' not in filename:
                if filename.lower() not in {"makefile", "dockerfile", "license", "readme", "procfile", "justfile"}:
                    continue
            content_lines = []
            past_header = False
            for line in block.splitlines():
                if not past_header:
                    if fname_match.group(0).strip() in line:
                        past_header = True
                    continue
                if line.startswith('+'):
                    content_lines.append(line[1:])
                elif not line.startswith('-') and not line.startswith('@@'):
                    content_lines.append(line)
            if content_lines:
                Executor._try_add_file(files, filename, "\n".join(content_lines))

        if files:
            return files

        # ── Pattern 3: text before code block mentions a file path ──
        # Scanned left to right, resuming past each block that is taken. This
        # pattern names the file from the line ABOVE the fence, so it is the
        # one that turned a README's "python main.py" example into a phantom
        # main.py — every line inside the document looked like another
        # filename introducing another block. Consuming the block closes that.
        pos = 0
        while True:
            m = Executor._NAME_BEFORE_FENCE_RE.search(text, pos)
            if not m:
                break
            fence_at = m.end()
            raw = (m.group(1) or m.group(2) or "").strip()
            filename = Executor._sanitize_filename(raw)
            usable = bool(filename) and (
                '/' in filename or '.' in filename
                or filename.lower() in {"makefile", "dockerfile", "license",
                                        "readme", "procfile", "justfile"})
            if not usable:
                pos = fence_at
                continue
            found = Executor._fenced_block_span(text, fence_at, filename)
            if found is None:
                pos = fence_at
                continue
            content, block_end = found
            Executor._try_add_file(files, filename, content)
            pos = block_end

        if files:
            return files

        # ── Pattern 4: first line of code block is a # filepath comment ──
        for _info, block, _bs, _be in Executor._iter_fenced_blocks(text):
            first_line = block.split("\n", 1)[0].strip()
            fname_match = re.match(r"^#\s*(\S+\.\w{1,5})\s*$", first_line)
            if fname_match:
                filename = Executor._sanitize_filename(fname_match.group(1))
                if filename and ('/' in filename or '.' in filename):
                    rest = block.split("\n", 1)[1] if "\n" in block else ""
                    Executor._try_add_file(files, filename, rest.rstrip("\n"))

        if files:
            return files

        # ── Pattern 5: KB Index Symbol Matching ──
        # If the LLM generates a pure anonymous code block, parse it
        # and match symbols against the knowledge base manifest.
        log.info("[Executor] Falling back to KB symbol matching (Pattern 5) for anonymous block")
        
        db_path = os.path.join(os.getcwd(), ".agentchanti", "kb", "local", "index.db")
        if not os.path.exists(db_path):
             return files
        
        try:
             from .kb.local.manifest import Manifest
             from .kb.local.parser import parse_code
             m = Manifest(db_path)
             symbol_occurrences = m.get_symbol_occurrences()
             symbol_to_files = {}
             # Create dict matching symbol names to their files
             for row in symbol_occurrences:
                 name, type_, path_ = row[0], row[1], row[2]
                 if name not in symbol_to_files:
                     symbol_to_files[name] = set()
                 symbol_to_files[name].add(path_)
                 
             for lang, block, _bs, _be in Executor._iter_fenced_blocks(text):

                 # Some language normalization
                 normalized_lang = lang.lower()
                 if normalized_lang in ("js", "javascript"):
                     normalized_lang = "javascript"
                 elif normalized_lang in ("ts", "typescript"):
                     normalized_lang = "typescript"
                 elif normalized_lang in ("py", "python"):
                     normalized_lang = "python"
                 elif normalized_lang in ("c++", "cpp"):
                     normalized_lang = "cpp"
                 
                 # It's an anonymous code block.
                 if not block.strip():
                     continue
                 
                 parsed = parse_code(block.encode('utf-8'), normalized_lang)
                 
                 # Collect unique symbols from the code block (only high-confidence ones)
                 block_symbols = set()
                 for func in parsed.functions:
                     if len(func.name) > 3 or len(parsed.functions) > 1:
                         block_symbols.add(func.name)
                 for cls in parsed.classes:
                     block_symbols.add(cls.name)
                     
                 # Score each file based on overlap
                 file_scores = {}
                 for sym in block_symbols:
                     if sym in symbol_to_files:
                         for fpath in symbol_to_files[sym]:
                             file_scores[fpath] = file_scores.get(fpath, 0) + 1
                             
                 # Find file with best score
                 if file_scores:
                     best_file = max(file_scores.keys(), key=lambda k: file_scores[k])
                     best_score = file_scores[best_file]
                     
                     if best_score > 0:
                         if not Executor._is_standalone_module(
                                 block, normalized_lang):
                             log.warning(
                                 f"[Executor] Pattern 5 declined {best_file}: "
                                 f"block is a fragment, not a whole file "
                                 f"(writing it would replace the module with "
                                 f"a partial definition)")
                             continue
                         log.info(f"[Executor] Pattern 5 assigned block to {best_file} with score {best_score}")
                         Executor._try_add_file(files, best_file, block.rstrip("\n"))
                         
        except Exception as e:
            log.warning(f"[Executor] Failed KB symbol matching (Pattern 5): {e}")

        return files

    # Dependency manifests and lock files that should NEVER be overwritten
    # by LLM-generated content.  These files are managed by package managers
    # and an LLM rewrite almost always drops dependencies, corrupting the
    # project.  They can still be *created* if they don't exist yet.
    _PROTECTED_FILENAMES: set[str] = {
        'package.json', 'package-lock.json',
        'yarn.lock', 'pnpm-lock.yaml',
        'go.mod', 'go.sum',
        'Cargo.toml', 'Cargo.lock',
        'Gemfile', 'Gemfile.lock',
        'composer.json', 'composer.lock',
        'Pipfile', 'Pipfile.lock', 'poetry.lock',
        'requirements.txt',
        '.agentchanti.yaml', '.agentchanti.yml',
        # Django / framework entry points — overwriting strips imports and
        # the if __name__ == '__main__' guard, causing silent no-op execution
        'manage.py', 'wsgi.py', 'asgi.py',
    }

    # Common mojibake patterns: UTF-8 bytes misinterpreted as Latin-1/cp1252.
    # Maps the corrupted string → correct Unicode character.
    _MOJIBAKE_MAP: dict[str, str] = {
        "â\x80\x94": "—",   # em dash
        "â\x80\x93": "–",   # en dash
        "â\x80\x99": "\u2019",  # right single quote
        "â\x80\x98": "\u2018",  # left single quote
        "â\x80\x9c": "\u201c",  # left double quote
        "â\x80\x9d": "\u201d",  # right double quote
        "â\x80\xa6": "…",   # ellipsis
        "â\x80\xa2": "•",   # bullet
        "â\x80\x9e": "\u201e",  # double low-9 quote
        "â\x84\xa2": "™",   # trademark
        "â\x80\x8b": "\u200b",  # zero-width space
        "Ã©": "é",
        "Ã¨": "è",
        "Ã¼": "ü",
        "Ã¶": "ö",
        "Ã¤": "ä",
        "Ã±": "ñ",
        "Ã§": "ç",
    }

    @staticmethod
    def _repair_mojibake(content: str) -> str:
        """Detect and repair common UTF-8→Latin-1 mojibake in LLM output.

        When an LLM regenerates a file containing multi-byte UTF-8 characters
        (like em dashes, smart quotes, etc.), it sometimes outputs the individual
        bytes as if they were Latin-1 characters, producing "mojibake".

        This method first tries a general fix (re-encode as Latin-1, decode as
        UTF-8), then falls back to replacing known mojibake patterns.
        """
        # Fast path: if there are no characters > 0x7F, nothing to fix
        if content.isascii():
            return content

        # Try the general fix: encode the string as Latin-1 and decode as UTF-8.
        # This reverses the most common corruption pattern.
        try:
            fixed = content.encode("latin-1").decode("utf-8")
            # Sanity check: the fix should not introduce more non-ASCII oddities
            # If the fixed version has fewer high bytes, it's probably correct
            if fixed != content:
                log.info("[Executor] Repaired mojibake via latin-1→utf-8 re-encoding")
                return fixed
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        # Fallback: replace known mojibake patterns individually
        result = content
        repaired = False
        for bad, good in Executor._MOJIBAKE_MAP.items():
            if bad in result:
                result = result.replace(bad, good)
                repaired = True

        if repaired:
            log.info("[Executor] Repaired mojibake via pattern replacement")

        return result

    @staticmethod
    def write_files(files: Dict[str, str], base_dir: str = ".",
                    allow_protected: "set[str] | None" = None) -> List[str]:
        """
        Writes files to disk. Returns list of written file paths.

        For Python files, automatically creates ``__init__.py`` in every
        parent directory so that imports like ``from src.module import X``
        work out of the box.

        Protected manifest files (package.json, go.mod, etc.) are never
        overwritten if they already exist — LLM-generated replacements
        almost always drop dependencies and corrupt the project.
        ``allow_protected`` lists paths exempt from that guard: content
        that was produced by an exact-match edit of the file's CURRENT
        on-disk content (e.g. the plan's FIND block matched verbatim) is
        grounded, not hallucinated — and editing a manifest is sometimes
        the entire task.
        """
        _allowed_norm = {p.replace("\\", "/").lstrip("./")
                         for p in (allow_protected or ())}
        written = []
        init_dirs: set[str] = set()

        # Track basenames we've already written to detect path conflicts
        written_basenames: dict[str, str] = {}  # basename → full relative path

        for filename, content in files.items():
            # Normalise separators before touching the filesystem: planner
            # output sometimes carries doubled backslashes, which Windows
            # silently collapses but Linux treats as literal characters in
            # a single (broken) filename.
            filename = re.sub(r"[\\/]+", "/", filename)
            filepath = os.path.join(base_dir, filename)
            dirpath = os.path.dirname(filepath)

            # Repair mojibake before writing
            content = Executor._repair_mojibake(content)

            # Hard block: never write internal memory-only paths to disk.
            # These are in-memory tracking entries (_cmd_output/, etc.)
            # that should never appear in the project directory.
            _norm_filename = filename
            if _norm_filename.startswith((
                    "_cmd_output/", "_fix_output/", "_search_context/")):
                continue

            # Hard block: never write inside node_modules/.
            # LLM-generated stubs placed here shadow real installed packages,
            # introducing import errors worse than the original failure.
            if "node_modules/" in _norm_filename or _norm_filename.startswith("node_modules"):
                log.warning(
                    f"[Executor] Blocked write to node_modules/: {filename} "
                    f"— writing here shadows real packages and corrupts the project."
                )
                continue

            # Guard: never overwrite dependency manifests / lock files
            basename = os.path.basename(filename)
            if basename in Executor._PROTECTED_FILENAMES and os.path.isfile(filepath):
                if _norm_filename.lstrip("./") in _allowed_norm:
                    log.info(f"[Executor] Writing protected file {filepath} — "
                             f"content derived from exact-match edit of the "
                             f"current file")
                else:
                    log.warning(f"[Executor] Skipping protected file: {filepath} "
                                f"(already exists — overwriting could corrupt dependencies)")
                    continue

            # Warn about potential path conflicts (same basename, different dir)
            if basename in written_basenames:
                prev_path = written_basenames[basename]
                if prev_path != filename:
                    log.warning(f"[Executor] Path conflict: '{filename}' has same "
                                f"basename as already-written '{prev_path}'")
            written_basenames[basename] = filename
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            log.info(f"Written: {filepath}")
            written.append(filepath)

            # Track directories that contain .py files
            if filename.endswith(".py") and dirpath and dirpath != base_dir:
                # Walk up to base_dir creating __init__.py at each level
                d = dirpath
                while d and d != base_dir and d != os.path.dirname(d):
                    init_dirs.add(d)
                    d = os.path.dirname(d)

        # Auto-create missing __init__.py so directories are importable packages.
        # Skip directories that are Django project roots (contain manage.py) —
        # making the project root a Python package causes Django's test runner to
        # import everything as <project>.app.* instead of app.*, breaking model
        # app_label resolution and test discovery.
        # Also skip asset directories (templates/, static/, media/) — they are
        # never Python packages and adding __init__.py there breaks Django's
        # template/static file loaders and can corrupt the import system.
        _ASSET_DIR_RE = re.compile(
            r'(?:^|[/\\])(?:templates|static|media|assets|public|dist|build)'
            r'(?:[/\\]|$)',
            re.IGNORECASE,
        )
        django_roots: set[str] = set()
        for d in init_dirs:
            if os.path.isfile(os.path.join(d, "manage.py")):
                django_roots.add(d)

        for dirpath in sorted(init_dirs):
            if _ASSET_DIR_RE.search(dirpath.replace(os.sep, "/")):
                log.debug(
                    f"[Executor] Skipping __init__.py for asset dir: {dirpath}/"
                )
                continue
            if dirpath in django_roots:
                log.debug(
                    f"[Executor] Skipping __init__.py for Django project root: {dirpath}/"
                )
                continue
            init_path = os.path.join(dirpath, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write("")
                log.info(f"Auto-created: {init_path}")
                written.append(init_path)

            # Remove any same-named .py stub that would shadow the package.
            # E.g. Django's `startapp` creates `tests.py`; if the agent later
            # writes files under `tests/`, both `tests.py` and `tests/` exist in
            # the same directory and Python's import system raises:
            #   ImportError: 'tests' module incorrectly imported from '…/tests'
            # Removing the stub (it is always empty or contains a comment only)
            # resolves the conflict without any data loss.
            pkg_name = os.path.basename(dirpath)
            parent_dir = os.path.dirname(dirpath)
            shadow_stub = os.path.join(parent_dir, pkg_name + ".py")
            if os.path.isfile(shadow_stub):
                try:
                    with open(shadow_stub, encoding="utf-8") as _f:
                        stub_content = _f.read().strip()
                    # Only remove if the file is a placeholder (empty or pure comments)
                    non_comment_lines = [
                        ln for ln in stub_content.splitlines()
                        if ln.strip() and not ln.strip().startswith("#")
                    ]
                    if not non_comment_lines:
                        os.remove(shadow_stub)
                        log.info(
                            f"Removed stub {shadow_stub} — shadowed by package {dirpath}/"
                        )
                except OSError:
                    pass

        return written

    # PowerShell cmdlets that cmd.exe cannot run directly
    _PS_CMDLETS = (
        'Get-ChildItem', 'Set-Location', 'Get-Content', 'Select-Object',
        'Where-Object', 'ForEach-Object', 'New-Item', 'Remove-Item',
        'Copy-Item', 'Move-Item', 'Test-Path', 'Invoke-WebRequest',
        'Write-Output', 'Out-File', 'Set-Content', 'Get-Command',
        'Get-Process', 'Stop-Process', 'Get-Service', 'Resolve-Path',
    )

    # Unicode look-alikes that LLMs sometimes emit instead of plain ASCII.
    # Maps each offending codepoint to its ASCII replacement.
    _UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
        # Hyphens / dashes → ASCII hyphen-minus
        ("\u2011", "-"),   # NON-BREAKING HYPHEN
        ("\u2013", "-"),   # EN DASH
        ("\u2014", "-"),   # EM DASH
        ("\u2212", "-"),   # MINUS SIGN
        ("\u00ad", "-"),   # SOFT HYPHEN
        ("\ufe58", "-"),   # SMALL EM DASH
        ("\ufe63", "-"),   # SMALL HYPHEN-MINUS
        ("\uff0d", "-"),   # FULLWIDTH HYPHEN-MINUS
        # Curly / typographic quotes → straight ASCII quotes
        ("\u2018", "'"),   # LEFT SINGLE QUOTATION MARK
        ("\u2019", "'"),   # RIGHT SINGLE QUOTATION MARK
        ("\u201c", '"'),   # LEFT DOUBLE QUOTATION MARK
        ("\u201d", '"'),   # RIGHT DOUBLE QUOTATION MARK
        ("\u00b4", "'"),   # ACUTE ACCENT
        ("\u2032", "'"),   # PRIME
    ]

    @staticmethod
    def _sanitize_unicode(text: str) -> str:
        """Replace Unicode look-alike characters with plain ASCII equivalents.

        LLMs occasionally emit typographic hyphens (U+2011, U+2013, U+2014)
        or curly quotes instead of their ASCII counterparts.  On Windows this
        creates directories / filenames with invisible Unicode characters that
        differ from what the plan expected (the ``responsive‑web‑page`` vs
        ``responsive-web-page`` problem).
        """
        for unicode_char, ascii_char in Executor._UNICODE_REPLACEMENTS:
            if unicode_char in text:
                text = text.replace(unicode_char, ascii_char)
        return text

    @staticmethod
    def _needs_powershell(cmd: str) -> bool:
        """Return True if *cmd* contains PowerShell-specific cmdlets."""
        for cmdlet in Executor._PS_CMDLETS:
            if cmdlet in cmd:
                return True
        return False

    # Known interactive commands and their non-interactive flags.
    # Each entry: (regex_pattern, flags_that_mean_already_handled, flag_to_append)
    _INTERACTIVE_REWRITES: list[tuple[str, tuple[str, ...], str]] = [
        (r'\bnpx\s+create-next-app\b', ('--yes',), ' --yes'),
        (r'\bnpm\s+init\b', ('--yes', '-y'), ' --yes'),
        (r'\byarn\s+init\b', ('--yes', '-y'), ' --yes'),
        (r'\bng\s+new\b', ('--defaults',), ' --defaults'),
        (r'\bcomposer\s+create-project\b', ('--no-interaction',), ' --no-interaction'),
    ]

    @staticmethod
    def _rewrite_interactive_cmd(cmd: str) -> str:
        """Rewrite known interactive commands to add non-interactive flags.

        Acts as a safety net so that even if the LLM forgets ``--yes``,
        the command won't hang waiting for stdin.
        """
        for pattern, existing_flags, add_flag in Executor._INTERACTIVE_REWRITES:
            if not re.search(pattern, cmd):
                continue
            if any(flag in cmd for flag in existing_flags):
                break  # already has a non-interactive flag
            cmd = cmd.rstrip() + add_flag
            log.info(f"[Executor] Auto-added non-interactive flag: {add_flag.strip()}")
            break
        return cmd

    @staticmethod
    def _is_likely_interactive(cmd: str) -> bool:
        """Return True if *cmd* matches patterns of commonly interactive CLI tools."""
        patterns = (
            r'\bcreate-next-app\b', r'\bcreate-react-app\b', r'\bcreate-vue\b',
            r'\bcreate-vite\b', r'\bnpm\s+init\b', r'\byarn\s+init\b',
            r'\bng\s+new\b', r'\bexpo\s+init\b', r'\bcomposer\s+create-project\b',
        )
        return any(re.search(p, cmd) for p in patterns)

    # ── POSIX shell compatibility rewrites ──

    @staticmethod
    def _rewrite_for_posix_sh(cmd: str) -> str:
        """Rewrite bash-specific constructs so they work under /bin/sh.

        subprocess uses /bin/sh by default on POSIX systems.  ``source`` is a
        bash builtin and not available in sh; replace it with ``.`` which is
        the POSIX-standard equivalent.
        """
        # Replace `source <path>` with `. <path>` (handles leading spaces/&&)
        rewritten = re.sub(r'(?<![.\w])source\s+', '. ', cmd)
        if rewritten != cmd:
            log.info("[Executor] Rewrote 'source' → '.' for /bin/sh compatibility")
        return rewritten

    # ── Unix → Windows command translation ──

    @staticmethod
    def _rewrite_unix_cmd_for_windows(cmd: str) -> str:
        """Rewrite common Unix/bash commands to Windows cmd.exe equivalents.

        LLMs frequently generate Unix-style shell commands regardless of the
        host OS.  This translates the most common ones so they work under
        ``cmd.exe`` on Windows.  Compound commands chained with ``&&`` or
        ``||`` are split, each segment is rewritten, and reassembled.
        """
        segments = re.split(r'(\s*&&\s*|\s*\|\|\s*)', cmd)
        rewritten = False
        result = []

        for seg in segments:
            if re.match(r'\s*(?:&&|\|\|)\s*$', seg):
                result.append(seg)
                continue
            original = seg.strip()
            new_seg = Executor._rewrite_single_unix_cmd(original)
            if new_seg != original:
                rewritten = True
                result.append(new_seg)
            else:
                result.append(seg)

        if rewritten:
            final = ''.join(result)
            log.info(f"[Executor] Rewrote Unix → Windows: {cmd!r}")
            return final
        return cmd

    @staticmethod
    def _rewrite_single_unix_cmd(cmd: str) -> str:
        """Translate a single Unix command to its Windows cmd.exe equivalent."""

        # mkdir -p dir1 dir2 → mkdir "dir1" 2>nul & mkdir "dir2" 2>nul
        m = re.match(r'^mkdir\s+-p\s+(.+)', cmd)
        if m:
            dirs = m.group(1).strip().split()
            return '; '.join(f'mkdir "{d}" 2>$nul' for d in dirs if d)

        # touch file1 file2 → create empty files
        m = re.match(r'^touch\s+(.+)', cmd)
        if m:
            files = m.group(1).strip().split()
            return ' & '.join(f'copy nul "{f}" >$nul 2>&1' for f in files if f)

        # rm [-rf] target(s) → rmdir /s /q or del /f /q
        m = re.match(r'^rm\s+((?:-\S+\s+)*)(.+)', cmd)
        if m:
            flag_str, targets_str = m.group(1), m.group(2).strip()
            flags = set()
            for tok in flag_str.split():
                if tok.startswith('-'):
                    flags.update(tok[1:])
            targets = targets_str.split()
            if 'r' in flags or 'R' in flags:
                return ' & '.join(
                    f'(rmdir /s /q "{t}" 2>$nul & del /f /q "{t}" 2>$nul)'
                    for t in targets if t
                )
            elif flags:
                return ' & '.join(
                    f'del /f /q "{t}" 2>$nul' for t in targets if t
                )
            else:
                return ' & '.join(
                    f'del /q "{t}" 2>$nul' for t in targets if t
                )

        # cp -r src dst → xcopy /E /I /Y "src" "dst"
        m = re.match(r'^cp\s+-[rR]\s+(\S+)\s+(\S+)$', cmd)
        if m:
            return f'xcopy /E /I /Y "{m.group(1)}" "{m.group(2)}"'

        # mv src dst → move /Y "src" "dst"
        m = re.match(r'^mv\s+(\S+)\s+(\S+)$', cmd)
        if m:
            return f'move /Y "{m.group(1)}" "{m.group(2)}"'

        # move dir\* dst → ALSO relocate dir's subdirectories.
        #
        # Not a Unix translation but a Windows-native repair, because
        # `move dir\*` silently moves only FILES. The standard scaffold
        # hoist —
        #
        #   npm create vite@latest scaffold -- --template react
        #   move scaffold\* . && ... && rmdir scaffold
        #
        # therefore leaves `src\` and `public\` behind, the `rmdir` fails
        # with "The directory is not empty", and the run continues with
        # TWO copies of every component. Observed twice: a leftover
        # `vite-react-scaffold\` and a nested `home_page\home_page\`, both
        # of which were then indexed, so semantic search served steps a
        # stale duplicate of the file they were editing.
        #
        # The source directory is deliberately left in place (empty)
        # rather than deleted: these commands are chained, and a later
        # segment of the SAME command line routinely still refers to it
        # (`type scaffold\.gitignore >> .gitignore && rmdir scaffold`).
        #
        # Parenthesised so the two halves stay ONE command. Without the
        # group, a caller's trailing `&& rmdir scaffold` binds to the FOR
        # BODY instead of to the rewrite as a whole, so it runs only when
        # a subdirectory happened to exist — and silently does nothing,
        # with exit 0, when the directory was flat.
        m = re.match(r'^move\s+((?:/[a-zA-Z]\s+)*)([^\s"]+)[\\/]\*\s+(\S+)$',
                     cmd, re.IGNORECASE)
        if m:
            flags, src, dst = m.group(1), m.group(2), m.group(3)
            return (f'(move {flags}{src}\\* {dst} & '
                    f'for /d %i in ({src}\\*) do @move {flags}"%i" {dst})')

        # chmod → no-op on Windows
        if re.match(r'^chmod\s+', cmd):
            return 'echo chmod skipped >nul'

        # export VAR=value → set VAR=value
        m = re.match(r'^export\s+(\w+)=(.*)', cmd)
        if m:
            return f'set "{m.group(1)}={m.group(2)}"'

        # which binary → where binary
        m = re.match(r'^which\s+(\S+)$', cmd)
        if m:
            return f'where "{m.group(1)}" 2>nul'

        # source path → call path (with venv bin→Scripts translation)
        # Also handles `. path` (dot-space shorthand for source)
        m = re.match(r'^(?:source|\.)\s+(.+)', cmd)
        if m:
            path = m.group(1).strip()
            path = re.sub(r'(\S+)/bin/activate', r'\1/Scripts/activate', path)
            path = path.replace('/', '\\')
            return f'call {path}'

        # ls [args] → dir [args]
        m = re.match(r'^ls(\s|$)(.*)', cmd)
        if m:
            args = m.group(2).strip()
            if not args or args.startswith('-'):
                return 'dir'
            return f'dir {args}'

        # cat file → type file (single file, not heredocs)
        m = re.match(r'^cat\s+(\S+)$', cmd)
        if m and '<<' not in cmd:
            return f'type {m.group(1)}'

        return cmd

    # Interpreters whose "run this inline script" flag takes an entire
    # program as ONE argument. Those scripts routinely contain `>`, `<`,
    # `|` and `&` as ordinary language operators, which is what makes them
    # unsafe to hand to cmd.exe. See _shell_free_argv.
    _INLINE_SCRIPT_INTERPRETERS = frozenset({
        'python', 'python3', 'py', 'node', 'deno', 'ruby', 'perl',
    })
    _INLINE_SCRIPT_FLAGS = frozenset({'-c', '-e', '-p', '--eval', '--print'})

    # cmd.exe metacharacters. Their presence in a script is what turns a
    # working command into a silently redirected one.
    _CMD_METACHARS = ('>', '<', '|', '&', '^')

    @staticmethod
    def _win_split(cmd: str) -> List[str] | None:
        """Split *cmd* into argv exactly as Windows itself would.

        Uses the real Win32 parser rather than shlex: the quoting rules
        (2n backslashes before a quote, `\\"` for a literal quote) are
        idiosyncratic, and an approximation here would produce a DIFFERENT
        argv than the child would otherwise have received — silently
        changing the command instead of fixing it.
        """
        import ctypes
        try:
            argc = ctypes.c_int(0)
            fn = ctypes.windll.shell32.CommandLineToArgvW
            fn.restype = ctypes.POINTER(ctypes.c_wchar_p)
            fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
            argv_ptr = fn(cmd, ctypes.byref(argc))
            if not argv_ptr:
                return None
            try:
                return [argv_ptr[i] for i in range(argc.value)]
            finally:
                ctypes.windll.kernel32.LocalFree(argv_ptr)
        except Exception:            # not Windows, or shell32 unavailable
            return None

    @staticmethod
    def _shell_free_argv(cmd: str) -> List[str] | None:
        """argv for an inline-script command that must NOT touch cmd.exe.

        `python -c "...assert n > 0..."` is ordinary Python, but under
        ``shell=True`` cmd.exe reads that `>` as redirection and writes the
        command's real stdout into a file literally named `0`, handing the
        caller empty output instead. Observed in a benchmark run: two
        agent-loop verification commands returned nothing, the model could
        not see why its code "failed", and the step burned all 8 turns and
        escalated. Escaped quotes (``\\"``) make it worse by breaking
        cmd.exe's quote tracking, so even a `>` written inside quotes is
        treated as an operator.

        Returns None (keep the shell) unless the command is a single
        inline-script invocation. The ``len(argv) == 3`` test is the
        load-bearing one: genuine shell syntax always survives parsing as
        extra argv entries — ``python -m pytest > out.txt`` splits into 5,
        with `>` and `out.txt` as their own arguments — so a 3-element argv
        proves there is no shell work to do and bypassing is equivalent.
        """
        if os.name != 'nt':
            return None
        argv = Executor._win_split(cmd)
        if not argv or len(argv) != 3:
            return None
        exe = os.path.basename(argv[0]).lower()
        if exe.endswith('.exe'):
            exe = exe[:-4]
        if exe not in Executor._INLINE_SCRIPT_INTERPRETERS:
            return None
        if argv[1] not in Executor._INLINE_SCRIPT_FLAGS:
            return None
        # Only divert commands actually at risk. A script with no
        # metacharacter runs identically either way, and leaving it on the
        # existing path keeps this change's blast radius to the bug.
        if not any(c in argv[2] for c in Executor._CMD_METACHARS):
            return None
        return argv

    @staticmethod
    def _env_path_key(run_env: dict) -> str:
        """The PATH key actually present in *run_env*.

        Windows env vars are case-insensitive ('Path' vs 'PATH'), but a
        plain dict is not.
        """
        return next((k for k in run_env if k.upper() == "PATH"), "PATH")

    @staticmethod
    def _venv_bin_dir(cwd: str | None = None) -> str | None:
        """Return the Scripts/bin dir of a project venv under *cwd*, if any.

        Only returns a directory that actually contains a python executable,
        so a half-created or foreign 'venv' folder is ignored.

        Scaffolded projects often live one level down (a CMD step ran
        ``mkdir user_portal && cd user_portal && python -m venv venv``)
        while probes and verification commands run from the pipeline root.
        When no venv exists directly under *cwd*, look one directory level
        deeper — otherwise every import probe resolves to the system
        interpreter and reports the project's own packages as missing.
        """
        root = cwd or os.getcwd()
        if os.name == 'nt':
            sub, py = "Scripts", "python.exe"
        else:
            sub, py = "bin", "python"

        def _check(base: str) -> str | None:
            for name in ("venv", ".venv"):
                bin_dir = os.path.join(base, name, sub)
                if os.path.isfile(os.path.join(bin_dir, py)):
                    return os.path.abspath(bin_dir)
            return None

        found = _check(root)
        if found:
            return found
        try:
            subdirs = [
                os.path.join(root, d) for d in os.listdir(root)
                if not d.startswith(".")
                and d not in ("node_modules", "venv", ".venv", "__pycache__")
                and os.path.isdir(os.path.join(root, d))
            ]
        except OSError:
            return None
        hits = [h for h in (_check(d) for d in subdirs) if h]
        # Only trust an unambiguous single sub-project venv.
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def _inject_venv_path(run_env: dict, cwd: str | None = None) -> None:
        """Prepend the project venv's bin dir to PATH in *run_env* (in place).

        Each command runs in a fresh shell, so a `venv\\Scripts\\activate` in
        an earlier step never persists.  Prepending the venv's bin dir makes
        bare `python`/`pip`/`pytest` resolve to the venv interpreter where the
        project's packages were installed.
        """
        venv_bin = Executor._venv_bin_dir(cwd)
        if not venv_bin:
            return
        # Windows env vars are case-insensitive ('Path' vs 'PATH') — reuse the
        # existing key to avoid passing duplicates to the subprocess.
        path_key = Executor._env_path_key(run_env)
        current = run_env.get(path_key, "")
        if venv_bin not in current.split(os.pathsep):
            run_env[path_key] = venv_bin + os.pathsep + current if current else venv_bin
        run_env.setdefault("VIRTUAL_ENV", os.path.dirname(venv_bin))

    def run_command(self, cmd: str, env: dict | None = None,
                    timeout: int = 120, background: bool = False,
                    cwd: str | None = None,
                    retry_on_crash: bool = True) -> Tuple[bool, str]:
        """Run a shell command, retrying once if the process CRASHED.

        A process that dies never reported a verdict, so believing its exit
        status is a category error — on Windows a pygame/SDL suite
        fast-fails (0xC0000409) or access-violates (0xC0000005) in a
        substantial fraction of invocations, printing ordinary-looking test
        output first. Read as a result, that green suite becomes "the tests
        regressed" and a correct file gets rolled back.

        The pipeline already knew this and hand-rolled the same
        detect-log-retry block at four call sites (``GateLedger.recheck``,
        the BulkTest plan gate, the BulkTest runner, ``AgentLoop._run_verify``
        — whose comment records it as having been *missing* there until
        someone hit it). Copy-paste coverage leaves every site the author
        did not think of unprotected, and an audit found 31 further
        test/gate invocations without the guard, including re-runs of the
        very ``base_cmd`` that is guarded elsewhere in the same function.
        Centralising it here makes the protection the default that new call
        sites inherit instead of a rule they must remember.

        Only an *abnormal* exit retries: a signal death or an NTSTATUS
        failure code. An ordinary non-zero status is a real verdict and is
        returned untouched, so genuine failures are never masked. Pass
        ``retry_on_crash=False`` for a command whose side effects must not
        be repeated. Background commands are never retried — they have not
        finished, so there is no exit status to judge.

        Returns (success, output); see :meth:`_run_command_once` for the
        command-rewriting behaviour.
        """
        ok, out = self._run_command_once(
            cmd, env=env, timeout=timeout, background=background, cwd=cwd)
        if ok or background or not retry_on_crash:
            return ok, out

        is_abnormal, describe, log_diagnostics = _crash_helpers()
        code = self.last_exit_code
        if not is_abnormal(code):
            return ok, out                       # a real verdict — believe it

        log.warning(
            f"[Executor] Command terminated abnormally "
            f"({describe(code) or code}) — retrying once before believing "
            f"it: {cmd}")
        log_diagnostics(code, cmd)
        ok, out = self._run_command_once(
            cmd, env=env, timeout=timeout, background=background, cwd=cwd)
        if ok:
            return ok, out

        if is_abnormal(self.last_exit_code):
            # Crashed twice. Still a failure to the caller — suppressing it
            # would invent a pass — but tagged, so a consumer that can act
            # on "inconclusive" (a gate ledger deciding whether to record a
            # regression) is able to tell it apart from a real red suite.
            log.warning(
                f"[Executor] Command crashed again "
                f"({describe(self.last_exit_code) or self.last_exit_code}) "
                f"— reporting as inconclusive: {cmd}")
            out = f"{out}\n{CRASHED_MARKER}".strip()
        return ok, out

    def _run_command_once(self, cmd: str, env: dict | None = None,
                          timeout: int = 120, background: bool = False,
                          cwd: str | None = None) -> Tuple[bool, str]:
        """
        Runs an arbitrary shell command. Returns (success, output).
        On Windows, auto-wraps PowerShell cmdlets so they don't fail
        in the default cmd.exe shell.

        If *background* is True, the process is started and tracked. The
        method waits briefly (3s) to see if it crashes; if not, it returns success.

        If *cwd* is set, the command runs in that directory instead of the
        current working directory.
        """
        try:
            cmd = Executor._sanitize_unicode(cmd)
            log.info(f"[Executor] Running {'background ' if background else ''}command: {cmd}"
                     f"{f' (cwd={cwd})' if cwd else ''}")
            # Fix bash-only constructs (source → .) so /bin/sh can run them
            if os.name != 'nt':
                cmd = Executor._rewrite_for_posix_sh(cmd)

            # Translate Unix commands to Windows cmd.exe equivalents
            if os.name == 'nt':
                cmd = Executor._rewrite_unix_cmd_for_windows(cmd)

            if os.name == 'nt' and Executor._needs_powershell(cmd):
                # Escape double quotes inside the command for PowerShell
                escaped = cmd.replace('"', '\\"')
                cmd = f'powershell -NoProfile -Command "{escaped}"'

            # Safety net: add non-interactive flags to known interactive commands
            cmd = Executor._rewrite_interactive_cmd(cmd)

            # Build environment — disable color codes and interactive prompts
            run_env = dict(env) if env else os.environ.copy()
            # Prefer the project venv's interpreter/tools over the system ones
            Executor._inject_venv_path(run_env, cwd)
            run_env.setdefault("NO_COLOR", "1")
            run_env.setdefault("FORCE_COLOR", "0")
            # Non-interactive: prevent CLI tools from prompting for input
            run_env.setdefault("CI", "true")
            run_env.setdefault("DEBIAN_FRONTEND", "noninteractive")
            run_env.setdefault("PIP_NO_INPUT", "1")
            run_env.setdefault("NPM_CONFIG_YES", "true")

            # An inline script (`python -c "...n > 0..."`) must bypass
            # cmd.exe, which would read its language operators as
            # redirection and swallow the output.
            argv = Executor._shell_free_argv(cmd)
            if argv:
                # subprocess does NOT honour env's PATH when resolving the
                # executable on Windows — CreateProcess searches the PARENT
                # process's PATH. Verified: with the project venv prepended
                # to run_env, `Popen(['python', ...])` still launched
                # C:\Python313\python.exe. Left implicit, this "fix" would
                # silently move every inline script off the venv
                # interpreter and report the project's own packages
                # missing. Resolve it explicitly, or keep the shell.
                resolved = shutil.which(
                    argv[0], path=run_env.get(Executor._env_path_key(run_env)))
                if resolved:
                    misread = [c for c in Executor._CMD_METACHARS
                               if c in argv[2]]
                    log.debug(
                        f"[Executor] Inline script — bypassing cmd.exe, "
                        f"which would misread {' '.join(misread)} as shell "
                        f"syntax and swallow the output")
                    argv = [resolved] + argv[1:]
                else:
                    argv = None

            # Read as raw bytes and decode manually. On Windows, text=True
            # uses cp1252 by default, but most tools (Node.js, Jest, Go)
            # output UTF-8.  This mismatch causes empty/garbled output.
            proc = subprocess.Popen(
                argv or cmd, shell=argv is None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=run_env,
                cwd=cwd,
            )

            if background:
                self._background_processes.append(proc)
                # Wait briefly to see if it dies instantly (e.g. port already in use)
                try:
                    stdout_bytes, _ = proc.communicate(timeout=3)
                    output = Executor._decode_output(stdout_bytes)
                    return proc.returncode == 0, output.strip()
                except subprocess.TimeoutExpired:
                    # Still running after 3s — assume background success for now
                    return True, "[Background process started]"

            stdout_bytes, _ = proc.communicate(timeout=timeout)
            output = Executor._decode_output(stdout_bytes)
            self.last_exit_code = proc.returncode
            log.info(f"[Executor] Exit code: {proc.returncode}, "
                     f"output={len(output)} chars")

            if not output.strip() and proc.returncode != 0:
                # Command failed silently — provide useful context
                interactive_hint = ""
                if Executor._is_likely_interactive(cmd):
                    interactive_hint = (
                        "- The command may require interactive input (prompts) "
                        "which is not available. Try adding --yes, -y, or "
                        "--defaults flag.\n"
                    )
                output = (
                    f"Command `{cmd}` exited with code {proc.returncode} "
                    f"{NO_OUTPUT_MARKER}.\n"
                    f"Possible causes:\n"
                    f"{interactive_hint}"
                    f"- The tool/binary is not installed or not on PATH\n"
                    f"- A required config file is missing\n"
                    f"- The command crashed before it could produce output"
                )
                log.warning(f"[Executor] {output}")

            return proc.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            log.warning(f"[Executor] Command timed out after {timeout}s: {cmd}")
            # Kill the entire process tree, not just the direct child.  On
            # Windows, proc.kill() only kills the immediate process — any
            # grandchildren (e.g. pyglet/pygame GUI threads spawned by a
            # hung pytest) inherit the stdout pipe and keep it open, which
            # makes the subsequent communicate() block indefinitely.  This
            # mirrors the cleanup() path which already does the right thing.
            if os.name == 'nt':
                try:
                    subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                        capture_output=True, timeout=5,
                    )
                except (subprocess.TimeoutExpired, OSError) as kill_exc:
                    log.warning(
                        f"[Executor] taskkill failed ({kill_exc}) — "
                        f"falling back to proc.kill()")
                    proc.kill()
            else:
                proc.kill()
            # Even after killing the tree, bound communicate() with a short
            # timeout in case any reader/writer is still in a weird state.
            try:
                stdout_bytes, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning(
                    "[Executor] Process still holding stdout pipe after "
                    "kill — closing pipes and abandoning collected output.")
                try:
                    if proc.stdout is not None:
                        proc.stdout.close()
                except OSError:
                    pass
                stdout_bytes = b""
            output = Executor._decode_output(stdout_bytes)
            return False, f"Command timed out after {timeout} seconds.\n{output}".strip()
        except Exception as e:
            log.error(f"[Executor] Exception running command: {e}")
            return False, str(e)

    @staticmethod
    def kill_process_tree(proc: subprocess.Popen) -> None:
        """Terminate *proc* and all of its children."""
        try:
            if proc.poll() is None:  # still running
                # On Windows, taskkill is often more reliable for tree cleanup
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                                   capture_output=True)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        except Exception as e:
            log.warning(f"[Executor] Failed to kill process {proc.pid}: {e}")

    def stop_background_processes_from(self, baseline_count: int) -> None:
        """Kill background processes started after *baseline_count* were tracked."""
        new_procs = self._background_processes[baseline_count:]
        for proc in new_procs:
            Executor.kill_process_tree(proc)
        del self._background_processes[baseline_count:]

    def cleanup(self):
        """Terminate all background processes."""
        if not self._background_processes:
            return
        log.info(f"[Executor] Cleaning up {len(self._background_processes)} background processes")
        for proc in self._background_processes:
            Executor.kill_process_tree(proc)
        self._background_processes.clear()

    @staticmethod
    def _decode_output(raw: bytes | None) -> str:
        """Decode subprocess output, trying UTF-8 first then system default."""
        if not raw:
            return ""
        try:
            return raw.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            pass
        try:
            import locale
            return raw.decode(locale.getpreferredencoding(False), errors="replace")
        except (UnicodeDecodeError, ValueError, LookupError):
            return raw.decode("ascii", errors="replace")

    def run_tests(self, test_command: str = "pytest", cwd: str | None = None) -> Tuple[bool, str]:
        """Run tests with the project root on PYTHONPATH.

        This ensures imports like ``from src.my_module import X`` resolve
        correctly regardless of how pytest discovers the tests.

        If the test runner binary is not found, returns a clear error
        message instead of a silent failure.
        """
        env = os.environ.copy()
        # Ensure PYTHONPATH always includes the base dir, even if running in a subdir
        base_dir = os.getcwd()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = base_dir + (os.pathsep + existing if existing else "")

        # Quick check: does the test runner binary exist?
        runner = test_command.split()[0]  # e.g. "pytest", "npx", "go"
        import shutil
        if not shutil.which(runner, path=env.get("PATH")):
            # Provide install hints appropriate to the tool
            _system_tools = {"go", "cargo", "rustc", "javac", "java", "dotnet", "gcc", "g++"}
            if runner in _system_tools:
                hint = (f"`{runner}` must be installed manually from its "
                        f"official website (not available via pip/npm).")
            else:
                hint = (f"Install it first (e.g. `pip install {runner}` or "
                        f"`npm install --save-dev {runner}`).")
            msg = f"Test runner `{runner}` is not installed or not on PATH.\n{hint}"
            log.warning(f"[Executor] {msg}")
            return False, msg

        return self.run_command(test_command, env=env, cwd=cwd)

    # ── Missing-package auto-install ──

    # JS/TS: globals that need explicit import in ESM projects
    _JS_GLOBAL_TO_IMPORT: dict[str, str] = {
        "expect": "@jest/globals",
        "describe": "@jest/globals",
        "test": "@jest/globals",
        "it": "@jest/globals",
        "beforeEach": "@jest/globals",
        "afterEach": "@jest/globals",
        "beforeAll": "@jest/globals",
        "afterAll": "@jest/globals",
        "jest": "@jest/globals",
    }

    # Well-known module → pip-package mappings where the names differ
    _MODULE_TO_PACKAGE = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "bs4": "beautifulsoup4",
        "dotenv": "python-dotenv",
        "gi": "PyGObject",
        "serial": "pyserial",
        "usb": "pyusb",
        "attr": "attrs",
        "dateutil": "python-dateutil",
        "jose": "python-jose",
        "jwt": "PyJWT",
        "magic": "python-magic",
        "lxml": "lxml",
    }

    # pytest fixtures that come from well-known plugins
    _FIXTURE_TO_PACKAGE = {
        "benchmark": "pytest-benchmark",
        "httpserver": "pytest-localserver",
        "mocker": "pytest-mock",
        "faker": "faker",
        "freezer": "pytest-freezegun",
        "celery_app": "pytest-celery",
        "async_client": "httpx",
        "anyio_backend": "anyio",
        "respx_mock": "respx",
    }

    @staticmethod
    def detect_missing_packages(test_output: str) -> List[str]:
        """Parse test output and return a list of packages to install.

        Detects:
        - ``ModuleNotFoundError: No module named 'xyz'``
        - ``ImportError: No module named 'xyz'``
        - ``fixture 'xyz' not found`` (pytest plugin fixtures)
        - ``ReferenceError: X is not defined`` (JS/TS missing globals)
        """
        packages: list[str] = []
        seen: set[str] = set()

        # Missing modules
        for m in re.finditer(
            r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]([^'\"]+)['\"]",
            test_output,
        ):
            module = m.group(1).split(".")[0]  # top-level package
            pkg = Executor._MODULE_TO_PACKAGE.get(module, module)
            if pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)

        # Missing pytest fixtures (plugin packages)
        for m in re.finditer(r"fixture ['\"](\w+)['\"] not found", test_output):
            fixture = m.group(1)
            pkg = Executor._FIXTURE_TO_PACKAGE.get(fixture)
            if pkg and pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)

        # JS/TS: ReferenceError for missing globals (expect, describe, etc.)
        for m in re.finditer(
            r"ReferenceError:\s*(\w+)\s+is not defined",
            test_output,
        ):
            name = m.group(1)
            pkg = Executor._JS_GLOBAL_TO_IMPORT.get(name)
            if pkg and pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)

        # Vite/Vitest: Failed to resolve import "pkg"
        # e.g. 'Failed to resolve import "@testing-library/user-event"'
        for m in re.finditer(
            r'Failed to resolve import\s+"([^"]+)"',
            test_output,
        ):
            pkg = m.group(1)
            # Skip relative imports
            if pkg.startswith("."):
                continue
            # Normalize scoped package subpaths: '@heroicons/react/outline' → '@heroicons/react'
            if pkg.startswith("@") and pkg.count("/") >= 2:
                pkg = "/".join(pkg.split("/")[:2])
            if pkg not in seen:
                packages.append(pkg)
                seen.add(pkg)

        # Vite/Vitest/Node: Cannot find module/package 'pkg'
        for m in re.finditer(
            r"Cannot find (?:module|package) ['\"]([^'\"]+)['\"]",
            test_output,
        ):
            pkg = m.group(1)
            # Skip relative imports — only catch package names
            if not pkg.startswith(".") and pkg not in seen:
                # Normalize scoped package subpaths: '@heroicons/react/24/outline' → '@heroicons/react'
                if pkg.startswith("@") and pkg.count("/") >= 2:
                    pkg = "/".join(pkg.split("/")[:2])
                if pkg not in seen:
                    packages.append(pkg)
                    seen.add(pkg)

        return packages

    def install_packages(self, packages: List[str], tool: str = "pip install", cwd: str | None = None) -> Tuple[bool, str]:
        """Install packages via the specified tool (default: `pip install`). Returns (all_succeeded, combined_output)."""
        if not packages:
            return True, ""
        cmd = f"{tool} {' '.join(packages)}"
        log.info(f"[Executor] Auto-installing: {cmd}")
        return self.run_command(cmd, cwd=cwd)

    @staticmethod
    def parse_step_dependencies(steps: List[str]) -> Tuple[List[str], Dict[int, set]]:
        """Parse ``(depends: N, M)`` markers from step text.

        Returns ``(cleaned_steps, dependencies)`` where *cleaned_steps*
        has dependency markers removed and *dependencies* maps each
        step index to a set of dependency indices (0-based).

        If **no** dependency markers are found at all, falls back to
        strict sequential ordering (each step depends on the previous)
        so that steps never run out of order.

        Handles LLM output formats like:
        - ``(depends: 1)``
        - ``(depends: 1, 3)``
        - ``(CMD, depends: 1):``
        - ``(CODE, depends: 2, 3):``
        """
        cleaned: List[str] = []
        deps: Dict[int, set] = {}
        # Match dependency markers in various LLM formats:
        # - Standalone: (depends: 1) or (depends: 1, 3)
        # - Combined with step type: (CMD, depends: 1) or (CODE, depends: 2, 3)
        # - With optional trailing colon: (depends: 1): or (CMD, depends: 1):
        dep_pattern = re.compile(
            r"\s*\([^)]*?depends?:\s*([\d,\s]+)\)[:\s]*$", re.IGNORECASE
        )
        # Also strip "(depends: none)" markers that carry no numeric deps
        none_dep_pattern = re.compile(
            r"\s*\([^)]*?depends?:\s*none\s*\)[:\s]*$", re.IGNORECASE
        )
        found_any_marker = False

        for idx, step in enumerate(steps):
            match = dep_pattern.search(step)
            if match:
                found_any_marker = True
                raw = match.group(1)
                # Parse comma-separated step numbers (1-based → 0-based)
                dep_indices = set()
                for num_str in raw.split(","):
                    num_str = num_str.strip()
                    if num_str.isdigit():
                        dep_indices.add(int(num_str) - 1)  # 1-based → 0-based
                deps[idx] = dep_indices
                cleaned.append(step[:match.start()].rstrip())
            else:
                # Strip "(depends: none)" so it doesn't pollute step text
                none_match = none_dep_pattern.search(step)
                if none_match:
                    found_any_marker = True
                    cleaned.append(step[:none_match.start()].rstrip())
                else:
                    cleaned.append(step)
                deps[idx] = set()

        # No markers at all → sequential: each step depends on its predecessor
        if not found_any_marker:
            for idx in range(1, len(cleaned)):
                deps[idx] = {idx - 1}

        return cleaned, deps

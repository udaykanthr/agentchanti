"""
Diff display — compute and show colored unified diffs before writing files.

Includes a Textual-based interactive diff viewer that pauses execution so the
user can review changes and approve/reject before files are written to disk.
"""

from __future__ import annotations

import os
import sys
import difflib
import threading
import time
import logging
from typing import TYPE_CHECKING
from .config import Config

if TYPE_CHECKING:
    from .cli_display import CLIDisplay

# Hazards that block execution or require explicit confirmation
HAZARD_BLOCK = "BLOCK"
HAZARD_WARN = "WARN"

# Get a logger instance consistent with the rest of the package
log = logging.getLogger("agentchanti")


def compute_diff(filepath: str, new_content: str, base_dir: str = ".") -> str | None:
    """Return unified diff string for modified or new files.

    For existing files: returns a unified diff if content differs.
    For new files: returns a diff showing all lines as additions.
    Returns None only if the file exists and content is unchanged.
    """
    # Repair mojibake in new content before diffing so the diff viewer
    # doesn't show false special-character changes
    from .executor import Executor
    new_content = Executor._repair_mojibake(new_content)

    full_path = os.path.join(base_dir, filepath)
    if not os.path.isfile(full_path):
        # New file — show entire content as additions
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            [], new_lines,
            fromfile="/dev/null",
            tofile=f"b/{filepath}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        return diff_text if diff_text.strip() else None

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            old_content = f.read()
    except OSError:
        return None

    if old_content == new_content:
        return None  # unchanged

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    return diff_text if diff_text.strip() else None


def format_colored_diff(diff_text: str) -> str:
    """Add ANSI colors to a unified diff string.

    Green for additions (+), red for deletions (-), cyan for @@ hunks.
    """
    lines = diff_text.splitlines()
    colored: list[str] = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            colored.append(f"\033[1m{line}\033[0m")  # bold
        elif line.startswith("@@"):
            colored.append(f"\033[36m{line}\033[0m")  # cyan
        elif line.startswith("+"):
            colored.append(f"\033[32m{line}\033[0m")  # green
        elif line.startswith("-"):
            colored.append(f"\033[31m{line}\033[0m")  # red
        else:
            colored.append(line)
    return "\n".join(colored)


def compute_diffs(files: dict[str, str], base_dir: str = ".") -> list[tuple[str, str]]:
    """Compute diffs for all files (modified and new). Returns list of (filepath, diff_text)."""
    diffs: list[tuple[str, str]] = []
    for filepath, content in files.items():
        diff = compute_diff(filepath, content, base_dir)
        if diff:
            diffs.append((filepath, diff))
    return diffs


def _detect_hazards(filepath: str, old_content: str, new_content: str) -> list[tuple[str, str]]:
    """Detect potential safety hazards in file changes.

    Returns list of (severity, message) tuples.
    Severity is HAZARD_BLOCK or HAZARD_WARN.
    """
    hazards = []
    fname = os.path.basename(filepath)

    # 1. Critical File Checks
    if fname in Config.CRITICAL_FILES:
        # Strict Block: Dependencies in package.json
        if fname == "package.json":
            import re
            # Check if dependencies section exists in old but is modified/removed in new
            # This is a heuristic: if "dependencies" string count changes or context suggests deletion
            # For robustness, we'll block ANY edit to dependencies/devDependencies unless via command
            # But parsing JSON with regex is fragile. simpler:
            # If "dependencies" or "devDependencies" key is present in old content,
            # and the new content looks like a manual edit (which often truncates),
            # we block it? No, that's too aggressive.
            # STICT RULE: If the prompt didn't strictly say "use npm install", the agent might match
            # "modify package.json".
            # We will use string containment.
            
            # Simple heuristic: if 'dependencies' is in old, it must be in new.
            if '"dependencies"' in old_content and '"dependencies"' not in new_content:
                hazards.append((HAZARD_WARN, "Critical section 'dependencies' removed!"))
            
            # If the user requested stricter control: BLOCK attempts to add dependencies manually.
            # We can't easily distinguish "adding dep" from "formatting".
            # But we can check if lines with "^weight" or version numbers are added.
            # Actually, the user requirement is: "package.json should be updated on npm commands only"
            # So, basically ANY change to package.json is suspect unless it's strictly 'scripts'.
            # Let's inspect the diff for lines adding dependencies.
             
            pass # We'll rely on the size check & general warning for now, plus the specific logic below.

    # 2. Significant Shrinkage (Truncation Risk)
    # Only flag truly destructive wipes — stricter threshold (< 25% of
    # original) avoids false positives during first-time project setup
    # where template files (e.g. App.jsx) are legitimately replaced with
    # smaller custom versions.
    if len(old_content) > 200 and len(new_content) < len(old_content) * 0.25:
        hazards.append((HAZARD_WARN,
                        f"Significant size reduction ({len(old_content)} -> {len(new_content)} chars). "
                        "Potential accidental truncation."))

    # 3. Strict Package.json Dependency Block
    if fname == "package.json":
         # Check if we are touching dependencies
         # Look for lines like `"react": "..."` in the new content that weren't there?
         # Or just simplistic: if the diff affects the dependencies block.
         # Since we don't have a JSON parser here easily without adding imports/complexity,
         # we'll use a regex check on the *diff* or content.
         
         # Better approach: Block ALL edits to package.json if they touch dependencies
         # "touched" = old content had them, new content has them different.
         # But the agent *overwrites* the file. So we compare full contents.
         
         # For this specific requirement, we will be aggressive.
         # If the diff shows changes to lines containing version numbers (e.g. "x": "^1.2.3"),
         # we flag it.
         pass
         
         # NOTE: For now, we'll enforce the "Warning" and rely on the PROMPT to stop the agent.
         # But if we really want to block, we can add:
         if '"dependencies"' in new_content or '"devDependencies"' in new_content:
             # Smart check: only flag if existing dependency versions are CHANGED
             # (not just if dependencies exist). Additive changes like new scripts
             # or new test deps are safe — SmartMerge handles protection at write time.
             import json as _json
             try:
                 old_pkg = _json.loads(old_content)
                 new_pkg = _json.loads(new_content)
                 changed_deps = []
                 for section in ('dependencies', 'devDependencies'):
                     old_deps = old_pkg.get(section, {})
                     new_deps = new_pkg.get(section, {})
                     for dep, ver in old_deps.items():
                         if dep in new_deps and new_deps[dep] != ver:
                             changed_deps.append(f"{dep}: {ver} → {new_deps[dep]}")
                 if changed_deps:
                     hazards.append((HAZARD_WARN,
                                     f"Dependency version changes detected: "
                                     f"{', '.join(changed_deps[:3])}. "
                                     f"Use `npm install` instead of manual edits."))
             except (ValueError, KeyError):
                 # JSON parse failed — treat as suspicious
                 hazards.append((HAZARD_WARN,
                                 "Could not verify package.json changes — "
                                 "verify manually."))

    # 4. Export statement loss detection (JS/TS/JSX/TSX)
    # Common LLM truncation: drops `export default X;` at end of file
    _, ext = os.path.splitext(filepath)
    if ext in ('.js', '.jsx', '.ts', '.tsx', '.mjs', '.mts'):
        import re
        # Check for export default loss
        old_has_export_default = bool(re.search(
            r'export\s+default\s+', old_content))
        new_has_export_default = bool(re.search(
            r'export\s+default\s+', new_content))
        if old_has_export_default and not new_has_export_default:
            hazards.append((HAZARD_WARN,
                            "Missing `export default` — likely truncated by LLM. "
                            "The original file exported a default symbol."))

        # Check for named export loss (e.g., export function, export const, export class)
        old_exports = set(re.findall(
            r'export\s+(?:function|const|let|var|class)\s+(\w+)', old_content))
        new_exports = set(re.findall(
            r'export\s+(?:function|const|let|var|class)\s+(\w+)', new_content))
        lost_exports = old_exports - new_exports
        if lost_exports:
            hazards.append((HAZARD_WARN,
                            f"Named export(s) lost: {', '.join(sorted(lost_exports))}. "
                            "Verify these were intentionally removed."))

    # 5. Unicode corruption (mojibake) detection
    # Check if the new content introduces common UTF-8→Latin-1 corruption
    # patterns that weren't present in the old content.
    import re
    _MOJIBAKE_PATTERNS = [
        r'â\x80[\x93\x94\x98\x99\x9c\x9d\x9e\xa2\xa6]',  # em/en dash, quotes, bullet, ellipsis
        r'Ã[©¨¼¶¤±§]',  # accented chars corrupted
    ]
    for pat in _MOJIBAKE_PATTERNS:
        old_matches = len(re.findall(pat, old_content))
        new_matches = len(re.findall(pat, new_content))
        if new_matches > old_matches:
            hazards.append((HAZARD_WARN,
                            f"Possible Unicode corruption (mojibake) detected — "
                            f"special characters may have been corrupted."))
            break  # one warning is enough

    return hazards


def show_diffs(files: dict[str, str], base_dir: str = ".",
               log_only: bool = False) -> list[str]:
    """Compute and display diffs for all files.

    When *log_only* is True, diffs are returned but not printed (for --auto mode).
    Returns list of diff strings.
    """
    from .cli_display import log

    diffs = compute_diffs(files, base_dir)
    diff_strings: list[str] = []

    for filepath, diff_text in diffs:
        colored = format_colored_diff(diff_text)
        diff_strings.append(diff_text)

        if log_only:
            log.info(f"Diff for {filepath}:\n{diff_text}")
        else:
            print(f"\n{'─' * 60}")
            print(colored)

    new_files = [f for f in files if not os.path.isfile(os.path.join(base_dir, f))]
    if new_files and not log_only:
        print(f"\n  New files: {', '.join(new_files)}")

    return diff_strings


# ══════════════════════════════════════════════════════════════════
#  Interactive Diff Approval — Textual TUI
# ══════════════════════════════════════════════════════════════════

# Module-level flag: once the user picks "Approve & don't ask again",
# all subsequent diffs are auto-approved for the rest of the session.
_approve_all: bool = False

# Serialize interactive approval — only one Textual/console prompt at a time.
# Without this, parallel wave threads launch competing Textual apps that
# fight for the terminal, causing blank screens and hangs.
_approval_lock = threading.Lock()


def prompt_diff_approval(files: dict[str, str], base_dir: str = ".",
                         auto: bool = False, display: CLIDisplay | None = None) -> bool:
    """Show diffs in an interactive Textual viewer and wait for approval.

    Returns ``True`` if the user approves (or if running in auto mode).
    Returns ``False`` if the user rejects.

    Uses ``_approval_lock`` to serialize interactive prompts so that
    parallel wave threads don't launch competing Textual apps (which
    causes blank screens, freezes, and unresponsive ESC on Windows).
    """
    global _approve_all

    diffs = compute_diffs(files, base_dir)
    new_files = [f for f in files if not os.path.isfile(os.path.join(base_dir, f))]

    # Nothing to review
    if not diffs and not new_files:
        return True

    # Auto mode or "approve all" — log diffs and approve (no lock needed)
    if auto or _approve_all:
        for filepath, diff_text in diffs:
            log.info(f"[auto] Diff for {filepath}:\n{diff_text}")
        if new_files:
            log.info(f"[auto] New files: {', '.join(new_files)}")
        return True

    # Serialize: only one interactive approval at a time
    with _approval_lock:
        # Re-check _approve_all — another thread may have set it while we waited
        if _approve_all:
            for filepath, diff_text in diffs:
                log.info(f"[auto] Diff for {filepath}:\n{diff_text}")
            if new_files:
                log.info(f"[auto] New files: {', '.join(new_files)}")
            return True

        # Try Textual TUI
        if display:
            log.debug(f"[diff] Pausing display wave_thread={threading.current_thread().name}")
            display.pause()
        try:
            time.sleep(0.1)  # Settle time for terminal
            try:
                log.debug(f"[diff] Launching Textual app wave_thread={threading.current_thread().name}")
                res = _textual_diff_approval(diffs, new_files, files, base_dir=base_dir)
                log.debug(f"[diff] Textual app result={res} wave_thread={threading.current_thread().name}")
                if res is not None:
                    return res
                log.warning("[diff] Textual app failed to start — falling back to console approval.")
            except ImportError:
                log.warning("Textual not installed — falling back to console diff approval.")
            except Exception as e:
                log.warning(f"Textual diff viewer failed: {e}")
                log.debug(f"[diff] Textual error: {e}", exc_info=True)

            # Fallback: console-based approval
            log.debug(f"[diff] Falling back to console approval wave_thread={threading.current_thread().name}")
            res = _console_diff_approval(diffs, new_files, files, base_dir=base_dir)
            log.debug(f"[diff] Console approval result={res} wave_thread={threading.current_thread().name}")
            return res
        finally:
            if display:
                log.debug(f"[diff] Resuming display wave_thread={threading.current_thread().name}")
                display.resume()





def _format_rich_diff(diff_text: str) -> str:
    """Convert unified diff text to Rich markup for Textual display."""
    lines = diff_text.splitlines()
    markup_lines: list[str] = []
    for line in lines:
        # Escape Rich markup characters in the line content
        escaped = line.replace("[", "\\[")
        if line.startswith("+++") or line.startswith("---"):
            markup_lines.append(f"[bold white]{escaped}[/bold white]")
        elif line.startswith("@@"):
            markup_lines.append(f"[cyan]{escaped}[/cyan]")
        elif line.startswith("+"):
            markup_lines.append(f"[green]{escaped}[/green]")
        elif line.startswith("-"):
            markup_lines.append(f"[red]{escaped}[/red]")
        else:
            markup_lines.append(escaped)
    return "\n".join(markup_lines)


def _textual_diff_approval(diffs: list[tuple[str, str]],
                           new_files: list[str],
                           files: dict[str, str],
                           base_dir: str = ".") -> bool | None:
    """Launch a Textual app to display diffs and get approval."""
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import Button, Footer, Static
    from textual.binding import Binding

    class DiffApprovalApp(App):
        """Interactive diff viewer with approve/reject."""

        CSS = """
        Screen {
            background: $surface;
        }
        #title-bar {
            dock: top;
            height: 3;
            background: #1a1a2e;
            color: #e94560;
            text-align: center;
            padding: 1;
            text-style: bold;
        }
        #diff-scroll {
            height: 1fr;
            margin: 1 2;
            border: round #444;
            padding: 1;
        }
        .file-header {
            color: #e9c46a;
            text-style: bold;
            margin: 1 0 0 0;
        }
        .diff-content {
            margin: 0 0 1 0;
        }
        .new-file-section {
            color: #2a9d8f;
            text-style: bold;
            margin: 1 0;
        }
        #action-buttons {
            dock: bottom;
            height: 3;
            align: center middle;
            padding: 0 2;
        }
        #action-buttons Button {
            margin: 0 2;
            min-width: 20;
        }
        #summary {
            dock: bottom;
            height: 1;
            text-align: center;
            color: #888;
        }
        """

        BINDINGS = [
            Binding("a", "approve", "Approve"),
            Binding("ctrl+s", "approve", "Approve"),
            Binding("escape", "reject", "Reject"),
            Binding("r", "reject", "Reject"),
            Binding("s", "approve_all", "Always Approve"),
        ]

        def __init__(self, diffs: list[tuple[str, str]],
                     new_files: list[str],
                     files: dict[str, str]) -> None:
            super().__init__()
            self._diffs = diffs
            self._new_files = new_files
            self._files = files
            self._approved: bool = False
            self._approve_all: bool = False
            self.started: bool = False
            self._hazards: list[str] = []

            # Pre-calc hazards
            for filepath, _ in diffs:
                full_path = os.path.join(base_dir, filepath)
                old_content = ""
                if os.path.isfile(full_path):
                     with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                         old_content = f.read()
                new_content = files.get(filepath, "")
                
                for severity, msg in _detect_hazards(filepath, old_content, new_content):
                    self._hazards.append(f"{filepath}: {msg}")

        def compose(self) -> ComposeResult:
            file_count = len(self._diffs) + len(self._new_files)
            title = f" ━━  Diff Review — {file_count} file(s) changed  ━━ "
            if self._hazards:
                title = f" ⚠️ SAFETY WARNING ⚠️  {len(self._hazards)} hazard(s) detected "
            yield Static(title, id="title-bar")
            
            if self._hazards:
                yield Static(
                    "\n".join([f"[bold red]! {h}[/bold red]" for h in self._hazards]),
                    id="hazard-banner",
                    classes="hazard-banner"
                )
            with VerticalScroll(id="diff-scroll"):
                for filepath, diff_text in self._diffs:
                    yield Static(
                        f"[bold yellow]{'─' * 58}[/bold yellow]\n"
                        f"[bold yellow]  {filepath}[/bold yellow]",
                        classes="file-header",
                    )
                    yield Static(
                        _format_rich_diff(diff_text),
                        classes="diff-content",
                    )
                if self._new_files:
                    yield Static(
                        f"[bold #2a9d8f]{'─' * 58}[/bold #2a9d8f]\n"
                        f"[bold #2a9d8f]  New files:[/bold #2a9d8f]",
                        classes="new-file-section",
                    )
                    for nf in self._new_files:
                        lines = self._files.get(nf, "")
                        line_count = len(lines.splitlines()) if lines else 0
                        yield Static(
                            f"  [green]+ {nf}[/green]  ({line_count} lines)",
                        )
            summary_parts = []
            if self._diffs:
                summary_parts.append(f"{len(self._diffs)} modified")
            if self._new_files:
                summary_parts.append(f"{len(self._new_files)} new")
            yield Static(
                f"  {' | '.join(summary_parts)}  —  "
                f"[bold]A[/bold] Approve  |  [bold]S[/bold] Always Approve  |  "
                f"[bold]R[/bold]/Esc Reject",
                id="summary",
            )
            with Horizontal(id="action-buttons"):
                if self._hazards:
                     yield Button(
                        "⚠️ CONFIRM", id="approve-btn", variant="error",
                    )
                else:
                    yield Button(
                        "✔ Approve", id="approve-btn", variant="success",
                    )
                yield Button(
                    "✔ Always Approve", id="approve-all-btn", variant="warning",
                )
                yield Button(
                    "✕ Reject", id="reject-btn", variant="error",
                )
            yield Footer()

        def on_mount(self) -> None:
            self.started = True
            # Focus the approve button so Tab+Enter works immediately
            # as a keyboard fallback for terminals with poor mouse support.
            btn = self.query_one("#approve-btn", Button)
            btn.focus()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            if event.button.id == "approve-btn":
                self._approved = True
            elif event.button.id == "approve-all-btn":
                self._approved = True
                self._approve_all = True
            elif event.button.id == "reject-btn":
                self._approved = False
            else:
                return
            # Defer exit so the mouse event finishes processing first —
            # calling self.exit() directly from a button handler can hang
            # on Windows terminals (alternate screen clears but event loop
            # never completes shutdown).
            self.call_later(self.exit)

        def action_approve(self) -> None:
            self._approved = True
            self.exit()

        def action_approve_all(self) -> None:
            self._approved = True
            self._approve_all = True
            self.exit()

        def action_reject(self) -> None:
            self._approved = False
            self.exit()

    global _approve_all
    app = DiffApprovalApp(diffs, new_files, files)
    
    start_t = time.monotonic()
    app.run()
    elapsed = time.monotonic() - start_t

    # Logic: if the app exits in < 0.3s AND never reached on_mount, it likely
    # failed to initialize the TUI (common in some thread/terminal combos).
    if not app.started or (elapsed < 0.3 and not app._approved):
        log.debug(f"[diff] Textual startup bail: started={app.started}, elapsed={elapsed:.3f}s")
        return None

    if app._approve_all:
        _approve_all = True
    return app._approved


def _console_diff_approval(diffs: list[tuple[str, str]],
                           new_files: list[str],
                           all_files: dict[str, str],
                           base_dir: str = ".") -> bool:
    """Fallback console-based diff approval when Textual is unavailable."""
    global _approve_all

    print("\n" + "=" * 60)
    print("  DIFF REVIEW")
    print("=" * 60)

    any_hazards = False
    for filepath, diff_text in diffs:
        print(f"\n{'─' * 60}")
        print(f"File: {filepath}")
        
        # Check for hazards
        full_path = os.path.join(base_dir, filepath)
        old_content = ""
        if os.path.isfile(full_path):
             with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                  old_content = f.read()
        
        new_content = all_files.get(filepath, "")
        hazards = _detect_hazards(filepath, old_content, new_content)
        for severity, msg in hazards:
            any_hazards = True
            print(f"\033[1;31m[!] SAFETY WARNING: {msg}\033[0m")

        print(format_colored_diff(diff_text))

    if new_files:
        print(f"\n  New files: {', '.join(new_files)}")

    print("\n" + "=" * 60)
    if any_hazards:
        print("  \033[1;31mHAZARDS DETECTED! Safety confirmation required.\033[0m")
        while True:
            try:
                sys.stdin.flush()
                choice = input("  Type [CONFIRM] to approve, or [R]eject: ").strip()
                if not choice:
                    continue
                return choice == "CONFIRM"
            except (EOFError, KeyboardInterrupt):
                log.debug(f"[diff] Console hazard input raised exception {threading.current_thread().name}", exc_info=True)
                return False
    else:
        print("  [A]pprove  |  [S] Always Approve (don't ask again)  |  [R]eject")
        while True:
            try:
                sys.stdin.flush()
                choice = input("  Your choice: ").strip().lower()
                if not choice:
                    continue
                if choice in ("a", "y", "yes", "approve"):
                    return True
                elif choice in ("s", "always", "skip"):
                    _approve_all = True
                    return True
                elif choice in ("r", "n", "no", "reject"):
                    return False
                else:
                    print("  Invalid choice. Use A, S, or R.")
            except (EOFError, KeyboardInterrupt):
                log.debug(f"[diff] Console choice input raised exception {threading.current_thread().name}", exc_info=True)
                return False

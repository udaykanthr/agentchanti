import logging
import os
import re
import subprocess
import sys
import shutil
import tempfile
import threading
import time as _time
from datetime import datetime

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.padding import Padding
    from rich import box as rich_box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

VERSION = "1.0"

# ── Design tokens ─────────────────────────────────────────────────────────────
# Centralised so tweaking palette only requires changes here.
_CLR = {
    "brand":    "bold yellow",       # ⚡ AgentChanti name
    "brand_dim":"dim yellow",         # version / subtitle
    "task":     "white",              # task description
    "model":    "dim",                # model info
    "done":     "bold green",         # ✔ success
    "failed":   "bold red",           # ✘ failure
    "active":   "bold bright_yellow", # ◉ active step / spinner
    "pending":  "dim",                # ○ not started
    "skipped":  "dim",                # – skipped
    "badge_code":  "bold cyan",
    "badge_cmd":   "bold magenta",
    "badge_test":  "bold blue",
    "badge_other": "dim",
    "metric_label":  "dim",
    "metric_value":  "bold white",
    "metric_tokens": "cyan",
    "metric_total":  "bold cyan",
    "metric_cost":   "bold yellow",
    "bar_filled":    "green",
    "bar_empty":     "dim",
    "bar_warn":      "yellow",        # partial pass (50-99 %)
    "panel_border":  "yellow",        # header border
    "section_border":"bright_black",  # section borders (subtle)
    "fail_detail":   "dim red",
}


# ── Token Tracker ─────────────────────────────────────────────────────────────

class TokenTracker:
    """Global tracker for token usage and cost across all LLM calls."""

    def __init__(self, pricing: dict | None = None):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.pricing = pricing or {}
        self.current_context_size = 0
        self._lock = threading.Lock()

    def set_context(self, tokens: int) -> None:
        with self._lock:
            self.current_context_size = tokens

    def record(self, prompt_tokens: int, completion_tokens: int, model_name: str | None = None):
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.call_count += 1
            self.current_context_size = prompt_tokens

        if model_name:
            self._calculate_cost(model_name, prompt_tokens, completion_tokens)

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.total_prompt_tokens, self.total_completion_tokens

    def _calculate_cost(self, model_name: str, prompt: int, completion: int):
        price_entry = None
        for pattern, prices in self.pricing.items():
            if pattern in model_name.lower():
                price_entry = prices
                break
        if price_entry:
            cost = (prompt * price_entry["input"] / 1_000_000) + \
                   (completion * price_entry["output"] / 1_000_000)
            with self._lock:
                self.total_cost += cost

    @property
    def total_tokens(self):
        return self.total_prompt_tokens + self.total_completion_tokens


# Global singleton
token_tracker = TokenTracker()


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str = ".agentchanti/logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"agent_{timestamp}.log")

    logger = logging.getLogger("multi_agent_coder")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)
    return logger


log = setup_logger()


# ── Sanitizers (kept for backward compatibility) ──────────────────────────────

_GIBBERISH_RE = re.compile(
    r'<\|[^|>]*\|?>|'
    r'<<[^>]*>>|'
    r'\[\|[^|\]]*\|?\]|'
    r'<\/?s>|'
    r'\[INST\]|\[\/INST\]|'
    r'\[UNUSED_TOKEN_\d+\]'
)
_READABLE_RE = re.compile(r'[a-zA-Z0-9\s]')


def _sanitize_line(text: str) -> str:
    if not text:
        return ""
    original_len = len(text)
    cleaned = _GIBBERISH_RE.sub('', text).strip()
    if not cleaned:
        return ""
    if original_len > 10 and len(cleaned) / original_len < 0.4:
        return ""
    cleaned = cleaned.strip("'\"[](){}<>,;:|`").strip()
    if not cleaned:
        return ""
    readable = len(_READABLE_RE.findall(cleaned))
    if len(cleaned) > 3 and readable / len(cleaned) < 0.4:
        return ""
    return cleaned


# ── Rich Live Renderable ──────────────────────────────────────────────────────

class _LiveRenderable:
    """Thin wrapper so Rich.Live can call back into CLIDisplay._build_panels()."""

    def __init__(self, display: "CLIDisplay"):
        self._d = display

    def __rich_console__(self, console, options):
        for renderable in self._d._build_panels():
            yield renderable


# ── Helper: format numbers ────────────────────────────────────────────────────

def _fmt_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


def _fill_bar(fraction: float, width: int = 20,
              filled_char: str = "▰", empty_char: str = "▱") -> tuple[str, str]:
    """Return (filled_part, empty_part) strings for a progress bar."""
    clamped = max(0.0, min(1.0, fraction))
    n_filled = int(clamped * width)
    n_empty = width - n_filled
    return filled_char * n_filled, empty_char * n_empty


def _format_elapsed(elapsed: float) -> str:
    total = int(elapsed)
    hours, remainder = divmod(total, 3600)
    mins, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# ── Main Display Class ────────────────────────────────────────────────────────

class CLIDisplay:
    """Rich-based full-screen display for AgentChanti pipeline."""

    ICONS = {
        "pending": "·",
        "active":  "▶",
        "done":    "✓",
        "failed":  "✗",
        "skipped": "⊘",
    }

    # Kept so external code that reads _GIBBERISH_RE / _READABLE_RE still works
    _GIBBERISH_RE = _GIBBERISH_RE
    _READABLE_RE  = _READABLE_RE

    # Quarter-circle spinner — renders reliably across terminals
    _SPINNER_FRAMES = "◐◓◑◒"
    # Legacy list kept for any external references
    _WAITING_PHRASES = [
        "Waiting for response", "Still thinking", "Processing", "Working on it",
    ]

    def __init__(self, task_description: str):
        self.task = task_description
        self.steps: list[dict] = []
        self.current_step: int = -1
        self.status_message: str = ""
        self.start_time: float = _time.monotonic()
        self.paused: bool = False

        self._llm_log: list[str] = []
        self._test_results: list[dict] = []   # [{file, passed, total, failures, duration}]
        self._wave_info: tuple[int, int] = (0, 0)
        self._model_info: str = ""
        # Intent investigation trail (shown in INVESTIGATING panel)
        self._intent_events: list[dict] = []  # {kind, label, result_info, done}
        self._intent_iteration: tuple[int, int] = (0, 0)  # (current, max)
        self._intent_last_response: str = ""   # full LLM response, shown below events
        self._lock = threading.RLock()
        self._last_stream_render: float = 0.0

        # Backward-compat attributes that external code may read
        self._header_end = 4
        self._left_pane_width = 24
        self.term_width = shutil.get_terminal_size((80, 24)).columns
        self.term_height = shutil.get_terminal_size((80, 24)).lines

        # Spinner background thread (kept for backward compat; also drives refresh)
        self._spinner_stop = threading.Event()
        self._spinner_thread: threading.Thread | None = None

        if _RICH_AVAILABLE:
            self._console = Console()
            self._renderable = _LiveRenderable(self)
            self._live = Live(
                self._renderable,
                console=self._console,
                refresh_per_second=8,
                screen=True,
                vertical_overflow="visible",
            )
            self._live.start()
        else:
            self._live = None
            self._console = None

    # ── Spinner (background thread kept for backward compat) ──────────────────

    def _start_spinner(self, message: str = ""):
        """Start background refresh thread (drives Live spinner animation)."""
        self._stop_spinner()
        self._spinner_stop.clear()
        self._spinner_thread = threading.Thread(
            target=self._spinner_loop, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self):
        if self._spinner_thread and self._spinner_thread.is_alive():
            self._spinner_stop.set()
            self._spinner_thread.join(timeout=1.0)
        self._spinner_thread = None

    def stop_spinner(self):
        """Public: stop spinner (called before interactive prompts)."""
        self._stop_spinner()

    def _spinner_loop(self):
        """Keeps Live refreshing during long-running steps."""
        while not self._spinner_stop.is_set():
            if not self.paused and self._live:
                try:
                    self._live.refresh()
                except Exception:
                    pass
            self._spinner_stop.wait(0.15)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def pause(self):
        """Pause rendering for interactive prompts."""
        self._stop_spinner()
        with self._lock:
            self.paused = True
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass

    def resume(self):
        """Resume rendering after interactive prompts."""
        with self._lock:
            self.paused = False
        if self._live:
            try:
                self._live.start()
            except Exception:
                pass
        self.render()

    def render(self):
        """Trigger a display refresh."""
        if self.paused or not self._live:
            return
        try:
            self._live.refresh()
        except Exception:
            pass

    # ── State setters ─────────────────────────────────────────────────────────

    def set_model_info(self, model: str, provider: str):
        """Set model/provider for display in header."""
        with self._lock:
            self._model_info = f"{model} ({provider})"
        self.render()

    def set_wave_info(self, current: int, total: int):
        """Set current/total wave numbers for footer display."""
        with self._lock:
            self._wave_info = (current, total)
        self.render()

    def set_steps(self, step_texts: list[str]):
        self._stop_spinner()
        with self._lock:
            self.steps = [
                {"text": t, "status": "pending", "type": "?"}
                for t in step_texts
            ]
        self.render()

    def show_status(self, message: str):
        """Show a planning/status message.

        Works in both phases:
          • Pre-step (no steps loaded yet): renders as the sole PLANNING panel
            via the if/elif chain in ``_build_panels``.
          • Post-step (after all steps complete): renders as a STATUS footer
            panel below the execution section, so long post-pipeline phases
            like wiring verification do not appear frozen.

        Pass an empty string to clear the message and stop the spinner.
        """
        with self._lock:
            self.status_message = message
        self.render()
        if message:
            self._start_spinner(message)
        else:
            self._stop_spinner()

    # ── Intent investigation trail ─────────────────────────────────────────────

    _INTENT_ICONS = {
        "preseed": ("⚡", "yellow"),
        "think":   ("◉",  "bright_yellow"),
        "reason":  ("  ↳", "dim"),   # LLM reasoning before tool call
        "kb":      ("🔍", "cyan"),
        "web":     ("🌐", "blue"),
        "cmd":     ("💻", "magenta"),
        "usage":   ("🔗", "cyan"),
        "spec":    ("✅", "green"),
        "reject":  ("✗",  "red"),
        "detail":  ("  ·", "dim"),   # indented detail line under spec
    }

    def show_intent_event(self, kind: str, label: str, result_info: str = "",
                          iteration: int = 0, max_iterations: int = 0):
        """Append a new row to the investigation trail panel.

        *kind* selects the icon (preseed/kb/web/cmd/usage/spec/reject).
        *label* is the short description shown on the left.
        *result_info* (optional) is appended on the right once the result arrives.
        Call update_last_intent_event() to fill in result_info after the fact.
        """
        with self._lock:
            self._intent_events.append({
                "kind": kind,
                "label": label,
                "result_info": result_info,
            })
            if iteration or max_iterations:
                self._intent_iteration = (iteration, max_iterations)
            self.status_message = label
        self.render()
        self._start_spinner(label)

    def update_last_intent_event(self, result_info: str):
        """Set result_info on the most recently added intent event."""
        with self._lock:
            if self._intent_events:
                self._intent_events[-1]["result_info"] = result_info
        self.render()

    def set_intent_response(self, text: str):
        """Store the latest full LLM response for display in the investigation panel."""
        with self._lock:
            self._intent_last_response = text
        self.render()

    def clear_intent_events(self):
        """Remove investigation trail (called when planning phase ends)."""
        with self._lock:
            self._intent_events = []
            self._intent_iteration = (0, 0)
            self._intent_last_response = ""

    def start_step(self, index: int, step_type: str = "?"):
        with self._lock:
            self.current_step = index
            self.steps[index]["status"] = "active"
            self.steps[index]["type"] = step_type
            self.steps[index]["info"] = []
            if "start_time" not in self.steps[index]:
                self.steps[index]["start_time"] = _time.monotonic()
            if "tokens" not in self.steps[index]:
                self.steps[index]["tokens"] = {"sent": 0, "recv": 0}
        self.render()

    def step_info(self, index: int, message: str):
        self._stop_spinner()
        with self._lock:
            if 0 <= index < len(self.steps):
                info_list = self.steps[index].get("info", [])
                if len(info_list) >= 5:
                    info_list.pop(0)
                info_list.append(message)
                self.steps[index]["info"] = info_list
        self.render()
        if any(kw in message.lower() for kw in (
            "generating", "coding", "classifying", "reviewing",
            "analyzing", "requesting", "running", "installing",
            "re-planning", "retrying", "searching", "sending",
            "resolving", "diagnosing", "fixing", "auto-fixing",
            "pre-install", "building", "processing", "waiting",
            "loading", "applying", "scoping", "testing",
        )):
            self._start_spinner(message)

    def step_tokens(self, index: int, sent: int, recv: int):
        with self._lock:
            if 0 <= index < len(self.steps):
                t = self.steps[index].get("tokens", {"sent": 0, "recv": 0})
                t["sent"] += sent
                t["recv"] += recv
                self.steps[index]["tokens"] = t
        self.render()

    def complete_step(self, index: int, status: str = "done"):
        self._stop_spinner()
        with self._lock:
            self.steps[index]["status"] = status
            if "start_time" in self.steps[index]:
                self.steps[index]["duration"] = (
                    _time.monotonic() - self.steps[index]["start_time"]
                )
        self.render()

    def add_llm_log(self, text: str, source: str = ""):
        added = False
        with self._lock:
            if source:
                self._llm_log.append(f"[{source}]")
            for line in text.splitlines():
                cleaned = _sanitize_line(line.strip())
                if cleaned:
                    self._llm_log.append(cleaned)
                    added = True
            if added:
                self._llm_log.append("")
        if added:
            self.render()

    def record_test_result(self, file_path: str, passed: int, total: int,
                           failures: list[dict] | None = None,
                           duration: float = 0.0):
        """Record a per-file test result for the TEST RESULTS panel.

        *failures* is a list of dicts with keys ``name`` and ``message``.
        Call this from step_handlers after each test file run.
        """
        with self._lock:
            # Update existing entry if same file re-runs (retries)
            for existing in self._test_results:
                if existing["file"] == file_path:
                    existing["passed"] = passed
                    existing["total"] = total
                    existing["failures"] = failures or []
                    existing["duration"] = duration
                    break
            else:
                self._test_results.append({
                    "file": file_path,
                    "passed": passed,
                    "total": total,
                    "failures": failures or [],
                    "duration": duration,
                })
        self.render()

    def update_streaming_progress(self, step_idx: int, tokens: int):
        now = _time.monotonic()
        if now - self._last_stream_render < 0.5:
            return
        self._last_stream_render = now
        self.step_info(step_idx, f"Generating... ({tokens} tokens)")

    def budget_check(self, limit: float) -> bool:
        if limit > 0 and token_tracker.total_cost >= limit:
            log.error(f"Budget exceeded: ${token_tracker.total_cost:.4f} >= ${limit:.2f}")
            return True
        return False

    # ── Rich panel builders ───────────────────────────────────────────────────

    def _spinner_char(self) -> str:
        elapsed = _time.monotonic() - self.start_time
        return self._SPINNER_FRAMES[int(elapsed * 8) % len(self._SPINNER_FRAMES)]

    def _build_header(self) -> Panel:
        task_preview = " ".join(self.task.split())
        if len(task_preview) > 110:
            task_preview = task_preview[:107] + "…"

        # ── Line 1: brand + version + model ──
        line1 = Text()
        line1.append("⚡ ", style="yellow")
        line1.append("AgentChanti", style=_CLR["brand"])
        line1.append(f"  v{VERSION}", style=_CLR["brand_dim"])
        if self._model_info:
            line1.append("   ·   ", style="dim")
            line1.append(self._model_info, style=_CLR["model"])

        # ── Line 2: task description ──
        line2 = Text()
        line2.append("   ", style="")
        line2.append(task_preview, style=_CLR["task"])

        # ── Line 3: live metrics (replaces the separate footer panel) ──
        t = token_tracker
        elapsed   = _time.monotonic() - self.start_time
        p, c      = t.snapshot()
        total_tok = p + c

        with self._lock:
            steps     = list(self.steps)
            wave_info = self._wave_info
        done_steps  = sum(1 for s in steps if s["status"] in ("done", "skipped", "failed"))
        total_steps = len(steps)

        line3 = Text()
        line3.append("   ", style="")           # align under ⚡
        line3.append("⏱ ", style="dim")
        line3.append(_format_elapsed(elapsed), style=_CLR["metric_value"])
        line3.append("   ↑ ", style=_CLR["metric_label"])
        line3.append(_fmt_k(p), style=_CLR["metric_tokens"])
        line3.append("  ↓ ", style=_CLR["metric_label"])
        line3.append(_fmt_k(c), style=_CLR["metric_tokens"])
        line3.append("  Σ ", style=_CLR["metric_label"])
        line3.append(_fmt_k(total_tok), style=_CLR["metric_total"])
        line3.append(f"   {t.call_count} calls", style="dim")
        if t.total_cost > 0:
            line3.append("   ", style="")
            line3.append(f"${t.total_cost:.4f}", style=_CLR["metric_cost"])
        if total_steps:
            line3.append(f"   ·   Steps {done_steps}/{total_steps}", style="dim")
        if wave_info[1] > 0:
            line3.append(f"  ·  Wave {wave_info[0]}/{wave_info[1]}", style="dim")

        return Panel(
            Group(line1, line2, line3),
            border_style=_CLR["panel_border"],
            box=rich_box.ROUNDED,
            padding=(0, 1),
        )

    def _build_planning_section(self, title_text: str = "PLANNING") -> Panel:
        spin = self._spinner_char()
        elapsed = _time.monotonic() - self.start_time
        t = Text()
        t.append(f" {spin} ", style=_CLR["active"])
        if self.status_message:
            t.append(f" {self.status_message}", style="white")
        t.append(f"  ·  {_format_elapsed(elapsed)}", style="dim")

        title = Text()
        title.append("◈ ", style="dim cyan")
        title.append(title_text, style="bold cyan")

        return Panel(
            t,
            title=title,
            title_align="left",
            border_style=_CLR["section_border"],
            box=rich_box.ROUNDED,
            padding=(0, 2),
        )

    def _build_execution_section(self) -> Panel:
        with self._lock:
            steps = list(self.steps)
            current = self.current_step

        total = len(steps)
        done = sum(1 for s in steps if s["status"] in ("done", "skipped", "failed"))
        frac = done / total if total else 0
        filled, empty = _fill_bar(frac, width=18)

        # Determine visible window (keep active step in view)
        MAX_VISIBLE = 18
        start_row = 0
        if total > MAX_VISIBLE:
            pivot = current if current >= 0 else done
            start_row = max(0, pivot - MAX_VISIBLE // 2)
            start_row = min(start_row, total - MAX_VISIBLE)
        visible_steps = list(enumerate(steps))[start_row:start_row + MAX_VISIBLE]

        table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        table.add_column("icon",     width=3,  no_wrap=True)
        table.add_column("type",     width=7,  no_wrap=True)
        table.add_column("desc",     ratio=1)
        table.add_column("activity", width=30, no_wrap=True)
        table.add_column("time",     width=7,  no_wrap=True)
        table.add_column("tokens",   width=15, no_wrap=True)

        spin = self._spinner_char()

        # Badge colours
        _BADGE = {
            "CODE": _CLR["badge_code"], "CMD": _CLR["badge_cmd"],
            "TEST": _CLR["badge_test"], "SEARCH": "dim yellow",
            "IGNORE": "dim",
        }

        for i, step in visible_steps:
            status = step["status"]
            icon_raw = self.ICONS.get(status, "·")
            _ICON_STYLE = {
                "pending": _CLR["pending"],
                "active":  _CLR["active"],
                "done":    _CLR["done"],
                "failed":  _CLR["failed"],
                "skipped": _CLR["skipped"],
            }
            icon = Text(f" {icon_raw}", style=_ICON_STYLE.get(status, "dim"))

            # Type badge  TEST  CODE  CMD
            stype = step.get("type", "?")
            if stype and stype not in ("?", "UNCLASSIFIED"):
                type_text = Text(f" {stype:<4} ", style=_BADGE.get(stype, "dim"))
            else:
                type_text = Text("")

            # Description — active step uses brighter white
            raw_desc = step.get("text", f"Step {i + 1}")
            if status == "active":
                desc = Text(raw_desc, style="bold white", no_wrap=True, overflow="ellipsis")
            elif status == "done":
                desc = Text(raw_desc, style="white", no_wrap=True, overflow="ellipsis")
            elif status == "failed":
                desc = Text(raw_desc, style="red", no_wrap=True, overflow="ellipsis")
            else:
                desc = Text(raw_desc, style="dim", no_wrap=True, overflow="ellipsis")

            # Activity column — only for the active step
            activity = Text("")
            if status == "active":
                step_elapsed = _time.monotonic() - step.get("start_time", _time.monotonic())
                elapsed_str = _format_elapsed(step_elapsed)
                info_list = step.get("info", [])
                last_info = info_list[-1] if info_list else "working…"
                if len(last_info) > 18:
                    last_info = last_info[:15] + "…"
                activity = Text()
                activity.append(f"{spin} ", style=_CLR["active"])
                activity.append(elapsed_str, style="dim yellow")
                activity.append(f"  {last_info}", style="dim white")

            # Duration (completed steps only)
            time_text = Text("")
            if status in ("done", "failed", "skipped") and "duration" in step:
                time_text = Text(_format_elapsed(step["duration"]), style="dim")

            # Per-step token counts
            tokens_text = Text("")
            tok = step.get("tokens")
            if tok and (tok["sent"] or tok["recv"]):
                tokens_text = Text()
                tokens_text.append(f"↑{_fmt_k(tok['sent'])}", style=_CLR["metric_tokens"])
                tokens_text.append(" ", style="")
                tokens_text.append(f"↓{_fmt_k(tok['recv'])}", style="dim cyan")

            table.add_row(icon, type_text, desc, activity, time_text, tokens_text)

        # Title: section label + inline progress bar
        title = Text()
        title.append("▸ ", style="dim")
        title.append("EXECUTION", style="bold white")
        title.append("  ", style="")
        title.append(filled, style=_CLR["bar_filled"])
        title.append(empty,  style=_CLR["bar_empty"])
        title.append(f"  {done}/{total}", style="dim")
        if total > MAX_VISIBLE:
            title.append(f"  ({start_row+1}–{start_row+len(visible_steps)})", style="dim")

        return Panel(
            table,
            title=title,
            title_align="left",
            border_style=_CLR["section_border"],
            box=rich_box.ROUNDED,
            padding=(0, 0),
        )

    def _build_tests_section(self) -> Panel:
        with self._lock:
            results = list(self._test_results)

        table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        table.add_column("icon",   width=3,  no_wrap=True)
        table.add_column("badge",  width=7,  no_wrap=True)
        table.add_column("file",   ratio=1)
        table.add_column("bar",    width=13, no_wrap=True)
        table.add_column("count",  width=7,  no_wrap=True)
        table.add_column("pct",    width=5,  no_wrap=True)
        table.add_column("time",   width=7,  no_wrap=True)

        for result in results:
            passed   = result.get("passed", 0)
            total    = result.get("total", 0) or 1
            duration = result.get("duration", 0.0)
            fpath    = result.get("file", "?")
            failures = result.get("failures", [])
            failed   = total - passed
            frac     = passed / total
            pct      = int(frac * 100)

            # Bar: ▰▰▰▰▰▱▱▱▱▱
            filled, empty = _fill_bar(frac, width=10)
            bar_t = Text()
            if failed == 0:
                bar_t.append(filled, style=_CLR["bar_filled"])
                bar_t.append(empty,  style=_CLR["bar_empty"])
                icon        = Text(" ✓", style=_CLR["done"])
                badge       = Text(" PASS ", style="bold green")
                file_style  = "green"
                pct_style   = "green"
            elif pct >= 50:
                bar_t.append(filled, style=_CLR["bar_warn"])
                bar_t.append(empty,  style="dim red")
                icon        = Text(" ✗", style="bold yellow")
                badge       = Text(" WARN ", style="bold yellow")
                file_style  = "yellow"
                pct_style   = "yellow"
            else:
                bar_t.append(filled, style="dim yellow")
                bar_t.append(empty,  style=_CLR["failed"])
                icon        = Text(" ✗", style=_CLR["failed"])
                badge       = Text(" FAIL ", style="bold red")
                file_style  = "red"
                pct_style   = "red"

            # Show only the filename; dim the directory prefix
            if "/" in fpath or "\\" in fpath:
                sep    = fpath.rfind("/") if "/" in fpath else fpath.rfind("\\")
                prefix = fpath[:sep + 1]
                fname  = fpath[sep + 1:]
            else:
                prefix = ""
                fname  = fpath

            file_text = Text(no_wrap=True, overflow="ellipsis")
            if prefix:
                file_text.append(prefix, style="dim")
            file_text.append(fname, style=file_style)

            table.add_row(
                icon, badge, file_text, bar_t,
                Text(f"{passed}/{total}", style="dim"),
                Text(f"{pct}%",           style=pct_style),
                Text(_format_elapsed(duration), style="dim"),
            )

            # Failure details (max 3 per file, indented)
            for failure in failures[:3]:
                name = failure.get("name", "")
                msg  = failure.get("message", "")
                if len(msg) > 50:
                    msg = msg[:47] + "…"
                detail = Text()
                detail.append("     └─ ", style="dim")
                detail.append(name, style="red")
                if msg:
                    detail.append(f"  {msg}", style=_CLR["fail_detail"])
                table.add_row(
                    Text(""), Text(""), detail,
                    Text(""), Text(""), Text(""), Text(""),
                )

        title = Text()
        title.append("◈ ", style="dim")
        title.append("TEST RESULTS", style="bold white")

        return Panel(
            table,
            title=title,
            title_align="left",
            border_style=_CLR["section_border"],
            box=rich_box.ROUNDED,
            padding=(0, 0),
        )

    def _build_investigation_section(self) -> Panel:
        """Rich panel showing the IntentAgent's investigation trail."""
        with self._lock:
            events = list(self._intent_events)
            iteration, max_iter = self._intent_iteration

        spin = self._spinner_char()
        table = Table(box=None, show_header=False, padding=(0, 1), expand=True)
        table.add_column("icon",   width=3,  no_wrap=True)
        table.add_column("label",  ratio=1)
        table.add_column("result", width=28, no_wrap=True)

        for ev in events:
            kind        = ev.get("kind", "kb")
            label       = ev.get("label", "")
            result_info = ev.get("result_info", "")

            icon_char, icon_color = self._INTENT_ICONS.get(kind, ("·", "dim"))
            icon_text = Text(f" {icon_char}", style=icon_color)

            # Truncate long labels
            if len(label) > 72:
                label = label[:69] + "…"

            if kind in ("detail", "reason"):
                label_style = "dim"
            elif kind == "spec":
                label_style = "bold green"
            elif kind == "reject":
                label_style = "red"
            elif kind == "think":
                label_style = "bright_yellow"
            else:
                label_style = "white"

            label_text = Text(label, style=label_style, no_wrap=True,
                              overflow="ellipsis")
            result_text = Text(
                result_info[:26] if result_info else "",
                style="dim", no_wrap=True)
            table.add_row(icon_text, label_text, result_text)

        # Full LLM response sub-panel (shown below the event trail)
        with self._lock:
            last_response = self._intent_last_response

        body_parts = [table]
        if last_response:
            resp_text = Text(last_response, style="dim", no_wrap=False)
            resp_panel = Panel(
                resp_text,
                title=Text("LLM Response", style="dim cyan"),
                title_align="left",
                border_style="dim",
                box=rich_box.SIMPLE,
                padding=(0, 1),
            )
            body_parts.append(resp_panel)

        # Title with iteration counter and spinner
        title = Text()
        title.append("◈ ", style="dim cyan")
        title.append("INVESTIGATING", style="bold cyan")
        if iteration:
            title.append(f"  iteration {iteration}", style="dim")
        title.append(f"  {spin}", style=_CLR["active"])

        return Panel(
            Group(*body_parts),
            title=title,
            title_align="left",
            border_style="cyan",
            box=rich_box.ROUNDED,
            padding=(0, 0),
        )

    def _build_panels(self) -> list:
        """Assemble all Rich renderables for the current state."""
        parts = []
        parts.append(self._build_header())

        with self._lock:
            has_steps       = bool(self.steps)
            has_status      = bool(self.status_message)
            has_tests       = bool(self._test_results)
            has_intent      = bool(self._intent_events)

        if has_steps:
            parts.append(self._build_execution_section())
        elif has_intent:
            parts.append(self._build_investigation_section())
        elif has_status:
            parts.append(self._build_planning_section())

        # Post-step status footer.  After all steps complete, long-running
        # phases like wiring verification or learning extraction call
        # show_status() to indicate progress.  Render those messages as a
        # STATUS panel below the execution section so the UI does not look
        # frozen during 60-90s LLM calls.
        if has_steps and has_status:
            parts.append(self._build_planning_section(title_text="STATUS"))

        if has_tests:
            parts.append(self._build_tests_section())

        return parts

    # ── Finish screen ─────────────────────────────────────────────────────────

    def finish(self, success: bool = True):
        """Show a final summary screen after pipeline completes."""
        self._stop_spinner()
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass

        t = token_tracker
        elapsed = _time.monotonic() - self.start_time

        if _RICH_AVAILABLE:
            console = Console()

            status_text = Text()
            if success:
                status_text.append("  ✓  ", style="bold green")
                status_text.append("All tasks completed successfully!", style="bold white")
            else:
                status_text.append("  ✗  ", style="bold red")
                status_text.append("Some tasks failed — check logs for details.", style="bold white")

            metrics = Text()
            metrics.append("  Tokens  ", style="dim")
            metrics.append(f"{t.total_tokens:,}", style="bold cyan")
            metrics.append("  ·  ↑ ", style="dim")
            metrics.append(f"{t.total_prompt_tokens:,}", style="cyan")
            metrics.append("  ↓ ", style="dim")
            metrics.append(f"{t.total_completion_tokens:,}", style="cyan")

            time_line = Text()
            time_line.append("  Time    ", style="dim")
            time_line.append(_format_elapsed(elapsed), style="bold white")
            if t.total_cost > 0:
                time_line.append("   ·   Cost ", style="dim")
                time_line.append(f"${t.total_cost:.4f}", style="bold yellow")

            title = Text()
            title.append("⚡ ", style="yellow")
            title.append("AgentChanti", style="bold yellow")
            title.append("  —  Done", style="dim")

            summary = Group(Text(""), status_text, Text(""), metrics, time_line, Text(""))
            console.print(Panel(summary, title=title, border_style="yellow",
                                box=rich_box.ROUNDED, padding=(0, 2)))
        else:
            symbol = "✔" if success else "✘"
            print(f"\n{symbol} Done  |  {_format_elapsed(elapsed)}"
                  f"  |  Tokens: {t.total_tokens:,}"
                  + (f"  |  Cost: ${t.total_cost:.4f}" if t.total_cost > 0 else ""))

    # ── Static helpers (backward compat) ──────────────────────────────────────

    @classmethod
    def _sanitize_line(cls, text: str) -> str:
        return _sanitize_line(text)

    @staticmethod
    def extract_explanation(response: str) -> str:
        lines = response.splitlines()
        result = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code or stripped.startswith("#### [FILE]:"):
                continue
            cleaned = _sanitize_line(stripped)
            if cleaned:
                result.append(cleaned)
        return "\n".join(result)

    @staticmethod
    def _format_elapsed(elapsed: float) -> str:
        return _format_elapsed(elapsed)

    # ── Interactive prompts ───────────────────────────────────────────────────

    @staticmethod
    def prompt_plan_approval(steps: list[str],
                             use_tui: bool = False) -> tuple[str, list[int], list[str] | None]:
        print("\n" + "=" * 60)
        print("  PROPOSED PLAN")
        print("=" * 60)
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print("=" * 60)
        print("  [A]pprove  |  [R]eplan  |  [E]dit (TUI)  |  [T]ext editor")
        print()

        while True:
            choice = input("  Your choice: ").strip().lower()
            if choice in ("a", "approve"):
                return "approve", [], None
            elif choice in ("r", "replan"):
                return "replan", [], None
            elif choice in ("e", "edit"):
                try:
                    from .tui_editor import launch_tui_editor
                    edited = launch_tui_editor(steps)
                    if edited:
                        return "edit", [], edited
                    print("  Edit cancelled or no changes.")
                except Exception as e:
                    print(f"  TUI editor failed ({e}). Try [T] for text editor.")
                    log.warning(f"TUI editor exception: {e}")
                print()
                print("  [A]pprove  |  [R]eplan  |  [E]dit (TUI)  |  [T]ext editor")
                print()
            elif choice in ("t", "text"):
                edited = CLIDisplay._edit_plan_in_editor(steps)
                if edited:
                    return "edit", [], edited
                else:
                    print("  No changes detected or empty plan.")
            else:
                print("  Invalid choice. Use A, R, E, or T.")

    @staticmethod
    def _edit_plan_in_editor(steps: list[str]) -> list[str] | None:
        content = "# Edit the plan below. One step per line.\n"
        content += "# Lines starting with '#' are ignored.\n"
        content += "# You may add, remove, or reorder steps.\n\n"
        for i, step in enumerate(steps, 1):
            content += f"{i}. {step}\n"

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="plan_", delete=False, encoding="utf-8"
        )
        try:
            tmp.write(content)
            tmp.close()
            editor = "notepad" if os.name == "nt" else os.environ.get("EDITOR", "vi")
            print(f"\n  Opening plan in {editor}...")
            print("  Save and close the editor when done.\n")
            subprocess.call([editor, tmp.name])
            with open(tmp.name, "r", encoding="utf-8") as f:
                edited_content = f.read()
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        edited_steps: list[str] = []
        for line in edited_content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"^\d+\.\s*", "", line)
            if line:
                edited_steps.append(line)
        return edited_steps if edited_steps else None

    @staticmethod
    def prompt_resume(checkpoint_state: dict) -> bool:
        print("\n" + "=" * 60)
        print("  CHECKPOINT FOUND")
        print("=" * 60)
        print(f"  Task: {checkpoint_state.get('task', '?')}")
        completed = checkpoint_state.get("completed_step", -1)
        total = len(checkpoint_state.get("steps", []))
        print(f"  Progress: {completed + 1}/{total} steps completed")
        print(f"  Language: {checkpoint_state.get('language', '?')}")
        print("=" * 60)
        print("  [R]esume  |  [S]tart fresh")
        print()

        while True:
            choice = input("  Your choice: ").strip().lower()
            if choice in ("r", "resume"):
                return True
            elif choice in ("s", "start", "fresh"):
                return False
            else:
                print("  Invalid choice. Use R or S.")

    @staticmethod
    def prompt_git_action(action: str) -> str:
        print("\n" + "=" * 60)
        if action == "complete":
            print("  TASK COMPLETED — Git Options")
            print("=" * 60)
            print("  [C]ommit changes  |  [S]kip (leave uncommitted)")
        else:
            print("  TASK FAILED — Git Options")
            print("=" * 60)
            print("  [R]ollback to checkpoint  |  [C]ommit as-is  |  [S]kip")
        print()

        while True:
            choice = input("  Your choice: ").strip().lower()
            if choice in ("c", "commit"):
                return "commit"
            elif choice in ("r", "rollback") and action != "complete":
                return "rollback"
            elif choice in ("s", "skip"):
                return "skip"
            else:
                print("  Invalid choice. Try again.")

"""
Bounded agent micro-loop — tool-calling step execution.

When ``agent_loop: true`` is configured and the provider supports native
tool calling, CODE and TEST steps run through this loop instead of the
generate → review → retry pipeline: the model edits files and runs
commands via :class:`~agentchanti.agent_tools.AgentTools`, observes real
execution output, and self-corrects — capped at a fixed number of turns
so cost stays predictable.

The system prompt below is deliberately byte-identical across all steps
of a run so provider prompt caches and local KV caches get a stable
prefix.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from threading import Lock

from ..agent_tools import AgentTools
from ..llm.chat_types import Message, ToolCall

_logger = logging.getLogger(__name__)


def truncate_middle(text: str, limit: int) -> str:
    """Length-cap *text* while keeping BOTH ends.

    Error output puts the conclusion at the end — a Python traceback names
    the exception on its last line — so a plain ``text[:limit]`` slice
    hands the model everything EXCEPT the actual error (observed: a
    recovery loop spent its whole budget on read-only turns hunting for a
    ``NoReverseMatch`` that the head-slice had cut off). Keeps ~1/4 head
    for the command/context and ~3/4 tail for the failure itself.
    """
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return (text[:head]
            + f"\n... [{len(text) - limit} chars truncated] ...\n"
            + text[-tail:])


# ── Telemetry ─────────────────────────────────────────────────────────
# Per-run loop statistics, consumed by the CLI summary and the A/B
# benchmark harness. Thread-safe: steps in the same wave run in parallel.

_stats_lock = Lock()
_loop_runs: list[dict] = []


def _record_loop_run(step_idx: int, turns_used: int,
                     tool_counts: Counter, outcome: str,
                     recovery: bool) -> None:
    with _stats_lock:
        _loop_runs.append({
            "step_idx": step_idx,
            "turns": turns_used,
            "tool_calls": dict(tool_counts),
            "outcome": outcome,
            "recovery": recovery,
        })
    _logger.info(
        "[AgentLoop] stats: step=%d turns=%d outcome=%s recovery=%s tools=%s",
        step_idx + 1, turns_used, outcome, recovery, dict(tool_counts))


def get_loop_stats() -> list[dict]:
    """All loop runs recorded in this process (copy)."""
    with _stats_lock:
        return [dict(r) for r in _loop_runs]


def reset_loop_stats() -> None:
    with _stats_lock:
        _loop_runs.clear()


# ── Cross-attempt memory ──────────────────────────────────────────────
# The retry ladder (loop → escalation → recovery → recovery escalation)
# gave each attempt a blank conversation carrying only the previous
# attempt's error string. It never learned what the previous attempt
# actually DID, so every attempt re-derived the same hypotheses and
# re-edited the same files. Observed on a Pygame run: four attempts,
# 8 turns each — 54 turns and 497k tokens — churning src/ghost.py,
# src/game.py and src/player.py with conflicting fixes, none of them
# finding the two-line bug. Record each attempt's edits, commands and
# verdict so the next one starts knowing which doors are already shut.

_JOURNAL_MAX_FILES = 8       # per attempt, in the rendered digest
_JOURNAL_MAX_COMMANDS = 4    # per attempt, most recent first
_JOURNAL_MAX_ATTEMPTS = 4    # most recent attempts shown
_JOURNAL_SUMMARY_CHARS = 240

_journal_lock = Lock()
_attempts: dict[int, list[dict]] = {}


def record_attempt(step_idx: int, label: str, outcome: str,
                   edited: list[str], commands: list[tuple[str, bool]],
                   summary: str) -> None:
    """Append one finished attempt to *step_idx*'s journal."""
    with _journal_lock:
        _attempts.setdefault(step_idx, []).append({
            "label": label,
            "outcome": outcome,
            "edited": list(edited),
            "commands": list(commands),
            "summary": (summary or "").strip(),
        })


def get_attempts(step_idx: int) -> list[dict]:
    with _journal_lock:
        return [dict(a) for a in _attempts.get(step_idx, [])]


def reset_attempt_journal() -> None:
    with _journal_lock:
        _attempts.clear()


def attempt_digest(step_idx: int) -> str:
    """Compact record of prior attempts, for the next attempt's context.

    Empty string when nothing has been tried yet, so callers can append
    it unconditionally. Deliberately terse: this rides in the opening
    user message of every retry, and the whole point of the ladder is
    that budget is already scarce by the time it is consulted.
    """
    attempts = get_attempts(step_idx)
    if not attempts:
        return ""

    lines = ["Previous attempts at this step ALL FAILED. What was already "
             "tried (do not simply repeat it):"]
    shown = attempts[-_JOURNAL_MAX_ATTEMPTS:]
    offset = len(attempts) - len(shown)
    for i, att in enumerate(shown, start=offset + 1):
        lines.append(f"  attempt {i} — {att['label']} — {att['outcome']}")

        edited = att["edited"]
        if edited:
            counts = Counter(edited)
            shown_files = counts.most_common(_JOURNAL_MAX_FILES)
            rendered = ", ".join(
                f"{path} (x{n})" if n > 1 else path
                for path, n in shown_files)
            if len(counts) > _JOURNAL_MAX_FILES:
                rendered += f", +{len(counts) - _JOURNAL_MAX_FILES} more"
            lines.append(f"    edited: {rendered}")
        else:
            lines.append("    edited: nothing")

        for cmd, ok in att["commands"][-_JOURNAL_MAX_COMMANDS:]:
            lines.append(f"    ran: {cmd[:160]} -> "
                         f"{'ok' if ok else 'FAILED'}")

        if att["summary"]:
            lines.append("    concluded: "
                         + truncate_middle(att["summary"],
                                           _JOURNAL_SUMMARY_CHARS)
                           .replace("\n", " "))

    # The single most useful signal: the same files keep being rewritten
    # and the gate stays red, so the cause is somewhere nobody has looked.
    all_edited = Counter(f for a in attempts for f in a["edited"])
    repeated = [f for f, n in all_edited.items() if n >= 2]
    if len(attempts) >= 2 and repeated:
        lines.append(
            "  NOTE: " + ", ".join(sorted(repeated)[:6])
            + " have been edited across multiple failed attempts. Re-editing "
              "them the same way will not work — look for the cause "
              "somewhere the previous attempts did not read.")
    return "\n".join(lines)


def loop_stats_summary() -> str | None:
    """One-line human summary for the end-of-run log, or None if unused."""
    runs = get_loop_stats()
    if not runs:
        return None
    outcomes = Counter(r["outcome"] for r in runs)
    total_turns = sum(r["turns"] for r in runs)
    recoveries = sum(1 for r in runs if r["recovery"])
    return (f"[AgentLoop] session: {len(runs)} loop run(s), "
            f"{total_turns} total turns "
            f"(avg {total_turns / len(runs):.1f}), "
            f"{recoveries} recovery run(s), outcomes: {dict(outcomes)}")


# Tools that inspect without changing anything — used to detect loops
# stuck in analysis mode.
_READ_ONLY_TOOLS = frozenset({"read_file", "list_files", "search_code"})

# Read-only intervention thresholds (consecutive inspection-only turns).
# Lowered from 3/4 → 2/3: with the step's target files pre-loaded into the
# opening message, the model rarely needs several read_file turns, so nudge
# it to act one turn sooner and stop paying to re-send inspection output on
# every subsequent turn.
_ACT_NOW_NUDGE_AT = 2
_WITHHOLD_READONLY_AT = 3

# Pre-load caps: keep the injected file bundle from dominating the prompt.
_PRELOAD_MAX_FILES = 6
_PRELOAD_MAX_CHARS = 12_000

# Stable prefix — keep byte-identical across steps (see module docstring).
# Step-specific data (task, context, platform quirks) belongs in the user
# message, never here.
AGENT_LOOP_SYSTEM_PROMPT = """\
You are a coding agent executing one step of a larger implementation plan.
Work autonomously with the provided tools until the step is complete.

Rules:
- Read a file before editing it; base edits on its actual current content.
- Prefer edit_file for changes to existing files; write_file only for new \
files or full rewrites.
- After making changes, verify them: run the relevant command or test and \
check its output. Do not claim success without evidence.
- Stay within the scope of this step. Do not refactor unrelated code.
- If a command fails, read the error and fix the cause; do not retry the \
same command unchanged.
- When the step is complete and verified, reply with a short plain-text \
summary (no tool calls). If you cannot complete it, explain what is blocking.
"""


def _platform_note() -> str:
    """Environment line for the loop's user message.

    Lives in the user message, not the system prompt, so the system
    prompt stays byte-identical across platforms (see module docstring).
    Windows needs it spelled out: an observed loop burned a turn on
    ``sed -n '1,200p' file`` (exit 1 — no sed on Windows) instead of
    calling read_file.
    """
    import os
    if os.name == "nt":
        return ("Environment: Windows (cmd.exe shell). POSIX text tools "
                "(sed, awk, grep, cat, head, tail, ls) are NOT available "
                "in run_command — use read_file / edit_file / search_code "
                "for file inspection and changes.")
    return ""


def _build_user_message(step_text: str, task: str, language: str | None,
                        context: str, preloaded: str = "") -> str:
    parts = [f"Overall task: {task}", f"Current step: {step_text}"]
    if language:
        parts.append(f"Project language: {language}")
    note = _platform_note()
    if note:
        parts.append(note)
    if context:
        parts.append(f"Project state:\n{context}")
    if preloaded:
        parts.append(preloaded)
    return "\n\n".join(parts)


def _preload_target_files(tools: AgentTools,
                          paths: list[str] | None) -> str:
    """Read the step's existing target files up front so the loop doesn't
    burn its first turns on read_file round-trips — whose output then rides
    along, re-sent, in every later turn.

    Only existing, non-empty files inside the project root are included;
    files the step will *create* are skipped (nothing to read yet). The
    bundle goes into the opening user message so it lands in the cacheable
    prefix instead of growing the conversation mid-loop.
    """
    if not paths:
        return ""
    seen: set[str] = set()
    blocks: list[str] = []
    total = 0
    for path in paths:
        rel = str(path).replace("\\", "/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        try:
            full = tools._resolve(path)
        except (ValueError, TypeError):
            continue  # outside the project root — skip
        if not os.path.isfile(full) or os.path.getsize(full) == 0:
            continue
        body = tools._tool_read_file(path)
        if body.startswith("ERROR"):
            continue
        if total + len(body) > _PRELOAD_MAX_CHARS:
            break
        total += len(body)
        blocks.append(body)
        if len(blocks) >= _PRELOAD_MAX_FILES:
            break
    if not blocks:
        return ""
    return ("Relevant existing files (already read for you — do NOT call "
            "read_file on these again):\n\n" + "\n\n".join(blocks))


def _cd_prefix(cmd: str | None) -> str:
    """The ``cd <dir> && `` prefix of *cmd* when it has one, else ``""``.

    Verify commands for sub-project layouts arrive as ``cd app && npm
    test``; an install that heals that verify must run in the same
    directory or it lands in the wrong package/venv.
    """
    if cmd:
        m = re.match(r"^(cd\s+\S+\s*&&\s*)", cmd)
        if m:
            return m.group(1)
    return ""


def _venv_python(project_root: str) -> str | None:
    """Absolute path to the project's own venv interpreter, or None.

    Dependency installs must land in the interpreter the gates run under
    (``venv\\Scripts\\python.exe``), not whatever bare ``python`` resolves
    to on PATH — installing into the system interpreter leaves the venv
    (and therefore the verify gate) still missing the package, while the
    test command run under system Python "passes" in the wrong place.
    """
    for rel in (os.path.join("venv", "Scripts", "python.exe"),
                os.path.join(".venv", "Scripts", "python.exe"),
                os.path.join("venv", "bin", "python"),
                os.path.join(".venv", "bin", "python")):
        cand = os.path.join(project_root, rel)
        if os.path.isfile(cand):
            return cand
    return None


def _flagless_tokens(cmd: str) -> list[str]:
    """Lower-cased non-flag tokens of a shell command, `&&` flattened."""
    toks: list[str] = []
    for seg in (cmd or "").split("&&"):
        for t in seg.split():
            t = t.strip("\"'")
            if t and not t.startswith("-"):
                toks.append(t.lower())
    return toks


def commands_equivalent_modulo_flags(candidate: str | None,
                                     original: str | None) -> bool:
    """True when *candidate* is *original* with only flag tokens changed.

    Same executables, same non-flag arguments, in the same order —
    differing only in ``-``/``--`` tokens (e.g. ``pip install pygame``
    vs ``pip install --yes pygame``). Identical strings return False:
    the gate already ran that exact command itself, so an earlier
    success proves nothing new.
    """
    if not candidate or not original or candidate.strip() == original.strip():
        return False
    a, b = _flagless_tokens(candidate), _flagless_tokens(original)
    return bool(a) and a == b


def attempt_env_self_heal(tools: AgentTools, verify_output: str,
                          language: str | None, healed: set[str],
                          verify_cmd: str | None = None,
                          planned_files: list[str] | None = None) -> bool:
    """Install a missing third-party dependency named in failing output.

    ``No module named X`` / ``Cannot find package 'X'`` are environment
    problems that editing project files can never fix — the loop that hit
    them burned its whole turn budget while the real fix was one pip
    install (mirrors the BulkTest self-heal, which the agent-loop path
    bypasses). *healed* accumulates already-attempted names so a dep that
    keeps failing is only installed once per loop.

    Returns True when an install succeeded and the caller should re-run
    verification.
    """
    lang = (language or "python").lower()
    from .pipeline import _missing_js_packages, _missing_third_party_module
    if lang in ("javascript", "typescript"):
        pkgs = [p for p in _missing_js_packages(verify_output)
                if p not in healed]
        if not pkgs:
            return False
        healed.update(pkgs)
        install_cmd = "npm install -D " + " ".join(pkgs)
    else:
        memory = getattr(tools, "_memory", None)
        project_files = list(memory.all_files()) if memory is not None else []
        # Files the step is about to create count as local too. A TEST step
        # whose gate is `python -m unittest -v test_main` fails with
        # "No module named test_main" before the file exists, and memory
        # cannot know better — so the heal tried `pip install test_main`,
        # a pointless call and the exact dependency-confusion hazard
        # _missing_third_party_module exists to prevent.
        project_files += [p for p in (planned_files or []) if p]
        mod = _missing_third_party_module(verify_output, project_files)
        if not mod or mod in healed:
            return False
        healed.add(mod)
        # Install into the venv the gates use, not bare `python`. The
        # absolute interpreter path also survives the `cd {sub} &&` prefix
        # added below.
        py = _venv_python(getattr(tools, "project_root", ".")) or "python"
        py_tok = f'"{py}"' if py != "python" else py
        install_cmd = f"{py_tok} -m pip install {mod}"
    install_cmd = _cd_prefix(verify_cmd) + install_cmd
    _logger.info("[AgentLoop] env self-heal: %s", install_cmd)
    result = tools.execute(ToolCall(name="run_command",
                                    arguments={"command": install_cmd},
                                    id="env-heal"))
    return result.startswith("exit: success")


def run_agent_loop(
    llm_client,
    tools: AgentTools,
    step_text: str,
    task: str,
    display=None,
    step_idx: int = 0,
    language: str | None = None,
    max_turns: int = 8,
    verify_cmd: str | None = None,
    context: str = "",
    preload_files: list[str] | None = None,
    _recovery: bool = False,
    attempt_label: str = "first attempt",
) -> tuple[bool, str]:
    """Run one step as a capped tool-calling loop.

    Exit conditions, in order:
    - Model stops calling tools AND ``verify_cmd`` (when given) passes
      → ``(True, summary)``. A failing ``verify_cmd`` is fed back to the
      model as a new user message and the loop continues.
    - Model stops calling tools without having used any tool at all
      → ``(False, ...)`` — a step that changed nothing cannot have
      succeeded.
    - ``max_turns`` exhausted → ``(False, ...)``.

    Returns the same ``(success, error_info)`` contract as the step
    handlers in ``step_handlers.py``.
    """
    preloaded = _preload_target_files(tools, preload_files)
    messages = [
        Message(role="system", content=AGENT_LOOP_SYSTEM_PROMPT),
        Message(role="user",
                content=_build_user_message(step_text, task, language,
                                            context, preloaded)),
    ]
    definitions = tools.definitions()
    action_definitions = [d for d in definitions
                          if d.name not in _READ_ONLY_TOOLS]
    any_tool_used = False
    tool_counts: Counter = Counter()
    read_only_streak = 0
    healed: set[str] = set()

    # Last run_command the model executed that exited 0 — evidence for
    # the unpassable-gate escape below (recovery loops only).
    last_ok_cmd: str | None = None

    # Cross-attempt memory: what this attempt actually touched and ran.
    edited_files: list[str] = []
    commands_run: list[tuple[str, bool]] = []

    def _finish(outcome: str, turns: int,
                result: tuple[bool, str]) -> tuple[bool, str]:
        _record_loop_run(step_idx, turns, tool_counts, outcome, _recovery)
        record_attempt(step_idx, attempt_label, outcome,
                       edited_files, commands_run, result[1])
        return result

    def _variant_gate_ok() -> bool:
        """Escape hatch for gates no correct work can pass.

        A failed CMD step's recovery gate is the original command
        verbatim; when that command is malformed (observed:
        `pip install --yes pygame` — pip has no --yes), the gate stays
        red forever even though the loop already ran the corrected
        command successfully. Accept a same-command-modulo-flags success
        the loop itself produced. Recovery loops only — CODE/TEST gates
        keep strict semantics.
        """
        return (_recovery and verify_cmd is not None
                and commands_equivalent_modulo_flags(last_ok_cmd, verify_cmd))

    def _run_verify() -> str:
        result = tools.execute_all([_verify_call(verify_cmd)])[0].content
        while (not result.startswith("exit: success")
               and attempt_env_self_heal(tools, result, language, healed,
                                         verify_cmd,
                                         planned_files=preload_files)):
            result = tools.execute_all([_verify_call(verify_cmd)])[0].content
        return result

    for turn in range(1, max_turns + 1):
        # Final turn: withhold tools so the model must produce a text
        # summary instead of burning the last turn on another tool call.
        final_turn = turn == max_turns
        if final_turn and any_tool_used:
            messages.append(Message(role="user", content=(
                "Turn budget exhausted — tools are no longer available. "
                "Reply now with a short summary of what you completed and "
                "whether it was verified.")))
        if final_turn:
            tools_for_turn = None
        elif read_only_streak >= _WITHHOLD_READONLY_AT:
            # The act-now nudge was ignored — withhold inspection tools so
            # the only moves left are the ones that change something.
            tools_for_turn = action_definitions
        else:
            tools_for_turn = definitions
        response = llm_client.chat(messages, tools=tools_for_turn)

        if response.has_tool_calls:
            any_tool_used = True
            tool_counts.update(tc.name for tc in response.tool_calls)
            names = ", ".join(tc.name for tc in response.tool_calls)
            _logger.info("[AgentLoop] step %d turn %d/%d: %s",
                         step_idx + 1, turn, max_turns, names)
            if display is not None:
                display.step_info(step_idx,
                                  f"Agent loop {turn}/{max_turns}: {names}")
            messages.append(response.to_message())
            _tool_msgs = tools.execute_all(response.tool_calls)
            messages.extend(_tool_msgs)
            for _tc, _tm in zip(response.tool_calls, _tool_msgs):
                _content = _tm.content or ""
                if _tc.name == "run_command":
                    _cmd = _tc.arguments.get("command", "")
                    _ok = _content.startswith("exit: success")
                    if _cmd:
                        commands_run.append((_cmd, _ok))
                    if _ok:
                        last_ok_cmd = _cmd
                elif _tc.name in ("write_file", "edit_file"):
                    # Errors come back as strings rather than raising, so
                    # only count edits that actually landed.
                    _path = (_tc.arguments.get("path") or "").replace("\\", "/")
                    if _path and not _content.lower().startswith("error"):
                        edited_files.append(_path)
            # Read-only intervention: some models settle into inspecting
            # file after file without ever acting (observed: whole 8-turn
            # budgets of read_file, twice in one run, even with the
            # failing output already in context). After a couple of
            # consecutive inspection-only turns, tell the model to act.
            if all(tc.name in _READ_ONLY_TOOLS for tc in response.tool_calls):
                read_only_streak += 1
            else:
                read_only_streak = 0
            if read_only_streak == _ACT_NOW_NUDGE_AT and turn <= max_turns - 2:
                _logger.info("[AgentLoop] step %d: %d read-only turns — "
                             "injecting act-now nudge", step_idx + 1,
                             _ACT_NOW_NUDGE_AT)
                messages.append(Message(role="user", content=(
                    f"You have spent {_ACT_NOW_NUDGE_AT} consecutive turns "
                    "only inspecting files. You have enough context — ACT "
                    "now: apply the fix with edit_file or write_file, or run "
                    "the command that completes this step. "
                    f"{max_turns - turn} turn(s) remain.")))
            elif read_only_streak == _WITHHOLD_READONLY_AT:
                # Nudge ignored (observed twice in one run: the model went
                # straight back to read_file). Escalate: from the next
                # turn only acting tools are offered.
                _logger.info("[AgentLoop] step %d: nudge ignored — "
                             "withholding read-only tools", step_idx + 1)
                messages.append(Message(role="user", content=(
                    "Inspection tools are now disabled. Only write_file, "
                    "edit_file and run_command are available — apply the "
                    "fix or run the completing command now.")))
            continue

        # Model stopped calling tools — it believes the step is done.
        summary = response.text.strip()
        if not any_tool_used:
            _logger.warning("[AgentLoop] step %d: model finished without "
                            "using any tool", step_idx + 1)
            return _finish("no-tools", turn, (False, (
                "Agent loop made no tool calls — no files were changed and "
                f"no commands were run. Model said: {summary[:500]}")))

        if verify_cmd:
            if display is not None:
                display.step_info(step_idx, f"Verifying: {verify_cmd}")
            result = _run_verify()
            if result.startswith("exit: success"):
                _logger.info("[AgentLoop] step %d verified in %d turn(s)",
                             step_idx + 1, turn)
                return _finish("verified", turn, (True, summary))
            if _variant_gate_ok():
                _logger.info(
                    "[AgentLoop] step %d: gate command fails but the loop "
                    "ran a flag-variant of it successfully (%r) — "
                    "accepting the variant as the gate", step_idx + 1,
                    last_ok_cmd)
                return _finish("verified-variant", turn, (True, summary))
            if final_turn:
                return _finish("verify-failed", turn, (False, (
                    f"Verification still failing after {max_turns} turns:\n"
                    f"{truncate_middle(result, 1000)}")))
            _logger.info("[AgentLoop] step %d: verification failed on "
                         "turn %d — feeding back", step_idx + 1, turn)
            messages.append(response.to_message())
            messages.append(Message(role="user", content=(
                f"Verification command failed:\n{verify_cmd}\n\n{result}\n\n"
                "The step is not complete. Fix the problem and verify again.")))
            continue

        _logger.info("[AgentLoop] step %d finished in %d turn(s)",
                     step_idx + 1, turn)
        return _finish("done", turn, (True, summary))

    # Exhausted without a final text answer (e.g. text-mode model ignored
    # the no-tools instruction). The work may still be done — let the
    # deterministic check have the last word.
    if verify_cmd and any_tool_used:
        result = _run_verify()
        if result.startswith("exit: success"):
            _logger.info("[AgentLoop] step %d: turns exhausted but "
                         "verification passes — accepting", step_idx + 1)
            return _finish("exhausted-verified", max_turns, (True, (
                "Step verified complete (turn budget exhausted "
                "before the model summarized).")))
        if _variant_gate_ok():
            _logger.info(
                "[AgentLoop] step %d: turns exhausted, gate command fails "
                "but a successful flag-variant ran (%r) — accepting",
                step_idx + 1, last_ok_cmd)
            return _finish("exhausted-verified-variant", max_turns, (True, (
                "Step complete: the gate command is malformed but the loop "
                f"ran an equivalent command successfully: {last_ok_cmd}")))

    return _finish("exhausted", max_turns, (False, (
        f"Agent loop exhausted {max_turns} turns without completing the "
        f"step: {step_text[:200]}")))


def run_agent_loop_with_escalation(llm_client, tools: AgentTools,
                                   step_text: str, task: str,
                                   escalation_client=None,
                                   **kw) -> tuple[bool, str]:
    """Run the loop; on failure, retry once with a stronger model.

    A weak model exhausting its turns is a capability floor, not a step
    the pipeline must fail on (observed: 8 read-only turns while the
    one-line fix sat in context). When ``models: escalation:`` names a
    stronger model, that client gets one fresh loop with the failed
    attempt's error in context. No escalation configured → identical to
    :func:`run_agent_loop`.
    """
    success, info = run_agent_loop(llm_client, tools, step_text, task, **kw)
    if success or escalation_client is None or escalation_client is llm_client:
        return success, info
    if not getattr(escalation_client, "supports_tools", lambda: False)():
        return success, info
    _logger.info(
        "[AgentLoop] step %d: loop failed — escalating to stronger model",
        kw.get("step_idx", 0) + 1)
    kw = dict(kw)
    # Digest = what was tried (narrative, heavily truncated). The full
    # error still rides alongside it: a verify failure's stack trace and
    # assertion text are the most actionable thing the next attempt has,
    # and the digest's per-attempt summary is far too short to carry them.
    _digest = attempt_digest(kw.get("step_idx", 0))
    kw["context"] = (
        (kw.get("context") or "")
        + (("\n\n" + _digest) if _digest else "")
        + "\n\nA previous attempt by another model FAILED:\n"
        + truncate_middle(info, 2000))
    kw["attempt_label"] = "escalation (stronger model)"
    return run_agent_loop(escalation_client, tools, step_text, task, **kw)


def _verify_call(verify_cmd: str):
    from ..llm.chat_types import ToolCall
    return ToolCall(name="run_command", arguments={"command": verify_cmd},
                    id="verify")


def verify_cmd_for_language(language: str | None,
                            project_root: str = ".") -> str | None:
    """Deterministic test command for the loop's exit verification.

    Returns None when no trustworthy command exists for the language —
    a wrong verify command is worse than none (the loop would chase
    failures in the verifier instead of the code).
    """
    import json
    import os

    lang = (language or "python").lower()
    if lang == "python":
        # Django projects test through manage.py — pytest is usually not
        # even installed there, so a pytest verifier fails regardless of
        # the app's real state (mirrors BulkTest's Django detection).
        if os.path.isfile(os.path.join(project_root, "manage.py")):
            return "python manage.py test --noinput"
        return "python -m pytest -q"
    if lang in ("javascript", "typescript"):
        # Only trust `npm test` when the project actually defines it;
        # guessing a runner (jest vs vitest) misfires too often.
        pkg_path = os.path.join(project_root, "package.json")
        try:
            with open(pkg_path, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            if (pkg.get("scripts") or {}).get("test"):
                return "npm test --silent"
        except (OSError, json.JSONDecodeError):
            pass
        return None
    if lang == "go":
        return "go test ./..."
    return None


# One-shot scaffolding commands fail on a SECOND invocation precisely when
# the first one (or the recovery) succeeded: mkdir → "already exists",
# django-admin startproject → "conflicts with existing", npm create →
# refuses a non-empty directory, python -m venv → half-usable dir. Any
# compound command containing one of these is excluded whole — planned
# commands routinely chain scaffold + install with `&&`.
# Venv must match CREATION invocations only — a bare `venv` alternative
# also matched the harmless activation path `venv\Scripts\activate`,
# which silently disqualified every plan-declared verify: that carried
# the planner's activation prefix (observed: all 12 CODE gates dropped).
_NON_REVERIFIABLE_RE = re.compile(
    r"\b(?:mkdir|(?:python3?|py)\s+-m\s+venv|virtualenv\s+\S+"
    r"|startproject|startapp"
    r"|git\s+init|npm\s+(?:create|init)|npx\s+create-|yarn\s+create"
    r"|cargo\s+(?:new|init)|rails\s+new|dotnet\s+new)\b",
    re.IGNORECASE,
)


def reverifiable_cmd(cmd: str | None) -> str | None:
    """Return *cmd* when re-running it is a safe deterministic verify gate
    for the recovery loop, else ``None``.

    A failed CMD step's own command is the natural ground truth for "did
    the recovery actually work" — without it the loop accepts the model's
    final summary on faith (observed: `npm run build:css` exited 1 twice,
    the model summarized what it had attempted, and the step was logged
    as recovered while the CSS was never built). Only commands that are
    not one-shot scaffolding qualify; for the rest, ``None`` keeps the
    summary-based exit (with its blocked-admission check).
    """
    if not cmd or _NON_REVERIFIABLE_RE.search(cmd):
        return None
    return cmd


def build_step_tools(executor, memory, kb_context_builder=None,
                     project_root: str = ".") -> AgentTools:
    """Assemble :class:`AgentTools` from the objects a step handler holds."""
    searcher = getattr(kb_context_builder, "_searcher", None) \
        if kb_context_builder is not None else None
    return AgentTools(project_root=project_root, executor=executor,
                      searcher=searcher, memory=memory)


def agent_loop_enabled(cfg, llm_client) -> bool:
    """True when the config opts in AND the provider can do native tools."""
    return (cfg is not None
            and getattr(cfg, "AGENT_LOOP", False)
            and llm_client is not None
            and getattr(llm_client, "supports_tools", lambda: False)())


# Marker prepended to error_info after a failed recovery attempt so the
# pipeline's diagnosis stage doesn't launch a second (redundant) loop.
RECOVERY_FAILED_MARKER = "[agent-loop-recovery-failed]"

# Exact line the recovery prompt asks the model to end with when it could
# NOT recover the step. Checked case-insensitively on the summary.
RECOVERY_BLOCKED_MARKER = "RECOVERY: blocked"


def run_recovery_loop(llm_client, tools: AgentTools, step_text: str,
                      task: str, error_info: str,
                      display=None, step_idx: int = 0,
                      language: str | None = None,
                      max_turns: int = 8,
                      verify_cmd: str | None = None,
                      escalation_client=None) -> tuple[bool, str]:
    """One bounded loop attempt to recover a failed step.

    Replaces the diagnose → fix → re-run machinery: the model gets the
    real error, inspects the actual project state with tools, fixes the
    cause and completes the step in place.

    Without a ``verify_cmd`` the loop's exit rests on the model's final
    summary, so the prompt asks for an explicit verdict line and a
    summary that admits the blocker is treated as a failure — previously
    an honest "the build still fails" summary was logged as a recovery.
    """
    context = (
        "A previous attempt at this step FAILED. Error:\n"
        f"{truncate_middle(error_info, 4000)}\n\n"
        "Investigate the actual state of the project, fix the cause, and "
        "complete the step. If the step is a failed shell command, prefer "
        "correcting and re-running that command (fix the path, drop a bad "
        "`cd`, adjust a flag) over recreating its effects by hand — do NOT "
        "hand-write the files a scaffolder would have generated. "
        "If the failure is an environment limitation "
        "you cannot fix (e.g. a required tool is not installed and cannot "
        "be installed), say so clearly in your summary and end it with "
        f"the exact line: {RECOVERY_BLOCKED_MARKER}")
    if verify_cmd:
        context += (
            f"\n\nThis step is complete ONLY when `{verify_cmd}` exits "
            "successfully — it will be run to verify your work.")

    def _attempt(client, label: str,
                 latest_error: str | None = None) -> tuple[bool, str]:
        # Recovery follows the main loop (and possibly its escalation), so
        # 1-3 failed attempts have already left edits in the working tree.
        # Recompute the digest per attempt so each one sees everything
        # tried so far, including the recovery attempt before it.
        digest = attempt_digest(step_idx)
        ctx = (context + "\n\n" + digest) if digest else context
        if latest_error:
            ctx += ("\n\nA previous recovery attempt by another model "
                    "FAILED:\n" + truncate_middle(latest_error, 2000))
        s, i = run_agent_loop(
            client, tools, step_text, task,
            display=display, step_idx=step_idx, language=language,
            max_turns=max_turns, verify_cmd=verify_cmd, context=ctx,
            _recovery=True, attempt_label=label)
        # Only meaningful when no verify_cmd gated the exit — a passing
        # deterministic check outranks the model's own pessimism.
        if s and not verify_cmd \
                and RECOVERY_BLOCKED_MARKER.lower() in i.lower():
            _logger.warning(
                "[AgentLoop] step %d: recovery summary admits the step is "
                "still blocked — not counting it as recovered",
                step_idx + 1)
            return False, f"Recovery loop reported itself blocked: {i[:800]}"
        return s, i

    # Recovery loops are where turn budgets die (observed repeatedly:
    # "mid-fix at turn 8"). Escalation is decided AFTER the blocked-
    # admission check — a "done" summary that admits the blocker is a
    # failure the stronger model should get a shot at (observed: a
    # blocked npx-tailwind recovery never escalated because the wrapper
    # saw the self-reported success).
    success, info = _attempt(llm_client, "recovery")
    if (not success and escalation_client is not None
            and escalation_client is not llm_client
            and getattr(escalation_client, "supports_tools",
                        lambda: False)()):
        _logger.info(
            "[AgentLoop] step %d: recovery failed — escalating to "
            "stronger model", step_idx + 1)
        success, info = _attempt(escalation_client, "recovery + escalation",
                                 latest_error=info)
    return success, info

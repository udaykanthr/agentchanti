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

from ..agent_tools import AgentTools
from ..llm.chat_types import Message

_logger = logging.getLogger(__name__)


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


def _build_user_message(step_text: str, task: str, language: str | None,
                        context: str) -> str:
    parts = [f"Overall task: {task}", f"Current step: {step_text}"]
    if language:
        parts.append(f"Project language: {language}")
    if context:
        parts.append(f"Project state:\n{context}")
    return "\n\n".join(parts)


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
    messages = [
        Message(role="system", content=AGENT_LOOP_SYSTEM_PROMPT),
        Message(role="user",
                content=_build_user_message(step_text, task, language, context)),
    ]
    definitions = tools.definitions()
    any_tool_used = False

    for turn in range(1, max_turns + 1):
        # Final turn: withhold tools so the model must produce a text
        # summary instead of burning the last turn on another tool call.
        final_turn = turn == max_turns
        if final_turn and any_tool_used:
            messages.append(Message(role="user", content=(
                "Turn budget exhausted — tools are no longer available. "
                "Reply now with a short summary of what you completed and "
                "whether it was verified.")))
        response = llm_client.chat(
            messages, tools=None if final_turn else definitions)

        if response.has_tool_calls:
            any_tool_used = True
            names = ", ".join(tc.name for tc in response.tool_calls)
            _logger.info("[AgentLoop] step %d turn %d/%d: %s",
                         step_idx + 1, turn, max_turns, names)
            if display is not None:
                display.step_info(step_idx,
                                  f"Agent loop {turn}/{max_turns}: {names}")
            messages.append(response.to_message())
            messages.extend(tools.execute_all(response.tool_calls))
            continue

        # Model stopped calling tools — it believes the step is done.
        summary = response.text.strip()
        if not any_tool_used:
            _logger.warning("[AgentLoop] step %d: model finished without "
                            "using any tool", step_idx + 1)
            return False, (
                "Agent loop made no tool calls — no files were changed and "
                f"no commands were run. Model said: {summary[:500]}")

        if verify_cmd:
            if display is not None:
                display.step_info(step_idx, f"Verifying: {verify_cmd}")
            result = tools.execute_all(
                [_verify_call(verify_cmd)])[0].content
            if result.startswith("exit: success"):
                _logger.info("[AgentLoop] step %d verified in %d turn(s)",
                             step_idx + 1, turn)
                return True, summary
            if final_turn:
                return False, (
                    f"Verification still failing after {max_turns} turns:\n"
                    f"{result[:1000]}")
            _logger.info("[AgentLoop] step %d: verification failed on "
                         "turn %d — feeding back", step_idx + 1, turn)
            messages.append(response.to_message())
            messages.append(Message(role="user", content=(
                f"Verification command failed:\n{verify_cmd}\n\n{result}\n\n"
                "The step is not complete. Fix the problem and verify again.")))
            continue

        _logger.info("[AgentLoop] step %d finished in %d turn(s)",
                     step_idx + 1, turn)
        return True, summary

    # Exhausted without a final text answer (e.g. text-mode model ignored
    # the no-tools instruction). The work may still be done — let the
    # deterministic check have the last word.
    if verify_cmd and any_tool_used:
        result = tools.execute_all([_verify_call(verify_cmd)])[0].content
        if result.startswith("exit: success"):
            _logger.info("[AgentLoop] step %d: turns exhausted but "
                         "verification passes — accepting", step_idx + 1)
            return True, ("Step verified complete (turn budget exhausted "
                          "before the model summarized).")

    return False, (
        f"Agent loop exhausted {max_turns} turns without completing the "
        f"step: {step_text[:200]}")


def _verify_call(verify_cmd: str):
    from ..llm.chat_types import ToolCall
    return ToolCall(name="run_command", arguments={"command": verify_cmd},
                    id="verify")


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

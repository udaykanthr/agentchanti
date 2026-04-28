"""
Normalizer — one-shot LLM pass that turns a raw trace into a semantic Spec.

Reads the JSONL trace written by ``Recorder``, builds a compact prompt
with the trace events and the target schema, asks the LLM for YAML, and
validates the result through ``Spec.validate`` before writing it to disk.

The LLM is called exactly once per run. This is the expensive step; the
replayer amortises it across many runs via the locator cache.

No heavy deps — this module imports only the local ``spec`` and ``trace``
modules plus ``yaml`` (already a core AgentChanti dep). It does not
require the ``[testing]`` extra to import, because users may want to
re-normalize an existing trace on a machine without Playwright installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .spec import ALLOWED_ACTIONS, ALLOWED_ASSERTIONS, SPEC_VERSION, Spec
from .trace import read_trace


class NormalizerError(Exception):
    """Raised when the LLM output can't be parsed as a valid Spec."""


class Normalizer:
    """Convert a raw recorder trace into a semantic Spec via one LLM pass.

    ``llm_client`` must expose ``generate_response(prompt: str) -> str``.
    Any of AgentChanti's LLM clients fits; tests can pass a fake.
    """

    def __init__(self, llm_client, *, spec_name: str | None = None):
        self.llm_client = llm_client
        self.spec_name = spec_name

    def normalize(
        self,
        raw_trace_path: str | Path,
        output_spec_path: str | Path,
    ) -> Path:
        events = list(read_trace(raw_trace_path))
        if not events:
            raise NormalizerError(f"trace {raw_trace_path} is empty")

        prompt = _build_prompt(events, spec_name=self.spec_name)
        raw_response = self.llm_client.generate_response(prompt)

        spec = _parse_llm_yaml_to_spec(raw_response)
        return spec.dump(output_spec_path)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_PROMPT_HEADER = f"""You are converting a raw browser session trace into a semantic test Spec (YAML).

SCHEMA (required):
  version: "{SPEC_VERSION}"
  name: <short description of the user journey>
  start_url: <url captured in the session_start event>
  metadata:
    recorded_by: agentchanti
    viewport: <copy verbatim from session_start.viewport when present —
               replay enforces this so coord=X,Y fallbacks hit the same
               screen positions; omit the key only if session_start has
               no viewport>
    user_agent: <copy from session_start.user_agent when present>
  steps:
    - id: step-<n>
      action: one of {sorted(ALLOWED_ACTIONS)}
      target:                           # required for every action except navigate
        label: <semantic intent, e.g. "Submit order button">
        fallbacks: [<best selector first>, <fallback>, ...]
      value: <required for fill/press/select>
      url: <required for navigate>
      expected_network:                 # attach the network events observed during this step
        - method: GET|POST|PUT|...
          path: <url path, use * for volatile segments>
          status: <int>
          response_schema: <JSON Schema, optional — infer from response_body>
  assertions:
    - id: assert-<n>
      kind: one of {sorted(ALLOWED_ASSERTIONS)}
      # url_equals       → url: <final URL>
      # dom_predicate    → selector + must_exist
      # natural_language → text: <business-rule sentence>

RULES:
  1. Every `interaction` event in the trace becomes a step. Preserve order.
  2. Synthesize the `label` from element.text | aria_label | nearby_label | role.
     Describe INTENT ("Place order button"), not selectors.
  3. Build `fallbacks` in priority order:
     data-testid > id > aria-label + tag > text selector > raw selector_used.
  4. Attach `network` events to the immediately preceding interaction step
     (events before any interaction belong to the initial `navigate`).
  5. For each network event, set status to what was observed. Only emit
     response_schema when the response_body is a JSON object — use JSON Schema
     with the observed required fields.
  6. Always end with one `url_equals` assertion matching the last known URL.
  7. Output ONLY valid YAML. No markdown fences. No prose.
"""


def _build_prompt(events: list[dict[str, Any]], *, spec_name: str | None) -> str:
    lines = [_PROMPT_HEADER]
    if spec_name:
        lines.append(f"NAME HINT (use as spec.name): {spec_name}")
    lines.append("\nTRACE (one event per line, ordered by seq):")
    # Keep the trace representation compact — the LLM sees exactly the
    # fields it needs. Drop noisy internal fields like precise timestamps
    # (seq preserves order; exact ts doesn't matter for the spec).
    for ev in events:
        compact = {k: v for k, v in ev.items() if k != "ts"}
        lines.append(_compact_json(compact))
    lines.append("\nYAML Spec:")
    return "\n".join(lines)


def _compact_json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:ya?ml)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _parse_llm_yaml_to_spec(raw: str) -> Spec:
    """Strip common LLM wrappings and validate the result as a Spec."""
    text = raw.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise NormalizerError(
            f"LLM output was not valid YAML: {e}\n--- output ---\n{raw[:2000]}"
        ) from e

    if not isinstance(data, dict):
        raise NormalizerError(
            f"LLM output did not produce a mapping at the top level.\n"
            f"--- output ---\n{raw[:2000]}"
        )

    try:
        return Spec.from_dict(data)
    except (KeyError, ValueError, TypeError) as e:
        raise NormalizerError(
            f"LLM output did not conform to the Spec schema: {e}\n"
            f"--- output ---\n{raw[:2000]}"
        ) from e

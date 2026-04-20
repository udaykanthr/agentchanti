"""Tests for agentchanti.testing.normalizer — trace -> Spec conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentchanti.testing.normalizer import Normalizer, NormalizerError
from agentchanti.testing.spec import Spec
from agentchanti.testing.trace import ElementContext, TraceWriter


class FakeLLM:
    """Minimal stand-in for an LLMClient — returns a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def generate_response(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def _write_simple_trace(tmp_path: Path) -> Path:
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as w:
        w.write_session_start(start_url="http://localhost:3000/login")
        w.write_interaction(
            action="fill", selector_used="input[name=email]",
            element=ElementContext(tag="input", nearby_label="Email"),
            value="user@example.com",
        )
        w.write_interaction(
            action="click", selector_used="button[type=submit]",
            element=ElementContext(tag="button", text="Sign in"),
        )
        w.write_network(
            request_id="r1", method="POST", url="/api/auth/login",
            status=200, request_body={"email": "user@example.com"},
            response_body={"token": "abc123"},
        )
        w.write_session_end()
    return path


VALID_SPEC_YAML = """\
version: "1"
name: "login flow"
start_url: "http://localhost:3000/login"
steps:
  - id: step-1
    action: fill
    target:
      label: "Email input"
      fallbacks: ["input[name=email]"]
    value: "user@example.com"
  - id: step-2
    action: click
    target:
      label: "Sign in button"
      fallbacks: ["button[type=submit]"]
    expected_network:
      - method: POST
        path: /api/auth/login
        status: 200
        response_schema:
          type: object
          required: [token]
assertions:
  - id: assert-1
    kind: url_equals
    url: /dashboard
"""


def test_normalize_happy_path_writes_valid_spec(tmp_path: Path):
    trace = _write_simple_trace(tmp_path)
    out = tmp_path / "spec.yaml"
    llm = FakeLLM(VALID_SPEC_YAML)

    result = Normalizer(llm).normalize(trace, out)

    assert result == out
    loaded = Spec.load(out)
    assert loaded.name == "login flow"
    assert len(loaded.steps) == 2
    assert loaded.steps[1].expected_network[0].path == "/api/auth/login"


def test_prompt_includes_every_trace_event(tmp_path: Path):
    trace = _write_simple_trace(tmp_path)
    llm = FakeLLM(VALID_SPEC_YAML)
    Normalizer(llm).normalize(trace, tmp_path / "spec.yaml")

    assert llm.last_prompt is not None
    prompt = llm.last_prompt
    # Every event type from the trace must survive into the prompt.
    assert "session_start" in prompt
    assert "interaction" in prompt
    assert "network" in prompt
    assert "user@example.com" in prompt  # value is preserved
    assert "/api/auth/login" in prompt   # network URL reaches the LLM


def test_spec_name_hint_reaches_prompt(tmp_path: Path):
    trace = _write_simple_trace(tmp_path)
    llm = FakeLLM(VALID_SPEC_YAML)
    Normalizer(llm, spec_name="login regression").normalize(
        trace, tmp_path / "spec.yaml"
    )
    assert "login regression" in llm.last_prompt


def test_markdown_fences_are_stripped(tmp_path: Path):
    trace = _write_simple_trace(tmp_path)
    fenced = "```yaml\n" + VALID_SPEC_YAML + "```\n"
    Normalizer(FakeLLM(fenced)).normalize(trace, tmp_path / "spec.yaml")
    assert Spec.load(tmp_path / "spec.yaml").name == "login flow"


def test_empty_trace_raises(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(NormalizerError, match="empty"):
        Normalizer(FakeLLM(VALID_SPEC_YAML)).normalize(empty, tmp_path / "s.yaml")


def test_invalid_yaml_raises_with_head_of_output(tmp_path: Path):
    trace = _write_simple_trace(tmp_path)
    llm = FakeLLM("this is: not: valid: yaml: at: all: {{{")
    with pytest.raises(NormalizerError, match="not valid YAML"):
        Normalizer(llm).normalize(trace, tmp_path / "s.yaml")


def test_yaml_that_fails_schema_validation_raises(tmp_path: Path):
    trace = _write_simple_trace(tmp_path)
    # Unknown action — schema will reject during validate()
    bad = """\
version: "1"
name: bad
start_url: "/"
steps:
  - id: step-1
    action: teleport
    target: { label: x, fallbacks: [] }
"""
    llm = FakeLLM(bad)
    with pytest.raises(NormalizerError, match="did not conform"):
        Normalizer(llm).normalize(trace, tmp_path / "s.yaml")


def test_non_mapping_output_raises(tmp_path: Path):
    trace = _write_simple_trace(tmp_path)
    llm = FakeLLM("- just\n- a\n- list\n")
    with pytest.raises(NormalizerError, match="mapping at the top level"):
        Normalizer(llm).normalize(trace, tmp_path / "s.yaml")

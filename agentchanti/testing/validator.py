"""
Validator — checks a replay run against the spec's assertions.

Validates three kinds of expectations:
  1. Navigation state (final URL, visible text, DOM predicates)
  2. Network contracts — request/response schema drift vs. the recorded baseline
  3. Natural-language assertions, delegated to the LLM with the observed state
"""

from __future__ import annotations

from pathlib import Path


class Validator:
    """Evaluate a Replayer run result against a semantic test spec."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def validate(self, spec_path: str | Path, run_result: dict) -> list[dict]:
        """Return a list of assertion results, one per spec-declared expectation."""
        raise NotImplementedError("Validator.validate is not implemented yet")

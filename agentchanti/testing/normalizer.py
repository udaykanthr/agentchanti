"""
Normalizer — converts a raw browser session trace into a semantic test spec.

Runs a single LLM pass over the raw event stream and produces a structured
YAML spec with semantic action labels, locator fallbacks, and expected
network request/response schemas. The output is the durable artifact that
``Replayer`` executes against a fresh browser session.
"""

from __future__ import annotations

from pathlib import Path


class Normalizer:
    """Convert a raw recorder trace into a semantic test spec via one LLM pass."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def normalize(self, raw_trace_path: str | Path, output_spec_path: str | Path) -> Path:
        """Read the raw trace, produce a semantic spec, and write it to disk."""
        raise NotImplementedError("Normalizer.normalize is not implemented yet")

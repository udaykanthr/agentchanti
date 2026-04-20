"""
Validator — evaluates a Spec's assertions against a RunResult.

Two sources of assertions are checked:

  1. Per-step ``expected_network`` contracts. For each step the validator
     pairs declared API expectations with the network events observed
     during that step, checks method / path / status, and (when
     ``jsonschema`` is installed) validates response bodies against the
     declared response_schema. This is where contract drift gets caught.

  2. Top-level ``assertions``:
     * ``url_equals``       — deterministic string compare
     * ``dom_predicate``    — lightweight walk of the accessibility snapshot
     * ``natural_language`` — delegated to the LLM when one was supplied

Validator is duck-typed on ``llm_client`` (same contract as Normalizer /
Replayer: ``generate_response(prompt) -> str``). Without one, NL
assertions are reported as skipped rather than silently marked as passing
— a skipped NL assertion is an explicit quality signal, not a free pass.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .replayer import RunResult, StepResult
from .spec import NetworkExpectation, Spec


@dataclass
class AssertionResult:
    id: str
    kind: str
    passed: bool
    detail: str = ""
    skipped: bool = False


class Validator:
    """Evaluate a Spec against a RunResult and return structured results."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def validate(self, spec: Spec, run_result: RunResult) -> list[AssertionResult]:
        results: list[AssertionResult] = []
        # 1. Network contracts — one AssertionResult per expected_network entry
        results.extend(self._validate_network(spec, run_result))
        # 2. Top-level assertions
        for a in spec.assertions:
            results.append(self._dispatch_assertion(a, run_result))
        return results

    # ---- Network contracts ----------------------------------------------

    def _validate_network(
        self, spec: Spec, run_result: RunResult,
    ) -> list[AssertionResult]:
        out: list[AssertionResult] = []
        # Pair spec.steps with run_result.steps positionally — replay is
        # fail-fast, so run_result may be shorter than spec.steps.
        for i, step in enumerate(spec.steps):
            if i >= len(run_result.steps):
                for j, exp in enumerate(step.expected_network):
                    out.append(AssertionResult(
                        id=f"{step.id}::net::{j}",
                        kind="network",
                        passed=False,
                        detail=(
                            f"step {step.id!r} did not run — skipped by "
                            f"fail-fast after an earlier failure"
                        ),
                    ))
                continue
            observed = run_result.steps[i].network_events
            for j, exp in enumerate(step.expected_network):
                out.append(_check_one_network_expectation(
                    assertion_id=f"{step.id}::net::{j}",
                    expected=exp, observed=observed,
                ))
        return out

    # ---- Top-level assertions -------------------------------------------

    def _dispatch_assertion(self, a, run_result: RunResult) -> AssertionResult:
        if a.kind == "url_equals":
            return _check_url_equals(a, run_result)
        if a.kind == "dom_predicate":
            return _check_dom_predicate(a, run_result)
        if a.kind == "natural_language":
            return self._check_natural_language(a, run_result)
        return AssertionResult(
            id=a.id, kind=a.kind, passed=False,
            detail=f"unknown assertion kind {a.kind!r}",
        )

    def _check_natural_language(self, a, run_result: RunResult) -> AssertionResult:
        if self.llm_client is None:
            return AssertionResult(
                id=a.id, kind=a.kind, passed=False, skipped=True,
                detail="no llm_client supplied — NL assertion skipped",
            )
        prompt = _build_nl_prompt(a.text or "", run_result)
        raw = (self.llm_client.generate_response(prompt) or "").strip()
        verdict, reason = _parse_nl_verdict(raw)
        return AssertionResult(
            id=a.id, kind=a.kind,
            passed=verdict is True,
            detail=reason or raw[:200],
        )


# ---------------------------------------------------------------------------
# Network check
# ---------------------------------------------------------------------------

def _check_one_network_expectation(
    *,
    assertion_id: str,
    expected: NetworkExpectation,
    observed: list,
) -> AssertionResult:
    match = None
    for ev in observed:
        if ev.method.upper() != expected.method.upper():
            continue
        if not _path_matches(expected.path, ev.url):
            continue
        match = ev
        break
    if match is None:
        return AssertionResult(
            id=assertion_id, kind="network", passed=False,
            detail=(
                f"no {expected.method} request matching {expected.path!r} "
                f"was observed during this step"
            ),
        )
    if match.status != expected.status:
        return AssertionResult(
            id=assertion_id, kind="network", passed=False,
            detail=(
                f"{expected.method} {expected.path} returned "
                f"{match.status}, expected {expected.status}"
            ),
        )
    if expected.response_schema and match.response_body is not None:
        err = _validate_json_schema(match.response_body, expected.response_schema)
        if err:
            return AssertionResult(
                id=assertion_id, kind="network", passed=False,
                detail=(
                    f"{expected.method} {expected.path} response schema drift: {err}"
                ),
            )
    return AssertionResult(
        id=assertion_id, kind="network", passed=True,
        detail=f"{expected.method} {expected.path} → {expected.status} (ok)",
    )


def _path_matches(pattern: str, observed_url: str) -> bool:
    """Match ``expected.path`` (which may have ``*`` globs) against the path
    portion of an observed URL — absolute or relative."""
    parsed = urlparse(observed_url)
    observed_path = parsed.path or observed_url
    return fnmatch.fnmatchcase(observed_path, pattern)


def _validate_json_schema(body: Any, schema: dict[str, Any]) -> str | None:
    """Return None when body conforms to schema, an error string otherwise.

    Uses ``jsonschema`` when available (it ships with the ``[testing]``
    extra). When missing, falls back to a minimal ``required`` + ``type``
    check so users without the extra still catch the most common drift —
    a dropped required field. Clearly inferior to jsonschema but useful.
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return _minimal_schema_check(body, schema)
    try:
        jsonschema.validate(instance=body, schema=schema)  # type: ignore[attr-defined]
        return None
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        return e.message


def _minimal_schema_check(body: Any, schema: dict[str, Any]) -> str | None:
    if schema.get("type") == "object":
        if not isinstance(body, dict):
            return "expected object"
        for req in schema.get("required", []):
            if req not in body:
                return f"missing required field {req!r}"
    return None


# ---------------------------------------------------------------------------
# url_equals
# ---------------------------------------------------------------------------

def _check_url_equals(a, run_result: RunResult) -> AssertionResult:
    observed = run_result.final_url or ""
    expected = a.url or ""
    # Compare just the path when the expected side is relative — a caller
    # writing `/dashboard` shouldn't fail against `http://host/dashboard`.
    if expected.startswith("/"):
        observed_path = urlparse(observed).path or observed
        ok = observed_path == expected
    else:
        ok = observed == expected
    return AssertionResult(
        id=a.id, kind=a.kind, passed=ok,
        detail=f"expected {expected!r}, got {observed!r}",
    )


# ---------------------------------------------------------------------------
# dom_predicate
# ---------------------------------------------------------------------------

def _check_dom_predicate(a, run_result: RunResult) -> AssertionResult:
    selector = a.selector or ""
    must_exist = True if a.must_exist is None else bool(a.must_exist)
    present = _snapshot_has_selector(run_result.final_snapshot, selector)
    ok = present if must_exist else (not present)
    verb = "present" if present else "absent"
    detail = f"selector {selector!r} is {verb}; must_exist={must_exist}"
    return AssertionResult(id=a.id, kind=a.kind, passed=ok, detail=detail)


def _snapshot_has_selector(snapshot: dict[str, Any], selector: str) -> bool:
    """Lightweight selector presence check against an accessibility snapshot.

    Handles three common cases explicitly; otherwise falls back to a
    substring match on the serialized snapshot. This is a deliberate MVP
    — when we wire a live Playwright MCP server we can swap this for the
    real snapshot-query tool.
    """
    if not snapshot or not selector:
        return False

    # [data-testid=foo] or data-testid=foo
    dt = _extract_data_testid(selector)
    if dt is not None:
        return _tree_any(snapshot, lambda n: _attr(n, "data_testid") == dt
                                             or _attr(n, "data-testid") == dt)

    # #foo  → id match
    if selector.startswith("#"):
        wanted = selector[1:]
        return _tree_any(snapshot, lambda n: _attr(n, "id") == wanted)

    # text=foo or text="foo"
    if selector.startswith("text="):
        needle = selector[len("text="):].strip().strip('"').strip("'")
        return _tree_any(snapshot, lambda n: needle in (_attr(n, "text") or "")
                                             or needle in (_attr(n, "name") or ""))

    # Fallback: substring on the serialized snapshot.
    return selector in json.dumps(snapshot, ensure_ascii=False)


def _extract_data_testid(selector: str) -> str | None:
    import re
    m = re.match(r"^\[?data-testid=([^\]]+)\]?$", selector)
    if m:
        return m.group(1).strip('"').strip("'")
    return None


def _attr(node: Any, key: str) -> Any:
    return node.get(key) if isinstance(node, dict) else None


def _tree_any(node: Any, predicate) -> bool:
    if isinstance(node, dict):
        if predicate(node):
            return True
        for v in node.values():
            if _tree_any(v, predicate):
                return True
    elif isinstance(node, list):
        for item in node:
            if _tree_any(item, predicate):
                return True
    return False


# ---------------------------------------------------------------------------
# natural_language
# ---------------------------------------------------------------------------

def _build_nl_prompt(assertion_text: str, run_result: RunResult) -> str:
    return (
        "Evaluate whether the following assertion holds about a completed "
        "browser test run.\n\n"
        f"ASSERTION: {assertion_text}\n\n"
        f"FINAL URL: {run_result.final_url!r}\n\n"
        f"FINAL DOM SNAPSHOT (truncated):\n"
        + json.dumps(run_result.final_snapshot, ensure_ascii=False)[:3000]
        + "\n\nObserved network traffic summary:\n"
        + json.dumps([
            {
                "step": s.step_id,
                "events": [
                    {"method": e.method, "url": e.url, "status": e.status}
                    for e in s.network_events
                ],
            }
            for s in run_result.steps
        ], ensure_ascii=False)[:1500]
        + "\n\nRespond on a single line with 'PASS: <reason>' or 'FAIL: <reason>'."
    )


def _parse_nl_verdict(raw: str) -> tuple[bool | None, str]:
    stripped = raw.strip().lstrip("-* ").strip()
    upper = stripped.upper()
    if upper.startswith("PASS"):
        return True, stripped.split(":", 1)[-1].strip() if ":" in stripped else ""
    if upper.startswith("FAIL"):
        return False, stripped.split(":", 1)[-1].strip() if ":" in stripped else ""
    return None, stripped

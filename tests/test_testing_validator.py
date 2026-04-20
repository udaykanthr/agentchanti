"""Tests for agentchanti.testing.validator."""

from __future__ import annotations

from agentchanti.testing.mcp_client import NetworkEvent
from agentchanti.testing.replayer import RunResult, StepResult
from agentchanti.testing.spec import (
    Assertion,
    Locator,
    NetworkExpectation,
    Spec,
    Step,
)
from agentchanti.testing.validator import Validator


def _run_result_with(**kwargs) -> RunResult:
    defaults = {
        "spec_name": "flow",
        "steps": [],
        "final_url": "/dashboard",
        "final_snapshot": {},
    }
    defaults.update(kwargs)
    return RunResult(**defaults)


# ---- Network expectations --------------------------------------------------

def test_network_ok_when_method_path_status_all_match():
    spec = Spec(
        name="f", start_url="/",
        steps=[Step(
            id="s1", action="click", target=Locator(label="x"),
            expected_network=[NetworkExpectation(
                method="POST", path="/api/orders", status=201,
            )],
        )],
    )
    rr = _run_result_with(steps=[StepResult(
        step_id="s1", action="click", success=True,
        network_events=[NetworkEvent(
            method="POST", url="http://localhost/api/orders", status=201,
        )],
    )])
    [net] = Validator().validate(spec, rr)
    assert net.passed
    assert "ok" in net.detail


def test_network_fails_when_no_matching_request():
    spec = Spec(
        name="f", start_url="/",
        steps=[Step(
            id="s1", action="click", target=Locator(label="x"),
            expected_network=[NetworkExpectation(
                method="POST", path="/api/orders", status=201)],
        )],
    )
    rr = _run_result_with(steps=[StepResult(
        step_id="s1", action="click", success=True,
        network_events=[NetworkEvent(method="GET", url="/api/other", status=200)],
    )])
    [net] = Validator().validate(spec, rr)
    assert not net.passed
    assert "no POST request matching" in net.detail


def test_network_fails_on_status_drift():
    spec = Spec(
        name="f", start_url="/",
        steps=[Step(
            id="s1", action="click", target=Locator(label="x"),
            expected_network=[NetworkExpectation(
                method="POST", path="/api/orders", status=201)],
        )],
    )
    rr = _run_result_with(steps=[StepResult(
        step_id="s1", action="click", success=True,
        network_events=[NetworkEvent(
            method="POST", url="/api/orders", status=500,
        )],
    )])
    [net] = Validator().validate(spec, rr)
    assert not net.passed
    assert "500" in net.detail


def test_network_path_supports_glob():
    spec = Spec(
        name="f", start_url="/",
        steps=[Step(
            id="s1", action="click", target=Locator(label="x"),
            expected_network=[NetworkExpectation(
                method="GET", path="/api/orders/*", status=200)],
        )],
    )
    rr = _run_result_with(steps=[StepResult(
        step_id="s1", action="click", success=True,
        network_events=[NetworkEvent(
            method="GET", url="/api/orders/42", status=200,
        )],
    )])
    [net] = Validator().validate(spec, rr)
    assert net.passed


def test_network_schema_drift_dropped_required_field():
    spec = Spec(
        name="f", start_url="/",
        steps=[Step(
            id="s1", action="click", target=Locator(label="x"),
            expected_network=[NetworkExpectation(
                method="POST", path="/api/orders", status=201,
                response_schema={
                    "type": "object",
                    "required": ["order_id", "total"],
                },
            )],
        )],
    )
    rr = _run_result_with(steps=[StepResult(
        step_id="s1", action="click", success=True,
        network_events=[NetworkEvent(
            method="POST", url="/api/orders", status=201,
            response_body={"order_id": "abc"},  # total missing
        )],
    )])
    [net] = Validator().validate(spec, rr)
    assert not net.passed
    assert "total" in net.detail


def test_network_assertions_for_unreached_steps_marked_failed():
    """Fail-fast replay means later steps didn't run — their network
    expectations must not silently pass."""
    spec = Spec(
        name="f", start_url="/",
        steps=[
            Step(id="s1", action="click", target=Locator(label="x")),
            Step(
                id="s2", action="click", target=Locator(label="y"),
                expected_network=[NetworkExpectation(
                    method="POST", path="/api/x", status=201)],
            ),
        ],
    )
    # Only s1 ran (and failed)
    rr = _run_result_with(steps=[StepResult(
        step_id="s1", action="click", success=False,
    )])
    net_results = [r for r in Validator().validate(spec, rr) if r.kind == "network"]
    assert len(net_results) == 1
    assert not net_results[0].passed
    assert "did not run" in net_results[0].detail


# ---- url_equals ------------------------------------------------------------

def test_url_equals_absolute_observed_matches_relative_expected():
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(id="a1", kind="url_equals", url="/dashboard")],
    )
    rr = _run_result_with(final_url="http://localhost:3000/dashboard")
    [r] = Validator().validate(spec, rr)
    assert r.passed


def test_url_equals_detail_shows_both_sides_on_mismatch():
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(id="a1", kind="url_equals", url="/dashboard")],
    )
    rr = _run_result_with(final_url="/login")
    [r] = Validator().validate(spec, rr)
    assert not r.passed
    assert "/dashboard" in r.detail and "/login" in r.detail


# ---- dom_predicate ---------------------------------------------------------

def test_dom_predicate_data_testid_found():
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(
            id="a1", kind="dom_predicate",
            selector="[data-testid=order-id]", must_exist=True,
        )],
    )
    rr = _run_result_with(final_snapshot={
        "role": "document",
        "children": [{"role": "text", "data_testid": "order-id", "text": "42"}],
    })
    [r] = Validator().validate(spec, rr)
    assert r.passed


def test_dom_predicate_must_exist_false_passes_when_absent():
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(
            id="a1", kind="dom_predicate",
            selector="#error-banner", must_exist=False,
        )],
    )
    rr = _run_result_with(final_snapshot={"role": "document", "children": []})
    [r] = Validator().validate(spec, rr)
    assert r.passed


def test_dom_predicate_text_selector_matches_nested():
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(
            id="a1", kind="dom_predicate",
            selector="text=Welcome back", must_exist=True,
        )],
    )
    rr = _run_result_with(final_snapshot={
        "children": [{"role": "heading", "text": "Welcome back, Uday"}],
    })
    [r] = Validator().validate(spec, rr)
    assert r.passed


# ---- natural_language ------------------------------------------------------

class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt: str | None = None

    def generate_response(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def test_nl_without_llm_is_skipped_not_silently_passing():
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(id="a1", kind="natural_language",
                              text="total equals sum of items")],
    )
    [r] = Validator().validate(spec, _run_result_with())
    assert r.skipped is True
    assert r.passed is False  # skipped is NOT pass


def test_nl_with_llm_pass_verdict():
    llm = FakeLLM("PASS: totals match as computed from the DOM.")
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(id="a1", kind="natural_language",
                              text="total equals sum of items")],
    )
    [r] = Validator(llm_client=llm).validate(spec, _run_result_with())
    assert r.passed
    assert "totals match" in r.detail


def test_nl_with_llm_fail_verdict():
    llm = FakeLLM("FAIL: displayed total is 10 but sum is 15.")
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(id="a1", kind="natural_language",
                              text="total equals sum of items")],
    )
    [r] = Validator(llm_client=llm).validate(spec, _run_result_with())
    assert not r.passed
    assert "15" in r.detail


def test_nl_ambiguous_verdict_treated_as_fail():
    llm = FakeLLM("I'm not sure what to say here honestly")
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(id="a1", kind="natural_language", text="x")],
    )
    [r] = Validator(llm_client=llm).validate(spec, _run_result_with())
    assert not r.passed


def test_nl_prompt_includes_final_url_and_snapshot():
    llm = FakeLLM("PASS: fine")
    spec = Spec(
        name="f", start_url="/",
        assertions=[Assertion(id="a1", kind="natural_language", text="looks ok")],
    )
    rr = _run_result_with(
        final_url="/orders/42",
        final_snapshot={"heading": "Order confirmed"},
    )
    Validator(llm_client=llm).validate(spec, rr)
    assert "/orders/42" in llm.last_prompt
    assert "Order confirmed" in llm.last_prompt

"""Roundtrip + validation tests for agentchanti.testing.spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentchanti.testing.spec import (
    Assertion,
    Locator,
    NetworkExpectation,
    Spec,
    Step,
)

EXAMPLE_SPEC = Path(__file__).parent.parent / "examples" / "testing" / "checkout_flow.yaml"


def _sample_spec() -> Spec:
    return Spec(
        name="sample",
        start_url="http://localhost:3000",
        steps=[
            Step(id="s1", action="navigate", url="/home"),
            Step(
                id="s2",
                action="click",
                target=Locator(label="Login button", fallbacks=["#login"]),
                expected_network=[
                    NetworkExpectation(method="POST", path="/api/login", status=200)
                ],
            ),
            Step(
                id="s3",
                action="fill",
                target=Locator(label="Email"),
                value="user@example.com",
            ),
        ],
        assertions=[
            Assertion(id="a1", kind="url_equals", url="/dashboard"),
        ],
    )


def test_roundtrip_preserves_all_fields(tmp_path: Path):
    original = _sample_spec()
    path = original.dump(tmp_path / "spec.yaml")
    loaded = Spec.load(path)
    assert loaded.name == original.name
    assert loaded.start_url == original.start_url
    assert len(loaded.steps) == len(original.steps)
    assert loaded.steps[1].target.label == "Login button"
    assert loaded.steps[1].expected_network[0].path == "/api/login"
    assert loaded.steps[2].value == "user@example.com"
    assert loaded.assertions[0].kind == "url_equals"


def test_validate_rejects_unknown_action():
    spec = Spec(
        name="bad", start_url="/",
        steps=[Step(id="s1", action="teleport", target=Locator(label="x"))],
    )
    with pytest.raises(ValueError, match="unknown action"):
        spec.validate()


def test_validate_rejects_duplicate_step_ids():
    spec = Spec(
        name="bad", start_url="/",
        steps=[
            Step(id="s1", action="navigate", url="/a"),
            Step(id="s1", action="navigate", url="/b"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate step id"):
        spec.validate()


def test_validate_rejects_fill_without_value():
    spec = Spec(
        name="bad", start_url="/",
        steps=[Step(id="s1", action="fill", target=Locator(label="x"))],
    )
    with pytest.raises(ValueError, match="fill requires 'value'"):
        spec.validate()


def test_example_yaml_loads_and_validates():
    """The shipped example spec must always parse — it's user-facing docs."""
    spec = Spec.load(EXAMPLE_SPEC)
    assert spec.name == "checkout flow — happy path"
    assert len(spec.steps) == 4
    assert len(spec.assertions) == 3
    # spot-check: the second order-POST network expectation should survive
    order_step = next(s for s in spec.steps if s.id == "step-4")
    assert order_step.expected_network[0].path == "/api/orders"
    assert order_step.expected_network[0].response_schema["required"] == [
        "order_id", "total", "items",
    ]

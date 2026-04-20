"""
Spec — the semantic, agent-understandable recording format.

This is the single most load-bearing design decision of the testing module:
too low-level (raw DOM events) → brittle; too high-level (pure natural
language) → ambiguous. The schema below is hybrid on purpose:

  * A semantic ``label`` describes *intent* ("Submit order button") so an
    LLM can re-locate the element if selectors change.
  * A list of ``fallbacks`` gives the replayer deterministic selectors it
    can cache and use without calling the LLM on every run.
  * ``NetworkExpectation`` records API contracts (method, path, status,
    request/response JSON schemas) so replay fails fast on schema drift,
    not only on UI change.
  * ``Assertion`` supports both DOM predicates (cheap, deterministic) and
    natural-language checks (LLM-evaluated, for cases where "correctness"
    is a business rule, not a selector).

The format is YAML-serializable. ``Spec.load`` / ``Spec.dump`` handle
round-trips. Keep the schema stable and bump ``version`` on breaking
changes so old recordings can be migrated forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SPEC_VERSION = "1"

# Allowed step actions. Keep narrow on purpose — every action the replayer
# needs to implement lives here. Extending this list is a deliberate choice,
# not a free-for-all.
ALLOWED_ACTIONS = {
    "navigate",   # url: str
    "click",      # target: Locator
    "fill",       # target: Locator, value: str
    "press",      # target: Locator, value: str (key, e.g. "Enter")
    "select",     # target: Locator, value: str
    "hover",      # target: Locator
    "wait_for",   # target: Locator
}

# Allowed assertion kinds.
ALLOWED_ASSERTIONS = {
    "natural_language",  # text: str — LLM-evaluated against observed state
    "dom_predicate",     # selector: str, must_exist: bool
    "url_equals",        # url: str
}


@dataclass
class Locator:
    """Identifies an element semantically, with deterministic fallbacks.

    ``label`` is consumed by the LLM when fallbacks fail to match (self-heal).
    ``fallbacks`` are tried in order — first hit wins and is cached.
    """
    label: str
    fallbacks: list[str] = field(default_factory=list)


@dataclass
class NetworkExpectation:
    """An API contract expected during a step.

    ``path`` is a glob-ish pattern (e.g. ``/api/orders/*``). Schema fields
    are JSON Schema documents validated by ``jsonschema`` at replay time.
    """
    method: str
    path: str
    status: int
    request_schema: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None


@dataclass
class Step:
    """A single action the replayer performs."""
    id: str
    action: str
    target: Locator | None = None
    value: str | None = None
    url: str | None = None
    expected_network: list[NetworkExpectation] = field(default_factory=list)


@dataclass
class Assertion:
    """A post-step or end-of-run check."""
    id: str
    kind: str
    text: str | None = None           # natural_language
    selector: str | None = None       # dom_predicate
    must_exist: bool | None = None    # dom_predicate
    url: str | None = None            # url_equals


@dataclass
class Spec:
    """Top-level recording. One spec = one user journey."""
    name: str
    start_url: str
    steps: list[TestStep] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    version: str = SPEC_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- I/O -------------------------------------------------------------

    def dump(self, path: str | Path) -> Path:
        """Write the spec to a YAML file. Returns the written path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                asdict(self),
                fh,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Spec:
        """Load a spec from a YAML file and validate its shape."""
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Spec:
        steps = [_step_from_dict(s) for s in data.get("steps", [])]
        assertions = [Assertion(**a) for a in data.get("assertions", [])]
        spec = cls(
            name=data["name"],
            start_url=data["start_url"],
            steps=steps,
            assertions=assertions,
            version=data.get("version", SPEC_VERSION),
            metadata=data.get("metadata", {}),
        )
        spec.validate()
        return spec

    # ---- Validation ------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError on any structural problem in the spec."""
        if self.version != SPEC_VERSION:
            raise ValueError(
                f"unsupported spec version {self.version!r}; "
                f"this build understands {SPEC_VERSION!r}"
            )
        seen_ids: set[str] = set()
        for step in self.steps:
            if step.action not in ALLOWED_ACTIONS:
                raise ValueError(f"step {step.id}: unknown action {step.action!r}")
            if step.id in seen_ids:
                raise ValueError(f"duplicate step id: {step.id!r}")
            seen_ids.add(step.id)
            if step.action == "navigate" and not step.url:
                raise ValueError(f"step {step.id}: navigate requires 'url'")
            if step.action in {"fill", "press", "select"} and step.value is None:
                raise ValueError(f"step {step.id}: {step.action} requires 'value'")
            if step.action != "navigate" and step.target is None:
                raise ValueError(f"step {step.id}: {step.action} requires 'target'")
        for a in self.assertions:
            if a.kind not in ALLOWED_ASSERTIONS:
                raise ValueError(f"assertion {a.id}: unknown kind {a.kind!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _step_from_dict(data: dict[str, Any]) -> Step:
    target = data.get("target")
    locator = Locator(**target) if target else None
    nets = [NetworkExpectation(**n) for n in data.get("expected_network", [])]
    return Step(
        id=data["id"],
        action=data["action"],
        target=locator,
        value=data.get("value"),
        url=data.get("url"),
        expected_network=nets,
    )

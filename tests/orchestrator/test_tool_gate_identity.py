"""`is` between two wrapper objects is never true, whatever the scene holds.

`platform_equivalent_variants` asks whether a SHELL gate means something
different under the other dialect. A tool gate has a dialect too — the host
application's Python API — and nothing was asking.

Measured 2026-10-07, a four-requirement Blender task. The plan gated the
node-material step on::

    any(y.to_node is r ... for y in l) and any(y.from_node is r ...)

`l.to_node` builds a **fresh Python wrapper** on every attribute access, so
`is` against a node fetched separately is False whatever the material
contains, while `bpy_struct.__eq__` compares the underlying RNA pointer and
works. No correct material could pass. The step spent 10 turns, the recovery
loop spent 10 more, and the run failed at 109,609 tokens — over a node graph
confirmed correct afterwards by reading `node_tree.links` directly:

    TEX_NOISE Fac  -> VALTORGB Fac
    VALTORGB Color -> BSDF_PRINCIPLED Base Color

The identical task the day before drew a planner that wrote `==` and passed
in six turns, which is what makes this luck of the draw rather than a
property of the task.

Two decisions these tests hold.

**`x is None` is never touched.** A comparison against any literal is left
exactly as written, which covers the one spelling `is` is genuinely for —
and rewriting it would change a correct gate's meaning.

**Believed only if it passes.** Like every dialect variant: if the original
gate was right the variant never runs, and if both fail nothing is adopted.
That is what keeps a heuristic about someone else's API from ever convicting
correct code.
"""
import json

import pytest

from agentchanti.orchestrator import tool_gates


class _Bridge:
    def __init__(self, *names):
        self.names = set(names)

    def owns(self, name):
        return name in self.names


@pytest.fixture
def bridge():
    return _Bridge("blender__execute_code", "blender__get_scene_info")


def _gate(code):
    return "mcp:blender__execute_code " + json.dumps({"code": code})


def _code_of(command):
    return json.loads(command.split(" ", 1)[1])["code"]


# ─── the incident ────────────────────────────────────────────────────

# The run's own gate, shortened to the two comparisons that decided it.
INCIDENT = ("import bpy\n"
            "p = next(x for x in n if x.type == 'BSDF_PRINCIPLED')\n"
            "assert any(y.to_node is r and y.to_socket.name == 'Fac' for y in l)\n"
            "assert any(y.from_node is r and y.to_node is p for y in l)")


def test_the_measured_gate_is_offered_as_equality(bridge):
    variants = tool_gates.equivalent_variants(_gate(INCIDENT), bridge)
    assert len(variants) == 1
    reason, command = variants[0]
    assert reason == "bpy-identity-vs-equality"
    code = _code_of(command)
    assert "y.to_node == r" in code
    assert "y.from_node == r" in code
    assert "y.to_node == p" in code
    assert " is " not in code


def test_the_variant_is_still_a_runnable_tool_gate(bridge):
    """A variant that no longer parses would be worse than no variant."""
    _reason, command = tool_gates.equivalent_variants(
        _gate(INCIDENT), bridge)[0]
    parsed = tool_gates.parse(command, bridge)
    assert parsed is not None and len(parsed) == 1
    assert parsed[0].tool == "blender__execute_code"


# ─── what must never be rewritten ────────────────────────────────────

@pytest.mark.parametrize("code", [
    "assert scene is not None",
    "assert obj.active_material is None",
    "assert flag is True",
    "assert x is False",
    "assert n is 0",
])
def test_a_comparison_with_a_literal_is_left_alone(bridge, code):
    assert tool_gates.equivalent_variants(_gate(code), bridge) == []


def test_a_gate_with_no_identity_comparison_offers_nothing(bridge):
    code = "import bpy\nassert len(bpy.context.scene.objects) == 5"
    assert tool_gates.equivalent_variants(_gate(code), bridge) == []


def test_equality_comparisons_are_untouched(bridge):
    code = ("assert a.b == c\n"
            "assert d is not None\n"
            "assert e.f is g")
    _reason, command = tool_gates.equivalent_variants(_gate(code), bridge)[0]
    out = _code_of(command)
    assert "a.b == c" in out
    assert "d is not None" in out          # the sentinel survived
    assert "e.f == g" in out               # the wrapper comparison moved


def test_a_payload_that_is_not_python_offers_nothing(bridge):
    """A JavaScript payload must not be accused on a Python parse failing."""
    code = "const x = 1; if (a !== b) { throw new Error('no') }"
    assert tool_gates.equivalent_variants(_gate(code), bridge) == []


def test_a_shell_gate_is_not_a_tool_gate(bridge):
    assert tool_gates.equivalent_variants("python -m pytest -q", bridge) == []


def test_no_bridge_offers_nothing():
    assert tool_gates.equivalent_variants(_gate(INCIDENT), None) == []


def test_a_call_with_no_code_argument_is_skipped(bridge):
    gate = "mcp:blender__get_scene_info {}"
    assert tool_gates.equivalent_variants(gate, bridge) == []


# ─── chained and multi-segment gates ─────────────────────────────────

def test_every_segment_of_a_chained_gate_is_rewritten(bridge):
    gate = (_gate("assert a.b is c") + " && "
            + _gate("assert d.e is f"))
    _reason, command = tool_gates.equivalent_variants(gate, bridge)[0]
    segments = command.split(" && ")
    assert len(segments) == 2
    assert "a.b == c" in _code_of(segments[0])
    assert "d.e == f" in _code_of(segments[1])


def test_a_chain_where_only_one_segment_changes_keeps_the_other(bridge):
    gate = (_gate("assert a.b is c") + " && "
            + _gate("assert len(x) == 3"))
    _reason, command = tool_gates.equivalent_variants(gate, bridge)[0]
    first, second = command.split(" && ")
    assert "a.b == c" in _code_of(first)
    assert "len(x) == 3" in _code_of(second)


def test_is_not_becomes_not_equal(bridge):
    _reason, command = tool_gates.equivalent_variants(
        _gate("assert a.b is not c.d"), bridge)[0]
    assert "a.b != c.d" in _code_of(command)


# ─── the wiring ──────────────────────────────────────────────────────

def test_the_loop_asks_a_tool_gate_for_tool_variants():
    """The defect is a missing CALL: `platform_equivalent_variants` had
    nothing to say about a tool gate and was the only thing being asked."""
    import inspect

    from agentchanti.orchestrator import agent_loop

    src = inspect.getsource(agent_loop)
    assert "def _gate_variants(" in src
    assert "tool_gates.equivalent_variants(cmd, bridge)" in src
    # and the retry must go through it rather than the shell-only source
    assert "for reason, variant in _gate_variants(verify_cmd):" in src

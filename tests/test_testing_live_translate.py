"""Tests for agentchanti.testing._live_translate — pure Python."""

from __future__ import annotations

from agentchanti.testing._live_translate import (
    extract_evaluate_result,
    synthesize_selector,
    translate_js_event,
)
from agentchanti.testing.trace import ElementContext


# ---- extract_evaluate_result ----------------------------------------------

REAL_OBJECT_RESPONSE = (
    '### Result\n'
    '{\n  "events": [\n    {"type": "click", "clientX": 42}\n  ],\n'
    '  "url": "https://example.com",\n  "missing": false\n}\n'
    '### Ran Playwright code\n```js\nawait page.evaluate(...);\n```'
)

REAL_STRING_RESPONSE = (
    '### Result\n"hello"\n'
    '### Ran Playwright code\n```js\n...\n```'
)


def test_extract_object_response():
    val = extract_evaluate_result(REAL_OBJECT_RESPONSE)
    assert isinstance(val, dict)
    assert val["url"] == "https://example.com"
    assert val["missing"] is False
    assert val["events"][0]["clientX"] == 42


def test_extract_string_response():
    assert extract_evaluate_result(REAL_STRING_RESPONSE) == "hello"


def test_extract_empty_input_returns_empty_dict():
    assert extract_evaluate_result("") == {}
    assert extract_evaluate_result(None) == {}  # type: ignore[arg-type]


def test_extract_missing_result_block_returns_empty():
    assert extract_evaluate_result("garbage with no markers") == {}


def test_extract_result_without_trailing_header():
    """If Playwright omits the trailing 'Ran Playwright code' header,
    we still try to recover the body."""
    text = '### Result\n{"a": 1}\n'
    val = extract_evaluate_result(text)
    assert val == {"a": 1}


def test_extract_array_response():
    text = (
        '### Result\n[1, 2, 3]\n'
        '### Ran Playwright code\n```js\n```'
    )
    assert extract_evaluate_result(text) == [1, 2, 3]


# ---- synthesize_selector --------------------------------------------------

def test_synthesize_priority_data_testid_wins():
    sel = synthesize_selector({
        "data_testid": "submit-btn",
        "id": "submit",
        "role": "button",
        "text": "Sign in",
        "tag": "button",
    })
    assert sel == "[data-testid=submit-btn]"


def test_synthesize_falls_back_to_id():
    sel = synthesize_selector({
        "id": "email",
        "role": "textbox",
        "text": "Email",
        "tag": "input",
    })
    assert sel == "#email"


def test_synthesize_role_name_when_no_id():
    sel = synthesize_selector({
        "role": "button",
        "text": "Sign in",
        "tag": "button",
    })
    assert sel == 'role=button name="Sign in"'


def test_synthesize_aria_label_preferred_over_text():
    sel = synthesize_selector({
        "role": "button",
        "aria_label": "Open menu",
        "text": "Some long visible text",
        "tag": "button",
    })
    assert sel == 'role=button name="Open menu"'


def test_synthesize_text_when_no_role():
    sel = synthesize_selector({"text": "Click me", "tag": "div"})
    assert sel == "text=Click me"


def test_synthesize_tag_only_when_nothing_else():
    sel = synthesize_selector({"tag": "div"})
    assert sel == "div"


def test_synthesize_handles_non_dict():
    assert synthesize_selector("not a dict") == "unknown"  # type: ignore[arg-type]


# ---- translate_js_event ---------------------------------------------------

def _click_event(**el_overrides):
    el = {
        "tag": "button",
        "id": None,
        "classes": ["btn"],
        "data_testid": "go-btn",
        "aria_label": None,
        "role": "button",
        "name": None,
        "text": "Go",
    }
    el.update(el_overrides)
    return {
        "type": "click",
        "clientX": 142,
        "clientY": 380,
        "button": 0,
        "element": el,
    }


def test_translate_click_returns_action_selector_element_coord():
    out = translate_js_event(_click_event())
    assert out is not None
    assert out["action"] == "click"
    assert out["selector_used"] == "[data-testid=go-btn]"
    assert isinstance(out["element"], ElementContext)
    assert out["element"].tag == "button"
    assert out["element"].data_testid == "go-btn"
    assert out["coord"] == {"x": 142, "y": 380}


def test_translate_click_without_coords_omits_coord_key():
    ev = _click_event()
    del ev["clientX"]
    out = translate_js_event(ev)
    assert "coord" not in out


def test_translate_input_carries_value():
    ev = {
        "type": "input",
        "value": "user@example.com",
        "element": {
            "tag": "input", "id": "email", "classes": [],
            "data_testid": None, "aria_label": "Email",
            "role": "textbox", "name": "email", "text": None,
        },
    }
    out = translate_js_event(ev)
    assert out["action"] == "fill"
    assert out["selector_used"] == "#email"
    assert out["value"] == "user@example.com"


def test_translate_input_redacted_value_passes_through():
    """Sensitive values are redacted in JS — Python must preserve the
    redaction marker so it lands in the trace as-is."""
    ev = {
        "type": "input", "value": "***REDACTED***",
        "element": {"tag": "input", "id": "pw", "type": "password",
                    "classes": [], "data_testid": None,
                    "aria_label": None, "role": None, "name": "password",
                    "text": None},
    }
    out = translate_js_event(ev)
    assert out["value"] == "***REDACTED***"


def test_translate_select_change_uses_select_action():
    ev = {
        "type": "change", "value": "uk",
        "element": {"tag": "select", "id": "country",
                    "classes": [], "data_testid": "country-select",
                    "aria_label": None, "role": None, "name": "country",
                    "text": None},
    }
    out = translate_js_event(ev)
    assert out["action"] == "select"
    assert out["value"] == "uk"


def test_translate_keydown_carries_key_in_value_slot():
    ev = {
        "type": "keydown", "key": "Enter",
        "element": {"tag": "input", "id": "search",
                    "classes": [], "data_testid": None,
                    "aria_label": None, "role": None, "name": None,
                    "text": None},
    }
    out = translate_js_event(ev)
    assert out["action"] == "press"
    assert out["value"] == "Enter"


def test_translate_drops_malformed_events():
    assert translate_js_event(None) is None  # type: ignore[arg-type]
    assert translate_js_event({}) is None
    assert translate_js_event({"type": "click"}) is None
    assert translate_js_event({"type": None, "element": {}}) is None


def test_translate_unknown_type_falls_back_to_click():
    """Future-proofing — a JS hook we haven't seen yet should produce
    a degraded but valid event rather than crash the polling loop."""
    ev = {
        "type": "future_event_type",
        "element": {"tag": "div", "id": None, "classes": [],
                    "data_testid": None, "aria_label": None,
                    "role": None, "name": None, "text": "thing"},
    }
    out = translate_js_event(ev)
    assert out["action"] == "click"

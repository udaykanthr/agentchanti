"""Tests for agentchanti.testing.snapshot — parser + selector resolver."""

from __future__ import annotations

from agentchanti.testing.snapshot import (
    Element,
    SelectorKind,
    classify_selector,
    parse_snapshot,
    resolve_selector,
)

REAL_PLAYWRIGHT_SNAPSHOT = """\
### Page
- Page URL: data:text/html,...
### Snapshot
```yaml
- generic [active] [ref=e1]:
  - heading "Test Page" [level=1] [ref=e2]
  - generic [ref=e3]:
    - text: Email
    - textbox "Email" [ref=e4]
    - button "Sign in" [ref=e5]
  - link "Open help" [ref=e6] [cursor=pointer]:
    - /url: /help
    - text: "?"
```
"""


# ---- Parsing ---------------------------------------------------------------

def test_parse_extracts_every_referenced_element():
    els = parse_snapshot(REAL_PLAYWRIGHT_SNAPSHOT)
    refs = [e.ref for e in els]
    assert refs == ["e1", "e2", "e3", "e4", "e5", "e6"]


def test_parse_captures_role_and_accessible_name():
    els = {e.ref: e for e in parse_snapshot(REAL_PLAYWRIGHT_SNAPSHOT)}
    assert els["e2"].role == "heading" and els["e2"].name == "Test Page"
    assert els["e4"].role == "textbox" and els["e4"].name == "Email"
    assert els["e5"].role == "button" and els["e5"].name == "Sign in"
    assert els["e6"].role == "link" and els["e6"].name == "Open help"


def test_parse_captures_bracket_attrs_excluding_ref():
    els = {e.ref: e for e in parse_snapshot(REAL_PLAYWRIGHT_SNAPSHOT)}
    assert els["e1"].attrs.get("active") == ""
    assert els["e2"].attrs.get("level") == "1"
    assert els["e6"].attrs.get("cursor") == "pointer"
    assert "ref" not in els["e1"].attrs


def test_parse_merges_text_children_into_parent_link():
    els = {e.ref: e for e in parse_snapshot(REAL_PLAYWRIGHT_SNAPSHOT)}
    # e6's `- text: "?"` child should surface as link.text
    assert els["e6"].text == "?"
    # And /url: /help should attach as an attr
    assert els["e6"].attrs.get("url") == "/help"


def test_parse_handles_empty_or_malformed_input():
    assert parse_snapshot("") == []
    assert parse_snapshot("no yaml block here") == []
    assert parse_snapshot("```yaml\n- not a valid row\n```") == []


def test_parse_handles_snapshot_without_fence():
    """Sometimes the raw text may not be wrapped in ```yaml``` — the parser
    should still work on bare tree text."""
    bare = "- button \"Go\" [ref=e1]"
    els = parse_snapshot(bare)
    assert len(els) == 1
    assert els[0].role == "button" and els[0].ref == "e1"


# ---- Classification --------------------------------------------------------

def test_classify_semantic_selectors():
    assert classify_selector("text=Sign in") is SelectorKind.SEMANTIC
    assert classify_selector("role=button") is SelectorKind.SEMANTIC
    assert classify_selector("role=button name=\"Sign in\"") is SelectorKind.SEMANTIC
    assert classify_selector("Sign in") is SelectorKind.SEMANTIC  # bare name


def test_classify_css_selectors():
    assert classify_selector("#email") is SelectorKind.CSS
    assert classify_selector(".btn-primary") is SelectorKind.CSS
    assert classify_selector("[data-testid=submit]") is SelectorKind.CSS
    assert classify_selector("button[type=submit]") is SelectorKind.CSS
    assert classify_selector("form > input") is SelectorKind.CSS


def test_classify_coord():
    assert classify_selector("coord=450,300") is SelectorKind.COORDINATE


def test_classify_empty_or_unknown():
    assert classify_selector("") is SelectorKind.UNKNOWN
    assert classify_selector("   ") is SelectorKind.UNKNOWN


# ---- Resolution ------------------------------------------------------------

def _els():
    return parse_snapshot(REAL_PLAYWRIGHT_SNAPSHOT)


def test_resolve_text_selector_matches_accessible_name():
    assert resolve_selector(_els(), "text=Sign in") == "e5"


def test_resolve_text_selector_matches_child_text():
    # e6 has text='?' — text='?' should resolve to e6 even though its
    # accessible name is 'Open help'.
    assert resolve_selector(_els(), "text=?") == "e6"


def test_resolve_role_selector_picks_first_match():
    # Two 'generic' roles in the snapshot — e1 is returned (first).
    assert resolve_selector(_els(), "role=generic") == "e1"


def test_resolve_role_with_name_narrows_the_match():
    assert resolve_selector(_els(), 'role=button name="Sign in"') == "e5"
    assert resolve_selector(_els(), 'role=textbox name="Email"') == "e4"


def test_resolve_bare_name_does_accessible_name_substring():
    # "Sign" should match "Sign in" button
    assert resolve_selector(_els(), "Sign") == "e5"


def test_resolve_returns_none_for_css_selectors():
    # CSS selectors should fall through — the caller dispatches via
    # browser_evaluate, not via ref.
    assert resolve_selector(_els(), "[data-testid=submit-btn]") is None
    assert resolve_selector(_els(), "#email") is None


def test_resolve_returns_none_when_no_match():
    assert resolve_selector(_els(), "text=Nonexistent") is None
    assert resolve_selector(_els(), 'role=button name="Wrong"') is None


def test_resolve_handles_empty_element_list():
    assert resolve_selector([], "text=Sign in") is None

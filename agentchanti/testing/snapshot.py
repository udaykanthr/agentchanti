"""
Parse Playwright MCP's accessibility-tree snapshot text + resolve selectors
to refs.

Playwright MCP returns snapshots as markdown with a ``yaml``-ish fenced
block that looks like::

    ### Snapshot
    ```yaml
    - generic [active] [ref=e1]:
      - heading "Test Page" [level=1] [ref=e2]
      - textbox "Email" [ref=e4]
      - button "Sign in" [ref=e5]
      - link "Open help" [ref=e6] [cursor=pointer]:
        - /url: /help
    ```

Important constraint: the snapshot only exposes accessibility data —
role, accessible name, level, a few a11y states, and the opaque ref. It
does NOT expose ``data-testid``, CSS classes, raw ids, or ``name``
attributes. Selectors that depend on those DOM-only hooks must fall
through to ``browser_evaluate``; ``classify_selector`` flags them.

Public API (kept small on purpose):

  * ``parse_snapshot(raw: str) -> list[Element]``
  * ``classify_selector(selector: str) -> SelectorKind``
  * ``resolve_selector(elements, selector) -> str | None``  # returns ref
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Element:
    """One row of the parsed accessibility tree."""
    ref: str
    role: str
    name: str | None = None
    attrs: dict[str, str] = field(default_factory=dict)
    # `text` is the inline text content Playwright surfaces as a child
    # `text: ...` line. Kept alongside the parent element for matching
    # convenience — it's what the user visually sees.
    text: str | None = None


class SelectorKind(Enum):
    """How a selector should be dispatched."""
    SEMANTIC = "semantic"        # resolvable against the snapshot → ref
    CSS = "css"                  # needs browser_evaluate / querySelector
    COORDINATE = "coordinate"    # coord=X,Y (P2b scope)
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_YAML_FENCE_RE = re.compile(r"```ya?ml\s*\n(.*?)\n```", re.DOTALL)
_REF_RE = re.compile(r"\[ref=([^\]]+)\]")
_BRACKET_ATTR_RE = re.compile(r"\[([A-Za-z_][\w-]*)(?:=([^\]]+))?\]")
# Line like:  - button "Sign in" [ref=e5]
#   or:      - generic [active] [ref=e1]:
#   or:      - text: Email
#   or:      - /url: /help
_ROW_RE = re.compile(
    r"""
    ^(?P<indent>\s*)
    -\s+
    (?P<body>.*?)
    :?\s*$
    """,
    re.VERBOSE,
)


def parse_snapshot(raw: str) -> list[Element]:
    """Extract elements with refs from a Playwright MCP snapshot payload.

    Unreferenced rows (``text: Email``, ``/url: /help``, anything without
    a ``[ref=...]`` attr) are merged into the most recent referenced
    element at a deeper indent — that's where they appear semantically.
    """
    if not raw:
        return []

    yaml_match = _YAML_FENCE_RE.search(raw)
    body = yaml_match.group(1) if yaml_match else raw

    elements: list[Element] = []
    # Stack of (indent, element) so that text/url children can attach to
    # the most recent ancestor.
    stack: list[tuple[int, Element]] = []

    for line in body.splitlines():
        if not line.strip() or not line.lstrip().startswith("-"):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        indent = len(m.group("indent"))
        content = m.group("body").strip()

        # Pop the stack to the current indent level.
        while stack and stack[-1][0] >= indent:
            stack.pop()

        ref_match = _REF_RE.search(content)
        if ref_match:
            el = _parse_element_row(content, ref_match.group(1))
            elements.append(el)
            stack.append((indent, el))
        else:
            # No ref — attach known fields (text:, /url:) to the closest
            # ancestor that does have a ref.
            if stack:
                _merge_child_into(stack[-1][1], content)
    return elements


def _parse_element_row(content: str, ref: str) -> Element:
    """Parse ``button "Sign in" [ref=e5] [cursor=pointer]`` → Element."""
    role = content.split(" ", 1)[0].strip()
    # Accessible name: the first double-quoted token after the role.
    name = None
    name_match = re.search(r'^\S+\s+"([^"]*)"', content)
    if name_match:
        name = name_match.group(1)
    # Other bracket attrs, minus ref.
    attrs: dict[str, str] = {}
    for m in _BRACKET_ATTR_RE.finditer(content):
        key = m.group(1)
        val = m.group(2) or ""
        if key == "ref":
            continue
        attrs[key] = val
    return Element(ref=ref, role=role, name=name, attrs=attrs)


def _merge_child_into(parent: Element, content: str) -> None:
    # Surface two common child kinds:
    #   text: <value>   — visible text content
    #   /url: <value>   — href for links
    if content.startswith("text:"):
        value = content[len("text:"):].strip().strip('"').strip("'")
        # Preserve the earliest-seen text; multiple text children do exist
        # but the first is usually what matters for matching.
        if parent.text is None:
            parent.text = value
    elif content.startswith("/url:"):
        parent.attrs.setdefault("url", content[len("/url:"):].strip())


# ---------------------------------------------------------------------------
# Selector classification + resolution
# ---------------------------------------------------------------------------

def classify_selector(selector: str) -> SelectorKind:
    """Decide how a selector should be dispatched.

    The intent is to keep the dispatcher simple: ref-based for anything
    that can be matched against the accessibility tree, CSS-based for
    everything else.
    """
    s = selector.strip()
    if not s:
        return SelectorKind.UNKNOWN
    if s.startswith("coord="):
        return SelectorKind.COORDINATE
    if s.startswith("text=") or s.startswith("role="):
        return SelectorKind.SEMANTIC
    # Anything that smells like CSS: leading #/./[ or contains brackets /
    # child combinator / pseudo-classes. A bare space is deliberately NOT
    # a CSS signal here — accessible names routinely contain spaces
    # ("Sign in", "Place order"). If you want a CSS descendant selector
    # write `form > input`, which the `>` catches.
    if s[0] in "#.[":
        return SelectorKind.CSS
    if any(c in s for c in "[]>:"):
        return SelectorKind.CSS
    # Bare word(s) — treat as an accessible-name substring probe.
    return SelectorKind.SEMANTIC


def resolve_selector(
    elements: list[Element],
    selector: str,
) -> str | None:
    """Return the ref of the first element matching ``selector``, or None.

    Only semantic selectors are resolved here; CSS/coordinate dialects
    return None so the caller dispatches them via the appropriate escape
    hatch.
    """
    kind = classify_selector(selector)
    if kind is not SelectorKind.SEMANTIC:
        return None

    s = selector.strip()
    role: str | None = None
    name_needle: str | None = None

    if s.startswith("text="):
        name_needle = s[len("text="):].strip().strip('"').strip("'")
    elif s.startswith("role="):
        role, name_needle = _parse_role_selector(s[len("role="):].strip())
    else:
        name_needle = s  # bare word → accessible-name substring match

    for el in elements:
        if role is not None and el.role != role:
            continue
        if name_needle is None:
            return el.ref
        if _matches_name(el, name_needle):
            return el.ref
    return None


def _parse_role_selector(rest: str) -> tuple[str, str | None]:
    """``button name="Sign in"`` → ("button", "Sign in")."""
    name_match = re.search(r'name\s*=\s*"([^"]*)"', rest)
    if name_match:
        role = rest[: name_match.start()].strip()
        return role, name_match.group(1)
    # Just the role.
    return rest.strip(), None


def _matches_name(el: Element, needle: str) -> bool:
    if not needle:
        return True
    hay_name = el.name or ""
    hay_text = el.text or ""
    return needle in hay_name or needle in hay_text

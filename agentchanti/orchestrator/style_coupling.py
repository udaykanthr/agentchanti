"""Do the classes the markup renders actually exist in the stylesheet?

Two files can each be individually correct and jointly wrong. A component
step writes `site-footer__content`; a stylesheet step, running in the same
wave and unable to see it, writes `.site-footer__inner`. Both steps pass
their own gates, the suite passes, the production build passes — an
unmatched CSS class is still valid CSS — and the page renders unstyled.

WHY THIS EXISTS
---------------
Four of six consecutive runs on one Vite/React project drifted this way,
once completely::

    run    classes used    styled
    13:12       7             3
    13:39       8             5
    21:28       7             0      <- every element unstyled
    21:51       8             4

The 21:51 run is the one that settles the argument. Its acceptance gate
had already been strengthened, and asserted eight separate structural
properties of the stylesheet — full-bleed override, background AND
colour, a max-width container, a grid, a hover treatment, a divider, a
flex utility row, a responsive stacking rule. Every one was true. All
eight described selectors the markup never renders, so the gate passed on
a visibly broken footer.

No amount of single-file assertion can catch this: neither file is wrong
on its own. Only the join is.

SCOPE AND REFUSALS
------------------
Only "used but never defined" is treated as a defect — that is the part a
visitor sees. Orphaned rules are reported as context, never as failure:
dead CSS is untidy, not broken, and a project may legitimately keep
styles for markup rendered elsewhere.

Everything ambiguous is refused rather than guessed, because a false
accusation here sends a correct run into a fix loop:

* a utility or component framework in the dependencies (Tailwind,
  Bootstrap, MUI, …) puts class names in the markup that the project's
  own stylesheets are not expected to define;
* Sass/SCSS nesting composes selectors (`.a { &__b {} }`) that no static
  scan of the text can reconstruct;
* CSS Modules rename classes at build time, so the literal text never
  matches;
* a dynamic `className={...}` cannot be resolved at all — those
  expressions are skipped, though string literals in the same file are
  still checked.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

_MARKUP_SUFFIXES = (".jsx", ".tsx")
_STYLE_SUFFIXES = (".css",)
_NESTED_STYLE_SUFFIXES = (".scss", ".sass", ".less", ".styl")
_SKIP_DIRS = {"node_modules", "dist", "build", ".git", "__pycache__",
              ".next", "coverage", ".agentchanti", "venv", ".venv"}

# Dependencies that ship their own class vocabulary. Their presence makes
# "this class is not in our CSS" meaningless.
_FRAMEWORK_MARKERS = (
    "tailwindcss", "bootstrap", "bulma", "foundation-sites", "@mui/",
    "antd", "@chakra-ui/", "semantic-ui", "materialize-css", "primereact",
    "@radix-ui/themes", "daisyui",
)

_CLASSNAME_RE = re.compile(r'className\s*=\s*"([^"]*)"')
_DYNAMIC_CLASSNAME_RE = re.compile(r'className\s*=\s*\{')
_SELECTOR_RE = re.compile(r'\.(-?[_a-zA-Z][\w-]*)')
_CSS_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_TAILWIND_DIRECTIVE_RE = re.compile(r'@tailwind\b|@apply\b')


@dataclass
class StyleDrift:
    """Classes rendered by markup that no project stylesheet defines."""

    unstyled: dict[str, list[str]] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)
    markup_files: int = 0
    style_files: int = 0

    @property
    def broken(self) -> bool:
        return bool(self.unstyled)

    def describe(self) -> str:
        lines = []
        for cls in sorted(self.unstyled):
            where = ", ".join(sorted(self.unstyled[cls])[:3])
            lines.append(f"  {cls}  (used in {where})")
        text = ("These classes are rendered by the markup but no project "
                "stylesheet defines them, so those elements render "
                "unstyled:\n" + "\n".join(lines))
        if self.orphans:
            text += ("\n\nLikely counterpart — rules defined but never "
                     "rendered (the two files disagree on naming):\n  "
                     + ", ".join(sorted(self.orphans)[:12]))
        return text


def _walk(root: str, suffixes: tuple[str, ...]) -> list[str]:
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if name.endswith(suffixes):
                found.append(os.path.join(base, name))
    return found


def _uses_a_class_framework(root: str) -> bool:
    path = os.path.join(root, "package.json")
    try:
        with open(path, encoding="utf-8") as fh:
            pkg = json.load(fh)
    except (OSError, ValueError):
        return False
    names = " ".join(list(pkg.get("dependencies") or {})
                     + list(pkg.get("devDependencies") or {}))
    return any(marker in names for marker in _FRAMEWORK_MARKERS)


def find_style_drift(root: str = ".") -> StyleDrift | None:
    """Classes the markup renders that no stylesheet defines, or None.

    None means "not judged" — an unrecognised or unsupported layout, not
    a clean bill of health.
    """
    if _uses_a_class_framework(root):
        return None
    if _walk(root, _NESTED_STYLE_SUFFIXES):
        return None                      # Sass nesting composes selectors

    markup = _walk(root, _MARKUP_SUFFIXES)
    styles = [p for p in _walk(root, _STYLE_SUFFIXES)
              if not p.endswith(".module.css")]
    if not markup or not styles:
        return None

    defined: set[str] = set()
    for path in styles:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = _CSS_COMMENT_RE.sub("", fh.read())
        except OSError:
            return None
        if _TAILWIND_DIRECTIVE_RE.search(text):
            return None                  # utility pipeline, not our names
        defined.update(_SELECTOR_RE.findall(text))

    used: dict[str, set[str]] = {}
    for path in markup:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        rel = os.path.relpath(path, root).replace("\\", "/")
        for literal in _CLASSNAME_RE.findall(text):
            for cls in literal.split():
                used.setdefault(cls, set()).add(rel)

    if not used:
        return None

    drift = StyleDrift(markup_files=len(markup), style_files=len(styles))
    for cls, where in used.items():
        if cls not in defined:
            drift.unstyled[cls] = sorted(where)
    # Orphans are reported only as the counterpart of a real break — on
    # their own they are dead CSS, which is untidy rather than wrong.
    if drift.unstyled:
        rendered = set(used)
        prefixes = {c.split("__")[0] for c in drift.unstyled}
        drift.orphans = sorted(
            d for d in defined
            if d not in rendered and d.split("__")[0] in prefixes)
    return drift


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    root = argv[0] if argv else "."
    drift = find_style_drift(root)
    if drift is None:
        print("style-coupling: not judged (framework, nesting or no markup)")
        return 0
    if not drift.broken:
        print(f"style-coupling: OK — every class in {drift.markup_files} "
              f"markup file(s) is defined across {drift.style_files} "
              f"stylesheet(s)")
        return 0
    print(drift.describe())
    return 1


if __name__ == "__main__":       # usable as a gate: exit 1 == drift
    raise SystemExit(main())

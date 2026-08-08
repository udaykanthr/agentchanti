"""A step's target must actually be part of the product.

Observed on a Vite/React project: `src/main.jsx` carried the app's only
stylesheet import (`./index.css`) and `src/App.jsx` imported no CSS at
all, so the scaffold's leftover `src/App.css` was never bundled.
Successive "restyle the header" runs targeted `App.css` and wrote twelve
`.site-header` rules including a full dark palette; the built bundle
contained one. Nothing in the browser changed across many runs, and no
gate could see it — the last asserted seven separate strings about
`App.css` and passed on all seven.
"""

import pytest

from agentchanti.orchestrator.plan_step import PlanStep
from agentchanti.orchestrator.reachability import (
    reachable_files,
    unreachable_stylesheet_reason,
)

# The real project's shape, reduced to what the module graph needs.
PROJECT = {
    "app/index.html": '<script type="module" src="/src/main.jsx"></script>',
    "app/package.json": "{}",
    "app/src/main.jsx": "import './index.css'\nimport App from './App.jsx'\n",
    "app/src/App.jsx": "import { HomePage } from './components/HomePage'\n",
    "app/src/components/HomePage.jsx": "export function HomePage(){}\n",
    "app/src/index.css": ".site-header { color: #000; }\n",
    "app/src/App.css": ".site-header { background: #1e293b; }\n",  # orphan
}


def _reader(files):
    return lambda path: files.get(path.replace("\\", "/"))


def _step(step_id, target, imports=None, imported_by=None):
    s = PlanStep(id=step_id, step_type="CODE", description="style")
    s.target_files = [target]
    if imports:
        s.imports_from = imports
    if imported_by:
        s.imported_by = imported_by
    return s


class TestReachableFiles:
    def test_walks_the_module_graph_from_the_entry_point(self):
        found = reachable_files("app", _reader(PROJECT))
        assert "app/src/main.jsx" in found
        assert "app/src/App.jsx" in found
        assert "app/src/components/HomePage.jsx" in found
        assert "app/src/index.css" in found

    def test_the_unimported_stylesheet_is_absent(self):
        assert "app/src/App.css" not in reachable_files("app", _reader(PROJECT))

    def test_an_unrecognisable_layout_is_refused_not_guessed(self):
        # No entry point → None ("cannot judge"), never an empty set,
        # which would report every file in the project as an orphan.
        assert reachable_files("app", _reader({"app/README.md": "x"})) is None


class TestUnreachableStylesheet:
    def test_flags_the_real_orphan(self):
        why = unreachable_stylesheet_reason(
            _step("1.1", "app/src/App.css"), [], _reader(PROJECT))
        assert why is not None
        assert "not reachable" in why

    def test_the_loaded_stylesheet_is_fine(self):
        assert unreachable_stylesheet_reason(
            _step("1.1", "app/src/index.css"), [], _reader(PROJECT)) is None

    def test_a_stylesheet_a_later_step_imports_is_not_an_orphan(self):
        """`create the file, then wire it up` is the ordinary shape."""
        css = _step("1.1", "app/src/theme.css")
        wire = _step("2.1", "app/src/main.jsx",
                     imports={"app/src/theme.css": []})
        assert unreachable_stylesheet_reason(
            css, [css, wire], _reader(PROJECT)) is None

    def test_imported_by_wires_the_step_s_own_target(self):
        """The two declarations point OPPOSITE ways.

        `imports_from` names the file a step consumes; `imported_by` names
        the consumers of the step's own target. Pooling them reported a
        step declaring `imported_by: main.jsx` as an orphan.
        """
        css = _step("1.1", "app/src/theme.css",
                    imported_by=["app/src/main.jsx"])
        assert unreachable_stylesheet_reason(
            css, [css], _reader(PROJECT)) is None

    def test_a_new_stylesheet_nothing_wires_is_still_flagged(self):
        lone = _step("1.1", "app/src/theme.css")
        assert unreachable_stylesheet_reason(
            lone, [lone], _reader(PROJECT)) is not None

    @pytest.mark.parametrize("target", ["app/src/App.jsx", "app/README.md"])
    def test_non_stylesheet_targets_are_not_judged(self, target):
        """The claim is only safe for CSS.

        An unreferenced JS module may still be reached by a dynamic
        import, a lazy route or `import.meta.glob`, so the same reasoning
        does not transfer.
        """
        assert unreachable_stylesheet_reason(
            _step("1.1", target), [], _reader(PROJECT)) is None

    def test_an_unreadable_project_is_refused(self):
        assert unreachable_stylesheet_reason(
            _step("1.1", "app/src/App.css"), [], lambda p: None) is None

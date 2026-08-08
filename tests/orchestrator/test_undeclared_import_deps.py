"""A step's real imports outrank the planner's `imports:` line.

`imports:` is the planner's opinion and it is optional. When a step that
edits an EXISTING file declares `imports: none`, two mechanisms fail at
once, because both read only that declaration:

  * `fix_import_dependencies` adds no edge, so producer and consumer land
    in the same wave and run CONCURRENTLY;
  * `build_step_context` injects no sibling, so neither step can see the
    other's content even if it wanted to.

Observed: `src/App.jsx` (first line `import './App.css'`) and
`src/App.css` were both declared `imports: none`, scheduled `[[0, 1]]`,
and written in parallel. The markup used `site-footer__nav-title` while
the stylesheet defined `site-footer__heading` — 3 of 8 classes unstyled,
6 CSS rules matching nothing. Tests and build both passed, because
unmatched CSS classes are valid CSS.
"""

import pytest

from agentchanti.orchestrator.plan_step import (
    PlanStep,
    _resolve_import_to_file,
    build_step_context,
    build_waves,
    fix_import_dependencies,
)


class _Memory:
    def __init__(self, files):
        self._f = dict(files)

    def get(self, path):
        return self._f.get(path)

    def all_files(self):
        return dict(self._f)


JSX = "import './App.css'\n\nexport default function App() { return null }\n"
CSS = ".site-footer { display: grid; }\n"


def _reader(files):
    return lambda p: files.get(p)


class TestRelativeImportResolution:
    """`./App.css` inside `src/App.jsx` means `src/App.css`."""

    MEM = _Memory({
        'src/App.jsx': JSX,
        'src/App.css': CSS,
        'src/components/Card.jsx': 'x',
        'src/styles/theme.css': 'y',
    })

    @pytest.mark.parametrize("spec,from_file,expected", [
        ('./App.css', 'src/App.jsx', 'src/App.css'),
        ('./components/Card', 'src/App.jsx', 'src/components/Card.jsx'),
        ('../App.css', 'src/components/Card.jsx', 'src/App.css'),
        ('../styles/theme.css', 'src/components/Card.jsx',
         'src/styles/theme.css'),
    ])
    def test_resolves_against_the_importing_file(self, spec, from_file,
                                                 expected):
        assert _resolve_import_to_file(
            spec, self.MEM, None, from_file=from_file) == expected

    def test_without_the_importer_it_cannot_be_resolved(self):
        # Pins the defect: this returned None, which is why the CSS was
        # never injected. `.replace('.', '/')` also mangles it to
        # '//App/css', so no later branch can rescue it either.
        assert _resolve_import_to_file('./App.css', self.MEM, None) is None

    def test_unknown_import_is_still_unresolved(self):
        assert _resolve_import_to_file(
            './Nope.css', self.MEM, None, from_file='src/App.jsx') is None


class TestWaveOrdering:
    def _steps(self):
        jsx = PlanStep(id='1.1', step_type='CODE', description='markup')
        jsx.target_files = ['src/App.jsx']
        jsx.index = 0
        css = PlanStep(id='1.2', step_type='CODE', description='styles')
        css.target_files = ['src/App.css']
        css.index = 1
        return jsx, css

    def test_undeclared_import_serialises_the_two_steps(self):
        jsx, css = self._steps()
        files = {'src/App.jsx': JSX, 'src/App.css': CSS}
        fixes = fix_import_dependencies([jsx, css], read_file=_reader(files))
        assert any('undeclared' in f for f in fixes), fixes
        assert css.id in jsx.depends_on
        waves = [[s.id for s in w] for w in build_waves([jsx, css])]
        assert waves == [['1.2'], ['1.1']], waves

    def test_without_a_reader_they_share_a_wave(self):
        """Pins the pre-fix behaviour — otherwise the test above proves nothing."""
        jsx, css = self._steps()
        fix_import_dependencies([jsx, css], read_file=None)
        waves = [[s.id for s in w] for w in build_waves([jsx, css])]
        assert waves == [['1.1', '1.2']], waves

    def test_a_file_the_plan_does_not_produce_adds_no_edge(self):
        jsx, _css = self._steps()
        # App.css exists but no step targets it — nothing to depend on.
        files = {'src/App.jsx': JSX}
        assert fix_import_dependencies([jsx], read_file=_reader(files)) == []
        assert jsx.depends_on == []

    def test_package_imports_are_ignored(self):
        jsx, css = self._steps()
        files = {'src/App.jsx': "import React from 'react'\n",
                 'src/App.css': CSS}
        fix_import_dependencies([jsx, css], read_file=_reader(files))
        assert jsx.depends_on == []

    def test_no_self_dependency(self):
        css = PlanStep(id='2.1', step_type='CODE', description='self')
        css.target_files = ['src/App.jsx', 'src/App.css']
        css.index = 0
        fix_import_dependencies([css], read_file=_reader(
            {'src/App.jsx': JSX, 'src/App.css': CSS}))
        assert css.depends_on == []


class TestContextInjection:
    def test_the_stylesheet_reaches_the_markup_step(self):
        step = PlanStep(id='1.1', step_type='CODE', description='markup')
        step.target_files = ['src/App.jsx']          # imports: none declared
        mem = _Memory({'src/App.jsx': JSX, 'src/App.css': CSS})
        files = build_step_context(step, [step], mem)
        assert 'src/App.css' in files, sorted(files)
        assert files['src/App.css'] == CSS

"""Markup and stylesheet must agree on class names.

Two files can each be individually correct and jointly wrong. A component
step writes `site-footer__content`; a stylesheet step in the same wave,
unable to see it, writes `.site-footer__inner`. Both gates pass, the
suite passes, the production build passes — unmatched CSS is still valid
CSS — and the page renders unstyled.

Four of six consecutive runs on one project drifted this way, once
completely (7 classes used, 0 styled). The decisive case had a gate
asserting eight structural properties of the stylesheet — background,
colour, max-width container, grid, hover, divider, flex utility row,
responsive stacking — all eight true, all eight describing selectors the
markup never rendered.
"""

import json

import pytest

from agentchanti.orchestrator.style_coupling import find_style_drift

FOOTER_JSX = '''
export function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer__content">
        <nav className="site-footer__nav"><a href="#x">Home</a></nav>
      </div>
    </footer>
  )
}
'''


def _project(tmp_path, files, with_pkg=True):
    if with_pkg:
        (tmp_path / "package.json").write_text(
            json.dumps({"dependencies": {"react": "^18"}}), encoding="utf-8")
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path)


class TestDetection:
    def test_the_real_drift_is_caught(self, tmp_path):
        # The stylesheet names the container __inner; the markup renders
        # __content. This is the shape that shipped an unstyled footer.
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer { background: #000; }\n"
                             ".site-footer__inner { width: 60rem; }\n"
                             ".site-footer__nav { display: grid; }\n",
        })
        drift = find_style_drift(root)
        assert drift is not None and drift.broken
        assert "site-footer__content" in drift.unstyled
        # and the counterpart is surfaced, because it is the explanation
        assert "site-footer__inner" in drift.orphans

    def test_total_drift_reports_every_class(self, tmp_path):
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".footer { background: #000; }\n"
                             ".footer__inner { width: 60rem; }\n",
        })
        drift = find_style_drift(root)
        assert set(drift.unstyled) == {
            "site-footer", "site-footer__content", "site-footer__nav"}

    def test_a_matching_pair_is_clean(self, tmp_path):
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n.site-footer__content {}\n"
                             ".site-footer__nav {}\n",
        })
        drift = find_style_drift(root)
        assert drift is not None
        assert not drift.broken
        assert drift.unstyled == {}

    def test_classes_may_live_in_any_project_stylesheet(self, tmp_path):
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n",
            "src/extra.css": ".site-footer__content {}\n.site-footer__nav {}\n",
        })
        assert not find_style_drift(root).broken

    def test_orphans_alone_are_not_a_failure(self, tmp_path):
        """Dead CSS is untidy, not broken — and may style markup elsewhere."""
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n.site-footer__content {}\n"
                             ".site-footer__nav {}\n.totally-unused {}\n",
        })
        drift = find_style_drift(root)
        assert not drift.broken


class TestRefusals:
    """Ambiguity must yield None — a false accusation costs a fix loop."""

    def test_a_utility_framework_is_not_judged(self, tmp_path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "package.json").write_text(
            json.dumps({"dependencies": {"tailwindcss": "^3"}}),
            encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "A.jsx").write_text(
            'export const A = () => <div className="flex gap-2" />',
            encoding="utf-8")
        (root / "src" / "i.css").write_text(".nothing {}", encoding="utf-8")
        assert find_style_drift(str(root)) is None

    def test_tailwind_directives_are_not_judged(self, tmp_path):
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": "@tailwind base;\n.something {}\n",
        })
        assert find_style_drift(root) is None

    def test_sass_nesting_is_not_judged(self, tmp_path):
        # `.a { &__b {} }` composes a selector no text scan reconstructs.
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n",
            "src/theme.scss": ".site-footer { &__content {} }\n",
        })
        assert find_style_drift(root) is None

    def test_css_modules_are_ignored_as_a_source_of_truth(self, tmp_path):
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n.site-footer__content {}\n"
                             ".site-footer__nav {}\n",
            "src/x.module.css": ".renamedAtBuildTime {}\n",
        })
        assert not find_style_drift(root).broken

    def test_dynamic_class_names_are_skipped(self, tmp_path):
        root = _project(tmp_path, {
            "src/A.jsx": 'const A = () => <div className={cx("a", b)} />',
            "src/index.css": ".unrelated {}\n",
        })
        # No string literal to judge → nothing used → not judged.
        assert find_style_drift(root) is None

    def test_a_dynamic_expression_does_not_hide_its_neighbours(self, tmp_path):
        root = _project(tmp_path, {
            "src/A.jsx": 'const A = () => (<><div className={x} />'
                         '<div className="static-one" /></>)',
            "src/index.css": ".unrelated {}\n",
        })
        drift = find_style_drift(root)
        assert "static-one" in drift.unstyled

    @pytest.mark.parametrize("files", [
        {"src/index.css": ".a {}"},                       # no markup
        {"src/A.jsx": 'const A = () => <b className="x" />'},  # no CSS
        {},                                                # empty
    ])
    def test_incomplete_projects_are_not_judged(self, tmp_path, files):
        assert find_style_drift(_project(tmp_path, files)) is None

    def test_vendor_directories_are_skipped(self, tmp_path):
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n.site-footer__content {}\n"
                             ".site-footer__nav {}\n",
            "node_modules/pkg/junk.jsx": '<div className="never-styled" />',
        })
        assert not find_style_drift(root).broken


class TestCommandLineGate:
    """Runnable as a verify command, so a fix loop can be held to it."""

    def test_exit_1_on_drift(self, tmp_path, capsys):
        from agentchanti.orchestrator.style_coupling import main
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n",
        })
        assert main([root]) == 1
        assert "site-footer__content" in capsys.readouterr().out

    def test_exit_0_when_clean(self, tmp_path):
        from agentchanti.orchestrator.style_coupling import main
        root = _project(tmp_path, {
            "src/Footer.jsx": FOOTER_JSX,
            "src/index.css": ".site-footer {}\n.site-footer__content {}\n"
                             ".site-footer__nav {}\n",
        })
        assert main([root]) == 0

    def test_exit_0_when_not_judged(self, tmp_path):
        from agentchanti.orchestrator.style_coupling import main
        assert main([_project(tmp_path, {})]) == 0

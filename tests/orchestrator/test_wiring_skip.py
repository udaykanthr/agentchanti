"""Regression tests for the post-pipeline wiring-verification skip gate.

Wiring verification is an expensive LLM call (60-90s) that checks for
cross-file integration issues. After a successful bulk test run, every
failure mode it looks for would already have crashed the test runner —
running it again is pure waste. ``should_run_wiring_verification`` is the
single source of truth for that gate; these tests pin its behaviour so
the optimisation can never silently regress.

The second test class in this file pins the **router-mount mismatch
override**: when production source uses react-router primitives but
no source file mounts a Router, the bulk-test green light is
*untrustworthy* (BulkTest's test-only retry can wrap the failing
test in <MemoryRouter> as a workaround). The detector forces wiring
verification to run anyway so the LLM-based fixer can land the real
source-side fix.

See: bugfix branch — pipeline.py:should_run_wiring_verification
                    pipeline.py:_detect_router_mount_missing
"""
import unittest
from unittest.mock import MagicMock

from multi_agent_coder.orchestrator.pipeline import (
    _detect_router_mount_missing,
    should_run_wiring_verification,
)


def _make_memory(files: dict[str, str]):
    """Build a minimal mock memory whose ``all_files`` returns *files*."""
    mem = MagicMock()
    mem.all_files.return_value = files
    return mem


class TestShouldRunWiringVerification(unittest.TestCase):
    """Pin the boolean truth table for the wiring-skip gate."""

    # ── Skip cases ─────────────────────────────────────────────────────────

    def test_skip_when_bulk_tests_existed_and_passed(self):
        """The headline optimisation: green bulk tests prove wiring."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/__tests__/App.test.jsx": "...",
        })
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=True,
                wiring_enabled=True,
            )
        )

    def test_skip_when_wiring_disabled_in_config(self):
        """Config opt-out wins regardless of bulk-test state."""
        memory = _make_memory({"src/App.jsx": "..."})
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=False,
            )
        )

    def test_skip_when_pipeline_failed(self):
        """No point verifying wiring on a failed pipeline."""
        memory = _make_memory({"src/App.jsx": "..."})
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=False,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )

    # ── Run cases ──────────────────────────────────────────────────────────

    def test_run_when_no_test_files_exist(self):
        """No tests = wiring is the only integration check we have."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/main.jsx": "...",
        })
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,  # bulk test didn't run
                wiring_enabled=True,
            )
        )

    def test_run_when_bulk_tests_existed_but_failed(self):
        """Failed bulk test does not prove wiring is correct."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/__tests__/App.test.jsx": "...",
        })
        # Note: in practice pipeline_success would also be False here, but
        # the helper handles that case independently — verify both axes.
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )

    def test_run_when_only_metadata_files_present(self):
        """Underscore-prefixed memory keys (e.g. _cmd_output/) are not tests."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "_cmd_output/step_1.txt": "...",
        })
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )

    def test_run_when_only_non_source_files_in_test_dir(self):
        """A snapshot.json inside __tests__/ is not a real test file."""
        memory = _make_memory({
            "src/App.jsx": "...",
            "src/__tests__/snapshot.json": "...",
        })
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=False,
                wiring_enabled=True,
            )
        )


class TestWiringSkipRealLogScenario(unittest.TestCase):
    """Reproduce the exact scenario from the bug report."""

    def test_user_test_fix_run_skips_wiring(self):
        """Task: 'fix all test cases' — bulk tests passed, wiring should skip.

        From the user's logs:
          02:13:01 [INFO] [BulkTest] All tests passed on first run.
          02:13:01 [INFO] [WiringVerification] Starting cross-file wiring check
        """
        memory = _make_memory({
            "myapp/src/App.jsx": "...",
            "myapp/src/main.jsx": "...",
            "myapp/src/components/Header.jsx": "...",
            "myapp/src/__tests__/App.test.jsx": "...",
            "myapp/src/__tests__/main.test.jsx": "...",
            "myapp/src/components/__tests__/Header.test.jsx": "...",
            "myapp/src/components/__tests__/HeroBanner.test.jsx": "...",
        })
        # After the test fix landed, bulk test re-ran green:
        result = should_run_wiring_verification(
            memory,
            pipeline_success=True,
            bulk_test_verif_ok=True,
            wiring_enabled=True,  # default config
        )
        self.assertFalse(
            result,
            "Wiring verification must be skipped after a green bulk test run",
        )


# ── Router-mount mismatch detection ────────────────────────────────────────


# Real Header.jsx pattern from a failing run: imports `Link` from
# react-router-dom and uses it in JSX, no Router mounted anywhere.
_HEADER_USING_LINK = """\
import { useState } from 'react'
import { Link } from 'react-router-dom'

export function Header({ title = 'MyApp' }) {
  return (
    <header>
      <Link to="#home">{title}</Link>
    </header>
  )
}
"""

# main.jsx that does NOT mount a Router — the production wiring bug.
_MAIN_NO_ROUTER = """\
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
"""

# main.jsx that DOES mount BrowserRouter — the healthy version.
_MAIN_WITH_BROWSER_ROUTER = """\
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
"""

# Test file using MemoryRouter — the workaround pattern that masks
# the source-side bug from BulkTest.
_HEADER_TEST_WITH_MEMORY_ROUTER = """\
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Header } from './Header.jsx'

describe('Header', () => {
  it('renders', () => {
    render(<MemoryRouter><Header /></MemoryRouter>)
    expect(screen.getByText('MyApp')).toBeInTheDocument()
  })
})
"""

# A simple App that doesn't import react-router at all.
_HEALTHY_APP = """\
export default function App() {
  return <div>hello</div>
}
"""


class TestDetectRouterMountMissing(unittest.TestCase):
    """Pin the source-side router-mount detector. Each test represents
    a class of project state the detector must categorise correctly."""

    def test_no_router_anywhere_returns_none(self):
        """Healthy single-page app — no react-router imported, nothing
        to verify."""
        memory = _make_memory({
            "myapp/src/App.jsx": _HEALTHY_APP,
            "myapp/src/main.jsx":
                "createRoot(...).render(<App/>)",
        })
        self.assertIsNone(_detect_router_mount_missing(memory))

    def test_router_imported_and_mounted_returns_none(self):
        """Healthy router setup — Link in source, BrowserRouter in
        entry point. Nothing to fix."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx": _MAIN_WITH_BROWSER_ROUTER,
            "myapp/src/App.jsx":
                "import { Header } from './components/Header.jsx'\n"
                "export default function App() { return <Header /> }",
        })
        self.assertIsNone(_detect_router_mount_missing(memory))

    def test_link_in_source_no_router_mounted_fires(self):
        """The reported bug: Header uses Link, main.jsx never wraps
        in BrowserRouter. Detector must catch it."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx": _MAIN_NO_ROUTER,
            "myapp/src/App.jsx":
                "import { Header } from './components/Header.jsx'\n"
                "export default function App() { return <Header /> }",
        })
        result = _detect_router_mount_missing(memory)
        self.assertIsNotNone(result)
        self.assertEqual(result["kind"], "router_mount_missing")
        self.assertIn(
            "myapp/src/components/Header.jsx",
            result["files_using_primitives"],
        )
        # main.jsx and App.jsx are recognised as entry-point candidates.
        self.assertIn("myapp/src/main.jsx", result["entry_candidates"])
        self.assertIn("myapp/src/App.jsx", result["entry_candidates"])

    def test_test_files_ignored_by_detector(self):
        """Tests using <MemoryRouter> must NOT count as a Router mount.
        That workaround is exactly what we're trying to detect."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx": _MAIN_NO_ROUTER,
            # Test file mounts MemoryRouter — must be ignored.
            "myapp/src/components/Header.test.jsx":
                _HEADER_TEST_WITH_MEMORY_ROUTER,
        })
        result = _detect_router_mount_missing(memory)
        self.assertIsNotNone(result)

    def test_router_imported_but_unused_returns_none(self):
        """If a file imports from react-router but uses no primitive
        from it (dead import), there's nothing to fix."""
        memory = _make_memory({
            "myapp/src/utils.js":
                "import 'react-router-dom'\n"
                "export const x = 1\n",
            "myapp/src/main.jsx": _MAIN_NO_ROUTER,
        })
        # No primitive used, so files_using_primitives is empty → None.
        self.assertIsNone(_detect_router_mount_missing(memory))

    def test_navigate_hook_in_source_no_mount_fires(self):
        """``useNavigate`` is also a router primitive — the detector
        must catch it as well as the JSX <Link> case."""
        memory = _make_memory({
            "myapp/src/components/LoginButton.jsx": (
                "import { useNavigate } from 'react-router-dom'\n"
                "export function LoginButton() {\n"
                "  const navigate = useNavigate()\n"
                "  return <button onClick={() => navigate('/x')} />\n"
                "}\n"
            ),
            "myapp/src/main.jsx": _MAIN_NO_ROUTER,
        })
        result = _detect_router_mount_missing(memory)
        self.assertIsNotNone(result)
        self.assertIn(
            "myapp/src/components/LoginButton.jsx",
            result["files_using_primitives"],
        )

    def test_router_provider_counts_as_mount(self):
        """The newer ``<RouterProvider>`` (from createBrowserRouter)
        is also a valid mount — must not trigger the detector."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx":
                "import { RouterProvider } from 'react-router-dom'\n"
                "createRoot(...).render(<RouterProvider router={r} />)\n",
        })
        self.assertIsNone(_detect_router_mount_missing(memory))


class TestRouterMismatchOverridesGate(unittest.TestCase):
    """Pin the gate override: tests-pass alone is not enough when the
    detector finds a router mismatch."""

    def test_router_mismatch_forces_wiring_run_even_when_tests_pass(self):
        """The headline regression — exact log scenario: tests pass
        (because the test wraps in MemoryRouter) but Header.jsx still
        uses Link without a production Router. Wiring verification
        MUST run."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx": _MAIN_NO_ROUTER,
            "myapp/src/App.jsx":
                "import { Header } from './components/Header.jsx'\n"
                "export default function App() { return <Header /> }",
            "myapp/src/components/Header.test.jsx":
                _HEADER_TEST_WITH_MEMORY_ROUTER,
        })
        self.assertTrue(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=True,    # tests "pass"
                wiring_enabled=True,
            ),
            "Wiring verification must run when a router mount is "
            "missing, even when bulk tests just passed.",
        )

    def test_router_mismatch_does_not_override_disabled_config(self):
        """Detector firing must NOT override a config-level opt-out."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx": _MAIN_NO_ROUTER,
        })
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=True,
                wiring_enabled=False,    # config opt-out
            ),
        )

    def test_router_mismatch_does_not_override_failed_pipeline(self):
        """If the pipeline already failed, no wiring verification —
        the detector override only matters when we'd otherwise be
        about to ship a green-but-broken result."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx": _MAIN_NO_ROUTER,
        })
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=False,    # already failed
                bulk_test_verif_ok=True,
                wiring_enabled=True,
            ),
        )

    def test_healthy_router_setup_still_skips_when_tests_pass(self):
        """Sanity check the optimization still applies when wiring is
        actually correct — Router mounted, tests green → skip."""
        memory = _make_memory({
            "myapp/src/components/Header.jsx": _HEADER_USING_LINK,
            "myapp/src/main.jsx": _MAIN_WITH_BROWSER_ROUTER,
            "myapp/src/App.jsx":
                "import { Header } from './components/Header.jsx'\n"
                "export default function App() { return <Header /> }",
            "myapp/src/components/Header.test.jsx":
                _HEADER_TEST_WITH_MEMORY_ROUTER,
        })
        self.assertFalse(
            should_run_wiring_verification(
                memory,
                pipeline_success=True,
                bulk_test_verif_ok=True,
                wiring_enabled=True,
            ),
            "Healthy router setup with green tests must still skip "
            "wiring verification — the optimisation still applies.",
        )


if __name__ == '__main__':
    unittest.main()

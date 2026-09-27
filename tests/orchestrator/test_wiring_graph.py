"""The code graph answers the wiring question before an LLM is paid.

Measured 2026-09-26 over three pre-scaffolded Next.js runs: the wiring
LLM call cost 6,719 prompt tokens per run — 20.7% of the whole run — and
answered `No wiring issues found` every time. Across every real run on
the machine it fired 3 times in 10.

The refusal to judge is what makes skipping safe: a graph that cannot
see the project is NOT a clean graph, and reporting one as the other is
the mistake `empty_suite_reason` exists to prevent.
"""
from agentchanti.orchestrator.wiring_graph import (findings_block,
                                                   wiring_suspects)

# The actual shape of the measured run: two files, one import edge, clean.
NEXT_APP = {
    "app/page.tsx": ("export default function Home() {\n"
                     "  return <main><h1>Northstar</h1></main>;\n}\n"),
    "app/layout.tsx": ("import type { Metadata } from 'next';\n"
                       "export const metadata: Metadata = { title: 'x' };\n"
                       "export default function RootLayout({ children }) {\n"
                       "  return <html><body>{children}</body></html>;\n}\n"),
}


class TestTheMeasuredCase:

    def test_a_clean_next_app_needs_no_llm_call(self):
        r = wiring_suspects(NEXT_APP, language="typescript")
        assert r.can_judge and r.clean and not r.suspects

    def test_the_summary_says_what_was_checked(self):
        """A skip has to be auditable, not silent."""
        r = wiring_suspects(NEXT_APP, language="typescript")
        assert "file(s)" in r.summary() and "suspect(s)" in r.summary()

    def test_prop_shapes_are_declared_unchecked(self):
        """Honest about the one thing the graph genuinely cannot see."""
        r = wiring_suspects(NEXT_APP, language="typescript")
        assert "prop shapes" in r.unjudged


class TestItCanActuallyFail:
    """A check that cannot fail is not a check."""

    def test_an_import_of_a_file_that_does_not_exist(self):
        files = dict(NEXT_APP)
        files["app/page.tsx"] = ("import Hero from './components/Hero';\n"
                                 + NEXT_APP["app/page.tsx"])
        r = wiring_suspects(files, language="typescript")
        assert r.can_judge
        assert any(s.kind == "broken-import" for s in r.suspects)
        assert not r.clean

    def test_a_default_import_from_a_module_with_no_default(self):
        files = {
            "src/App.jsx": "import Button from './Button';\nexport default App;\n",
            "src/Button.jsx": "export function Button() { return null; }\n",
        }
        r = wiring_suspects(files, language="javascript")
        assert any(s.kind == "missing-default-export" for s in r.suspects)

    def test_a_router_mismatch_is_carried_through(self):
        r = wiring_suspects(
            NEXT_APP, language="typescript",
            router_mismatch={"file": "src/main.jsx",
                             "description": "useNavigate with no Router"})
        assert any(s.kind == "router-mount" for s in r.suspects)
        assert not r.clean


class TestItRefusesRatherThanGuessing:
    """`can_judge=False` must never be read as clean."""

    def test_no_files_is_not_clean(self):
        r = wiring_suspects({})
        assert not r.can_judge and not r.clean

    def test_a_language_the_graph_cannot_parse_is_not_clean(self):
        r = wiring_suspects({"main.zig": "const std = @import(\"std\");"})
        assert not r.can_judge and not r.clean
        assert "parses" in r.unjudged

    def test_a_third_party_import_is_never_called_broken(self):
        """`react` and `next/image` are the package manager's problem."""
        files = {"app/page.tsx": ("import Image from 'next/image';\n"
                                  "import React from 'react';\n"
                                  "export default function H() { return null; }\n")}
        r = wiring_suspects(files, language="typescript")
        assert r.clean, [str(s) for s in r.suspects]

    def test_a_file_on_disk_but_not_in_memory_is_not_broken(self, tmp_path):
        """Memory holds what the RUN touched, not what the project has.

        Judging from memory alone would report a correct import of an
        untouched file as broken — and send the model to rewrite code
        that was never wrong.
        """
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "Hero.tsx").write_text("export default 1;",
                                                   encoding="utf-8")
        files = {"app/page.tsx": "import Hero from './Hero';\nexport default 1;\n"}
        r = wiring_suspects(files, language="typescript",
                            project_root=str(tmp_path))
        assert r.clean, [str(s) for s in r.suspects]


class TestPythonRelativeImports:
    """A Python relative import is a dotted MODULE path, not `./path`.

    Reading the dots as `./` resolved `from .models import X` to
    `.models.py`, which exists nowhere — so every ordinary Python package
    would have been reported broken and the model sent to rewrite correct
    code.
    """

    def test_a_sibling_module_resolves(self):
        files = {"app/main.py": "from .models import Thing\n",
                 "app/models.py": "class Thing: pass\n"}
        assert wiring_suspects(files, language="python").clean

    def test_a_package_init_resolves(self):
        files = {"app/main.py": "from .models import Thing\n",
                 "app/models/__init__.py": "class Thing: pass\n"}
        assert wiring_suspects(files, language="python").clean

    def test_climbing_a_package_resolves(self):
        files = {"app/api/main.py": "from ..util import helper\n",
                 "app/util.py": "def helper(): pass\n"}
        assert wiring_suspects(files, language="python").clean

    def test_a_genuinely_missing_module_is_still_caught(self):
        files = {"app/main.py": "from .nope import Thing\n"}
        r = wiring_suspects(files, language="python")
        assert any(s.kind == "broken-import" for s in r.suspects)

    def test_climbing_past_the_root_is_not_a_finding(self):
        """Unresolvable to a path is not the same as proven missing."""
        files = {"main.py": "from ...far import x\n"}
        assert wiring_suspects(files, language="python").clean


class TestTheEscalationPrompt:

    def test_findings_are_named_so_the_model_fixes_not_hunts(self):
        files = dict(NEXT_APP)
        files["app/page.tsx"] = "import X from './nope';\n"
        block = findings_block(wiring_suspects(files,
                                               language="typescript").suspects)
        assert "./nope" in block
        assert "do not look for other issues" in block

    def test_no_suspects_means_no_block(self):
        assert findings_block([]) == ""

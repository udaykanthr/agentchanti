"""When the files answer the language, do not pay a model to answer it.

Measured 2026-09-23 on a Next.js project: pre-analysis spent 2,845 output
tokens to reply "typescript" for a tree holding `tsconfig.json`,
`next-env.d.ts` and `.tsx` files. `detect_language_llm` exists for the
ambiguous case — a `go.mod` inside a Django project — and that case is the
only one it should cost anything for.
"""
from unittest.mock import MagicMock

import pytest

from agentchanti.agents.planner import PlannerAgent
from agentchanti.language import unambiguous_language

NEXT_TS = [
    "my-app/package.json", "my-app/tsconfig.json", "my-app/next-env.d.ts",
    "my-app/next.config.ts", "my-app/app/page.tsx", "my-app/app/layout.tsx",
    "my-app/app/globals.css",
]


class TestTheFilesAnswer:

    def test_the_measured_next_project(self):
        assert unambiguous_language(NEXT_TS) == "typescript"

    def test_a_js_config_does_not_vote_for_javascript(self):
        """next.config.js in a TypeScript project is not evidence of JS."""
        files = NEXT_TS[:3] + ["my-app/next.config.js",
                               "my-app/tailwind.config.js",
                               "my-app/app/page.tsx"]
        assert unambiguous_language(files) == "typescript"

    @pytest.mark.parametrize("files,expected", [
        (["manage.py", "app/models.py", "app/views.py", "requirements.txt"],
         "python"),
        (["go.mod", "main.go", "server/handler.go"], "go"),
        (["Cargo.toml", "src/main.rs", "src/lib.rs"], "rust"),
        (["package.json", "src/index.js", "src/app.js", "src/util.js"],
         "javascript"),
    ])
    def test_ordinary_projects(self, files, expected):
        assert unambiguous_language(files) == expected


class TestWhenItDeclines:

    def test_two_decisive_manifests_disagree(self):
        """The case detect_language_llm was written for."""
        assert unambiguous_language(
            ["manage.py", "requirements.txt", "go.mod", "app/views.py"]) is None

    def test_a_manifest_contradicted_by_the_sources(self):
        assert unambiguous_language(
            ["go.mod", "app.py", "models.py", "views.py", "urls.py"]) is None

    def test_a_mixed_tree_with_no_majority(self):
        assert unambiguous_language(
            ["a.py", "b.py", "c.go", "d.go", "e.rs"]) is None

    def test_no_source_files(self):
        assert unambiguous_language(["README.md", "LICENSE"]) is None
        assert unambiguous_language([]) is None


class TestTheCallIsSkipped:

    def _planner(self):
        llm = MagicMock()
        llm.generate_response.return_value = "typescript"
        return PlannerAgent("Planner", "role", "goal", llm), llm

    def test_no_llm_call_when_the_files_are_clear(self, monkeypatch):
        planner, llm = self._planner()
        calls = []
        monkeypatch.setattr("agentchanti.language.detect_language_llm",
                            lambda **kw: calls.append(kw) or "typescript")

        planner.pre_analyze(task="build a home page", language="javascript",
                            source_files={p: "" for p in NEXT_TS})

        assert calls == [], "the files already answered it"
        assert planner._detected_language == "typescript"

    def test_the_llm_still_runs_when_the_tree_disagrees(self, monkeypatch):
        planner, _ = self._planner()
        calls = []

        def _fake(**kw):
            calls.append(kw)
            return "python"

        monkeypatch.setattr("agentchanti.language.detect_language_llm", _fake)
        planner.pre_analyze(
            task="fix the tests", language="go",
            source_files={p: "" for p in
                          ["go.mod", "manage.py", "requirements.txt",
                           "app/views.py"]})

        assert len(calls) == 1
        assert planner._detected_language == "python"

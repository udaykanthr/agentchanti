"""Tests for the inline test-coverage file filter.

Coverage generation must only target real code files — package markers,
manifests, and docs produce wasted LLM calls or fragile tests (e.g. a test
for src/__init__.py that imports the whole package and drags in GUI deps).
"""

from agentchanti.orchestrator.pipeline import _has_code_ext


class TestHasCodeExt:
    def test_python_source(self):
        assert _has_code_ext("src/snake.py")

    def test_js_source(self):
        assert _has_code_ext("src\\App.jsx")

    def test_markdown_skipped(self):
        assert not _has_code_ext("README.md")

    def test_requirements_skipped(self):
        assert not _has_code_ext("requirements.txt")

    def test_no_extension_skipped(self):
        assert not _has_code_ext("Makefile")

    def test_yaml_skipped(self):
        assert not _has_code_ext(".agentchanti.yaml")

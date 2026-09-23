"""Next names prerendered HTML after the ROUTE, never `page.html`.

Measured 2026-09-23, kimi-k2.7-code. The gate read
`my-app/.next/server/app/page.html`; `app/page.tsx` serves `/`, which the
build emits as `index.html`. 8 turns plus 8 of recovery against a
FileNotFoundError no edit could fix, and the run was failed over an app
whose contract passes 7/7 by hand.
"""
import pytest

from agentchanti.orchestrator.gate_integrity import (
    next_html_output_variant, platform_equivalent_variants)

MEASURED = ("python -c \"import pathlib;html=pathlib.Path("
            "'my-app/.next/server/app/page.html').read_text(encoding='utf-8');"
            "assert 'Start building today' in html\"")


def test_the_measured_gate_is_pointed_at_index_html():
    out = next_html_output_variant(MEASURED)
    assert "my-app/.next/server/app/index.html" in out
    assert "page.html" not in out


def test_it_is_offered_as_a_variant():
    assert "next-route-html" in dict(platform_equivalent_variants(MEASURED))


@pytest.mark.parametrize("path,expected", [
    (".next/server/app/page.html", ".next/server/app/index.html"),
    (".next/server/app/about/page.html", ".next/server/app/about.html"),
    (".next/server/app/blog/post/page.html",
     ".next/server/app/blog/post.html"),
    ("my-app\\.next\\server\\app\\page.html",
     "my-app\\.next\\server\\app\\index.html"),
])
def test_routes_map_to_their_emitted_file(path, expected):
    assert next_html_output_variant(f"type {path}") == f"type {expected}"


@pytest.mark.parametrize("cmd", [
    "type my-app/.next/server/app/index.html",       # already correct
    "node -e \"require('./page.html')\"",            # not under .next
    "type dist/page.html",                           # another bundler's output
    "cd my-app && npm run build",
])
def test_other_paths_are_untouched(cmd):
    assert next_html_output_variant(cmd) is None

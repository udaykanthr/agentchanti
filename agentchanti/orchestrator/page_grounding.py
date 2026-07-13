"""Ground web-page tasks in the ACTUAL rendered pages.

Two consumers:

1. **Pre-analysis** (before planning): render the project's no-argument
   pages and report which lines quoted in the task already appear on a
   plain GET. Users paste the broken screen into the task ("example of
   current screen: ..."); those lines are ground truth. Without this the
   intent stage guesses which template mechanism produces the text —
   observed: help_text misclassified as bound-form validation errors, so
   the plan gated errors on ``is_bound`` while the text the user wanted
   gone kept rendering on load, and the pipeline finished green.

2. **Post-pipeline acceptance** (smoke stage): parse the machine-checkable
   ``Acceptance:`` assertions the task briefing emits and hand them to the
   Django probe, which asserts them against the re-rendered pages. This
   closes the loop: the briefing's "Expected output" used to be prose that
   nothing ever executed.

Django-only for now — the mechanism needs a way to render pages without a
browser, and Django's test client is the one we have.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile

_logger = logging.getLogger(__name__)

# ── Acceptance check parsing ──────────────────────────────────────────

# One assertion per line inside the briefing's ``Acceptance:`` block:
#   - GET /accounts/signup/ MUST_NOT_CONTAIN "150 characters or fewer"
#   - GET /accounts/signup/ MUST_CONTAIN "Create an account"
# The shape is distinctive enough to scan the whole briefing text.
_ACCEPTANCE_LINE_RE = re.compile(
    r'^\s*-\s*GET\s+(\S+)\s+(MUST_CONTAIN|MUST_NOT_CONTAIN)\s+"([^"]+)"',
    re.IGNORECASE | re.MULTILINE,
)

_MAX_ACCEPTANCE_CHECKS = 8


def parse_acceptance_checks(briefing: str) -> list[dict]:
    """Extract acceptance assertions from a task-briefing block.

    Returns ``[{"url": str, "kind": "must_contain"|"must_not_contain",
    "needle": str}, ...]`` — the JSON shape the Django probe consumes.
    """
    if not isinstance(briefing, str):
        # _task_briefing is attached to FileMemory dynamically — anything
        # non-string (absent, mocked, wrong type) simply means no checks.
        return []
    checks: list[dict] = []
    seen: set[tuple] = set()
    for m in _ACCEPTANCE_LINE_RE.finditer(briefing or ""):
        url, kind, needle = m.group(1), m.group(2).lower(), m.group(3)
        if not url.startswith("/"):
            continue
        key = (url, kind, needle)
        if key in seen:
            continue
        seen.add(key)
        checks.append({"url": url, "kind": kind, "needle": needle})
        if len(checks) >= _MAX_ACCEPTANCE_CHECKS:
            break
    return checks


# ── Page rendering (pre-analysis) ─────────────────────────────────────

# Renders every no-argument named route with the test client and writes
# {url: {"status": int, "html": str}} as JSON to argv[2].
# ``reverse_dict`` only lists non-namespaced names, which conveniently
# excludes the admin. ASCII-only: written to disk and run by an arbitrary
# interpreter.
_PAGES_PROBE = '''\
import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", sys.argv[1])
import django
django.setup()
from django.conf import settings
from django.test import Client
from django.urls import NoReverseMatch, get_resolver, reverse

if "testserver" not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS.append("testserver")

urls = ["/"]
for key in list(get_resolver().reverse_dict.keys()):
    if not isinstance(key, str) or "logout" in key:
        continue
    try:
        u = reverse(key)
    except NoReverseMatch:
        continue
    if u not in urls:
        urls.append(u)

client = Client()
pages = {}
for u in urls[:20]:
    try:
        resp = client.get(u, follow=True)
        pages[u] = {"status": resp.status_code,
                    "html": resp.content.decode("utf-8", "replace")[:40000]}
    except Exception as e:
        pages[u] = {"status": -1,
                    "html": "%s: %s" % (type(e).__name__, e)}

with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(pages, f)
print("PAGES_PROBE_DONE")
'''


def _find_django_root(subproject_cwd: str | None = None) -> str | None:
    """Directory holding manage.py: the detected sub-project, the cwd, or
    a first-level child of the cwd."""
    candidates = []
    if subproject_cwd:
        candidates.append(subproject_cwd)
    candidates.append(".")
    try:
        candidates.extend(
            e for e in os.listdir(".")
            if os.path.isdir(e) and not e.startswith((".", "_")))
    except OSError:
        pass
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "manage.py")):
            return cand
    return None


def render_project_pages(executor, django_dir: str) -> dict[str, dict]:
    """Render all no-arg pages of the Django project at *django_dir*.

    Returns ``{url: {"status": int, "html": str}}`` or ``{}`` on any
    failure — this is best-effort context, never a blocker.
    """
    from .smoke_test import _django_settings_module

    settings_module = _django_settings_module(django_dir)
    if settings_module is None:
        return {}

    tmp_dir = tempfile.mkdtemp(prefix="agentchanti_pages_probe_")
    script = os.path.join(tmp_dir, "pages_probe.py")
    out_file = os.path.join(tmp_dir, "pages.json")
    try:
        with open(script, "w", encoding="utf-8") as f:
            f.write(_PAGES_PROBE)
        cmd = f'python "{script}" {settings_module} "{out_file}"'
        ok, out = executor.run_command(
            cmd, timeout=120, cwd=None if django_dir == "." else django_dir)
        if not ok or "PAGES_PROBE_DONE" not in (out or ""):
            _logger.debug("[PageGrounding] Probe did not complete: %s",
                          (out or "")[:300])
            return {}
        with open(out_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        _logger.debug("[PageGrounding] Render failed (non-fatal): %s", exc)
        return {}
    finally:
        for p in (script, out_file):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ── Task-line extraction and matching ─────────────────────────────────

_MIN_LINE_CHARS = 12
_MAX_TASK_LINES = 60
_MAX_REPORT_LINES_PER_URL = 10


def extract_task_page_lines(task: str) -> list[str]:
    """Lines from the task that could be quoted page content.

    Deliberately permissive: prose lines that match no rendered page are
    simply never reported, so over-extraction is harmless.
    """
    lines: list[str] = []
    for raw in (task or "").splitlines():
        line = raw.strip().strip("`")
        if len(line) < _MIN_LINE_CHARS or line in lines:
            continue
        lines.append(line)
        if len(lines) >= _MAX_TASK_LINES:
            break
    return lines


def _normalize(text: str) -> str:
    return " ".join(text.split())


def build_page_grounding(task: str, executor,
                         subproject_cwd: str | None = None) -> str:
    """Pre-planning grounding block, or "" when not applicable.

    Renders the current app and reports which lines quoted in the task
    already appear on a plain GET — the fact the intent stage needs to
    identify WHAT produces the text the user is describing.
    """
    django_dir = _find_django_root(subproject_cwd)
    if django_dir is None:
        return ""
    lines = extract_task_page_lines(task)
    if not lines:
        return ""

    pages = render_project_pages(executor, django_dir)
    if not pages:
        return ""

    matches: dict[str, list[str]] = {}
    for url, page in pages.items():
        if page.get("status") != 200:
            continue
        html = _normalize(page.get("html", ""))
        hits = [ln for ln in lines if _normalize(ln) in html]
        if hits:
            matches[url] = hits[:_MAX_REPORT_LINES_PER_URL]

    rendered_urls = ", ".join(sorted(pages))
    if not matches:
        return (
            "PAGE GROUNDING (live render of the CURRENT app, before any "
            "changes):\n"
            "None of the lines quoted in the task appear on a plain GET of "
            f"the rendered pages ({rendered_urls}). If the task describes "
            "visible text, it only appears after user interaction (e.g. a "
            "form POST) or on a page that needs arguments/login.\n"
        )

    parts = [
        "PAGE GROUNDING (live render of the CURRENT app, before any "
        "changes):",
        "The following lines quoted in the task ALREADY RENDER on a plain "
        "GET of these pages — they are visible on page load, before any "
        "user input or form submission:",
    ]
    for url in sorted(matches):
        parts.append(f"- GET {url} (HTTP 200) contains:")
        parts.extend(f'    "{ln}"' for ln in matches[url])
    parts.append(
        "Use this to identify WHICH template mechanism produces the text "
        "the user is describing: text visible on a fresh GET cannot come "
        "from bound-form validation errors — look at help_text and static "
        "template content instead.")
    parts.append(f"(Pages rendered: {rendered_urls})")
    _logger.info("[PageGrounding] %d line(s) from the task found on %d "
                 "rendered page(s)", sum(len(v) for v in matches.values()),
                 len(matches))
    return "\n".join(parts) + "\n"

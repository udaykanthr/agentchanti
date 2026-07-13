"""Deterministic Django wiring lints — zero LLM cost.

Two bug classes killed consecutive benchmark runs and are perfectly
machine-checkable, yet invisible to ``manage.py check``:

1. **URL-name namespace mismatches**: ``app_name = 'x'`` in an app's
   urls.py namespaces every route as ``x:name``, but views keep calling
   ``redirect('name')`` / ``reverse('name')`` and templates keep writing
   ``{% url 'name' %}`` — NoReverseMatch at request time.
2. **Template tags without their load line**: ``{% static %}`` used
   without ``{% load static %}`` — TemplateSyntaxError on every render.

The checks are conservative: a bare name is only flagged when it is NOT
reachable unnamespaced anywhere but IS defined under some ``app_name``
namespace — so third-party names (admin, auth) never false-positive.
"""

from __future__ import annotations

import re

_APP_NAME_RE = re.compile(
    r"^app_name\s*=\s*['\"]([\w.-]+)['\"]", re.MULTILINE)
_URL_NAME_RE = re.compile(r"name\s*=\s*['\"]([\w-]+)['\"]")
# redirect('name') / reverse('name') / reverse_lazy('name') with a
# literal first argument. URL paths ('/dashboard/') and dotted view
# paths ('app.views.home') are excluded by the character class.
_PY_NAME_REF_RE = re.compile(
    r"\b(?:redirect|reverse|reverse_lazy)\(\s*['\"]([\w-]+)['\"]")
_TEMPLATE_URL_RE = re.compile(r"{%\s*url\s+['\"]([\w:-]+)['\"]")
_STATIC_TAG_RE = re.compile(r"{%\s*static\b")
_LOAD_STATIC_RE = re.compile(r"{%\s*load\s+[^%]*\bstatic\b")


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def check_django_project(files: dict[str, str]) -> list[str]:
    """Lint *files* (path -> content) for Django wiring bugs.

    Returns human-readable error strings, each naming the file, the bad
    reference, and the exact fix — precise enough for a recovery loop to
    apply mechanically. Empty list = clean (or not a Django project:
    without urls.py there are no namespaces to violate).
    """
    namespaced: dict[str, str] = {}   # route name -> app_name
    plain: set[str] = set()           # names reachable without a namespace

    for path, content in files.items():
        if not content or not _norm(path).endswith("urls.py"):
            continue
        names = set(_URL_NAME_RE.findall(content))
        m = _APP_NAME_RE.search(content)
        if m:
            ns = m.group(1)
            for name in names:
                namespaced.setdefault(name, ns)
        else:
            plain |= names

    errors: list[str] = []
    for path, content in files.items():
        p = _norm(path)
        if not content or p.startswith("_"):
            continue

        if p.endswith(".py"):
            for name in sorted(set(_PY_NAME_REF_RE.findall(content))):
                if name in plain:
                    continue
                if name in namespaced:
                    errors.append(
                        f"{path}: redirect/reverse('{name}') — this route "
                        f"is namespaced; use "
                        f"'{namespaced[name]}:{name}' instead")

        elif p.endswith(".html"):
            for ref in sorted(set(_TEMPLATE_URL_RE.findall(content))):
                if ":" in ref or ref in plain:
                    continue
                if ref in namespaced:
                    errors.append(
                        f"{path}: {{% url '{ref}' %}} — this route is "
                        f"namespaced; use "
                        f"'{namespaced[ref]}:{ref}' instead")
            if _STATIC_TAG_RE.search(content) \
                    and not _LOAD_STATIC_RE.search(content):
                errors.append(
                    f"{path}: uses {{% static %}} without "
                    f"{{% load static %}} — add {{% load static %}} at "
                    f"the top of the template")

    return errors

"""Deterministic Django wiring lints — zero LLM cost.

Two bug classes killed consecutive benchmark runs and are perfectly
machine-checkable, yet invisible to ``manage.py check``:

1. **URL-name namespace mismatches**: ``app_name = 'x'`` in an app's
   urls.py namespaces every route as ``x:name``, but views keep calling
   ``redirect('name')`` / ``reverse('name')`` and templates keep writing
   ``{% url 'name' %}`` — NoReverseMatch at request time.
2. **Template tags without their load line**: ``{% static %}`` used
   without ``{% load static %}`` — TemplateSyntaxError on every render.
3. **@login_required without LOGIN_URL**: the project serves login at a
   custom route, but settings never sets ``LOGIN_URL``, so Django
   redirects to its default ``/accounts/login/`` — a 404. Every
   protected view breaks for anonymous users.
4. **tests.py shadowed by a tests/ package**: ``startapp`` scaffolds
   ``app/tests.py``; a later step creates ``app/tests/`` — Django test
   discovery then dies with "module incorrectly imported". Disk-aware:
   the scaffold stub is rarely in the run's file memory.
5. **References to routes that exist nowhere**: ``{% url 'logout' %}``
   with no ``name='logout'`` in any urls.py and contrib.auth urls not
   mounted — NoReverseMatch on every render of that template. Safe to
   assert in generated-from-scratch projects, where every route is
   project-defined; Django's own auth names are exempted whenever
   ``django.contrib.auth.urls`` is mounted.

Namespaced third-party references (``admin:index``) are never flagged.
"""

from __future__ import annotations

import os
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
_LOGIN_REQUIRED_RE = re.compile(
    r"\blogin_required\b|\bLoginRequiredMixin\b")
_LOGIN_URL_RE = re.compile(r"^LOGIN_URL\s*=", re.MULTILINE)
# Django's default LOGIN_URL is served when contrib.auth urls are
# mounted (include('django.contrib.auth.urls') or an accounts/ path).
_AUTH_URLS_RE = re.compile(
    r"django\.contrib\.auth\.urls|['\"]accounts/login")
# Route names provided by django.contrib.auth.urls when mounted.
_AUTH_URL_NAMES = frozenset({
    "login", "logout", "password_change", "password_change_done",
    "password_reset", "password_reset_done", "password_reset_confirm",
    "password_reset_complete",
})


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
    auth_urls_mounted = False
    urlconf_seen = False

    for path, content in files.items():
        if not content or not _norm(path).endswith("urls.py"):
            continue
        urlconf_seen = True
        names = set(_URL_NAME_RE.findall(content))
        if _AUTH_URLS_RE.search(content):
            auth_urls_mounted = True
        m = _APP_NAME_RE.search(content)
        if m:
            ns = m.group(1)
            for name in names:
                namespaced.setdefault(name, ns)
        else:
            plain |= names

    def _unknown(name: str) -> bool:
        """A bare route name that no urls.py defines anywhere."""
        if not urlconf_seen or ":" in name:
            return False
        if name in plain or name in namespaced:
            return False
        if auth_urls_mounted and name in _AUTH_URL_NAMES:
            return False
        return True

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
                elif _unknown(name):
                    errors.append(
                        f"{path}: redirect/reverse('{name}') — no route "
                        f"named '{name}' is defined in any urls.py. Add "
                        f"path(..., name='{name}') to the app's urls.py "
                        f"(or mount django.contrib.auth.urls if it is a "
                        f"Django auth route)")

        elif p.endswith(".html"):
            for ref in sorted(set(_TEMPLATE_URL_RE.findall(content))):
                if ":" in ref or ref in plain:
                    continue
                if ref in namespaced:
                    errors.append(
                        f"{path}: {{% url '{ref}' %}} — this route is "
                        f"namespaced; use "
                        f"'{namespaced[ref]}:{ref}' instead")
                elif _unknown(ref):
                    errors.append(
                        f"{path}: {{% url '{ref}' %}} — no route named "
                        f"'{ref}' is defined in any urls.py. Add "
                        f"path(..., name='{ref}') to the app's urls.py "
                        f"(or mount django.contrib.auth.urls if it is a "
                        f"Django auth route)")
            if _STATIC_TAG_RE.search(content) \
                    and not _LOAD_STATIC_RE.search(content):
                errors.append(
                    f"{path}: uses {{% static %}} without "
                    f"{{% load static %}} — add {{% load static %}} at "
                    f"the top of the template")

    errors.extend(_check_login_url(files, namespaced, plain,
                                   auth_urls_mounted))
    errors.extend(_check_tests_shadow(files))
    return errors


def _check_tests_shadow(files: dict[str, str]) -> list[str]:
    """Both ``app/tests.py`` and ``app/tests/`` existing kills test
    discovery ("module incorrectly imported") — struck three benchmark
    runs. The tests.py side is usually the untracked startapp stub, so
    check the disk as well as the run's file memory.
    """
    apps_with_pkg: set[str] = set()
    apps_with_file: set[str] = set()
    for path in files:
        p = _norm(path)
        if p.startswith("_"):
            continue
        m = re.match(r"^(.+)/tests/[^/]+\.py$", p)
        if m:
            apps_with_pkg.add(m.group(1))
        elif p.endswith("/tests.py"):
            apps_with_file.add(p[: -len("/tests.py")])

    errors: list[str] = []
    for app in sorted(apps_with_pkg):
        stub = f"{app}/tests.py"
        if app in apps_with_file or os.path.isfile(stub):
            errors.append(
                f"{stub}: both tests.py and a tests/ package exist in "
                f"'{app}' — Django test discovery fails with 'module "
                f"incorrectly imported'. Delete {stub} (usually the "
                f"startapp stub) or move its cases into {app}/tests/")
    return errors


def _check_login_url(files: dict[str, str], namespaced: dict[str, str],
                     plain: set[str], auth_urls_mounted: bool) -> list[str]:
    """@login_required with a custom login route needs LOGIN_URL set.

    Without it Django redirects anonymous users to its default
    ``/accounts/login/`` — a 404 unless contrib.auth urls are mounted.
    Observed as the sole surviving failure of an otherwise-green run:
    every file individually correct, one settings line missing.
    """
    settings_path = None
    uses_protection = False
    for path, content in files.items():
        p = _norm(path)
        if not content or p.startswith("_") or not p.endswith(".py"):
            continue
        if p.endswith("settings.py"):
            settings_path = path
        elif _LOGIN_REQUIRED_RE.search(content):
            uses_protection = True

    if not uses_protection or settings_path is None:
        return []
    if _LOGIN_URL_RE.search(files[settings_path] or ""):
        return []
    if auth_urls_mounted:
        return []  # Django's default /accounts/login/ actually resolves

    if "login" in plain:
        suggestion = "LOGIN_URL = 'login'"
    elif "login" in namespaced:
        suggestion = f"LOGIN_URL = '{namespaced['login']}:login'"
    else:
        suggestion = "LOGIN_URL = '<your login route name>'"
    return [
        f"{settings_path}: views use @login_required but LOGIN_URL is "
        f"not set — anonymous users get redirected to Django's default "
        f"/accounts/login/ which has no route (404). Add: {suggestion}"
    ]

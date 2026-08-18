"""Runtime smoke verification — actually launch the app after the pipeline.

Tests can pass while the application crashes or renders nothing at runtime
(GUI apps especially: tests mock the graphics library and never draw a
frame).  This stage launches the project's entry point briefly through the
Executor (which resolves the project venv):

- process crashes within the launch window → the traceback is fed to an LLM
  fix loop (bounded attempts), then the launch is retried;
- process is still running after the window → healthy launch, kill it;
- process exits 0 quickly → CLI-style script, success.

The whole stage is best-effort: no entry point, non-Python projects, or a
headless environment (no display) all skip silently rather than fail.
"""

import logging
import os
import re

from ..cli_display import status_only as _status_only

_logger = logging.getLogger(__name__)

# A module with a __main__ guard is a runnable entry point
_MAIN_GUARD_RE = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']')

# Preferred entry-point basenames, most likely first
_ENTRY_PRIORITY = ("__main__.py", "main.py", "app.py", "run.py", "cli.py", "game.py")

# Crash signatures caused by the environment (headless CI), not the code
_HEADLESS_SIGNATURES = (
    "NoSuchDisplayException",
    "ScreenNotFoundException",
    "no display",
    "cannot connect to display",
    "couldn't connect to display",
    "DISPLAY environment variable",
)

# Traceback file references: File "path", line N
_TB_FILE_RE = re.compile(r'File "([^"]+)"')

_MAX_FIX_ATTEMPTS = 2


def find_python_entrypoint(memory_files: dict[str, str]) -> str | None:
    """Pick the most likely runnable Python entry point from session files."""
    from .pipeline import _is_test_file

    candidates = []
    for path, content in memory_files.items():
        norm = path.replace("\\", "/")
        if not norm.endswith(".py") or norm.startswith("_"):
            continue
        base = norm.rsplit("/", 1)[-1]
        if base == "conftest.py" or _is_test_file(norm):
            continue
        if not isinstance(content, str) or not _MAIN_GUARD_RE.search(content):
            continue
        candidates.append(norm)

    if not candidates:
        return None

    def rank(p: str):
        base = p.rsplit("/", 1)[-1]
        try:
            prio = _ENTRY_PRIORITY.index(base)
        except ValueError:
            prio = len(_ENTRY_PRIORITY)
        return (prio, p.count("/"), len(p))

    return min(candidates, key=rank)


def build_run_command(entry_path: str) -> str:
    """Build the launch command.

    ``python -m pkg.mod`` when the file's own directory is a package
    (has ``__init__.py``) — running such files as scripts breaks their
    relative imports.  Parent levels may be namespace packages (no
    ``__init__.py``); ``-m`` handles those fine.  Otherwise ``python path``.
    """
    norm = entry_path.replace("\\", "/").lstrip("./")
    parts = norm.split("/")
    if len(parts) > 1 and os.path.isfile(
        os.path.join(*parts[:-1], "__init__.py")
    ):
        if parts[-1] == "__main__.py":
            module = ".".join(parts[:-1])
        else:
            module = ".".join(parts)[: -len(".py")]
        return f"python -m {module}"
    return f'python "{norm}"'


def _is_headless_failure(output: str) -> bool:
    low = output.lower()
    return any(sig.lower() in low for sig in _HEADLESS_SIGNATURES)


def _launch(executor, cmd: str) -> tuple[bool, str]:
    """Launch *cmd* in the background and classify the result.

    The Executor's background mode waits ~3s: a crash inside that window
    returns ``(False, traceback)``; still-running means a healthy launch
    (we kill the process tree afterwards).
    """
    baseline = len(executor._background_processes)
    try:
        ok, out = executor.run_command(cmd, background=True)
    finally:
        executor.stop_background_processes_from(baseline)
    return ok, out or ""


# A launch crash that names a package the project environment lacks is an
# ENVIRONMENT finding, and the fix loop below cannot see that: the only
# repair it has is an edit, so the only repair it makes is an edit.
# Measured 2026-08-18 on both benchmark paths — a venv holding nothing but
# pip made `python main.py` crash on `import pygame`, and the fix the
# model returned changed the graphical entry point into a silent fallback
# to headless mode. The app then "launched successfully" and every gate
# stayed green over a Pac-Man that never opens a window. Repairing the
# environment first leaves the app saying what it meant to say.
_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([A-Za-z0-9_.-]+)['\"]")
# The graceful spelling of the same crash, and the reason a bare
# ModuleNotFoundError match is not enough: an app that catches its own
# ImportError prints advice instead of a traceback ("Pygame is required.
# Install it with: python -m pip install pygame") and exits non-zero.
_PIP_HINT_RE = re.compile(
    r"pip install\s+(?:-U\s+|--upgrade\s+)?([A-Za-z0-9_.-]+)", re.IGNORECASE)

# Two, because a project can genuinely be missing two packages, and a
# third round of the same repair means the install is not taking.
_MAX_ENV_REPAIRS = 2


def _dependency_named_by(output: str, memory_files: dict[str, str]) -> str | None:
    """Third-party distribution the crash blames, or None.

    Refuses a name the project itself defines and a name from the standard
    library — installing either fetches whatever squats on PyPI under that
    spelling, which is a far worse outcome than the crash. Module and
    distribution names can differ (``cv2`` ships as ``opencv-python``);
    when they do the install simply fails and the ordinary LLM fix path
    runs untouched, so the guess costs one command and never a verdict.
    """
    import sys

    project = {p.replace("\\", "/").rsplit("/", 1)[-1]
               for p in (memory_files or {})}
    for rx in (_MISSING_MODULE_RE, _PIP_HINT_RE):
        m = rx.search(output or "")
        if not m:
            continue
        name = m.group(1).split(".")[0].strip()
        if len(name) < 2 or name.lower() == "pip":
            continue
        if f"{name}.py" in project or os.path.exists(name):
            continue                      # the project's own module
        if name in getattr(sys, "stdlib_module_names", frozenset()):
            continue
        return name
    return None


def _repair_missing_dependency(name: str, executor) -> bool:
    """Install *name* into the interpreter the app is launched with.

    Silent — and returning False — when the project has no venv of its
    own: installing into the ambient interpreter would be this pipeline
    writing to the user's machine on a guess, the same refusal
    ``PKG_PRESENT`` makes when it resolves ``UNKNOWN``.
    """
    from .agent_loop import _venv_python

    root = os.getcwd()
    py = _venv_python(root)
    if not py:
        _logger.info(
            "[SmokeTest] %s is missing but the project has no venv of its "
            "own — leaving the ambient interpreter alone", name)
        return False
    ok, out = executor.run_command(f'"{py}" -c "import {name}"')
    if ok:
        return False                      # importable — not what crashed us
    _logger.warning(
        "[SmokeTest] launch failed on a missing dependency (%s) — installing "
        "it into the project venv instead of editing the app, which would "
        "fix the crash by removing the feature", name)
    ok, out = executor.run_command(f'"{py}" -m pip install {name}', timeout=300)
    if not ok:
        _logger.warning("[SmokeTest] could not install %s: %s",
                        name, (out or "")[-300:])
    return bool(ok)


def _files_from_traceback(output: str, memory_files: dict[str, str]) -> list[str]:
    """Map traceback file paths back to session files (max 4)."""
    matched: list[str] = []
    for tb_path in _TB_FILE_RE.findall(output):
        tb_norm = tb_path.replace("\\", "/")
        for mem_path in memory_files:
            mem_norm = mem_path.replace("\\", "/").lstrip("./")
            if tb_norm.endswith(mem_norm) and mem_path not in matched:
                matched.append(mem_path)
                break
        if len(matched) >= 4:
            break
    return matched


# Tokens that commonly appear with call syntax in generic error prose and
# would over-match nearly every source file.
_ERROR_SYMBOL_STOPWORDS = frozenset({
    "print", "warn", "warning", "warnings", "exit", "main", "run", "init",
    "str", "int", "type", "format", "len", "open", "input", "call",
    "called", "self", "super",
})


def _files_mentioning_error_symbols(
    crash_output: str, memory_files: dict[str, str]
) -> list[str]:
    """Session files whose content references a symbol the crash text
    names with call syntax (e.g. ``start_render() can only be called
    once``).

    Rescues fix targeting when an app-level try/except swallowed the
    traceback: with no ``File "..."`` lines the loop previously fell back
    to the entry point alone and rewrote the wrong file every attempt
    while the failing call sat in a file the error message named.
    """
    tokens = {
        t for t in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(", crash_output or "")
        if t.lower() not in _ERROR_SYMBOL_STOPWORDS
    }
    if not tokens:
        return []
    from .pipeline import _is_test_file
    matched: list[str] = []
    for path, content in memory_files.items():
        norm = path.replace("\\", "/")
        if not norm.endswith(".py") or _is_test_file(norm):
            continue
        if not isinstance(content, str):
            continue
        if any(f"{t}(" in content for t in tokens):
            matched.append(path)
    return matched


def _same_crash(a: str, b: str) -> bool:
    """True when two crash outputs describe the same failure."""
    if not a or not b:
        return False
    return a.strip()[-400:] == b.strip()[-400:]


def _installed_versions_line(executor, memory_files=None) -> str:
    """Installed-packages prompt block so fixes target the real versions."""
    try:
        from .api_grounding import (get_installed_package_versions,
                                    grounding_packages)
        versions = get_installed_package_versions(executor=executor)
        pkgs = grounding_packages(versions, memory_files)
    except Exception:
        return ""
    if not pkgs:
        return ""
    return (
        "=== INSTALLED PACKAGES (write code against these EXACT versions) "
        "===\n" + ", ".join(pkgs) + "\n\n"
    )


def _probe_fix_files(fix_files: dict[str, str], executor, memory_files) -> list[str]:
    """API-probe a proposed fix before it is written. Best-effort."""
    py_fixes = {p: c for p, c in fix_files.items() if p.endswith(".py")}
    if not py_fixes:
        return []
    try:
        from .api_grounding import (local_top_levels_from_files,
                                    probe_api_usage)
        return probe_api_usage(
            py_fixes, executor,
            local_top_levels=local_top_levels_from_files(memory_files.keys()),
        )
    except Exception as exc:
        _logger.debug("[SmokeTest] API probe failed (non-fatal): %s", exc)
        return []


def _attempt_fix(
    crash_output: str,
    cmd: str,
    memory,
    executor,
    coder,
    entry_path: str,
    stuck_note: str = "",
) -> list[str]:
    """One LLM fix attempt from the crash output. Returns the list of
    files written (empty when no fix was applied)."""
    memory_files = memory.all_files()
    fix_targets = _files_from_traceback(crash_output, memory_files)
    if not fix_targets and entry_path in memory_files:
        fix_targets = [entry_path]
    # Widen with files the error text implicates by symbol name — the
    # traceback may have been swallowed by an app-level try/except.
    for extra in _files_mentioning_error_symbols(crash_output, memory_files):
        if extra not in fix_targets and len(fix_targets) < 4:
            fix_targets.append(extra)
    if not fix_targets:
        return []

    sources = []
    for p in fix_targets:
        sources.append(f"--- {p} ---\n```\n{memory_files[p]}\n```")

    prompt = (
        f"The application crashed when launched with `{cmd}`.\n\n"
        f"=== CRASH OUTPUT ===\n{crash_output[-3000:]}\n\n"
        + _installed_versions_line(executor, memory_files)
        + "=== SOURCE FILES ===\n" + "\n\n".join(sources) + "\n\n"
        "Fix the crash. Pay close attention to the exact error message — it "
        "often states the correct API to use. Reply with the COMPLETE "
        "corrected content of each file that needs changes, using exactly "
        "this format for each file:\n"
        "#### [FILE]: path/to/file.py\n"
        "```python\n"
        "# complete file content — never abbreviate\n"
        "```\n"
        "Only modify the files shown above."
        + stuck_note
    )

    allowed = {p.replace("\\", "/") for p in fix_targets}
    feedback = ""
    fix_files: dict[str, str] = {}
    for ground_attempt in range(2):
        try:
            response = coder.llm_client.generate_response(prompt + feedback)
        except Exception as exc:
            _logger.warning("[SmokeTest] Fix LLM call failed: %s", exc)
            return []

        fix_files = executor.parse_code_blocks(response)
        if not fix_files:
            fix_files = executor.parse_code_blocks_fuzzy(response)

        # Only accept fixes for files the crash implicated
        fix_files = {
            p: c for p, c in (fix_files or {}).items()
            if p.replace("\\", "/").lstrip("./") in allowed
        }
        if not fix_files:
            _logger.warning(
                "[SmokeTest] No applicable fix files in LLM response")
            return []

        # A fix that swaps one missing API for another just burns a launch
        # attempt — probe it first and re-ask with the probe's suggestions.
        api_errs = _probe_fix_files(fix_files, executor, memory_files)
        if not api_errs:
            break
        _logger.warning(
            "[SmokeTest] Proposed fix uses missing APIs (attempt %d): %s",
            ground_attempt + 1,
            "; ".join(e.split(" — ")[0] for e in api_errs))
        feedback = (
            "\n\n=== PREVIOUS ATTEMPT REJECTED ===\n"
            "Your previous fix used APIs that do NOT exist in the installed "
            "package versions:\n"
            + "\n".join(f"- {e}" for e in api_errs)
            + "\nUse the suggested replacements above."
        )
    else:
        _logger.warning(
            "[SmokeTest] Fix still uses missing APIs after re-ask — rejecting")
        return []

    executor.write_files(fix_files)
    memory.update(fix_files)
    _logger.info("[SmokeTest] Applied fix to: %s", list(fix_files.keys()))
    return list(fix_files.keys())


# Django page/template verification.
#
# Executed with the project venv's interpreter inside the Django root.
# argv: 1=settings module, 2=json file of [template_name, expected_path]
# pairs, 3=comma-separated URLs to render, 4=(optional) json file of
# acceptance checks [{"url", "kind", "needle"}]. Exits non-zero on
# failures so it doubles as the recovery loop's verify_cmd. ASCII-only:
# this source is written to disk and parsed by an arbitrary interpreter.
_DJANGO_PROBE = '''\
import json
import os
import sys

# `python <script>` puts the SCRIPT dir on sys.path, not the cwd -- the
# settings package lives in the cwd (the Django project root).
sys.path.insert(0, os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", sys.argv[1])
import django
django.setup()
from django.conf import settings
from django.template.loader import get_template
from django.test import Client
from django.urls import NoReverseMatch, get_resolver, reverse

failures = []

with open(sys.argv[2], encoding="utf-8") as f:
    checks = json.load(f)
for name, expected in checks:
    try:
        t = get_template(name)
        got = os.path.realpath(t.origin.name)
        if os.path.realpath(expected) != got:
            failures.append(
                "SHADOWED: template %r resolves to %s, NOT the created %s. "
                "The created file is invisible to Django - it must live in "
                "an app templates/ dir (APP_DIRS) or a directory listed in "
                "TEMPLATES[0][\\'DIRS\\']." % (name, got, expected))
    except Exception as e:
        failures.append("UNREACHABLE: template %r cannot be loaded: %s: %s"
                        % (name, type(e).__name__, e))

if "testserver" not in settings.ALLOWED_HOSTS:
    settings.ALLOWED_HOSTS.append("testserver")
client = Client()

# Every no-argument named route is cheap ground truth -- render them all,
# not just the explicit list (observed: a task modified signup.html and
# the probe never rendered /accounts/signup/). reverse_dict only lists
# non-namespaced names, which conveniently excludes the admin.
urls = [u for u in sys.argv[3].split(",") if u]
for key in list(get_resolver().reverse_dict.keys()):
    if not isinstance(key, str) or "logout" in key:
        continue
    try:
        u = reverse(key)
    except NoReverseMatch:
        continue
    if u not in urls:
        urls.append(u)

for url in urls[:20]:
    try:
        resp = client.get(url)
        if resp.status_code >= 500:
            failures.append("RENDER_FAILED: GET %s -> HTTP %s"
                            % (url, resp.status_code))
    except Exception as e:
        failures.append("RENDER_CRASH: GET %s raised %s: %s"
                        % (url, type(e).__name__, e))

# Acceptance checks from the task briefing: raw-HTML substring
# assertions on a plain GET (redirects followed). These encode the
# task's requirements -- a green run that fails one of these did not
# accomplish what the user asked for.
if len(sys.argv) > 4 and sys.argv[4]:
    with open(sys.argv[4], encoding="utf-8") as f:
        acceptance = json.load(f)
    for chk in acceptance:
        if chk.get("kind") == "must_resolve":
            # Task-pinned URL: it must exist at this exact path. Any
            # response but 404/5xx counts -- redirects (auth guards)
            # are fine; the address just has to be served.
            try:
                resp = client.get(chk["url"])
                if resp.status_code == 404 or resp.status_code >= 500:
                    failures.append(
                        "ACCEPTANCE_FAILED: GET %s -> HTTP %s, but the "
                        "task names this exact path -- serve it there "
                        "(any 2xx/3xx response)."
                        % (chk["url"], resp.status_code))
            except Exception as e:
                failures.append("ACCEPTANCE_FAILED: GET %s raised %s: %s"
                                % (chk["url"], type(e).__name__, e))
            continue
        try:
            resp = client.get(chk["url"], follow=True)
            html = resp.content.decode("utf-8", "replace")
            status = resp.status_code
        except Exception as e:
            failures.append("ACCEPTANCE_FAILED: GET %s raised %s: %s"
                            % (chk["url"], type(e).__name__, e))
            continue
        if status != 200:
            failures.append(
                "ACCEPTANCE_FAILED: GET %s -> HTTP %s (cannot check %r)"
                % (chk["url"], status, chk["needle"]))
            continue
        present = chk["needle"] in html
        if chk["kind"] == "must_contain" and not present:
            failures.append(
                "ACCEPTANCE_FAILED: GET %s response HTML must contain %r "
                "but it is MISSING" % (chk["url"], chk["needle"]))
        elif chk["kind"] == "must_not_contain" and present:
            failures.append(
                "ACCEPTANCE_FAILED: GET %s response HTML must NOT contain "
                "%r but it is PRESENT (it still renders on a plain GET)"
                % (chk["url"], chk["needle"]))

for line in failures:
    print(line)
print("DJANGO_PROBE_DONE")
sys.exit(1 if failures else 0)
'''


def _find_django_project_dir(memory_files: dict[str, str]) -> str | None:
    """Directory holding manage.py for this run, or None."""
    candidates = ["."]
    for path in memory_files:
        norm = path.replace("\\", "/")
        if "/" in norm:
            top = norm.split("/", 1)[0]
            if top not in candidates and not top.startswith("_"):
                candidates.append(top)
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "manage.py")):
            return cand
    return None


def _django_settings_module(django_dir: str) -> str | None:
    """Find the '<pkg>.settings' module inside the Django root."""
    try:
        for entry in os.listdir(django_dir):
            if os.path.isfile(os.path.join(django_dir, entry, "settings.py")):
                return f"{entry}.settings"
    except OSError:
        pass
    return None


def _django_template_checks(memory_files: dict[str, str],
                            django_dir: str) -> list[tuple[str, str]]:
    """(template_name, absolute_expected_path) for session template files.

    The template NAME is the path after the last ``templates/`` segment —
    what Django's loader is asked for. Pre-existing files resolve to
    themselves and pass; a new file shadowed by an old one under the same
    name is exactly the failure this exists to catch.
    """
    checks: list[tuple[str, str]] = []
    for path in memory_files:
        # Collapse duplicate separators — malformed keys otherwise derive
        # loader names like '/main//base.html' that fail for a working app.
        norm = re.sub(r"/+", "/", path.replace("\\", "/"))
        if "/templates/" not in norm or not norm.endswith(".html"):
            continue
        name = norm.rsplit("/templates/", 1)[1].lstrip("/")
        if name and os.path.isfile(norm):
            checks.append((name, os.path.abspath(norm)))
    return checks


def _run_django_verification(memory, executor, coder, display,
                             task: str, language: str | None,
                             cfg, django_dir: str) -> tuple[bool, str]:
    """Ground-truth check for Django: template reachability + page render.

    Catches the class of failure where a run creates template/static files
    at paths Django never consults — every static check passes, yet the
    rendered site is unchanged.
    """
    import json as _json
    import tempfile as _tempfile

    from .page_grounding import parse_acceptance_checks, pinned_urls_from_task

    settings_module = _django_settings_module(django_dir)
    if settings_module is None:
        _logger.info("[SmokeTest] Django settings module not found — skipping")
        return True, ""
    checks = _django_template_checks(memory.all_files(), django_dir)
    # Requirement-derived assertions from the task briefing — the piece
    # that catches "everything renders but the task wasn't accomplished".
    acceptance = parse_acceptance_checks(
        getattr(memory, "_task_briefing", "") or "")
    # URLs the task text pins verbatim become resolvability checks: the
    # user named the address, so delivering the feature at another route
    # is a miss the generated tests won't catch (observed: /dashboard/
    # pinned, dashboard served elsewhere, pipeline green).
    _covered = {c["url"] for c in acceptance}
    for _url in pinned_urls_from_task(task):
        if _url not in _covered:
            acceptance.append(
                {"url": _url, "kind": "must_resolve", "needle": ""})

    tmp_dir = _tempfile.mkdtemp(prefix="agentchanti_django_probe_")
    script = os.path.join(tmp_dir, "django_probe.py")
    checks_file = os.path.join(tmp_dir, "checks.json")
    acceptance_file = os.path.join(tmp_dir, "acceptance.json")
    try:
        with open(script, "w", encoding="utf-8") as f:
            f.write(_DJANGO_PROBE)
        with open(checks_file, "w", encoding="utf-8") as f:
            _json.dump(checks, f)
        with open(acceptance_file, "w", encoding="utf-8") as f:
            _json.dump(acceptance, f)

        probe_cmd = (f'python "{script}" {settings_module} '
                     f'"{checks_file}" / "{acceptance_file}"')
        verify_cmd = (probe_cmd if django_dir == "."
                      else f"cd {django_dir} && {probe_cmd}")
        _logger.info("[SmokeTest] Django verification: %d template check(s) "
                     "+ page render + %d acceptance check(s) (cwd=%s)",
                     len(checks), len(acceptance), django_dir)
        _show_status = getattr(display, "show_status", None)
        if callable(_show_status):
            _show_status("Verifying Django templates and page render...")
        ok, out = executor.run_command(
            probe_cmd, timeout=180,
            cwd=None if django_dir == "." else django_dir)
        if "DJANGO_PROBE_DONE" not in (out or ""):
            _logger.info("[SmokeTest] Django probe did not complete — "
                         "skipping (non-blocking): %s", (out or "")[:300])
            return True, ""
        if ok:
            _logger.info("[SmokeTest] Django verification passed")
            return True, ""

        from .agent_loop import (
            agent_loop_enabled, build_step_tools, run_recovery_loop,
            truncate_middle,
        )
        _logger.warning("[SmokeTest] Django verification FAILED:\n%s",
                        truncate_middle(out or "", 1500))
        llm_client = getattr(coder, "llm_client", None)
        if agent_loop_enabled(cfg, llm_client):
            tools = build_step_tools(executor, memory)
            try:
                recovered, info = run_recovery_loop(
                    llm_client, tools,
                    step_text=("Make the Django verification pass: created "
                               "templates reachable by the template loader, "
                               "every page renders without errors, and all "
                               "ACCEPTANCE checks derived from the task hold "
                               "(they assert what the task requires on the "
                               "rendered HTML)."),
                    task=task,
                    error_info=f"Django verification failed:\n{out}",
                    display=_status_only(display, "Django verification"),
                    step_idx=0, language=language,
                    max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
                    verify_cmd=verify_cmd,
                    escalation_client=getattr(coder, "escalation_client", None))
            except Exception as rec_exc:
                # Recovery is best-effort. A dead LLM endpoint (e.g. a
                # misrouted escalation model 404ing) must not crash the
                # whole run — report the original verification failure.
                _logger.warning(
                    "[SmokeTest] Django recovery loop errored (non-fatal): %s",
                    rec_exc)
                return False, ("Django verification failed: "
                               f"{truncate_middle(out or '', 800)}")
            if recovered:
                _logger.info("[SmokeTest] Django verification recovered: %s",
                             info[:200])
                return True, ""
            return False, ("Django verification failing: "
                           f"{truncate_middle(info, 600)}")
        return False, ("Django verification failed: "
                       f"{truncate_middle(out or '', 800)}")
    finally:
        for p in (script, checks_file, acceptance_file):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def _find_js_project_dir(memory_files: dict[str, str]) -> str | None:
    """Locate the directory holding package.json for this run's JS files.

    Session file keys are relative to the run cwd (possibly prefixed with
    a scaffolded sub-project, e.g. ``my-app/src/main.jsx``). Checks each
    candidate prefix on disk, most-specific first, then the cwd itself.
    """
    prefixes: list[str] = []
    for path in memory_files:
        norm = path.replace("\\", "/")
        if "/" in norm:
            top = norm.split("/", 1)[0]
            if top not in prefixes and not top.startswith("_"):
                prefixes.append(top)
    for candidate in prefixes + ["."]:
        if os.path.isfile(os.path.join(candidate, "package.json")):
            return candidate
    return None


def _run_js_build_verification(memory, executor, coder, display,
                               task: str, language: str | None,
                               cfg=None) -> tuple[bool, str]:
    """Run the project's production build as the JS ground-truth check.

    Failure gets one agent-loop recovery attempt (when enabled) with the
    build command as the deterministic verify_cmd; otherwise the failure
    is reported as-is.
    """
    import json

    project_dir = _find_js_project_dir(memory.all_files())
    if project_dir is None:
        _logger.info("[SmokeTest] No package.json found — skipping JS build check")
        return True, ""
    try:
        with open(os.path.join(project_dir, "package.json"),
                  encoding="utf-8") as fh:
            pkg = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning("[SmokeTest] Could not read package.json: %s", exc)
        return True, ""
    if not (pkg.get("scripts") or {}).get("build"):
        _logger.info("[SmokeTest] No build script in package.json — skipping")
        return True, ""

    build_cmd = "npm run build"
    _logger.info("[SmokeTest] JS build check: %s (cwd=%s)",
                 build_cmd, project_dir)
    _show_status = getattr(display, "show_status", None)
    if callable(_show_status):
        _show_status("Verifying production build...")
    cwd = None if project_dir == "." else project_dir
    ok, out = executor.run_command(build_cmd, timeout=300, cwd=cwd)
    if ok:
        _logger.info("[SmokeTest] JS build passed")
        return _verify_style_coupling(
            memory, executor, coder, display, task, language, cfg,
            project_dir)

    _logger.warning("[SmokeTest] JS build FAILED:\n%s", (out or "")[-1500:])

    from .agent_loop import (
        agent_loop_enabled, build_step_tools, run_recovery_loop,
    )
    llm_client = getattr(coder, "llm_client", None)
    if agent_loop_enabled(cfg, llm_client):
        tools = build_step_tools(executor, memory,
                                 project_root=project_dir if cwd else ".")
        recovered, info = run_recovery_loop(
            llm_client, tools,
            step_text=f"Make the production build pass: {build_cmd}",
            task=task,
            error_info=f"Command `{build_cmd}` failed.\nOutput:\n{out}",
            display=_status_only(display, "Production build"),
            step_idx=0, language=language,
            max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
            verify_cmd=build_cmd,
            escalation_client=getattr(coder, "escalation_client", None))
        if recovered:
            _logger.info("[SmokeTest] JS build recovered: %s", info[:200])
            return True, ""
        return False, f"Production build failing after recovery: {info[:500]}"

    return False, f"Production build failed: {(out or '')[-1000:]}"


def _verify_style_coupling(memory, executor, coder, display, task,
                           language, cfg, project_dir: str
                           ) -> tuple[bool, str]:
    """Do the classes the markup renders exist in the stylesheet?

    Runs only after the build is green, because it answers a question the
    build cannot: an unmatched CSS class is valid CSS, so a page whose
    every element is unstyled compiles perfectly. Observed on four of six
    consecutive runs of one project — once with 7 classes rendered and 0
    styled, while a gate asserting eight structural properties of the
    stylesheet passed on all eight, every one of them describing a
    selector the markup never rendered.

    The check refuses far more often than it fires (see
    :mod:`.style_coupling`); a false accusation here would send a correct
    run into a fix loop.
    """
    from .style_coupling import find_style_drift

    try:
        drift = find_style_drift(project_dir)
    except Exception as exc:                       # never fail a run
        _logger.debug("[SmokeTest] style-coupling check skipped: %s", exc)
        return True, ""
    if drift is None or not drift.broken:
        if drift is not None:
            _logger.info("[SmokeTest] Style coupling OK — every rendered "
                         "class is defined")
        return True, ""

    _logger.warning("[SmokeTest] Style coupling BROKEN:\n%s", drift.describe())
    print(f"  [SmokeTest] {len(drift.unstyled)} rendered class(es) have no "
          f"stylesheet rule — those elements render unstyled.")

    from .agent_loop import (
        agent_loop_enabled, build_step_tools, run_recovery_loop,
    )
    llm_client = getattr(coder, "llm_client", None)
    if not agent_loop_enabled(cfg, llm_client):
        # Reported rather than silently accepted: the run is finished, so
        # this is the last chance to say the page will look wrong.
        return True, ""

    # The same check, as a command, so the fix is held to the real thing
    # rather than the model's opinion of it.
    gate = (f"python -m agentchanti.orchestrator.style_coupling "
            f"{project_dir or '.'}")
    tools = build_step_tools(executor, memory, project_root=".")
    recovered, info = run_recovery_loop(
        llm_client, tools,
        step_text=("Make the stylesheet and the markup agree on class "
                   "names. Rename the CSS selectors to match what the "
                   "components already render — do NOT rename the "
                   "className values in the markup unless the component "
                   "is clearly the wrong one."),
        task=task,
        error_info=drift.describe(),
        display=_status_only(display, "Style coupling"),
        step_idx=0, language=language,
        max_turns=getattr(cfg, "AGENT_LOOP_MAX_TURNS", 8),
        verify_cmd=gate,
        escalation_client=getattr(coder, "escalation_client", None))
    if recovered:
        _logger.info("[SmokeTest] Style coupling repaired: %s", info)
        return True, ""
    _logger.warning("[SmokeTest] Style coupling still broken: %s", info)
    return True, ""      # advisory outcome — never fails an otherwise green run


def run_smoke_verification(
    memory,
    executor,
    coder,
    display,
    task: str,
    language: str | None,
    cfg=None,
    max_fix_attempts: int = _MAX_FIX_ATTEMPTS,
) -> tuple[bool, str]:
    """Launch the app entry point and fix launch crashes. Returns (ok, error)."""
    if cfg is not None and not getattr(cfg, "SMOKE_TEST_ENABLED", True):
        return True, ""
    if language in ("javascript", "typescript"):
        # JS/TS apps have no launchable entry point, but the production
        # build is a cheap deterministic ground truth: a broken import or
        # JSX error fails `npm run build` even when no tests exist.
        return _run_js_build_verification(
            memory, executor, coder, display, task, language, cfg)
    if language not in (None, "python"):
        return True, ""  # Python entry points only, for now

    memory_files = memory.all_files()

    # Django projects have no launchable entry point, but template
    # reachability + a test-client page render are cheap ground truth
    # (observed: a run created 7 template/static files at paths Django
    # never consults — green pipeline, unchanged site).
    _django_dir = _find_django_project_dir(memory_files)
    if _django_dir is not None:
        return _run_django_verification(
            memory, executor, coder, display, task, language, cfg,
            _django_dir)

    entry = find_python_entrypoint(memory_files)
    if not entry:
        _logger.info("[SmokeTest] No runnable Python entry point — skipping")
        return True, ""

    cmd = build_run_command(entry)
    _logger.info("[SmokeTest] Launching app: %s", cmd)

    out = ""
    prev_out = ""
    prev_fixed: list[str] = []
    attempt = 0
    env_repairs: list[str] = []
    while True:
        ok, out = _launch(executor, cmd)
        if ok:
            _logger.info("[SmokeTest] App launched successfully (%s)", entry)
            return True, ""
        if _is_headless_failure(out):
            _logger.info(
                "[SmokeTest] No display available — skipping runtime check")
            return True, ""

        _logger.warning(
            "[SmokeTest] App crashed on launch (attempt %d/%d):\n%s",
            attempt + 1, max_fix_attempts + 1, out[-1500:])
        # An environment repair is not a fix attempt: nothing about the
        # code was wrong, so spending one of the code budget on it would
        # push a genuine bug out of reach. Tried before the LLM sees the
        # crash at all — asked to stop a missing import from crashing the
        # app, a model deletes the import, and the feature with it.
        if len(env_repairs) < _MAX_ENV_REPAIRS:
            dep = _dependency_named_by(out, memory_files)
            if dep and dep not in env_repairs and _repair_missing_dependency(
                    dep, executor):
                env_repairs.append(dep)
                continue

        if attempt >= max_fix_attempts:
            break
        _show_status = getattr(display, "show_status", None)
        if callable(_show_status):
            _show_status("[SmokeTest] App crashed on launch — attempting fix")
        # Tell the LLM when its previous fix changed nothing — otherwise
        # it re-polishes the same wrong file every attempt (observed:
        # three identical crashes, three rewrites of the entry point,
        # while the failing call lived in another file).
        stuck_note = ""
        if prev_fixed and _same_crash(prev_out, out):
            _logger.warning(
                "[SmokeTest] Crash unchanged after fixing %s — redirecting",
                prev_fixed)
            stuck_note = (
                "\n\nIMPORTANT: A previous fix modified "
                + ", ".join(f"`{p}`" for p in prev_fixed)
                + " but the crash output is UNCHANGED. The bug is almost "
                "certainly in a DIFFERENT file — fix the file that actually "
                "contains the failing call named in the error message."
            )
        prev_out = out
        fixed = _attempt_fix(out, cmd, memory, executor, coder, entry,
                             stuck_note=stuck_note)
        if not fixed:
            break
        prev_fixed = fixed
        attempt += 1

    return False, (
        f"[SmokeTest] Application crashes when launched with `{cmd}`:\n"
        f"{out[-1500:]}"
    )

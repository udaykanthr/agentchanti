"""API grounding — align LLM codegen with the *installed* library versions.

LLMs write code against the library versions dominant in their training data,
which is frequently a major version behind what is actually installed (e.g.
arcade 2.x idioms against an installed arcade 3.x, where ``start_render`` /
``draw_rectangle_filled`` changed or were removed).  Two capabilities close
that gap:

1. :func:`get_installed_package_versions` — query the target interpreter's
   installed distributions (``pip list``) so agent prompts can say
   ``arcade==3.3.3`` instead of just ``arcade``.

2. :func:`probe_api_usage` — after Python files are written, statically
   collect every ``module.attr`` usage of imported third-party modules and
   verify each exists in the installed package by importing it in the target
   interpreter (the Executor resolves the project venv).  Catches removed or
   renamed APIs before tests or runtime ever execute the code.
"""

import ast
import json
import logging
import os
import re
import tempfile

_logger = logging.getLogger(__name__)


def get_installed_package_versions(
    cwd: str | None = None,
    executor=None,
    timeout: int = 60,
    language: str | None = None,
) -> dict[str, str]:
    """Return ``{package_name_lower: version}`` for the project's ecosystem.

    Python (default): ``pip list`` through the Executor so the project
    venv's interpreter is used when one exists. JavaScript/TypeScript:
    ``npm ls`` in the project directory — probing pip on a JS project
    grounds agents in the wrong package universe. Returns an empty dict
    on any failure — grounding is best-effort, never blocking.
    """
    if executor is None:
        from ..executor import Executor
        executor = Executor()

    if language in ("javascript", "typescript"):
        return _get_npm_versions(cwd, executor, timeout)

    ok, out = executor.run_command(
        "python -m pip list --format=json --disable-pip-version-check",
        cwd=cwd, timeout=timeout,
    )
    if not ok or not out:
        return {}
    try:
        # pip may print warnings around the JSON — extract the array
        start = out.index("[")
        end = out.rindex("]") + 1
        data = json.loads(out[start:end])
        return {
            d["name"].lower(): d["version"]
            for d in data
            if isinstance(d, dict) and d.get("name") and d.get("version")
        }
    except (ValueError, json.JSONDecodeError, TypeError):
        _logger.debug("[ApiGrounding] Could not parse pip list output")
        return {}


def _get_npm_versions(cwd: str | None, executor,
                      timeout: int) -> dict[str, str]:
    """Top-level npm dependency versions for a JS/TS project."""
    import os
    if not os.path.isfile(os.path.join(cwd or ".", "package.json")):
        return {}
    # npm ls exits non-zero on peer-dep warnings but still prints the
    # JSON tree — parse the output regardless of the exit code.
    _ok, out = executor.run_command(
        "npm ls --json --depth=0", cwd=cwd, timeout=timeout)
    if not out:
        return {}
    try:
        start = out.index("{")
        end = out.rindex("}") + 1
        data = json.loads(out[start:end])
        deps = data.get("dependencies") or {}
        return {
            name.lower(): info["version"]
            for name, info in deps.items()
            if isinstance(info, dict) and info.get("version")
        }
    except (ValueError, json.JSONDecodeError, TypeError):
        _logger.debug("[ApiGrounding] Could not parse npm ls output")
        return {}


def format_packages_with_versions(
    packages: list[str],
    versions: dict[str, str],
) -> list[str]:
    """Render package names as ``name==version`` where the version is known."""
    out = []
    for p in packages:
        v = versions.get(p.lower()) if versions else None
        out.append(f"{p}=={v}" if v else p)
    return out


# ── Static usage collection ───────────────────────────────────────


class _UsageCollector(ast.NodeVisitor):
    """Collect ``module.attr`` usages of imported modules in one file.

    Imports AND attribute usages inside ``try`` blocks are ignored —
    generated code commonly uses try/except fallbacks (import fallbacks,
    or ``try: new_api() except: old_api()`` version shims).  Code with
    explicit failure handling is runtime-safe by construction; flagging
    it rejects functionally correct fixes.
    """

    def __init__(self):
        self.aliases: dict[str, str] = {}   # bound name -> module path
        self.checks: set[tuple[str, str | None]] = set()  # (module, attr|None)
        self._try_depth = 0

    def visit_Try(self, node):
        self._try_depth += 1
        self.generic_visit(node)
        self._try_depth -= 1

    def visit_Import(self, node):
        if self._try_depth == 0:
            for a in node.names:
                if a.asname:
                    self.aliases[a.asname] = a.name
                else:
                    # `import x.y` binds `x`; usages on `x` probe module `x`
                    top = a.name.split(".")[0]
                    self.aliases[top] = top
                for mod in {a.name, a.name.split(".")[0]}:
                    self.checks.add((mod, None))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # Relative imports (level > 0) are project-local — skip
        if self._try_depth == 0 and node.level == 0 and node.module:
            for a in node.names:
                if a.name == "*":
                    continue
                self.checks.add((node.module, a.name))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self._try_depth == 0 and isinstance(node.value, ast.Name):
            mod = self.aliases.get(node.value.id)
            if mod:
                self.checks.add((mod, node.attr))
        self.generic_visit(node)


def _collect_checks(
    py_files: dict[str, str],
    local_top_levels: set[str] | None = None,
) -> set[tuple[str, str | None]]:
    """AST-scan *py_files* for (module, attr) pairs worth probing."""
    checks: set[tuple[str, str | None]] = set()
    for path, content in py_files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue  # syntax errors are caught by other validators
        collector = _UsageCollector()
        collector.visit(tree)
        checks |= collector.checks

    filtered: set[tuple[str, str | None]] = set()
    for mod, attr in checks:
        top = mod.split(".")[0]
        if top.startswith("__"):            # __future__ etc.
            continue
        if local_top_levels and top in local_top_levels:
            continue                        # project-local module
        if attr and attr.startswith("__"):  # dunder attrs always exist-ish
            continue
        filtered.add((mod, attr))
    return filtered


def collect_api_usages(
    py_files: dict[str, str],
    local_top_levels: set[str] | None = None,
) -> list[str]:
    """Sorted, human-readable ``module.attr`` usages found in *py_files*.

    Used to arm reviewer prompts with the exact list of probe-verified
    APIs after a clean probe — without it, reviewers re-litigate API
    existence from training data and can FAIL correct code.
    """
    checks = _collect_checks(py_files, local_top_levels)
    return sorted({f"{m}.{a}" if a else m for m, a in checks})


_PROBE_TEMPLATE = '''\
import importlib
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    checks = json.load(f)

# Framework modules (Django especially) refuse to import without app
# settings configured. That means "needs app context", not "missing".
_INCONCLUSIVE = ("ImproperlyConfigured", "AppRegistryNotReady")

failed_modules = set()
for mod, attr in checks:
    if mod in failed_modules:
        continue
    try:
        m = importlib.import_module(mod)
    except BaseException as exc:
        if type(exc).__name__ in _INCONCLUSIVE:
            failed_modules.add(mod)
            continue
        failed_modules.add(mod)
        print("MODULE_MISSING::%s::%s: %s" % (mod, type(exc).__name__, exc))
        continue
    if attr and not hasattr(m, attr):
        # May be a submodule the package __init__ doesn't re-export
        try:
            importlib.import_module(mod + "." + attr)
        except BaseException as sub_exc:
            if type(sub_exc).__name__ in _INCONCLUSIVE:
                continue
            import difflib
            ver = str(getattr(m, "__version__", "") or getattr(m, "VERSION", ""))
            close = difflib.get_close_matches(
                attr, [a for a in dir(m) if not a.startswith("_")],
                n=3, cutoff=0.6)
            print("API_MISSING::%s.%s::%s::%s" % (mod, attr, ver, ",".join(close)))
print("PROBE_DONE")
'''


def probe_api_usage(
    py_files: dict[str, str],
    executor,
    cwd: str | None = None,
    local_top_levels: set[str] | None = None,
    timeout: int = 90,
) -> list[str]:
    """Verify every ``module.attr`` used in *py_files* exists as installed.

    Runs a probe script through the Executor (which resolves the project
    venv) and returns human-readable error strings for missing modules or
    attributes.  Returns ``[]`` when everything checks out or when the probe
    itself cannot run — this is a tripwire, never a blocker.
    """
    checks = _collect_checks(py_files, local_top_levels)
    if not checks:
        return []

    tmp_dir = tempfile.mkdtemp(prefix="agentchanti_probe_")
    script_path = os.path.join(tmp_dir, "probe.py")
    checks_path = os.path.join(tmp_dir, "checks.json")
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_PROBE_TEMPLATE)
        with open(checks_path, "w", encoding="utf-8") as f:
            json.dump(sorted(checks, key=lambda c: (c[0], c[1] or "")), f)
        ok, out = executor.run_command(
            f'python "{script_path}" "{checks_path}"', cwd=cwd, timeout=timeout,
        )
    finally:
        for p in (script_path, checks_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if "PROBE_DONE" not in (out or ""):
        _logger.debug("[ApiGrounding] Probe did not complete — skipping check")
        return []

    errors: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("API_MISSING::"):
            parts = line.split("::")
            usage = parts[1] if len(parts) > 1 else ""
            ver = parts[2] if len(parts) > 2 else ""
            close = ([c for c in parts[3].split(",") if c]
                     if len(parts) > 3 else [])
            mod_path = usage.rsplit(".", 1)[0] if "." in usage else usage
            top = usage.split(".")[0]
            ver_note = f" {ver}" if ver else ""
            hint = ""
            if close:
                hint = (" Did you mean: "
                        + ", ".join(f"`{mod_path}.{c}`" for c in close) + "?")
            errors.append(
                f"`{usage}` does not exist in the installed {top}{ver_note} — "
                f"this API was likely removed or renamed in this version."
                f"{hint}"
            )
        elif line.startswith("MODULE_MISSING::"):
            _, mod, detail = (line.split("::", 2) + ["", ""])[:3]
            errors.append(
                f"Module `{mod}` failed to import in the project environment: "
                f"{detail}"
            )
    if errors:
        _logger.info(
            "[ApiGrounding] %d API usage issue(s) found: %s",
            len(errors), "; ".join(e.split(" — ")[0] for e in errors[:5]),
        )
    return errors


# Commands that change the set of installed packages.  `pip install`,
# `python -m pip install`, and JS/other package managers.
_INSTALL_CMD_RE = re.compile(
    r'\b(?:pip3?|python3?\s+-m\s+pip|npm|yarn|pnpm|poetry|uv|conda)\b'
    r'[^&|;]*\b(?:install|add)\b',
    re.IGNORECASE,
)


def is_install_command(cmd: str) -> bool:
    """True if *cmd* likely installs packages (any segment of a && chain)."""
    return bool(_INSTALL_CMD_RE.search(cmd or ""))


def refresh_installed_versions(project_context, executor=None,
                               cwd: str | None = None,
                               language: str | None = None) -> None:
    """Re-probe installed versions after packages were installed mid-run.

    On a blank project the initial snapshot is taken before the venv even
    exists, so the coder would never learn e.g. ``arcade==3.3.3``.  Also
    merges newly installed package names into ``installed_packages`` so the
    version annotation actually renders in agent prompts.
    """
    if project_context is None or not hasattr(project_context,
                                              "installed_versions"):
        return
    versions = get_installed_package_versions(cwd=cwd, executor=executor,
                                              language=language)
    if not versions:
        return
    project_context.installed_versions = versions
    _skip = {"pip", "setuptools", "wheel"}
    existing = {p.lower() for p in project_context.installed_packages}
    for name in sorted(versions):
        if name not in _skip and name not in existing:
            project_context.installed_packages.append(name)
    _logger.info(
        "[ApiGrounding] Refreshed installed versions: %d package(s)",
        len(versions))


def local_top_levels_from_files(file_paths) -> set[str]:
    """Derive the project-local top-level module names from file paths.

    ``src/snake_game/main.py`` contributes ``src``; ``utils.py`` contributes
    ``utils`` — anything importable from the project itself must not be
    probed as a third-party package.
    """
    tops: set[str] = set()
    for p in file_paths:
        norm = p.replace("\\", "/").lstrip("./")
        first = norm.split("/")[0]
        if first.endswith(".py"):
            first = first[:-3]
        if first:
            tops.add(first)
    return tops

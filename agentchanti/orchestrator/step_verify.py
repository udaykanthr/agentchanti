"""Per-step execution verification — load each just-written file against
the real project environment.

The registry maps language → the cheapest native command that resolves a
file's imports for real (venv, node_modules) right after it is written,
instead of discovering integration failures at the end of the pipeline.
For compiled languages the compiler *is* the verifier (``tsc --noEmit``,
``go build``, ``cargo check``) — deterministic and side-effect free.
Python has no compile step, so its check is an import in a subprocess:
it executes module top-level code, which is exactly what makes it catch
missing modules, broken relative imports, and dead APIs referenced at
class level.

Unknown languages skip silently — this is a tripwire, never a blocker.
"""

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

# Test files are exercised by the test machinery (BulkTest / TEST steps)
# and often use sys.path manipulation that breaks a bare import check.
_SKIP_BASENAME_PARTS = ("test_", ".test.", ".spec.", "conftest")


def _python_import_target(file_path: str) -> Optional[tuple[str, str]]:
    """Derive ``(module, sys_path_root)`` for importing *file_path*.

    Walks up while parent directories are packages (contain
    ``__init__.py``); the first non-package directory becomes the
    sys.path root.  This handles flat layouts (``main.py`` → ``main``,
    root ``.``) and src-layouts (``src/pkg/mod.py`` → ``pkg.mod``,
    root ``src``) from the same rule.
    """
    norm = file_path.replace("\\", "/").lstrip("./")
    if not norm.endswith(".py"):
        return None
    parts = norm.split("/")
    base = parts[-1][:-3]
    dirs = parts[:-1]
    if base == "__init__":
        if not dirs:
            return None
        base = dirs[-1]
        dirs = dirs[:-1]
    if not base.isidentifier():
        return None
    i = len(dirs)
    pkg_parts: list[str] = []
    while i > 0 and os.path.isfile(os.path.join(*dirs[:i], "__init__.py")):
        pkg_parts.insert(0, dirs[i - 1])
        i -= 1
    if any(not p.isidentifier() for p in pkg_parts):
        return None
    root = "/".join(dirs[:i]) or "."
    module = ".".join(pkg_parts + [base])
    return module, root


def _python_verify_cmd(file_path: str, cwd: Optional[str]) -> Optional[str]:
    target = _python_import_target(file_path)
    if target is None:
        return None
    module, root = target
    return (
        f"python -c \"import sys; sys.path.insert(0, '{root}'); "
        f"import {module}\""
    )


def _ts_verify_cmd(file_path: str, cwd: Optional[str]) -> Optional[str]:
    # tsc verifies the whole project against tsconfig — imports, types,
    # and API existence in one deterministic pass, with no execution.
    if not file_path.replace("\\", "/").endswith((".ts", ".tsx")):
        return None
    if not os.path.isfile(os.path.join(cwd or ".", "tsconfig.json")):
        return None
    return "npx tsc --noEmit"


@dataclass(frozen=True)
class StepVerifier:
    """One language's per-step load/compile check.

    ``build_command`` maps ``(file_path, cwd)`` to a shell command, or
    ``None`` when the file cannot be checked (wrong extension, missing
    toolchain config).  ``executes_code`` documents whether the check
    runs project code (import) or only compiles it.
    """
    build_command: Callable[[str, Optional[str]], Optional[str]]
    executes_code: bool
    timeout: int = 60


VERIFIERS: dict[str, StepVerifier] = {
    "python": StepVerifier(_python_verify_cmd, executes_code=True),
    "typescript": StepVerifier(_ts_verify_cmd, executes_code=False,
                               timeout=180),
}


def _should_skip(file_path: str) -> bool:
    base = file_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(part in base for part in _SKIP_BASENAME_PARTS)


def verify_step_files(
    files,
    language: Optional[str],
    executor,
    cwd: Optional[str] = None,
) -> list[str]:
    """Run the language's per-step verifier on each written file.

    Returns human-readable error strings; empty when everything loads,
    when the language has no registered verifier, or when a check cannot
    run.  Errors carry the failing command and the traceback tail so the
    retry loop's feedback is actionable.
    """
    verifier = VERIFIERS.get(language or "")
    if verifier is None:
        return []
    errors: list[str] = []
    seen_cmds: set[str] = set()
    for path in files:
        if _should_skip(path):
            continue
        try:
            cmd = verifier.build_command(path, cwd)
        except Exception:
            continue
        if not cmd or cmd in seen_cmds:
            continue
        seen_cmds.add(cmd)
        try:
            ok, out = executor.run_command(
                cmd, cwd=cwd, timeout=verifier.timeout)
        except Exception as exc:
            _logger.debug("[StepVerify] %s could not run: %s", cmd, exc)
            continue
        if not ok:
            # Django app modules refuse to import outside an app context
            # (settings not configured / app registry not ready). That is
            # framework behavior, not a broken file — manage.py-driven
            # checks and tests cover these modules instead.
            if ("AppRegistryNotReady" in (out or "")
                    or "ImproperlyConfigured" in (out or "")):
                _logger.debug(
                    "[StepVerify] %s needs app context (Django) — "
                    "inconclusive, skipping", path)
                continue
            tail = "\n".join((out or "").strip().splitlines()[-12:])
            errors.append(
                f"`{path}` fails to load in the project environment "
                f"(ran `{cmd}`):\n{tail}"
            )
    if errors:
        _logger.info(
            "[StepVerify] %d file(s) failed to load in the project "
            "environment", len(errors))
    return errors

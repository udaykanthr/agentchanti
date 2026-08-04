"""Python syntax validation that matches what the interpreter enforces.

``ast.parse`` is NOT a syntax check. It stops at the parse stage, and a
few errors are only raised when the module is compiled — most importantly
future-import placement::

    >>> ast.parse("X = 1\\nfrom __future__ import annotations\\n")
    <ast.Module ...>                      # accepted
    >>> import that same file
    SyntaxError: from __future__ imports must occur at the beginning of the file

Every write guard in this project was built on ``ast.parse``, so a chunk
edit that spliced a module header into the middle of a file passed
validation, was written to disk, and only surfaced as an ImportError from
the test command afterwards. Observed on a Pac-Man run: diagnosis
replaced the ``Player`` chunk with content that restated the module
header, leaving a second ``from __future__ import annotations`` at line
26; two diagnosis attempts then failed against a file that could never
import, and the pipeline halted.

``compile(..., "exec")`` runs the same front end the interpreter does, so
it catches that placement error and everything ``ast.parse`` catches.
"""

from __future__ import annotations

__all__ = ["check_python_syntax", "is_valid_python"]


def check_python_syntax(source: str, filename: str = "<string>") -> str | None:
    """Return a human-readable error, or ``None`` when *source* is valid.

    ``dont_inherit=True`` matters: this module itself uses
    ``from __future__ import annotations``, and without it the compiler
    would inherit that flag and judge *source* under rules the target
    file will not actually be run with.
    """
    try:
        compile(source, filename, "exec", dont_inherit=True)
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        return f"{exc.msg}{where}"
    except ValueError as exc:
        # e.g. source containing null bytes — compile raises ValueError,
        # not SyntaxError, and it is still content we must not write.
        return str(exc)
    return None


def is_valid_python(source: str, filename: str = "<string>") -> bool:
    """Boolean form of :func:`check_python_syntax`."""
    return check_python_syntax(source, filename) is None

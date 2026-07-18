"""Post-mortem diagnostics for silent process deaths.

Some process terminations bypass Python's ``faulthandler`` *and* its
excepthook entirely, so the run vanishes mid-instruction with no
traceback and no exit message:

* **Windows fast-fail** (exception code ``0xc0000409``) — raised by
  ``RtlFailFast`` on heap corruption / security-cookie failure / CRT
  ``abort()``. It terminates the process *without* dispatching through
  the SEH chain that ``faulthandler`` hooks, so no crash-log stack is
  written. This is the documented failure mode when a native extension
  (SQLite, tree-sitter, numpy) is misused across threads.
* **OS hard-kills** — ``SIGKILL`` / OOM-killer / Job-object termination.
  Nothing in-process can catch these.

``faulthandler`` (armed in ``cli._arm_faulthandler``) still covers the
*catchable* native crashes; this module covers the rest by leaving
durable breadcrumbs so the *next* run can explain what happened to the
*previous* one:

1. A heartbeat file (``.agentchanti/heartbeat.json``) rewritten every
   few seconds with the current pipeline activity. It survives any form
   of death — its last contents pinpoint the moment. A clean shutdown
   deletes it, so its mere presence at startup means the prior run died
   abnormally.
2. A ``threading.excepthook`` that routes exceptions from daemon worker
   threads (KB watchers, background embedders) into the log — otherwise
   they print to a stderr the Rich live display has taken over and are
   lost.
3. A startup scavenger that, when a stale heartbeat is found, queries
   the Windows Application event log for the matching ``python.exe``
   fault and writes a consolidated post-mortem into ``crash.log``.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from datetime import datetime

from ..cli_display import log

# Rewrite cadence for the heartbeat file. Small file, atomic replace —
# cheap enough to run continuously, slow enough to be negligible I/O.
_HEARTBEAT_INTERVAL = 3.0

_state: dict = {
    "activity": "starting",
    "heartbeat_path": None,
    "stop": threading.Event(),
    "thread": None,
    "session_start": None,
    "pid": os.getpid(),
    "installed": False,
}


def set_activity(activity: str) -> None:
    """Record what the pipeline is currently doing.

    Called at wave/step boundaries so a silent death's last heartbeat
    names the exact step that was executing when the process vanished.
    Persists synchronously: the background heartbeat only ticks every few
    seconds, so a crash within that window would otherwise lose the most
    recent (and most relevant) transition. Safe to call before
    :func:`install_crash_diagnostics`.
    """
    _state["activity"] = activity
    if _state["installed"]:
        _write_heartbeat()


def _heartbeat_path(project_root: str) -> str:
    return os.path.join(project_root, ".agentchanti", "heartbeat.json")


def _write_heartbeat() -> None:
    path = _state["heartbeat_path"]
    if not path:
        return
    # Live thread names matter for async heap corruption: the fast-fail
    # is tripped later, on a different thread, than the native write that
    # corrupted the heap — so the set of concurrent native-touching
    # threads (KB watcher, background embedders) is the real evidence.
    try:
        threads = sorted(t.name for t in threading.enumerate() if t.is_alive())
    except Exception:
        threads = []
    payload = {
        "ts": datetime.now().isoformat(),
        "pid": _state["pid"],
        "activity": _state["activity"],
        "threads": threads,
        "session_start": _state["session_start"],
    }
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", errors="replace") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)  # atomic on both Windows and POSIX
    except OSError:
        pass  # heartbeat is best-effort; never let it break a run


def _heartbeat_loop() -> None:
    stop = _state["stop"]
    # stop.wait returns True once set — loop exits without another write.
    while not stop.wait(_HEARTBEAT_INTERVAL):
        _write_heartbeat()


def _thread_excepthook(args) -> None:
    """Surface exceptions raised in worker threads.

    Python's default sends these to ``stderr``, which the Rich live
    display hides. SystemExit in a thread is normal control flow and is
    ignored.
    """
    if args.exc_type is SystemExit:
        return
    thread_name = args.thread.name if args.thread else "?"
    log.error(
        "[CrashDiag] Uncaught exception in worker thread %r", thread_name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _query_windows_faults() -> str:
    """Return recent python.exe Application-Error events, or ''.

    Only called when a stale heartbeat already indicates a prior crash,
    so the (relatively slow) PowerShell query never runs on a healthy
    startup.
    """
    if sys.platform != "win32":
        return ""
    ps = (
        "Get-WinEvent -FilterHashtable @{LogName='Application';"
        "ProviderName='Application Error','Windows Error Reporting'} "
        "-MaxEvents 40 -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Message -match 'python' } | "
        "Select-Object -First 5 TimeCreated, Id, "
        "@{n='Msg';e={($_.Message -split \"`n\")[0..3] -join ' | '}} | "
        "Format-List | Out-String"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=25,
        )
        return (out.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _append_crash_log(lines: list[str]) -> None:
    project_dir = os.path.dirname(_state["heartbeat_path"] or ".agentchanti/x")
    try:
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, "crash.log"), "a",
                  encoding="utf-8", errors="replace") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        pass


def _scavenge_prior_crash() -> None:
    """If the previous run left a stale heartbeat, report how it died."""
    path = _state["heartbeat_path"]
    if not path or not os.path.exists(path):
        return  # clean prior shutdown, or first run — pay no PowerShell cost

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        data = {}

    prev_pid = data.get("pid")
    activity = data.get("activity")
    ts = data.get("ts")
    headline = (
        f"Previous run (pid={prev_pid}) ended abnormally — no clean exit. "
        f"Last activity: {activity!r} at {ts}."
    )
    log.warning("[CrashDiag] %s", headline)

    report = [
        f"\n=== POST-MORTEM (detected {datetime.now().isoformat()}) ===",
        headline,
    ]
    faults = _query_windows_faults()
    if faults:
        report.append("Correlated Windows fault events:\n" + faults)
        report.append(
            "NOTE: exception code 0xc0000409 is a fast-fail (heap "
            "corruption / abort), typically a native extension misused "
            "across threads. faulthandler cannot catch it."
        )
    elif sys.platform == "win32":
        report.append(
            "No matching python.exe fault in the Application event log — "
            "consistent with an OS hard-kill (OOM / SIGKILL / Job-object "
            "termination) rather than an in-process fault."
        )
    _append_crash_log(report)
    log.warning("[CrashDiag] Post-mortem appended to .agentchanti/crash.log")

    # Clear the stale heartbeat so we report each crash exactly once.
    try:
        os.remove(path)
    except OSError:
        pass


def _on_exit() -> None:
    """Clean-shutdown marker: stop the heartbeat and remove its file.

    Its absence at next startup is how the scavenger distinguishes a
    clean exit from an abnormal death.
    """
    _state["stop"].set()
    thread = _state["thread"]
    if thread is not None:
        thread.join(timeout=1.0)
    path = _state["heartbeat_path"]
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def install_crash_diagnostics(project_root: str = ".") -> None:
    """Arm the heartbeat, thread excepthook, and prior-crash scavenger.

    Idempotent. Call once at process start, after ``faulthandler`` is
    armed. ``project_root`` is where the ``.agentchanti`` dir lives (the
    cwd, matching ``faulthandler``'s crash.log location).
    """
    if _state["installed"]:
        return
    _state["installed"] = True
    _state["heartbeat_path"] = _heartbeat_path(project_root)
    _state["session_start"] = datetime.now().isoformat()
    try:
        os.makedirs(os.path.dirname(_state["heartbeat_path"]), exist_ok=True)
    except OSError:
        pass

    # Inspect the previous run's trail BEFORE we overwrite it.
    _scavenge_prior_crash()

    threading.excepthook = _thread_excepthook
    atexit.register(_on_exit)

    _write_heartbeat()  # seed immediately so a fast death is still captured
    t = threading.Thread(target=_heartbeat_loop, name="crash-heartbeat",
                         daemon=True)
    _state["thread"] = t
    t.start()

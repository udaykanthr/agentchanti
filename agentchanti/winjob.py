"""Windows job objects, so a command's whole process tree can be killed.

`taskkill /T` walks parent PIDs, and a process started with `start /b` or
re-parented after its launcher exits is not on that walk. Measured
2026-09-22: an agent ran

    cd my-app && start /b npm run dev && timeout /t 6 /nobreak >nul && node -e "fetch(...)"

The command timed out at 120s and was killed, but `next dev` survived,
holding the command's stdout pipe open — and the pipeline hung for over four
hours waiting on that pipe. A job object has no such blind spot: every
process created inside it stays in it unless it explicitly breaks away.

Jobs are created with KILL_ON_JOB_CLOSE and their handles are kept for the
life of the pipeline, so a server a command left running is still reaped
when agentchanti exits, and a command that times out is terminated as a
whole. Everything here returns quietly on failure — the caller falls back to
`taskkill`, which is what ran before this module existed.
"""
from __future__ import annotations

import os

_JOBS: list[int] = []

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]

    _JobObjectExtendedLimitInformation = 9
    _KILL_ON_JOB_CLOSE = 0x2000


def contain(proc) -> int | None:
    """Put *proc* (and everything it will spawn) in a job. Returns the job."""
    if os.name != "nt":
        return None
    try:
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(
                job, _JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            _k32.CloseHandle(job)
            return None
        if not _k32.AssignProcessToJobObject(job, int(proc._handle)):
            _k32.CloseHandle(job)
            return None
        _JOBS.append(job)
        return job
    except Exception:
        return None


def terminate(job: int | None) -> bool:
    """Kill every process in *job*. True when the job was terminated."""
    if not job or os.name != "nt":
        return False
    try:
        return bool(_k32.TerminateJobObject(job, 1))
    except Exception:
        return False

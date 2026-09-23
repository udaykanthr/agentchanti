"""A timed-out command must not hang the pipeline on a surviving child.

Measured 2026-09-22: `cd my-app && start /b npm run dev && ...` timed out,
`taskkill /T` missed the `start /b` server, and closing the stdout pipe the
server still held blocked the pipeline for over four hours.
"""
import os
import subprocess
import sys
import time

import pytest

from agentchanti.executor import Executor

pytestmark = pytest.mark.skipif(os.name != "nt", reason="cmd.exe start /b")


def _alive(marker):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "
         f"'*{marker}*' -and $_.Name -eq 'python.exe' }} | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True).stdout
    return [p for p in out.split() if p.strip().isdigit()]


def test_a_start_b_child_neither_hangs_nor_survives(tmp_path):
    marker = f"orphan_marker_{os.getpid()}_{int(time.time())}"
    py = sys.executable
    # The intermediate `cmd /c` exits at once, so the sleeper's parent PID
    # names a dead process — the shape of the measured `next dev`, whose
    # npm launcher was gone, and the one `taskkill /T` cannot walk to.
    cmd = (f'start /b "" cmd /c start /b "" "{py}" -c '
           f'"import time; {marker}=1; time.sleep(90)" '
           f'&& "{py}" -c "import time; time.sleep(60)"')

    t0 = time.monotonic()
    ok, out = Executor().run_command(cmd, timeout=3, cwd=str(tmp_path),
                                     retry_on_crash=False)
    elapsed = time.monotonic() - t0

    assert ok is False and "timed out" in out
    assert elapsed < 30, f"hung for {elapsed:.0f}s on the orphan's pipe"
    time.sleep(1)
    assert not _alive(marker), "the start /b child outlived its command"

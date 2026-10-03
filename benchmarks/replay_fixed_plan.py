"""Replay ONE fixed plan through several code paths, to make a 20% effect
visible.

The problem this exists for, measured repeatedly this session: the plan a
run draws dominates its cost. step-execution varied 6 -> 13 calls between
two rounds of the identical task, a +165% swing, so the wiring gate's real
-20,157 token saving was invisible at the run level and totals went UP.
Every optimisation so far has landed in "worked, but unmeasurable".

A checkpoint already carries `plan_steps`, `task`, `language` and
`project_context`, and `--resume` restores them and skips briefing, the
global KB and the planner. So: capture one plan, then replay that SAME
plan in each arm. The planner stops being a variable and the remaining
difference is the code under test.

    python replay_bench.py capture <task-file> <template-dir> <out-dir> [--image IMG]
    python replay_bench.py replay  <out-dir> <label> [runs] [--image IMG]
    python replay_bench.py report  <out-dir>

`template-dir` is the project state BEFORE the run - an empty directory
for a greenfield task, the scaffold for a pre-scaffolded one. Each replay
starts from a byte copy of it, so no arm inherits another's work.

`--image` runs agentchanti INSIDE that container image instead of on the
host, which is what makes this usable for Go, Rust, Java and C. Without it
the harness can only replay a plan whose toolchain the host happens to
have - and the host has no gcc, make or cmake, so the one question this
was most recently needed for could not be asked at all.

BOTH arms must use the same image, or the comparison measures the image.
The image must already contain the agentchanti build under test, because
that is the thing being compared; rebuild it between arms.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT = ".agentchanti_checkpoint.json"
CONTAINER_TIMEOUT = 2700

# Both of these are machine-specific, so they come from the environment with
# a best-effort default rather than being baked in.
#
# AGENTCHANTI is only used for a HOST run (no --image); a containerised run
# takes the one on the image's PATH. CONFIG supplies the provider keys and
# is copied into each workdir, which is also how a containerised run gets
# them — the key travels with the project rather than through `-e`, so one
# less thing differs between arms.
AGENTCHANTI = Path(os.environ.get(
    "AGENTCHANTI_BIN",
    shutil.which("agentchanti") or "agentchanti"))
CONFIG = Path(os.environ.get(
    "AGENTCHANTI_BENCH_CONFIG",
    Path.home() / ".agentchanti.yaml"))
TOKENS_RE = re.compile(
    r"Total tokens: (\d+) \(sent=(\d+)(?: \[cached=(\d+) \((\d+)%\), "
    r"full-price=(\d+)\])?, recv=(\d+)\)")


def _fresh(dst: Path, template: Path):
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    if template and template.is_dir():
        shutil.copytree(template, dst, ignore=shutil.ignore_patterns(
            ".agentchanti", ".agentchanti_checkpoint.json", "node_modules",
            "venv", ".venv", "__pycache__"))
    else:
        dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, dst / ".agentchanti.yaml")


def _container_argv(cwd: Path, args, image: str, name: str):
    """`docker run` that executes agentchanti against /work.

    The config file already lives in the workdir (`_fresh` copies it), so
    the API key arrives with the project rather than through the
    environment - one less thing to differ between arms.

    `timeout` runs INSIDE the container deliberately: killing the docker
    CLI on the host leaves the container running, which is how an earlier
    benchmark left a dev server alive for four hours.
    """
    inner = "timeout %d agentchanti %s" % (
        CONTAINER_TIMEOUT, " ".join(subprocess.list2cmdline([a]) for a in args))
    return ["docker", "run", "--rm", "--name", name,
            "-v", f"{cwd}:/work", "-w", "/work", image, "sh", "-c", inner]


def _launch(cwd: Path, args, image: str | None = None):
    out = (cwd / "stdout.txt").open("w", encoding="utf-8", errors="replace")
    err = (cwd / "stderr.txt").open("w", encoding="utf-8", errors="replace")
    if image:
        name = f"replay-{cwd.name}-{int(time.time())}"
        argv = _container_argv(cwd, args, image, name)
        proc = subprocess.Popen(argv, stdout=out, stderr=err)
        # Remember the container so _kill can reach it: killing the docker
        # CLI does not stop what it started.
        proc._replay_container = name
        return proc, out, err
    return subprocess.Popen([str(AGENTCHANTI)] + args, cwd=str(cwd),
                            stdout=out, stderr=err), out, err


def _kill(proc):
    if proc.poll() is not None:
        return
    name = getattr(proc, "_replay_container", None)
    if name:
        # The container, not the client. `docker kill` on a --rm container
        # also removes it, so nothing is left behind for the next arm.
        subprocess.run(["docker", "kill", name], capture_output=True)
    else:
        subprocess.run(["taskkill", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def capture(task_file: Path, template: Path, out: Path,
            image: str | None = None):
    """Run until a checkpoint carrying a plan exists, then stop.

    Only the PLAN is wanted, so the run is killed as soon as one is on
    disk - a full run would cost the tokens this harness exists to save,
    and a successful run CLEARS its checkpoint on the way out.
    """
    out.mkdir(parents=True, exist_ok=True)
    work = out / "capture"
    _fresh(work, template)
    shutil.copy2(task_file, work / "task.txt")
    proc, fo, fe = _launch(work, ["--prompt-from-file", "task.txt", "--auto"],
                           image)
    cp, deadline = work / CHECKPOINT, time.time() + 1800
    captured = None
    try:
        while time.time() < deadline:
            if cp.is_file():
                try:
                    state = json.loads(cp.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    time.sleep(2)
                    continue
                if state.get("plan_steps"):
                    captured = state
                    break
            if proc.poll() is not None:
                break
            time.sleep(2)
    finally:
        _kill(proc)
        fo.close()
        fe.close()

    if not captured:
        print("FAILED: no checkpoint with a plan appeared")
        return 1

    # Reset to "nothing done yet" so a replay executes the whole plan.
    captured["completed_step"] = -1
    captured["step_results"] = {}
    captured["file_memory"] = {}
    (out / "plan_checkpoint.json").write_text(
        json.dumps(captured, indent=2), encoding="utf-8")
    shutil.copy2(task_file, out / "task.txt")
    if template and template.is_dir():
        tpl = out / "template"
        if tpl.exists():
            shutil.rmtree(tpl, ignore_errors=True)
        shutil.copytree(template, tpl, ignore=shutil.ignore_patterns(
            ".agentchanti", CHECKPOINT, "node_modules", "venv", "__pycache__"))
    steps = captured["plan_steps"]
    print(f"captured a {len(steps)}-step plan for: "
          f"{captured.get('task', '')[:70]}")
    for s in steps:
        print(f"   {s.get('index', '?')} [{s.get('step_type', '?')}] "
              f"{str(s.get('description', ''))[:64]}")
    return 0


def replay(out: Path, label: str, runs: int, image: str | None = None):
    state_file = out / "plan_checkpoint.json"
    if not state_file.is_file():
        print("no captured plan - run `capture` first")
        return 1
    template = out / "template"
    results = []
    for i in range(1, runs + 1):
        work = out / f"{label}-{i}"
        _fresh(work, template if template.is_dir() else None)
        shutil.copy2(state_file, work / CHECKPOINT)
        # The task is positional and required even when resuming: with no
        # task agentchanti prints its usage and exits 0, which read as
        # three zero-second "runs". The checkpoint still supplies the plan,
        # so the planner is skipped either way.
        shutil.copy2(out / "task.txt", work / "task.txt")
        t0 = time.time()
        proc, fo, fe = _launch(work, ["--prompt-from-file", "task.txt",
                                      "--auto"], image)
        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            _kill(proc)
        finally:
            fo.close()
            fe.close()
        secs = int(time.time() - t0)
        logs = sorted(work.rglob(".agentchanti/logs/agent_*.log"),
                      key=lambda p: p.stat().st_mtime)
        sent = recv = cached = total = 0
        if logs:
            text = logs[-1].read_text(encoding="utf-8", errors="replace")
            m = None
            for m in TOKENS_RE.finditer(text):
                pass
            if m:
                total, sent = int(m.group(1)), int(m.group(2))
                cached, recv = int(m.group(3) or 0), int(m.group(6))
        row = {"label": label, "run": i, "exit": proc.returncode,
               "secs": secs, "sent": sent, "cached": cached, "recv": recv,
               "total": total, "image": image or "host"}
        results.append(row)
        print(f"  {label} run {i}: exit={row['exit']} secs={secs} "
              f"sent={sent:,} cached={cached:,} recv={recv:,} "
              f"total={total:,}")
    path = out / f"results-{label}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def report(out: Path):
    rows = []
    for f in sorted(out.glob("results-*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    if not rows:
        print("no results yet")
        return 1
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    print(f"{'arm':<18}{'runs':>5}{'avg sent':>11}{'avg recv':>10}"
          f"{'avg total':>11}{'cached%':>9}{'exit0':>7}")
    print("-" * 71)
    base = None
    for label, rs in sorted(by.items()):
        n = len(rs)
        s = sum(r["sent"] for r in rs) / n
        c = sum(r["cached"] for r in rs)
        v = sum(r["recv"] for r in rs) / n
        t = sum(r["total"] for r in rs) / n
        ok = sum(1 for r in rs if r["exit"] == 0)
        pct = round(100 * c / sum(r["sent"] for r in rs)) if s else 0
        delta = "" if base is None else f"  ({100 * (t - base) / base:+.1f}%)"
        if base is None:
            base = t
        print(f"{label:<18}{n:>5}{s:>11,.0f}{v:>10,.0f}{t:>11,.0f}"
              f"{pct:>8}%{ok:>7}{delta}")
    # Report the spread, not just the means. A fixed plan removes the
    # planner as a variable; it does NOT make a small difference
    # significant, and the first real use of this harness showed why —
    # arms of 151,055 vs 123,145 (-18.5%) where one arm's own range was
    # 111,447, four times the effect. Printing only the delta invites
    # exactly the reading the harness exists to prevent.
    print("\nSame plan in every arm, so the planner is not a variable.")
    for label, rs in by.items():
        ts = [r["total"] for r in rs]
        if len(ts) > 1:
            print(f"  {label:<18} spread {max(ts) / max(min(ts), 1):.2f}x "
                  f"({min(ts):,} - {max(ts):,})")
    if len(by) == 2:
        (_, a), (_, b) = by.items()
        ta = [r["total"] for r in a]
        tb = [r["total"] for r in b]
        eff = abs(sum(ta) / len(ta) - sum(tb) / len(tb))
        worst = max(max(ta) - min(ta), max(tb) - min(tb))
        if eff and worst > eff:
            print(f"  NOTE: the noisier arm's own range ({worst:,.0f}) is "
                  f"{worst / eff:.1f}x the difference between arms "
                  f"({eff:,.0f}). Treat the delta as a direction, not a "
                  f"result, until the runs per arm grow.")
    return 0


def _take_image(argv):
    """Pull `--image X` out of argv, leaving the positional arguments."""
    image = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--image" and i + 1 < len(argv):
            image = argv[i + 1]
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    return image, rest


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    image, argv = _take_image(sys.argv[1:])
    cmd = argv[0]
    if image:
        probe = subprocess.run(["docker", "image", "inspect", image],
                               capture_output=True)
        if probe.returncode != 0:
            print(f"image not present locally: {image}")
            return 2
    if cmd == "capture":
        return capture(Path(argv[1]).resolve(),
                       Path(argv[2]).resolve() if len(argv) > 2 else None,
                       Path(argv[3]).resolve(), image)
    if cmd == "replay":
        return replay(Path(argv[1]).resolve(), argv[2],
                      int(argv[3]) if len(argv) > 3 else 3, image)
    if cmd == "report":
        return report(Path(argv[1]).resolve())
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""A/B benchmark harness: agent_loop on/off × plan_mode content/intent.

Runs each benchmark task through the full agentchanti CLI once per mode
combination in isolated work directories, then measures:

  ground truth   — do the task's success_cmds pass? (primary metric)
  pipeline claim — did the pipeline report success?
  tokens         — total LLM tokens (parsed from the run log)
  wall time      — end-to-end seconds
  loop stats     — [AgentLoop] session line from the log, when present

Usage (from the repo root):
  python benchmarks/run_ab.py --config path/to/.agentchanti.yaml
  python benchmarks/run_ab.py --config cfg.yaml --tasks bugfix,cmd-recovery
  python benchmarks/run_ab.py --config cfg.yaml --modes on --truststore

  # The Phase-4 decision run — intent vs content planning, loop on:
  python benchmarks/run_ab.py --config cfg.yaml --modes on \\
      --plan-modes content,intent --truststore

The config file supplies provider/model/API keys; the harness overrides
only the agent_loop / plan_mode flags. Results are written to
benchmarks/results/ab_<timestamp>.json and printed as a table.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.tasks import TASKS  # noqa: E402

RUN_TIMEOUT_S = 600

_TOKENS_RE = re.compile(r"Total tokens:\s*([\d,]+)")
_LOOP_STATS_RE = re.compile(r"\[AgentLoop\] session: (.+)")
_FAILED_RE = re.compile(r"Pipeline failed")
_FINISHED_RE = re.compile(r"Finished\. Total tokens")


def parse_total_tokens(log_text: str) -> int | None:
    """Last 'Total tokens: N' in the log (final summary line)."""
    matches = _TOKENS_RE.findall(log_text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def parse_pipeline_claim(log_text: str) -> bool | None:
    """True/False for the pipeline's own verdict, None if undetermined."""
    if _FAILED_RE.search(log_text):
        return False
    if _FINISHED_RE.search(log_text):
        return True
    return None


def parse_loop_stats(log_text: str) -> str | None:
    m = _LOOP_STATS_RE.search(log_text)
    return m.group(1) if m else None


def _build_config(base_config: str, agent_loop: bool,
                  plan_mode: str | None = None) -> str:
    """Base yaml with harness overrides appended (last key wins in yaml
    loaders that dict-merge; agentchanti's loader reads top-level keys,
    so strip any existing overridden lines first)."""
    lines = [
        ln for ln in base_config.splitlines()
        if not ln.strip().startswith(("agent_loop:", "agent_loop_max_turns:",
                                      "plan_mode:"))
    ]
    lines += [
        f"agent_loop: {'true' if agent_loop else 'false'}",
        "agent_loop_max_turns: 8",
    ]
    if plan_mode:
        lines.append(f"plan_mode: {plan_mode}")
    return "\n".join(lines) + "\n"


def _bootstrap_code(task_text: str, use_truststore: bool) -> str:
    ts = ("import truststore; truststore.inject_into_ssl(); "
          if use_truststore else "")
    return (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); {ts}"
        f"sys.argv = ['agentchanti', {task_text!r}, '--auto', '--no-report']; "
        f"from agentchanti.orchestrator.cli import main; main()"
    )


def _read_run_log(workdir: Path) -> str:
    log_dir = workdir / ".agentchanti" / "logs"
    if not log_dir.is_dir():
        return ""
    logs = sorted(log_dir.glob("agent_*.log"))
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in logs)


def run_one(task: dict, agent_loop: bool, base_config: str,
            use_truststore: bool, keep_workdirs: bool,
            plan_mode: str | None = None) -> dict:
    mode = "on" if agent_loop else "off"
    label = mode + (f"-{plan_mode}" if plan_mode else "")
    workdir = Path(tempfile.mkdtemp(
        prefix=f"ab_{task['id']}_{label}_"))
    for rel, content in task["files"].items():
        dest = workdir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
    (workdir / ".agentchanti.yaml").write_text(
        _build_config(base_config, agent_loop, plan_mode), encoding="utf-8")

    print(f"  [{task['id']} / loop={mode}"
          f"{f' / plan={plan_mode}' if plan_mode else ''}] "
          f"running in {workdir} ...",
          flush=True)
    started = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-c",
             _bootstrap_code(task["task"], use_truststore)],
            cwd=workdir, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=RUN_TIMEOUT_S,
            env={**__import__("os").environ,
                 "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        stdout_tail = (proc.stdout or "")[-2000:]
        stderr_tail = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout_tail = "(timed out)"
        stderr_tail = ""
    wall_s = round(time.monotonic() - started, 1)

    log_text = _read_run_log(workdir)

    # Ground truth: every success command must pass in the workdir.
    ground_truth = True
    check_outputs = []
    for cmd in task["success_cmds"]:
        try:
            chk = subprocess.run(cmd, shell=True, cwd=workdir,
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace",
                                 stdin=subprocess.DEVNULL, timeout=120)
            check_outputs.append(
                f"$ {cmd}\n(exit {chk.returncode}) "
                f"{(chk.stdout + chk.stderr)[-300:].strip()}")
            if chk.returncode != 0:
                ground_truth = False
        except subprocess.TimeoutExpired:
            check_outputs.append(f"$ {cmd}\n(timed out after 120s)")
            ground_truth = False

    result = {
        "task": task["id"],
        "agent_loop": agent_loop,
        "plan_mode": plan_mode or "(config default)",
        "ground_truth": ground_truth,
        "pipeline_claim": parse_pipeline_claim(log_text),
        "tokens": parse_total_tokens(log_text),
        "wall_s": wall_s,
        "timed_out": timed_out,
        "loop_stats": parse_loop_stats(log_text),
        "workdir": str(workdir),
        "check_outputs": check_outputs,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if not keep_workdirs:
        shutil.rmtree(workdir, ignore_errors=True)
        result["workdir"] = "(removed)"
    return result


def print_table(results: list[dict]) -> None:
    hdr = (f"{'task':<20} {'loop':<5} {'plan':<9} {'truth':<6} {'claim':<6} "
           f"{'tokens':>8} {'time_s':>7}  loop_stats")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        plan = r.get("plan_mode") or "-"
        if plan == "(config default)":
            plan = "-"
        print(f"{r['task']:<20} {'on' if r['agent_loop'] else 'off':<5} "
              f"{plan:<9} "
              f"{'PASS' if r['ground_truth'] else 'FAIL':<6} "
              f"{str(r['pipeline_claim']):<6} "
              f"{str(r['tokens'] or '-'):>8} {r['wall_s']:>7}  "
              f"{(r['loop_stats'] or '-')[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Path to a .agentchanti.yaml with provider/keys")
    ap.add_argument("--tasks", default="",
                    help="Comma-separated task ids (default: all)")
    ap.add_argument("--modes", default="on,off",
                    help="Which agent_loop modes to run: on, off, or on,off")
    ap.add_argument("--plan-modes", default="",
                    help="Optional plan_mode axis: content, intent, or "
                         "content,intent (empty = use the config's value)")
    ap.add_argument("--truststore", action="store_true",
                    help="Inject truststore into child runs (TLS-intercepted "
                         "environments)")
    ap.add_argument("--keep-workdirs", action="store_true",
                    help="Keep per-run work directories for inspection")
    args = ap.parse_args()

    base_config = Path(args.config).read_text(encoding="utf-8")
    wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
    tasks = [t for t in TASKS if not wanted or t["id"] in wanted]
    modes = [m.strip() == "on" for m in args.modes.split(",") if m.strip()]
    plan_modes: list[str | None] = [
        p.strip() for p in args.plan_modes.split(",") if p.strip()
    ] or [None]
    for p in plan_modes:
        if p is not None and p not in ("content", "intent"):
            ap.error(f"invalid --plan-modes value: {p}")

    print(f"Running {len(tasks)} task(s) x {len(modes)} loop mode(s) x "
          f"{len(plan_modes)} plan mode(s)")
    results = []
    for task in tasks:
        for agent_loop in modes:
            for plan_mode in plan_modes:
                results.append(run_one(task, agent_loop, base_config,
                                       args.truststore, args.keep_workdirs,
                                       plan_mode=plan_mode))

    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ab_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_table(results)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

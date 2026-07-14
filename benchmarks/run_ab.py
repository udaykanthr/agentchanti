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


# Clears VERIFY_X509_STRICT on urllib3's SSL contexts. Python 3.13-era
# strictness rejects the TLS-interception root on this class of machine
# ("Basic Constraints of CA cert not marked critical") even when the CA
# bundle contains it — Windows' own verifier (and truststore) tolerate
# the same cert. Patching both binding sites: urllib3.connection
# from-imports the factory at import time.
_LAX_TLS_SNIPPET = """\
import ssl as _ssl
try:
    import urllib3.util.ssl_ as _u1
    import urllib3.connection as _u2
    _orig_cuc = _u1.create_urllib3_context
    def _lax_cuc(*a, **k):
        _c = _orig_cuc(*a, **k)
        try:
            _c.verify_flags &= ~_ssl.VERIFY_X509_STRICT
        except Exception:
            pass
        return _c
    _u1.create_urllib3_context = _lax_cuc
    if getattr(_u2, 'create_urllib3_context', None) is _orig_cuc:
        _u2.create_urllib3_context = _lax_cuc
except Exception:
    pass
"""


def _bootstrap_code(task_text: str, use_truststore: bool,
                    lax_tls: bool = False) -> str:
    ts = ("import truststore; truststore.inject_into_ssl()\n"
          if use_truststore else "")
    lax = _LAX_TLS_SNIPPET if lax_tls else ""
    return (
        f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n{ts}{lax}"
        f"sys.argv = ['agentchanti', {task_text!r}, '--auto', '--no-report']\n"
        f"from agentchanti.orchestrator.cli import main\nmain()"
    )


def _build_ca_bundle() -> str | None:
    """Combined certifi + Windows system-store CA bundle for child runs.

    Replaces ``truststore.inject_into_ssl()`` in the child process: the
    injected verification hooks race under concurrent SSL from worker
    threads (KB embedder + main-thread LLM call) and intermittently
    hard-abort the whole process with 0xC0000409 — observed as 4-6s
    "failures" with empty output. A plain CA file handed to `requests`
    via REQUESTS_CA_BUNDLE trusts the same interception certs with no
    native hooks in the verification path. Returns the bundle path, or
    None to fall back to injection.
    """
    try:
        import ssl

        import certifi
        parts = [Path(certifi.where()).read_text(encoding="utf-8",
                                                 errors="replace")]
        if sys.platform == "win32":
            for store in ("ROOT", "CA"):
                for cert, enc, _trust in ssl.enum_certificates(store):
                    if enc == "x509_asn":
                        parts.append(ssl.DER_cert_to_PEM_cert(cert))
        out = REPO_ROOT / "benchmarks" / "results" / "ca_bundle.pem"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(parts), encoding="utf-8")
        return str(out)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  (CA bundle build failed: {exc} — falling back to "
              f"truststore injection)", flush=True)
        return None


def _read_run_log(workdir: Path) -> str:
    log_dir = workdir / ".agentchanti" / "logs"
    if not log_dir.is_dir():
        return ""
    logs = sorted(log_dir.glob("agent_*.log"))
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in logs)


def run_one(task: dict, agent_loop: bool, base_config: str,
            use_truststore: bool, keep_workdirs: bool,
            plan_mode: str | None = None,
            ca_bundle: str | None = None) -> dict:
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
    crash_retries = 0
    returncode = None
    child_env = {**__import__("os").environ,
                 "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    inject_truststore = use_truststore
    if use_truststore and ca_bundle:
        # Env-var CA bundle instead of in-process injection (see
        # _build_ca_bundle for why injection is not thread-safe here).
        inject_truststore = False
        child_env["REQUESTS_CA_BUNDLE"] = ca_bundle
        child_env["SSL_CERT_FILE"] = ca_bundle
        child_env["CURL_CA_BUNDLE"] = ca_bundle
    while True:
        try:
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", "-c",
                 _bootstrap_code(task["task"], inject_truststore,
                                 lax_tls=not inject_truststore
                                 and ca_bundle is not None)],
                cwd=workdir, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                stdin=subprocess.DEVNULL,
                timeout=RUN_TIMEOUT_S,
                env=child_env,
            )
            returncode = proc.returncode
            stdout_tail = (proc.stdout or "")[-2000:]
            stderr_tail = (proc.stderr or "")[-2000:]
        except subprocess.TimeoutExpired:
            timed_out = True
            stdout_tail = "(timed out)"
            stderr_tail = ""

        log_text = _read_run_log(workdir)

        # A native hard abort (observed: intermittent 0xC0000409
        # STATUS_STACK_BUFFER_OVERRUN during KB startup — empty stdout
        # AND stderr, seconds of wall time, no pipeline verdict in the
        # log) is an infrastructure flake, not a task result. One retry
        # keeps the benchmark honest; a repeat is reported as-is.
        crashed = (not timed_out
                   and returncode not in (0, 1)
                   and parse_pipeline_claim(log_text) is None)
        if crashed and crash_retries < 2:
            crash_retries += 1
            print(f"    native crash (rc={returncode:#x}) — "
                  f"retry {crash_retries}/2 ...", flush=True)
            continue
        break
    wall_s = round(time.monotonic() - started, 1)

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
        "returncode": returncode,
        "crash_retries": crash_retries,
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

    ca_bundle = _build_ca_bundle() if args.truststore else None
    if ca_bundle:
        print(f"TLS via CA bundle: {ca_bundle}")

    print(f"Running {len(tasks)} task(s) x {len(modes)} loop mode(s) x "
          f"{len(plan_modes)} plan mode(s)")
    results = []
    for task in tasks:
        for agent_loop in modes:
            for plan_mode in plan_modes:
                results.append(run_one(task, agent_loop, base_config,
                                       args.truststore, args.keep_workdirs,
                                       plan_mode=plan_mode,
                                       ca_bundle=ca_bundle))

    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ab_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_table(results)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

"""`agentchanti mcp list` / `get` — read the MCP config, and health-check it.

WHY THIS EXISTS

Configuring an MCP server was a write-and-hope affair: the only way to learn
whether the entry worked was to spend a whole run and read the log. Every
mistake available costs the same amount to discover, and three of them were
hit by hand within one afternoon of using the feature:

* the optional `mcp` extra not installed, so no server can start at all;
* a server with neither `read_only: true` nor an `allow:` list, which
  `load_specs` REJECTS outright rather than merely limiting — the entry is
  dropped and the run proceeds with no external tools;
* no `snapshot:` pair and no `state_probe:`, which costs the run its undo
  and leaves `observe_gate_verdict` silent by construction.

None of those needs a model to diagnose, and none of them should need a run.

It deliberately needs **no provider and no API key**, the same argument
`--restore` makes: starting a configured MCP server needs neither, and
someone checking why their server is not working must not be asked for
credentials to find out.

NAMING

`list` and `get` rather than a new `test` verb, because `claude mcp list`
and `claude mcp get` already health-check and the muscle memory is worth
more than a name of our own. A reader who knows one knows the other.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Fields whose ABSENCE is worth reporting, with what it costs. Phrased as
# the consequence rather than the field, because "no state_probe" means
# nothing to someone who has not read external_state.py.
_MISSING_COSTS = {
    "snapshot": ("no undo — `agentchanti --restore` cannot put this system "
                 "back, because there is no general way to snapshot one"),
    "state_probe": ("the stall detector is blind to it — a tool-only step "
                    "writes no files, so without a probe it cannot tell a "
                    "broken gate from a model that changed nothing"),
}


def _load(config_path: str | None):
    """(specs, problems, cfg, path). Never raises; a bad config is a report.

    The path comes from `_find_config_file`, the same function `Config.load`
    calls, so the two cannot disagree about which file was read — `Config`
    itself does not record it.

    Printing it is not decoration. `_find_config_file` returns the FIRST
    match, CWD then home, and never merges: a project `.agentchanti.yaml`
    shadows a home one entirely. "I added a server and nothing happened" is
    most often that, and naming the file answers it immediately.
    """
    from .config import Config, _find_config_file
    from .mcp_bridge import load_specs
    path = _find_config_file(config_path)
    cfg = Config.load(config_path)
    specs, problems = load_specs(getattr(cfg, "MCP", None))
    return specs, problems, cfg, path


def _describe_target(spec: Any) -> str:
    if spec.transport == "stdio":
        return " ".join([spec.command or "?", *spec.args])
    return spec.url or "?"


def _fence(spec: Any) -> str:
    if spec.read_only:
        return "read_only: true"
    if spec.allow:
        return f"allow: {', '.join(spec.allow)}"
    # Unreachable from `_load` — such a spec never survives `load_specs` —
    # but printed rather than assumed away, because a future loader change
    # must not make this silently say nothing.
    return "NO FENCE — this entry is rejected at load"


def _health(specs, cfg, want: str | None):
    """Start the bridge and report what each server offers.

    Returns (bridge_or_None, note, checked). `checked` separates two
    outcomes a single None conflated: the health check could not be
    PERFORMED (the extra is absent, nothing to connect to), versus it was
    performed and the server FAILED it. Reporting the second as "not
    checked" is the `empty_suite_reason` mistake in miniature — a verdict
    read as an absence of one — and it also decided the exit code, so a
    server that could not start reported success.
    """
    from . import mcp_bridge
    # Anything the bridge logs goes to stderr while print() is buffered, so
    # without this the warnings land above a header that was written first.
    sys.stdout.flush()
    ok, why = mcp_bridge.mcp_available()
    if not ok:
        return None, why, False
    if not specs:
        return None, "no usable server to connect to", False
    # A filtered Config so `get` starts only the server asked about: a
    # health check on one server must not pay for every other server's
    # handshake, and must not fail because an unrelated one is down.
    raw = getattr(cfg, "MCP", None)
    if want:
        entries = [{"name": s.name, "transport": s.transport,
                    "command": s.command, "args": list(s.args),
                    "env": dict(s.env), "url": s.url,
                    "headers": dict(s.headers), "read_only": s.read_only,
                    "allow": list(s.allow), "timeout": s.timeout,
                    "snapshot": dict(s.snapshot),
                    "state_probe": s.state_probe}
                   for s in specs if s.name == want]
        raw = {"servers": entries}

    class _Cfg:
        MCP = raw

    bridge = mcp_bridge.ensure_started(_Cfg())
    if bridge is None:
        return None, "no tool was offered", True
    return bridge, "", True


def _offered_by_server(bridge) -> dict:
    out: dict[str, list[str]] = {}
    for d in bridge.definitions():
        server, _, tool = d.name.partition("__")
        out.setdefault(server, []).append(tool)
    return out


def _withheld_by_server(bridge) -> dict:
    out: dict[str, list[tuple[str, str]]] = {}
    for w in getattr(bridge, "withheld", ()):
        server, _, tool = w.qualified.partition("__")
        out.setdefault(server, []).append((tool, w.reason))
    return out


def _print_problems(problems) -> None:
    for p in problems:
        print(f"  ! {p}")
    if problems:
        print()


def _cmd_list(args) -> int:
    specs, problems, cfg, path = _load(args.config)
    print(f"\nMCP servers in {path or '(no .agentchanti.yaml found)'}\n")
    _print_problems(problems)
    if not specs:
        print("  No usable server configured.")
        print("  Declare one under `mcp:` in .agentchanti.yaml, or run "
              "`agentchanti mcp get --help`.\n")
        return 0 if not problems else 1

    if args.no_health:
        bridge, note, checked = None, "", False
    else:
        bridge, note, checked = _health(specs, cfg, None)
    offered = _offered_by_server(bridge) if bridge else {}
    unhealthy = 0
    try:
        for spec in specs:
            tools = offered.get(spec.name)
            if args.no_health:
                status = ""
            elif not checked:
                status = f"  [not checked: {note}]"
            elif tools:
                status = f"  [{len(tools)} tool(s) offered]"
            else:
                status = "  [FAILED — offers nothing; see the warnings]"
            if checked and not tools:
                unhealthy += 1
            print(f"  {spec.name}{status}")
            print(f"      {spec.transport}: {_describe_target(spec)}")
            print(f"      {_fence(spec)}")
            missing = [k for k in _MISSING_COSTS if not getattr(spec, k, None)]
            for key in missing:
                print(f"      no {key} — {_MISSING_COSTS[key]}")
            print()
    finally:
        if bridge is not None:
            from . import mcp_bridge
            mcp_bridge.stop_active()
    print(f"  {len(specs)} server(s). `agentchanti mcp get <name>` for one "
          f"in full.\n")
    # Non-zero when a check was performed and something failed it, so this
    # is usable from a script and from CI. A health check that reports a
    # dead server and then exits 0 says the opposite of what it printed.
    if problems or unhealthy:
        return 1
    if not args.no_health and not checked:
        return 1
    return 0


def _cmd_get(args) -> int:
    specs, problems, cfg, path = _load(args.config)
    _print_problems(problems)
    match = [s for s in specs if s.name == args.name]
    if not match:
        known = ", ".join(s.name for s in specs) or "none"
        print(f"\n  No usable server named {args.name!r} in "
              f"{path or '(no .agentchanti.yaml found)'}. Configured: "
              f"{known}\n")
        return 1
    spec = match[0]

    print(f"\n{spec.name}\n")
    print(f"  transport    {spec.transport}")
    print(f"  {'command' if spec.transport == 'stdio' else 'url':<12} "
          f"{_describe_target(spec)}")
    print(f"  fence        {_fence(spec)}")
    print(f"  timeout      {spec.timeout:g}s")
    if spec.env:
        print(f"  env          {', '.join(sorted(spec.env))}")
    if spec.headers:
        # Names only. A header's VALUE is usually a bearer token, and a
        # command people paste into issues must not print credentials.
        print(f"  headers      {', '.join(sorted(spec.headers))} "
              f"(values hidden)")
    for key, cost in _MISSING_COSTS.items():
        value = getattr(spec, key, None)
        print(f"  {key:<12} {'declared' if value else 'MISSING — ' + cost}")
    print()

    if args.no_health:
        return 0
    bridge, note, checked = _health([spec], cfg, spec.name)
    if bridge is None:
        label = "FAILED" if checked else "NOT CHECKED"
        print(f"  health: {label} — {note}; see the warnings above\n")
        return 1
    try:
        offered = _offered_by_server(bridge).get(spec.name, [])
        withheld = _withheld_by_server(bridge).get(spec.name, [])
        print(f"  health: connected — {len(offered)} offered, "
              f"{len(withheld)} withheld")
        for tool in sorted(offered):
            print(f"    + {tool}")
        for tool, reason in sorted(withheld):
            print(f"    - {tool}  ({reason})")
        print()
        return 0 if offered else 1
    finally:
        from . import mcp_bridge
        mcp_bridge.stop_active()


def mcp_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentchanti mcp",
        description="Inspect and health-check configured MCP servers.")
    parser.add_argument("--config", default=None,
                        help="Path to .agentchanti.yaml")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List configured MCP servers")
    p_list.add_argument("--no-health", action="store_true",
                        help="Read the config only; do not start any server")
    p_list.set_defaults(func=_cmd_list)

    p_get = sub.add_parser("get", help="Details for one MCP server")
    p_get.add_argument("name")
    p_get.add_argument("--no-health", action="store_true",
                       help="Read the config only; do not start the server")
    p_get.set_defaults(func=_cmd_get)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    # The bridge reports every configuration fault through `log.warning`
    # — the missing extra, a withheld tool, a server that will not start.
    # Without a handler those reach stderr through logging's last resort
    # with a `WARNING:agentchanti.mcp_bridge:` prefix; this prints the
    # message itself, which is what the reader needs.
    #
    # The level is set on the `agentchanti` logger and not just on root,
    # because importing the package leaves that logger at DEBUG: with root
    # alone at WARNING, three INFO/DEBUG lines still reached the handler —
    # a tool count this command already summarises, and a teardown note
    # (`shutdown raised, continuing`) that is `log.debug` precisely because
    # it is not actionable. A report whose own chatter buries its finding
    # is the thing this command exists to replace.
    import logging
    logging.basicConfig(level=logging.WARNING, format="  ! %(message)s")
    logging.getLogger("agentchanti").setLevel(logging.WARNING)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130

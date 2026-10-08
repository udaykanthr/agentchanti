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
import re
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


# ── writing the config back ──────────────────────────────────────────
#
# `--generate-config` rewrites `.agentchanti.yaml` wholesale from
# `cfg.to_yaml()`. Doing that here would be destructive in a way nobody
# asked for: a real MCP config is heavily commented — the one this feature
# was developed against carries the reasoning for `allow:`, `env:` and the
# snapshot pair inline — and a PyYAML round-trip drops every comment, every
# blank line and the key order with them.
#
# So the only operation performed on an existing file is INSERTION of new
# lines. Nothing already in it is reformatted, reordered or re-quoted. When
# the shape cannot be determined with certainty the edit is REFUSED and the
# block is printed for the reader to paste — the `unrunnable_gate_reason`
# temperament, where declining beats half-doing it.

_TOP_LEVEL_MCP = re.compile(r"^mcp:\s*(?:#.*)?$")
_SERVERS_KEY = re.compile(r"^(\s+)servers:\s*(?:#.*)?$")


def _server_block(entry: dict, indent: str) -> str:
    """*entry* as a YAML list item, indented to *indent*.

    Dumped by PyYAML rather than formatted by hand so that quoting is the
    library's problem: a Windows command path, a header value and a
    `mcp:`-syntax snapshot payload are all full of characters that decide
    whether YAML reads a string or something else.
    """
    import yaml
    body = yaml.safe_dump([entry], default_flow_style=False, sort_keys=False,
                          allow_unicode=True)
    return "".join(indent + line if line.strip() else line
                   for line in body.splitlines(keepends=True))


def _insert_server(text: str, entry: dict) -> tuple[str | None, str]:
    """(new_text, note). `None` means refused — the note says why."""
    lines = text.splitlines(keepends=True)
    newline = "\r\n" if text.count("\r\n") > len(lines) / 2 else "\n"

    mcp_at = next((i for i, ln in enumerate(lines)
                   if _TOP_LEVEL_MCP.match(ln.rstrip("\r\n"))), None)
    if mcp_at is None:
        # No `mcp:` at all: appending a whole section touches nothing.
        block = _server_block(entry, "    ")
        tail = "" if not text or text.endswith(("\n", "\r")) else newline
        section = (f"{tail}{newline}# External tools from MCP servers."
                   f"{newline}mcp:{newline}  servers:{newline}")
        return text + section + block, "appended a new `mcp:` section"

    # Find `servers:` inside the mcp block — i.e. before the next line that
    # starts at column 0 and is neither blank nor a comment.
    end = len(lines)
    for i in range(mcp_at + 1, len(lines)):
        stripped = lines[i].rstrip("\r\n")
        if stripped and not stripped[0].isspace() and not stripped.startswith("#"):
            end = i
            break

    servers_at = servers_indent = None
    for i in range(mcp_at + 1, end):
        m = _SERVERS_KEY.match(lines[i].rstrip("\r\n"))
        if m:
            servers_at, servers_indent = i, m.group(1)
            break
    if servers_at is None:
        return None, ("`mcp:` exists but has no plain `servers:` key — it may "
                      "be a list or inline mapping, and inserting into one "
                      "blindly could corrupt it")

    # The list items, and where the list stops.
    item_indent = None
    last_item_line = servers_at
    for i in range(servers_at + 1, end):
        raw = lines[i].rstrip("\r\n")
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lead = raw[:len(raw) - len(raw.lstrip())]
        if len(lead) <= len(servers_indent):
            break                      # back out to a sibling key of servers
        if raw.lstrip().startswith("- ") and item_indent is None:
            item_indent = lead
        last_item_line = i
    if item_indent is None:
        item_indent = servers_indent + "  "

    block = _server_block(entry, item_indent)
    if not lines[last_item_line].endswith(("\n", "\r")):
        lines[last_item_line] = lines[last_item_line] + newline
    return ("".join(lines[:last_item_line + 1]) + block
            + "".join(lines[last_item_line + 1:]),
            "inserted into the existing `mcp.servers` list")


def _mcp_block(lines: list[str]) -> tuple[int, int] | None:
    """(start, end) line indices of the top-level `mcp:` block, or None.

    `end` is the first line at column 0 that is neither blank nor a
    comment — i.e. the next top-level key.
    """
    start = next((i for i, ln in enumerate(lines)
                  if _TOP_LEVEL_MCP.match(ln.rstrip("\r\n"))), None)
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].rstrip("\r\n")
        if stripped and not stripped[0].isspace() and not stripped.startswith("#"):
            end = i
            break
    return start, end


def _servers_list(lines: list[str]) -> tuple[int, str, list[tuple[int, int]]] | None:
    """(servers_line, item_indent, [(first, last)]) for each list item.

    None when the shape is not a plain block `servers:` with block items —
    flow style and the bare-list form of `mcp:` both land here, and both are
    refused rather than guessed at.
    """
    block = _mcp_block(lines)
    if block is None:
        return None
    start, end = block
    servers_at = servers_indent = None
    for i in range(start + 1, end):
        m = _SERVERS_KEY.match(lines[i].rstrip("\r\n"))
        if m:
            servers_at, servers_indent = i, m.group(1)
            break
    if servers_at is None:
        return None

    items: list[tuple[int, int]] = []
    item_indent = None
    for i in range(servers_at + 1, end):
        raw = lines[i].rstrip("\r\n")
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lead = raw[:len(raw) - len(raw.lstrip())]
        if len(lead) <= len(servers_indent):
            break
        if raw.lstrip().startswith("- "):
            if item_indent is None:
                item_indent = lead
            if len(lead) == len(item_indent):
                items.append((i, i))
                continue
        if items:
            items[-1] = (items[-1][0], i)
    if item_indent is None:
        item_indent = servers_indent + "  "
    return servers_at, item_indent, items


def _item_name(lines: list[str], first: int, last: int) -> str | None:
    """The `name:` of one list item, wherever in the item it sits."""
    for i in range(first, last + 1):
        text = lines[i].rstrip("\r\n").lstrip()
        if text.startswith("- "):
            text = text[2:]
        if text.startswith("name:"):
            value = text[len("name:"):].strip()
            if "#" in value:
                value = value.split("#", 1)[0].strip()
            return value.strip("'\"") or None
    return None


def _remove_server(text: str, name: str) -> tuple[str | None, str]:
    """(new_text, note). `None` means refused or not found — note says which.

    Deletion only, in the same spirit as insertion: the item's lines go and
    nothing else is reformatted. Removing the LAST server rewrites
    `servers:` to `servers: []` rather than leaving the key valueless —
    `load_specs` reads a bare `servers:` as None and reports
    `should be a list or mapping of servers, got NoneType`, so tidying the
    file would otherwise leave it complaining.
    """
    lines = text.splitlines(keepends=True)
    found = _servers_list(lines)
    if found is None:
        return None, ("no plain `mcp: servers:` block to edit — it may be "
                      "inline or a bare list, and editing one blindly could "
                      "corrupt it")
    servers_at, _item_indent, items = found
    matches = [(a, b) for a, b in items if _item_name(lines, a, b) == name]
    if not matches:
        return None, f"no server named {name!r} in this file"

    first, last = matches[0]
    note = "removed"
    if len(items) == 1:
        # Keep the section and its comments; empty the list explicitly.
        raw = lines[servers_at]
        newline = raw[len(raw.rstrip("\r\n")):] or "\n"
        indent = _SERVERS_KEY.match(raw.rstrip("\r\n")).group(1)
        lines[servers_at] = f"{indent}servers: []{newline}"
        note = "removed — it was the last server, so `servers:` is now empty"
    del lines[first:last + 1]
    if len(matches) > 1:
        note += (f"; {len(matches) - 1} more entr(y/ies) named {name!r} "
                 f"remain — the file had duplicates")
    return "".join(lines), note


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


# ── add ──────────────────────────────────────────────────────────────

def _kv(pairs, sep, what):
    """`KEY=value` / `Header: value` pairs into a dict, or raise."""
    out = {}
    for item in pairs or ():
        key, found, value = item.partition(sep)
        if not found or not key.strip():
            raise ValueError(f"{what} must be `KEY{sep}value`, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def _discover(entry: dict) -> tuple[list[str], str]:
    """Every tool the server offers, by connecting once. ([], why) on failure.

    Discovery uses an in-memory spec with `read_only=True` and NO allow
    list, which is the one combination `tool_defs_for` passes through
    unfiltered — the point is to see everything so the operator can choose.
    It is never written: what lands in the file is the fence they picked.
    """
    from . import mcp_bridge
    ok, why = mcp_bridge.mcp_available()
    if not ok:
        return [], why
    probe = dict(entry)
    probe.pop("allow", None)
    probe["read_only"] = True

    class _Cfg:
        MCP = {"servers": [probe]}

    sys.stdout.flush()
    bridge = mcp_bridge.ensure_started(_Cfg())
    if bridge is None:
        return [], "the server offered no tools — see the warnings above"
    try:
        return sorted(d.name.partition("__")[2]
                      for d in bridge.definitions()), ""
    finally:
        mcp_bridge.stop_active()


def _write_entry(args, entry: dict, path: str | None) -> int:
    """Insert *entry*, verify it parses, and report. Returns an exit code."""
    specs, _problems, _cfg, _p = _load(args.config)
    if any(s.name == entry["name"] for s in specs):
        print(f"\n  A server named {entry['name']!r} is already configured in "
              f"{path}.\n  Edit it there, or choose another name — names "
              f"qualify tool names and must be unique.\n")
        return 1

    target = path or os.path.join(os.getcwd(), ".agentchanti.yaml")
    text = ""
    if os.path.isfile(target):
        with open(target, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()

    updated, note = _insert_server(text, entry)
    if updated is None:
        print(f"\n  Not editing {target} automatically: {note}.\n"
              f"  Add this under `mcp: servers:` yourself:\n")
        print(_server_block(entry, "    "))
        return 1

    with open(target, "w", encoding="utf-8", newline="") as fh:
        fh.write(updated)
    print(f"\n  Added {entry['name']!r} to {target} ({note}).")

    # Verified by re-reading the file rather than trusting the dict we
    # wrote: the question is whether the PIPELINE will accept it, and the
    # only thing that answers that is the loader the pipeline uses.
    specs, problems, _cfg, _p = _load(target)
    for p in problems:
        print(f"  ! {p}")
    if not any(s.name == entry["name"] for s in specs):
        print(f"  ! it did not load back — the entry is in the file but the "
              f"loader rejected it.\n")
        return 1
    print(f"  Check it any time with `agentchanti mcp get "
          f"{entry['name']}`.\n")
    return 0


def _cmd_add(args) -> int:
    if args.scope != "project":
        print(f"\n  --scope {args.scope} is not supported yet. "
              f"`.agentchanti.yaml` is found CWD-first and NEVER merged, so "
              f"a server written to a home config is silently ignored in "
              f"every project that has its own file — which is every project "
              f"that sets a provider. Use --scope project (the default).\n")
        return 2
    try:
        env = _kv(args.env, "=", "--env")
        headers = _kv(args.header, ":", "--header")
    except ValueError as exc:
        print(f"\n  {exc}\n")
        return 2

    entry: dict = {"name": args.name, "transport": args.transport}
    if args.transport == "stdio":
        entry["command"] = args.command_or_url
        if args.args:
            entry["args"] = list(args.args)
    else:
        entry["url"] = args.command_or_url
        if args.args:
            print("\n  Positional arguments are only meaningful for a stdio "
                  "server; ignoring them for an http one.\n")
    if env:
        entry["env"] = env
    if headers:
        entry["headers"] = headers
    if args.timeout:
        entry["timeout"] = args.timeout

    _specs, _problems, _cfg, path = _load(args.config)

    # The fence. A server with neither is REJECTED by `load_specs`, so
    # writing one would produce a file that parses and a run with no
    # external tools — the failure this whole command exists to prevent.
    allow = list(args.allow or ())
    if args.read_only:
        entry["read_only"] = True
    elif allow:
        entry["allow"] = allow
    else:
        found, why = ([], "discovery skipped") if args.no_discover else \
            _discover(entry)
        if not found:
            print(f"\n  Need a fence before this can be written: pass "
                  f"--allow <tool...> or --read-only.\n"
                  f"  Could not list the server's tools to help "
                  f"({why}).\n")
            return 1
        print(f"\n  {args.name} offers {len(found)} tool(s): "
              f"{', '.join(found)}")
        # `isatty()` is not the question, and on Windows it is not even an
        # answer: the CRT reports every CHARACTER DEVICE as a tty, so a
        # process whose stdin is NUL — which is what a CI runner and
        # `subprocess.DEVNULL` both give — reports True and then raises
        # EOFError on the first read. Measured here before it shipped.
        # Asking and handling the EOF is the only reliable test, and
        # `_prompt_for_acceptance_cmds` already draws the same conclusion:
        # a closed stdin is DECLINING, not an error.
        try:
            reply = input("  Allow which? (blank = all, or space-separated "
                          "names, or `q`): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  Nothing to read from, so nothing is assumed and "
                  f"nothing was written. Re-run with one of:\n"
                  f"    --allow {' '.join(found)}\n"
                  f"    --read-only\n")
            return 1
        if reply.lower() == "q":
            print("  Nothing written.\n")
            return 1
        chosen = reply.split() if reply else list(found)
        unknown = [t for t in chosen if t not in found]
        if unknown:
            print(f"\n  This server does not offer: {', '.join(unknown)}\n")
            return 1
        entry["allow"] = chosen

    return _write_entry(args, entry, path)


def _cmd_add_json(args) -> int:
    """The escape hatch, and the only sane home for `snapshot:`.

    A snapshot command is a `mcp:`-syntax tool call carrying a JSON payload
    full of quotes and backslashes. Putting that on a command line is how
    this project has repeatedly been burned — cmd.exe losing quote tracking
    on `\\"`, reading `<3` as a redirect — so those fields get a JSON
    document instead of three more flags.
    """
    import json
    if args.scope != "project":
        print(f"\n  --scope {args.scope} is not supported yet; see "
              f"`agentchanti mcp add --help`.\n")
        return 2
    try:
        entry = json.loads(args.json)
    except ValueError as exc:
        print(f"\n  That is not valid JSON: {exc}\n")
        return 2
    if not isinstance(entry, dict):
        print(f"\n  Expected a JSON object describing one server, got "
              f"{type(entry).__name__}.\n")
        return 2
    entry = {"name": args.name, **{k: v for k, v in entry.items()
                                   if k != "name"}}
    _specs, _problems, _cfg, path = _load(args.config)
    return _write_entry(args, entry, path)


def _cmd_remove(args) -> int:
    if args.scope != "project":
        print(f"\n  --scope {args.scope} is not supported yet; see "
              f"`agentchanti mcp add --help`.\n")
        return 2
    _specs, _problems, _cfg, path = _load(args.config)
    if not path or not os.path.isfile(path):
        print("\n  No .agentchanti.yaml to edit.\n")
        return 1
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()

    updated, note = _remove_server(text, args.name)
    if updated is None:
        print(f"\n  {note}.\n")
        return 1
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(updated)
    print(f"\n  {args.name}: {note} ({path}).")

    # Re-read through the pipeline's own loader, exactly as `add` does: the
    # question is not whether the text changed but whether what is left is
    # a config the run will accept.
    specs, problems, _cfg, _p = _load(path)
    for p in problems:
        print(f"  ! {p}")
    if any(s.name == args.name for s in specs):
        print(f"  ! it is still being loaded — the removal did not take.\n")
        return 1
    remaining = ", ".join(s.name for s in specs) or "none"
    print(f"  Remaining: {remaining}\n")
    return 1 if problems else 0


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

    p_add = sub.add_parser(
        "add", help="Add an MCP server to .agentchanti.yaml",
        description=(
            "Add an MCP server.\n\n"
            "Examples:\n"
            "  agentchanti mcp add blender -- python blender_server.py\n"
            "  agentchanti mcp add thing --allow read_x,read_y -- npx my-mcp\n"
            "  agentchanti mcp add api --transport http https://x.test/mcp \\\n"
            "      -H 'Authorization: Bearer ...' --read-only\n\n"
            "With neither --allow nor --read-only the server is started once "
            "and its tools listed, so the fence can be chosen from what it "
            "actually offers."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_add.add_argument("name")
    p_add.add_argument("command_or_url", metavar="commandOrUrl")
    p_add.add_argument("args", nargs="*",
                       help="Arguments for a stdio command (after `--`)")
    p_add.add_argument("--transport", choices=["stdio", "http"],
                       default="stdio")
    p_add.add_argument("-e", "--env", action="append", metavar="KEY=value",
                       help="Environment variable for a stdio server")
    p_add.add_argument("-H", "--header", action="append",
                       metavar="'Name: value'",
                       help="Header for an http server")
    p_add.add_argument("-s", "--scope", default="project",
                       help="Config scope (project only, for now)")
    p_add.add_argument("--allow", nargs="+", metavar="TOOL",
                       help="Tools the agent may call")
    p_add.add_argument("--read-only", action="store_true",
                       help="Assert the whole server is read-only")
    p_add.add_argument("--timeout", type=float, default=None,
                       help="Per-call timeout in seconds")
    p_add.add_argument("--no-discover", action="store_true",
                       help="Do not start the server to list its tools")
    p_add.set_defaults(func=_cmd_add)

    p_json = sub.add_parser(
        "add-json", help="Add a server from a JSON object",
        description=("Add a server from a JSON object — the home for "
                     "`snapshot:` and `state_probe:`, whose values are "
                     "tool calls carrying JSON payloads that no shell "
                     "quotes comfortably."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p_json.add_argument("name")
    p_json.add_argument("json", metavar="JSON")
    p_json.add_argument("-s", "--scope", default="project")
    p_json.set_defaults(func=_cmd_add_json)

    p_rm = sub.add_parser("remove", help="Remove a server from the config")
    p_rm.add_argument("name")
    p_rm.add_argument("-s", "--scope", default="project")
    p_rm.set_defaults(func=_cmd_remove)

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

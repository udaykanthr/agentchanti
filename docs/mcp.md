# External Tools over MCP

AgentChanti can reach a **live external application** through a [Model
Context Protocol](https://modelcontextprotocol.io) server — a running
Blender session, a FreeCAD document, a database, anything with an MCP
server in front of it. The agent edits that application directly instead
of writing a script and hoping.

This is entirely opt-in. A run that configures no server never imports the
MCP SDK, never mentions it, and its prompts are byte-identical to a build
without the extra installed.

---

## Contents

1. [Install the extra](#1-install-the-extra)
2. [Add a server](#2-add-a-server)
3. [The allow list, and why it is mandatory](#3-the-allow-list-and-why-it-is-mandatory)
4. [Check it before you spend a run](#4-check-it-before-you-spend-a-run)
5. [Two keys worth setting](#5-two-keys-worth-setting)
6. [Scopes: project and user](#6-scopes-project-and-user)
7. [Writing gates that call tools](#7-writing-gates-that-call-tools)
8. [Independent evidence for a tool-only task](#8-independent-evidence-for-a-tool-only-task)
9. [Undoing what a run changed](#9-undoing-what-a-run-changed)
10. [Command reference](#10-command-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Install the extra

The MCP SDK is an **optional extra**, not a dependency. A plain install
never downloads it — it pulls roughly thirty packages including a web
server stack, and most runs configure no server at all.

```bash
pipx install "agentchanti[mcp]"        # fresh install, with MCP
pipx inject agentchanti mcp            # already installed via pipx
python -m pip install -e ".[mcp]"      # from a clone
```

Nothing is installed behind your back, and nothing is fetched on first
use. If you configure a server without the extra, the run says so once and
carries on without external tools:

```
[MCP] the `mcp` package is not importable (ModuleNotFoundError);
      install it with `pip install agentchanti[mcp]`
```

---

## 2. Add a server

```bash
# stdio — a local process AgentChanti launches and talks to over pipes
agentchanti mcp add blender -- python path/to/blender_server.py

# with arguments and environment variables
agentchanti mcp add thing -e API_KEY=xxx -- npx -y my-mcp-server --flag

# http
agentchanti mcp add api --transport http https://example.test/mcp \
    -H 'Authorization: Bearer ...' --read-only
```

Everything after `--` is the command and its arguments, so a server's own
flags are never mistaken for AgentChanti's.

This writes into `.agentchanti.yaml`. It **only ever inserts lines** —
your comments, key order, quoting and formatting are left exactly as they
were — and it refuses rather than guesses if the file's shape is ambiguous,
printing the block for you to paste instead.

The resulting entry looks like this:

```yaml
mcp:
  servers:
    - name: blender
      transport: stdio
      command: python
      args: ["path/to/blender_server.py"]
      allow: [execute_code, get_scene_info]
```

> **A stdio server does not inherit your whole environment.** The SDK hands
> it a minimal safe set — on Windows that is twelve variables including
> `PATH`. Anything else your server needs (`UV_SYSTEM_CERTS`, a proxy
> variable, an API key) must be passed with `-e`.

---

## 3. The allow list, and why it is mandatory

Every server needs **either** an `allow:` list **or** `read_only: true`.
A server with neither is **rejected when the config loads** — the entry is
dropped and the run proceeds with no external tools at all:

```
[MCP] server 'blender' is neither marked `read_only: true` nor given an
      `allow:` list. This release offers read-only MCP tools only, because
      a tool that writes files would bypass every guard on
      AgentTools.write_file — including the one that keeps the acceptance
      instrument unwritable.
```

This is deliberate. The MCP protocol's own `readOnlyHint` is optional and
advisory, so a server can simply not set it, and trusting its absence
either way would be guessing about something that can write files. The
fence is **your** assertion, not the server's.

You do not have to guess tool names. Run `add` with neither flag and it
starts the server once, lists what it actually offers, and asks:

```
$ agentchanti mcp add blender -- python blender_server.py

  blender offers 3 tool(s): execute_code, get_object_info, get_scene_info
  Allow which? (blank = all, or space-separated names, or `q`):
```

Nothing is written until you choose. With no terminal to ask at — CI, a
script — it prints the ready-made flag and writes nothing:

```
  Re-run with one of:
    --allow execute_code get_object_info get_scene_info
    --read-only
```

---

## 4. Check it before you spend a run

```bash
agentchanti mcp list          # every server: transport, fence, health
agentchanti mcp get blender   # one server in full, offered vs withheld
```

Both connect briefly, list the server's tools, and stop. They need **no
provider and no API key** — you should not have to supply credentials to
find out why a server is not working. They exit non-zero when a server
fails its check, so they work in CI.

```
$ agentchanti mcp list

MCP servers in /home/you/project/.agentchanti.yaml

  blender  [3 tool(s) offered]
      stdio: python blender_server.py
      allow: execute_code, get_scene_info, get_object_info

  1 server(s). `agentchanti mcp get <name>` for one in full.
```

`list` names the config file it read. That matters: `.agentchanti.yaml` is
found **CWD-first**, and for everything except the `mcp:` section a project
file replaces a home one entirely.

A server that cannot start says so plainly, and the exit code agrees:

```
  ghosty  [FAILED — offers nothing; see the warnings]
  ! [MCP] MCP server 'ghosty' did not start: FileNotFoundError ...
```

Use `--no-health` to read the config without starting anything.

---

## 5. Two keys worth setting

Every safety net in AgentChanti is made of files and git — wave snapshots,
gate rollback, `agentchanti --restore`. A run that reaches a live external
system has **none of them**, because the artifact is not on disk. Two
optional keys put that right, and both are declared by you because there
is no general way to guess them: saving a `.blend`, dumping a database and
exporting a browser profile share no vocabulary.

### `state_probe:` — how to fingerprint the system

A read-only tool call whose output identifies the system's current state.

```yaml
      state_probe: 'mcp:blender__get_scene_info {}'
```

Without it, the one check designed to stop a run burning its budget on a
gate that cannot pass is **silent by construction**. It needs to see that
the artifact changed while the gate's answer did not, and a tool-only step
writes no files for it to compare.

### `snapshot:` — how to capture and restore

```yaml
      snapshot:
        capture: 'mcp:blender__execute_code {"code": "import bpy\nbpy.ops.wm.save_as_mainfile(filepath=r\"{path}\", copy=True)"}'
        restore: 'mcp:blender__execute_code {"code": "import bpy\nbpy.ops.wm.open_mainfile(filepath=r\"{path}\")"}'
```

`{path}` is substituted with a run-scoped file under
`.agentchanti/external/<server>/`. **Both halves are required** — a
capture with no restore is a file nobody can use, and either alone *looks*
like an undo, which is worse than plainly having none.

Because these values are tool calls carrying JSON payloads full of quotes
and backslashes, add them with `add-json` rather than fighting your shell:

```bash
agentchanti mcp add-json blender '{
  "transport": "stdio", "command": "python", "args": ["blender_server.py"],
  "allow": ["execute_code", "get_scene_info"],
  "state_probe": "mcp:blender__get_scene_info {}",
  "snapshot": {"capture": "...", "restore": "..."}
}'
```

A server that declares neither is warned about **once, before the first
step**, while the answer is still cheap to change.

---

## 6. Scopes: project and user

```bash
agentchanti mcp add blender -- python server.py                  # project
agentchanti mcp add shared --scope user --read-only -- python s.py  # user
```

| scope | file | offered in |
|---|---|---|
| `project` (default) | `./.agentchanti.yaml` | this project |
| `user` | `~/.agentchanti.yaml` | **every** project |

The `mcp:` section is the **one** part of the config that merges across
the two files. Everything else is first-file-wins: a provider or API key
resolved half from one file and half from another would be worse than one
source, while a list of servers is additive by nature.

On a name collision the **project wins**, and the shadowed entry is
reported rather than silently dropped. `mcp list` marks where each server
came from, and `remove` edits the file the server actually came from
unless you name a scope.

There is no `local` scope — AgentChanti has no per-project private store,
only the file itself.

---

## 7. Writing gates that call tools

A plan step's `verify:` can be a tool call, so a step that does its work
through an external tool is measured by one:

```
verify: mcp:blender__execute_code {"code": "import bpy\nassert 'Tree' in bpy.data.objects"}
```

Several calls can be chained with `&&`; all must pass. A bare qualified
tool name (`blender__get_scene_info`) is accepted too.

**Put the assertion inside a code-executing tool.** A gate that only calls
a reader passes whenever the server is reachable — *including over the
state that existed before the step ran* — and AgentChanti says so:

```
[ToolGate] step 2: the gate calls blender__get_scene_info and asserts
nothing about the result, so it passes whenever the server is reachable —
including over the state that existed before this step ran. Put the check
inside a tool that executes code and let it fail: e.g.
mcp:<server>__execute_code {"code": "...; assert <the concrete condition>"}
```

Two traps worth knowing, because both have cost real runs:

- **`is` is never true between two wrapper objects.** Most host APIs
  rebuild a Python wrapper on every attribute access, so `link.to_node is
  node` is false whatever the scene contains, while `==` compares the
  underlying object and works. AgentChanti now retries such a gate under
  the `==` reading and adopts it only if it passes, but writing `==` in
  the first place saves the detour.
- **A gate sees only the end state.** It cannot express "this did not
  change" — "unchanged", "still has its original vertex count", "nothing
  else was touched" all relate two points in time, and a gate holds one.
  That is what the next section is for.

---

## 8. Independent evidence for a tool-only task

AgentChanti separates *completed* (the plan ran) from *verified*
(something the run did not write agreed). For an ordinary project it can
seed its own contract — but a seeded contract is a **file** check and
cannot judge a live application, so a task whose artifact exists only
inside another program has nothing to verify it.

Supply `acceptance_cmds`: a script you wrote, which the agent can read but
**cannot edit**.

```yaml
acceptance_cmds:
  - python acceptance_check.py
```

Make it compare against a baseline recorded **before** the run. That is
the one thing the pipeline's own machinery structurally cannot do for
external state, and it is what catches the half-done work a gate cannot:

```bash
python acceptance_check.py --snapshot   # before: record the baseline
agentchanti --prompt-from-file task.txt # the run
python acceptance_check.py              # after: compare
```

Record enough that a placeholder cannot pass. For a 3D scene, comparing
names and positions was not enough — a repair round once satisfied such a
check by creating an *empty object with the right name*. Geometry counts,
data-block names and animation curve counts are cheap to record and cannot
be forged by an empty stand-in.

---

## 9. Undoing what a run changed

```bash
agentchanti --restore
```

Restores the project files **and** every external system that declared a
`snapshot:` pair. It needs no config, provider or API key.

**Nothing is ever restored automatically.** A run that fails can still
leave the application in exactly the state you wanted, so rolling back on
failure would destroy the work you asked for. Recovery is always explicit.

Files whose contents changed after the snapshot was taken are copied to
`.agentchanti/pre-restore/` before being overwritten, and named in the
output — an undo should not itself destroy work.

---

## 10. Command reference

```
agentchanti mcp list [--no-health]
agentchanti mcp get <name> [--no-health]
agentchanti mcp add <name> <commandOrUrl> [args...]
        --transport stdio|http     -e, --env KEY=value
        -H, --header 'Name: value' -s, --scope project|user
        --allow TOOL [TOOL...]     --read-only
        --timeout SECONDS          --no-discover
agentchanti mcp add-json <name> '<json>'  [-s project|user]
agentchanti mcp remove <name>             [-s project|user]
```

| key | meaning |
|---|---|
| `name` | Unique; qualifies tool names as `<name>__<tool>` |
| `transport` | `stdio` (a local process) or `http` |
| `command`, `args` | stdio only |
| `url`, `headers` | http only |
| `env` | Passed to a stdio server; it does **not** inherit yours |
| `allow` | Tools the agent may call — required unless `read_only` |
| `read_only` | Your assertion that the whole server is safe |
| `timeout` | Per-call timeout in seconds (default 60) |
| `state_probe` | Read-only call that fingerprints the system |
| `snapshot` | `capture` / `restore` pair, `{path}` substituted |

---

## 11. Troubleshooting

**"I added a server and nothing happened."**
Run `agentchanti mcp list`. It names the config file it actually read —
most often a project `.agentchanti.yaml` is shadowing the home one you
edited. Note that only the `mcp:` section merges across the two.

**The run says the `mcp` package is not importable.**
The extra is not installed: `pipx inject agentchanti mcp`, or
`pip install "agentchanti[mcp]"`.

**`mcp list` shows `[FAILED — offers nothing]`.**
The server started but exposed no tool, or did not start at all. The
warning beside it carries the server's own error. Check the command runs
by hand, and that anything it needs from the environment is passed with
`-e`.

**All my tools are withheld.**
`mcp get <name>` lists offered versus withheld with a reason for each.
Usually the `allow:` list names a tool the server does not have, or
spells one differently.

**The agent wrote a plan step that just `echo`s a sentence about a tool.**
That is caught and re-routed — a tool call can only happen inside a
CODE/TEST step, which runs as a tool-calling conversation. Nothing to do.

**A gate fails over work that looks correct.**
Check it for `is` between two objects fetched separately (see
[§7](#7-writing-gates-that-call-tools)), and for asserting something the
end state cannot express. `agentchanti mcp get <name>` plus running the
gate's payload by hand through the server is the fastest way to tell a
broken instrument from broken work.

---

## See also

- [README — Optional: external tools over MCP](../README.md#optional-external-tools-over-mcp)
- [documentation.md](../documentation.md) — full CLI and configuration reference

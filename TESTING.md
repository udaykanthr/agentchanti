# AgentChanti Testing Module

**Record a real user session in a real browser. Replay it as a self-healing E2E test that catches UI drift, locator drift, and API contract drift in one pass.**

The `agentchanti.testing` module is a second product line shipped alongside the main coding agent. It uses [Playwright MCP](https://github.com/microsoft/playwright-mcp) under the hood — same browser engine the rest of the industry runs E2E tests on, exposed as a Model Context Protocol server so an LLM (or this module) can drive it.

Pipeline: **record → normalize → replay → report.**

---

## Why use this instead of writing Playwright tests by hand?

| Problem with hand-written E2E tests | What this module does |
|-------------------------------------|------------------------|
| Brittle CSS selectors break on every UI tweak | Captures multi-tier locators (data-testid, id, role+name, text, screen coords) and falls through automatically |
| When a selector dies, the test just fails | Optional LLM "self-heal" step picks a working locator from a live snapshot and pins it for next run |
| API contracts drift silently | Records every network request during the session and validates them on replay against JSON Schema |
| Writing the spec is the slowest part | The spec is generated from a real recording — one click, type, select session in your browser is the entire authoring step |
| Coordinate-based replay breaks at different viewports | Captures viewport with the recording; replay enforces the same dimensions before navigation |

Everything that's hard about E2E test maintenance is the part this module automates.

---

## One-time setup

### 1. Install AgentChanti with the testing extras

The `testing` extra pulls in the optional dependencies (`mcp`, `jsonschema`) — without them, the rest of AgentChanti still works, but `agentchanti test` won't.

**From PyPI (most users):**

```bash
pip install "agentchanti[testing]"
```

The quotes matter — without them, your shell may interpret `[testing]` as a glob pattern and silently drop it.

If you already have AgentChanti installed and want to add the extras to an existing install:

```bash
pip install --upgrade "agentchanti[testing]"
```

**From source (contributors / pre-release builds):**

```bash
git clone https://github.com/udaykanthr/agentchanti.git
cd agentchanti
pip install -e ".[testing]"
```

### 2. Install + run the Playwright MCP server

Playwright MCP is a **Node** tool, not a Python package — it can't be bundled into the AgentChanti install. Run it in a separate terminal from any folder:

```bash
npx @playwright/mcp@latest --port 8931
```

The server listens at `http://localhost:8931/mcp`. A Chromium window opens when the first `agentchanti test` command connects.

> **Tip — pinning Node version.** If `npx` fails on Node ≥ 22 with native build errors, fall back to Node 20 LTS or pin a specific version: `npx @playwright/mcp@0.0.x --port 8931`.

---

## The three-stage workflow

### Stage 1 — Record

```bash
agentchanti test record \
  --url https://your-app.example.com/login \
  --out trace.jsonl \
  --viewport 1280x720
```

What happens:

1. The browser opens at `--url`. AgentChanti injects a small JS event hook on the page (capture-phase listeners on `document`) that streams clicks, typing, selections, and key presses into a buffer.
2. **You drive the session manually** — clicking around, filling forms, navigating, exactly as a real user. Every interaction streams into `trace.jsonl` with full element metadata (tag, id, classes, `data-testid`, aria-label, role, text, nearby label) plus the captured `clientX/clientY` coordinates.
3. URL changes are detected and recorded as `navigate` events; the JS hook auto-reinjects on every new page so you can navigate freely without losing capture.
4. Sensitive inputs are redacted **at the JS layer**, not after the fact: `<input type="password">` and any element with the `data-sensitive` attribute have their values replaced with `***REDACTED***` before they ever leave the browser. Plaintext never reaches disk.
5. **Press Ctrl+C** in the terminal to finalize the trace. A `session_end` event closes the file.

Result: `trace.jsonl` — one JSON object per line, append-only, JSONL-friendly.

```jsonl
{"seq":1,"ts":"...","type":"session_start","start_url":"...","viewport":{"width":1280,"height":720}}
{"seq":2,"ts":"...","type":"navigate","url":"...","status":null}
{"seq":3,"ts":"...","type":"interaction","action":"click","selector_used":"[data-testid=signin]","element":{...},"coord":{"x":482,"y":231}}
{"seq":4,"ts":"...","type":"interaction","action":"fill","selector_used":"#email","value":"user@example.com",...}
{"seq":5,"ts":"...","type":"session_end","reason":"user_stopped"}
```

### Stage 2 — Normalize

```bash
agentchanti test normalize \
  --trace trace.jsonl \
  --out login.spec.yaml \
  --name "user login flow"
```

This is **one LLM call** that turns the raw event stream into a stable, semantic spec. The LLM:

- Synthesizes `label` from text/aria-label/nearby-label/role: `"Place order button"`, not `[ref=e5]`
- Builds prioritized fallbacks: `data-testid > id > role+name > text > selector_used > coord=X,Y`
- Attaches network events to the step they fired during, with inferred JSON schemas
- Emits a final `url_equals` assertion matching the last observed URL
- Copies the recorded viewport into `metadata.viewport` so replay can enforce it

LLM provider is read from your `.agentchanti.yaml`; override with `--provider`/`--model`. This is the single LLM call in the pipeline — replay amortizes it across many runs via the locator cache.

Output is YAML so it's reviewable and editable by hand:

```yaml
version: "1"
name: user login flow
start_url: https://your-app.example.com/login
metadata:
  recorded_by: agentchanti
  viewport: {width: 1280, height: 720}
steps:
  - id: step-1
    action: fill
    target:
      label: Email input
      fallbacks: ["[data-testid=email]", "#email", "coord=482,231"]
    value: user@example.com
  - id: step-2
    action: click
    target:
      label: Sign in button
      fallbacks: ["[data-testid=signin]", "button[type=submit]", "coord=482,290"]
    expected_network:
      - method: POST
        path: /api/auth/login
        status: 200
        response_schema:
          type: object
          required: [token, user_id]
assertions:
  - id: assert-1
    kind: url_equals
    url: /dashboard
```

### Stage 3 — Replay

```bash
agentchanti test replay \
  --spec login.spec.yaml \
  --report report.json
```

What happens:

1. Connects to the same Playwright MCP server.
2. **Resizes the viewport** to `metadata.viewport` before the first navigate so coord fallbacks land on the same screen positions.
3. For each step, runs three-tier locator resolution:
   - **Cache hit** — pinned selector from a previous successful run (the fast path that makes CI cheap)
   - **Walk fallbacks** — try each in order; first match wins and is cached
   - **LLM self-heal** — when all fallbacks fail and an `llm_client` is configured, ask the LLM to pick a working selector from a live accessibility snapshot, then cache it
4. Diffs the browser's network log per step. `expected_network` entries are matched by method + path glob + status, with optional JSON Schema validation of response bodies.
5. Evaluates assertions: `url_equals` (deterministic), `dom_predicate` (snapshot walk), `natural_language` (LLM-evaluated, marked `skipped` without an LLM).
6. Writes a console summary + JSON report. Exit code is `1` if any assertion failed — drop-in for CI.

Add `--no-llm` to skip self-healing + NL assertions when you want a pure deterministic replay.

---

## Why replay is fast on CI

The locator cache (`.agentchanti/testing/locator-cache.json`) is the single fast path. After one successful run, every step has a known-good selector — no LLM calls, no fallback walks, no snapshots. A cache hit looks like:

```
selector resolution: cache hit → "[data-testid=signin]"
```

The cache invalidates an entry the moment a step's action fails or the cached selector stops matching, so you'll never have a stale entry silently masking a real regression.

---

## Spec format primer

Five action verbs. Narrow on purpose: every action the replayer needs to support is in `spec.py:ALLOWED_ACTIONS`. Adding to it is a deliberate choice.

| Action | Required fields | What it does |
|--------|-----------------|--------------|
| `navigate` | `url` | Browser navigation |
| `click` | `target` | Click the resolved element |
| `fill` | `target`, `value` | Focus + type into an input |
| `press` | `target`, `value` | Focus the element, then press a key (e.g. `Enter`) |
| `select` | `target`, `value` | Set a `<select>` value and dispatch `change` |
| `hover` | `target` | Dispatch hover events |
| `wait_for` | `target` | Probe until the locator matches or times out |

Three assertion kinds:

| Kind | Use for |
|------|---------|
| `url_equals` | Deterministic — final URL matches expected (path-only when expected is relative) |
| `dom_predicate` | Lightweight snapshot walk — check a selector is present/absent |
| `natural_language` | LLM-evaluated — business rules where "correctness" isn't a selector ("the order summary should mention free shipping") |

---

## Selector dialects

The replayer dispatches selectors by kind:

| Selector | Example | Dispatch path |
|----------|---------|---------------|
| Bare name | `Sign in` | Accessibility-tree match → ref |
| `text=` | `text=Sign in` | Same — accessible-name substring |
| `role=` | `role=button name="Sign in"` | Same — role-filtered name match |
| CSS | `#email`, `[data-testid=foo]`, `button[type=submit]` | `browser_evaluate` + `document.querySelector` |
| Coord | `coord=482,231` | `browser_evaluate` + `document.elementFromPoint` |

The accessibility tree only exposes role + accessible name + a few states. CSS selectors are the escape hatch for everything the tree elides (`data-testid`, raw id, classes). Coord is the last-ditch fallback when the DOM has drifted but the page layout hasn't.

---

## CI integration

```bash
# In your CI job — run the recorded spec, fail the build on assertion failure
agentchanti test replay \
  --spec specs/login.spec.yaml \
  --report artifacts/login-report.json \
  --no-llm
```

`--no-llm` keeps CI deterministic and fast: no API calls, no self-heal, no NL assertions. Network/URL/DOM assertions still run. The JSON report is structured per assertion (`{id, kind, passed, detail, skipped}`) — easy to parse for dashboards.

For richer regression detection in pre-merge runs, drop `--no-llm` and configure a provider in `.agentchanti.yaml`. Self-healed selectors get pinned in the cache so the *next* run is fast again.

---

## Programmatic API

Everything the CLI does is accessible as a library:

```python
from pathlib import Path
from agentchanti.testing import (
    Recorder, Normalizer, Replayer, Validator, Reporter,
    Spec, BrowserMCPClient, LocatorCache,
)

# Record
with Recorder.from_url("http://localhost:8931/mcp", "trace.jsonl") as rec:
    rec.start("https://app.example.com/login", viewport={"width": 1280, "height": 720})
    rec.subscribe_to_live_events()
    input("Press Enter when done...")
    rec.stop()

# Normalize (assumes you've already built `llm_client` from your config)
Normalizer(llm_client, spec_name="login flow").normalize("trace.jsonl", "login.spec.yaml")

# Replay
spec = Spec.load("login.spec.yaml")
with BrowserMCPClient("http://localhost:8931/mcp") as mcp:
    run = Replayer(mcp, LocatorCache(".agentchanti/testing/locator-cache.json"),
                   llm_client=llm_client).replay(spec)
    results = Validator(llm_client=llm_client).validate(spec, run)

print(Reporter().render_console(results))
```

Heavy deps (Playwright MCP transport, JSON Schema) are lazy-imported via PEP 562, so `import agentchanti.testing` is safe even without the `[testing]` extra installed.

---

## Troubleshooting

**`Playwright MCP not reachable at http://localhost:8931/mcp`**
The `npx @playwright/mcp@latest --port 8931` server isn't running, or it's listening on a different port. Start it in a separate terminal first. The path *must* include `/mcp` — the bare root returns 404.

**Replay clicks the wrong element on a different machine**
Almost always a viewport mismatch. Re-record with the `--viewport WxH` flag (e.g. `--viewport 1920x1080`) and confirm `metadata.viewport` is populated in the spec. The replayer will resize before navigation.

**Password value showed up in the trace anyway**
The redaction triggers on `type=password` or `[data-sensitive]`. Custom password components rendered as `<input type="text">` won't be detected — add `data-sensitive` to those elements in your app, or drive a `[data-sensitive]` wrapping div around the input.

**Network assertion always says "no request matching ..."**
Path patterns use `fnmatch` syntax (`*` matches anything in a path segment, `?` matches one character). `path: /api/orders/*` matches `/api/orders/42` but **not** `/api/orders/42/items` — use `/api/orders/**` style globs by adjusting your pattern.

**LLM self-heal picks a wrong selector**
The cache pins the last successful selector regardless of whether a human would call it "right." Open `.agentchanti/testing/locator-cache.json`, delete the offending entry, and re-run — or improve the spec's `fallbacks` list so the LLM doesn't get invoked.

---

## Architecture at a glance

```
recorder.py        ──→ trace.jsonl
   │                       │
   │ JS injection            │
   │ + polling drain         │ (one LLM call)
   ↓                         ↓
BrowserMCPClient    normalizer.py ──→ spec.yaml
   │                                       │
   │ Playwright MCP                        │
   ↓                                       ↓
Chromium                          replayer.py + validator.py
                                          │
                                          │ three-tier locator
                                          │ + network diff
                                          │ + viewport enforce
                                          ↓
                                       reporter.py ──→ report.json
```

Source: `agentchanti/testing/`. Tests under `tests/test_testing_*.py` — unit tests run in default `pytest`, integration tests are opt-in via `pytest -m integration` (require a running Playwright MCP server).

# Changelog

All notable user-facing changes to `agentchanti` land here. This
project follows [Semantic Versioning](https://semver.org): breaking
changes bump the minor (until 1.0), bugfixes bump the patch.

## Unreleased

## 0.7.0 — 2026-08-18

### Fixed

- **A dependency install now lands in the venv the run actually uses.**
  A planner step spelled `python -m venv venv && python3 -m pip install
  pygame`. A Windows venv ships `python.exe` and no `python3.exe`, so the
  name fell through to the ambient interpreter and pip reported "already
  satisfied" against *its* site-packages; the project venv ended the run
  holding nothing but pip, and every later command ran under it.

  `Executor._inject_venv_path` cannot cover this, twice over: it is
  computed before the command runs, so a venv the command itself creates
  is not on PATH for the install that follows it, and prepending
  `venv\Scripts` does nothing for a spelling that directory has no entry
  for. Installs are now redirected to the interpreter by absolute path,
  which settles both without depending on name resolution, and `--user`
  is dropped, since pip refuses it inside a venv. Silent when no project
  venv is in play — a project on the ambient interpreter is never
  redirected into one — and silent after a `cd`.

- **A missing package is repaired in the environment, not by editing the
  app.** Everything downstream of that empty venv ran under it, so the
  smoke test's `python main.py` crashed on `import pygame` every time.
  The smoke test's only repair is an edit, so the only repair it made was
  an edit: measured on both benchmark paths, it came back with a
  graphical entry point that silently falls back to headless mode. The
  app then "launched successfully" and every gate stayed green over a
  game that never opens a window.

  A launch crash naming a package the environment lacks is now installed
  into the project venv before the crash is ever shown to a model, and
  that repair does not consume a code-fix attempt, because nothing about
  the code was wrong. Both crash shapes are read: the
  `ModuleNotFoundError` traceback, and the advice an app prints when it
  caught its own `ImportError`. Names the project defines and stdlib
  names are refused — installing those fetches whatever squats on PyPI
  under the spelling — and with no venv of its own the pipeline declines
  rather than write to the ambient interpreter.

- **`tests-never-collected` now asks where a test file sits, not only
  what it contains.** unittest discovery recurses only into importable
  packages, so a suite written to `tests/` with no `__init__.py`
  contributes nothing to `python -m unittest`. Measured: a step spent
  eight of ten turns watching its gate pass on the seeded acceptance
  contract at the root while the six tests it had just written were
  collected by nothing.

  The check fires only when the plan's own command is bare root
  discovery, since that is the only spelling under which importability
  decides whether tests exist — `discover -s tests` makes that directory
  its own top level, and a named target is not discovery at all. Ghost
  heal closes the gap by creating the marker, which is the `EXISTS`
  healer's rule applied to a file the plan did not name: empty *is* the
  correct content of an `__init__.py`, so it invents nothing and instead
  connects a declared target to a declared gate. Replayed against the
  incident's tree, `python -m unittest` goes from 8 collected to 14.

- **A gate that cannot pass on this platform no longer fails the step.**
  `verify_passed` already encoded "exit 0 is not proof"; this adds the
  mirror — exit 1 is not proof either, because a gate can be an invalid
  instrument rather than a failing test.

  Observed on a React/Vite run. The planner declared a `node -e` gate
  whose regex was double-escaped (`[\\s\\S]` rather than `[\s\S]`). Under
  a POSIX shell the quoting collapses one level of backslashes before
  node sees them and the regex means "any character"; under `cmd.exe`
  there is no such collapsing, so node compiled a character class
  matching a literal backslash, `s` or `S`. Identical plan text was
  therefore satisfiable on Linux and unsatisfiable on Windows. The CSS
  edit was correct on the first turn, but the primary loop, the
  escalation to a stronger model and the recovery loop all grade against
  that one command — so a single broken gate defeated all three at once:
  24 turns, ~182k tokens, and a failed run on working code. The escalated
  model even proved the gate wrong by printing each sub-condition, and
  had nowhere to put the finding.

  On a still-red verdict the loop now re-runs the gate once under the
  other shell dialect's reading of the *identical text*. Only a variant
  that PASSES is believed — that proves the original was unsatisfiable,
  since the two forms differ solely by an escaping step one shell
  performs and the other does not. A variant that also fails proves
  nothing and leaves the original verdict untouched, so genuinely broken
  code still fails. The repair is recorded and reused when the monotonic
  ledger re-checks the gate, which would otherwise re-run the form
  already shown incapable of passing and report a regression on unchanged
  code.

  Deliberately narrow: variants come from one whitelisted pure transform,
  are never authored by a model, and are never semantically different
  commands. A looser rule would be a machine for manufacturing false
  greens — mutate the gate until something passes — which is precisely
  what this project's verification layers exist to prevent. The check is
  language-agnostic by construction: it tests behaviour rather than
  parsing a payload, so it covers `python -c`, `node -e`, `ruby -e` and
  anything else equally. (A syntax check could not have caught this one:
  the broken payload is valid JavaScript.)

- **A log file is created only when there is something to log.**
  `setup_logger()` runs at module scope, so *any* import of the package
  opened a timestamped log — including short-lived subprocess utilities
  that never write a line. The style-coupling gate made this visible by
  running `python -m agentchanti...` once per check: a single run left
  ten zero-byte files beside its real 47 KB log, cluttering exactly the
  directory someone opens to find out what happened. The handler now
  defers opening its file until the first record.

- **Markup and stylesheet are checked for agreement on class names.**
  Two files can each be individually correct and jointly wrong. A
  component step writes `site-footer__content`; a stylesheet step in the
  same wave, unable to see it, writes `.site-footer__inner`. Both gates
  pass, the suite passes, the production build passes — an unmatched CSS
  class is still valid CSS — and the page renders unstyled.

  Four of six consecutive runs on one project drifted this way, once
  completely (7 classes rendered, 0 styled). The decisive case had a gate
  asserting eight structural properties of the stylesheet — background,
  colour, max-width container, grid, hover, divider, flex utility row,
  responsive stacking. All eight were true; all eight described selectors
  the markup never rendered. No single-file assertion can catch this,
  because neither file is wrong on its own — only the join is.

  Runs after the build is green, since it answers what the build cannot,
  and is held to the same check re-run as a command so a repair is
  verified rather than believed. Only "rendered but never defined" counts
  as a defect; orphaned rules are reported as the explanation, never as
  the failure. Refuses rather than guesses whenever the answer is
  unclear — a utility or component framework in the dependencies, Sass
  nesting, CSS Modules, or a dynamic `className` — because a false
  accusation sends a correct run into a fix loop.

- **A self-locating suite gate is no longer given the sub-project cwd too.**
  BulkTest's preflight runs the plan-declared gate before substituting the
  framework default. It passed `cwd=<subproject>` unconditionally — so a
  gate that already opens with its own `cd` had the prefix applied twice:
  `cd react-home && npm test -- --run` launched from `react-home/` looked
  for `react-home/react-home`, and cmd.exe answered "The system cannot
  find the path specified" with exit 1.

  The result was a spurious `Plan-declared gate did not pass`, demoting a
  perfectly good gate to the framework runner — precisely the substitution
  this preflight exists to prevent. Every other caller ran the identical
  command from the repo root and it passed. The test mirrors the one
  `_gate_on_declared_verify` already uses before ADDING such a prefix, so
  the two cannot disagree about what "self-locating" means.

- **A step targeting a file the application never loads is reported.**
  A gate can assert seven true things about a file that is not in the
  build, and every check downstream will agree the step went fine: the
  tests render components rather than styles, the build does not error on
  a file it never bundles, and the smoke test only proves the build
  succeeded.

  Observed on a Vite/React project. `src/main.jsx` carried the app's only
  stylesheet import (`./index.css`) while `src/App.jsx` imported no CSS
  at all, so the scaffold's leftover `src/App.css` was never bundled.
  Successive "restyle the header" runs targeted it and wrote twelve
  `.site-header` rules including a full dark palette; the built bundle
  contained one. Nothing in the browser changed across many runs.

  Stylesheets only, and only where reachability can be established: a CSS
  file no entry point can reach is inert by construction, which is a fact
  about the module graph rather than a heuristic. The same claim about a
  JS module is far weaker — dynamic import, lazy routes,
  `import.meta.glob` — so it is deliberately not made. A file the plan
  itself intends to wire up is not an orphan, and anything ambiguous is
  not judged.

  It replans rather than merely warning. It shipped advisory for exactly
  one run, on the reasoning that the right repair varies between
  retargeting and adding the import — and that run settled the question
  the other way: the warning fired correctly, nothing consumed it, the
  step edited the dead file anyway, every gate went green, and the
  interface was unchanged for the fourth time. An advisory nobody acts on
  is indistinguishable from silence. The correction tells the planner to
  fix the TARGET rather than the verify command, since the gate is not
  what is wrong.

- **A gate that cannot observe its own step's file is rejected.**
  `shallow_gate_reason` clears any command matching a test runner, on the
  reasonable ground that a suite asserts real behaviour — but a suite only
  asserts what it can reach, and a stylesheet is not reachable. No CSS
  edit can turn `npm test` red.

  Observed: a step briefed to add a whole footer layout — brand area,
  navigation grid, legal row, responsive breakpoints — was gated on
  `cd react-home && npm test -- --run`. It deleted two words from a
  selector, wrote no footer styling at all, and passed on turn 2. The
  markup shipped with eight classes and not one of them styled, while the
  suite, the build and the smoke test were all green, because none of
  them could see the difference.

  Deliberately narrow: flagged only when EVERY target is a stylesheet AND
  the gate is nothing but a runner invocation. A gate that also asserts
  something about the file is fine, and any step touching executable code
  is left alone.

- **The export-promise warning stopped crying wolf.** Planners spell a
  default export `default Footer`; the JavaScript extractor reports
  `['Footer', 'default']`. Comparing the two literally warned that
  `default Footer` was missing from a file exporting precisely that — on
  every run for a week, never once correctly. The bare-name form is also
  reconciled against a file whose only export is the default, which is
  the same symbol with the name flattened away. A genuinely absent export
  is still reported.

- **A malformed `node -e` gate is rejected at plan time, not paid for.**
  `unrunnable_gate_reason` already refused a `python -c` gate whose
  payload could not be parsed; JavaScript payloads went unchecked. The
  same defect in the other language therefore cost a whole run: a plan
  put `&& npm --prefix react-home run build` INSIDE the quoted script,
  which is a syntax error, and the failure only surfaced after the loop,
  an escalation and a recovery had each spent their turn budget against
  it. Rejected at plan time it costs one replan instead.

  Dispatch is a table of payload-kind → checker, so a further language is
  a row rather than a branch. Silence is the safe answer throughout: no
  node on PATH, a timeout, or any other surprise means "not judged",
  never "rejected", because a false rejection replans a plan that was
  fine.

- **A valid gate using escaped quotes is no longer called unrunnable.**
  The payload is captured from between the command's quotes, so a `\"`
  written for the shell is still backslash-quote in the capture while the
  interpreter receives a plain `"`. Checking the raw text reported good
  gates as broken — verified for both
  `python -c "... open(\"p.json\") ..."` and its JS equivalent — sending
  the planner off to rewrite a command that was never at fault. A
  latent false positive in the existing Python check, found while adding
  the JavaScript one.

- **A gate accepted via an equivalent form is now recorded that way.**
  The loop had two ways to satisfy a gate that no correct work could
  pass — the platform re-reading above, and the older flag-variant escape
  hatch — but only one of them told the monotonic ledger. The other
  passed the step and then let the ledger re-run the *original*, which
  failed exactly as it always had, which read as a regression.

  Observed: a plan wrote `&& npm --prefix react-home run build` INSIDE
  the `node -e "..."` string, making the payload a JavaScript syntax
  error that no code could satisfy. The loop diagnosed it, ran the
  correct form, and recovered the step — then the ledger rechecked the
  malformed original, reported `REGRESSION`, rolled the wave back and
  failed the run. The work had been correct for two minutes.

  The resolution now happens in `_record_passed_gate`, the single
  chokepoint every path funnels through (loop, recovery, classic),
  rather than at individual call sites where it could drift.

- **A step's real imports now outrank the planner's `imports:` line.**
  `imports:` is the planner's opinion and it is optional, yet two
  mechanisms read only that declaration. When a step editing an existing
  file declared `imports: none`, both failed at once:
  `fix_import_dependencies` added no edge, so producer and consumer
  landed in the same wave and ran *concurrently*; and
  `build_step_context` injected no sibling, so neither could see the
  other even if it had wanted to.

  Observed: `src/App.jsx` — whose first line is `import './App.css'` —
  and `src/App.css` were both declared `imports: none`, scheduled as
  `[[0, 1]]`, and written in parallel. The markup used
  `site-footer__nav-title` while the stylesheet defined
  `site-footer__heading`: 3 of 8 classes unstyled and 6 CSS rules
  matching nothing. Tests and the build both passed, because unmatched
  CSS classes are still valid CSS, and the string-presence acceptance
  gates checked each file in isolation.

  Two fixes, both deterministic. `_resolve_import_to_file` now takes the
  importing file, so a relative specifier resolves at all — `./App.css`
  inside `src/App.jsx` means `src/App.css`, which nothing could know
  without the importer's directory (and which the existing
  `.replace(".", "/")` branch mangled into `//App/css`). And plan fixing
  now derives edges from the files themselves, consulting only files that
  already exist — for a file the run is about to create there is nothing
  to read, and the declared imports remain the only signal.

- **Hoisting a scaffold no longer leaves a second copy of the project.**
  `move dir\*` on Windows moves files but *not* subdirectories, so the
  standard hoist an agent writes after scaffolding —
  `npm create vite@latest scaffold -- --template react` followed by
  `move scaffold\* . && ... && rmdir scaffold` — silently left `src\` and
  `public\` behind. The `rmdir` then failed with "The directory is not
  empty" (exit 1) and the run carried on with two copies of every
  component. Seen twice in one afternoon: a leftover
  `vite-react-scaffold\` and a nested `home_page\home_page\`.

  The cost is not just clutter. Both orphan trees were indexed, so
  semantic search served later steps a stale duplicate of the very file
  they were editing, and the project scan reported 17 files where 11
  existed. The rewrite now adds a subdirectory pass, and leaves the
  (emptied) source directory in place because a later segment of the same
  chained command routinely still refers to it.

  Grouped in parentheses, which is load-bearing: ungrouped, a caller's
  trailing `&& rmdir scaffold` binds to the `for` body rather than to the
  rewrite as a whole, so it runs only when a subdirectory happens to
  exist and silently does nothing — with exit 0 — when the directory is
  flat.

- **A command that fails silently is no longer met with "fix the code".**
  The repeated-failing-command nudge told the model *"the failure is in
  the code, not in how the command is invoked. Read the error above."*
  When the command produced no output there is no error to read, so the
  advice was unactionable — and its certainty was sometimes simply wrong,
  as in the mis-escaped gate above, where the source had been correct
  since the first turn. In that run the model eventually printed the
  check's conditions one at a time and found every one of them true; that
  was the right move, reached far too late.

  A silent repeated failure now asks for that evidence directly: make the
  failure observable, run a form that prints each condition separately,
  and if all of them hold say so and quote the output. The original
  wording is unchanged whenever there *is* an error to read. This cannot
  be used to dodge a step — the acceptance gate is run by the harness,
  not by the model, so talking itself out of the work still does not
  finish it.

Two further defects found by a fourth A/B iteration over the Pac-Man task,
where both modes passed and both artifacts held every wall invariant under
an independent drive (fixed 1/60, jittery, hostile, and dt=1.0). Neither
defect broke the run — both quietly made it more expensive.

### Fixed

- **Inline scripts no longer have their output swallowed by `cmd.exe`.**
  `Executor.run_command` runs everything through `shell=True`, so a
  perfectly ordinary `python -c "...assert n > 0..."` had its `>` read as
  redirection: the command's real stdout was written to a file literally
  named `0` in the project root, and the caller received an empty string.
  Escaped quotes make it worse — they break cmd.exe's quote tracking, so
  even a `>` written inside quotes is treated as an operator. Observed
  live: two agent-loop verification commands returned nothing, the model
  could not see why its own code "failed", and the step burned all eight
  turns and escalated to the stronger model.

  A single inline-script invocation now bypasses the shell entirely. The
  gate is deliberately narrow: the command is parsed with the real Win32
  parser (`CommandLineToArgvW`, not an approximation that would silently
  produce a different argv), and is diverted only when it splits into
  exactly three arguments — interpreter, inline-script flag, script.
  Genuine shell syntax always survives parsing as extra arguments
  (`python -m pytest > out.txt` splits into five), so a three-element argv
  proves there is no shell work to do. Scripts containing no shell
  metacharacter keep the existing path. The interpreter is resolved
  explicitly against the run's PATH, because `subprocess` does *not*
  honour env's PATH when locating an executable on Windows — without that,
  the fix would have quietly moved every inline script off the project
  venv and onto the system interpreter.

- **A crashed test runner is retried instead of believed.** The
  plan-declared suite gate already treated an abnormal exit as "no verdict"
  and retried it; the framework runner it falls back to did not. A green
  10-test pygame suite access-violated (`0xC0000005`) inside an iterative
  BFS — no SDL, no recursion — and the crash was read as a real failure,
  starting a fix cascade against code that was never broken. It now gets
  the same single retry, which passed with zero code changes.

## 0.6.5 — 2026-08-07

Closes the gap left open by 0.6.4's FileMemory work: memory was made to
agree with the filesystem, but nothing stopped the write reaching the
filesystem in the first place.

### Fixed

- **The agent loop cannot replace a manifest it did not write.** The
  classic writer has always refused to overwrite a dependency manifest it
  did not create (`Executor.write_files`); the loop's `write_file` went
  straight to disk with no such check. A model regenerating
  `requirements.txt` or `package.json` from memory could therefore replace
  the project's real one with a shorter version, dropping dependencies —
  and every later step would build and test against a different dependency
  set than the project actually has, which is the kind of failure that
  looks like a code bug for a long time.

  The test is create-versus-overwrite, not existence: creating a manifest
  is legitimate and common (5 of 8 benchmark runs did it), and a run that
  wrote one must be able to update it. Within a step the instance tracks
  what it created; across steps the answer comes from FileMemory, because
  `build_step_tools()` builds a fresh `AgentTools` per step. Only a
  manifest that existed before the run began is refused, and the refusal
  names `edit_file` — exact-match and single-occurrence, so it cannot
  silently discard the rest of the file — as the way to add a dependency.
  Ordinary source files are unaffected.

## 0.6.4 — 2026-08-07

The two defects observed alongside the 0.6.3 fence bug, neither of which
caused that failure but both of which would hide or repeat one — plus four
more found by a third A/B iteration over the Pac-Man task, where both modes
passed (loop 4:45 / $0.43, classic 16:25 / $1.34) and every artifact was
checked against an independent wall-invariance drive.

### Fixed

- **A halted pipeline exits non-zero.** The failure branch logged "Pipeline
  failed", wrote the HTML report, and fell through returning `None`, so the
  process exited 0. A benchmark run that stopped at step 11 of 12 after three
  failed diagnosis attempts, having never written its tests, still returned
  `EXIT=0` — indistinguishable from success to CI, a `&&` chain, or any
  harness reading `$?`. `_main_impl` now returns 0/1 from the
  `pipeline_success` flag it already had and `main` exits with it. Returned
  rather than raised, so watcher and executor cleanup still run first.
  `--version`, the `kb` subcommand and an aborted prompt return `None` and
  stay 0; SIGINT stays 130; an unhandled exception is still re-raised.
- **The fuzzy parser no longer truncates at, or invents files from, inner
  fences.** `parse_code_blocks_fuzzy` searched every pattern's body with an
  unanchored non-greedy `` (.*?)``` ``, so a block ended at the first ```
  anywhere. Two consequences. It returned the same truncated document the
  strict parser did on 0.6.3's README — and then, because Pattern 3 takes its
  filename from the line above a fence, the document's own usage examples
  kept matching: a README's install and run instructions were emitted as
  phantom `requirements.txt` and `main.py` files to write. Separately, an
  unanchored search ends a diff block on its own `+``` ` line, so a diff
  touching any Markdown file was cut at the first fence it added. Fence
  detection is now anchored to line starts and shares one span helper with
  the strict parser; the patterns walk top-level blocks only, and Pattern 3
  resumes past each block it takes so a document's interior can never be
  re-read as further filenames.
- **Sanitising a verify gate no longer corrupts `set VAR=value` on
  Windows.** `_declared_verify_cmd` split a gate on `&&`, stripped every
  segment and rejoined with `" && "` unconditionally — rewriting whitespace
  even when it dropped nothing. On `cmd.exe` that is not cosmetic:
  `set VAR=dummy && next` assigns `"dummy "`, trailing space included. A
  planner wrote `set SDL_VIDEODRIVER=dummy&& ...` precisely to avoid that;
  the space put back made SDL look for a display driver named `"dummy "`,
  failed the gate, and cost a diagnosis round that "fixed" it by adding an
  environment-scrubbing function to the *generated* project's `main.py` —
  harness damage shipped in the delivered artifact. The repair now runs
  unconditionally on the resolved gate — the planner writes the spaced form
  about as often as this module reintroduced one, and a first, narrower fix
  that only repaired chains the sanitiser had reassembled left the
  planner-authored case broken (caught by the next benchmark run: the same
  gate failed three times in a row while the generated `main.py` was
  correct all along). Non-assignment segments keep their spacing, so
  `cd app && npm test` is unaffected.
- **A manifest a step just created is tracked in FileMemory.** The
  protected-basename guard exists to stop a hallucinated replacement
  clobbering a real manifest, but it tests `os.path.isfile()` — true the
  moment the file is written — so a manifest the run had just created
  looked pre-existing and was dropped. The content then stayed invisible to
  dependency checks, context injection and the checkpoint for the rest of
  the run, while the log claimed a skip that protected nothing. Fixed on
  both paths: the agent loop's `AgentTools._record` runs *after* its write
  has landed, so it now records unconditionally; the classic path
  distinguishes create-from-overwrite using the pre-write existence set it
  already computed. A genuinely pre-existing manifest is still protected on
  disk and in memory. Seen in 5 of 8 benchmark runs.
- **pytest is not installed for a runner that never runs.**
  `_ensure_pytest_available` ran before the plan-declared suite gate, and
  that gate passed on every run of a unittest-based project, so each run
  paid a pip install and a network round-trip for a runner never invoked.
  Moved to the fallback path that actually needs it.

### Benchmarks

- **`verify_dt_invariance.py` no longer reports a working game as broken.**
  Two of six artifacts in one session got `VERDICT: FAIL - game raised ...`
  from exceptions raised by the harness's *own* probes rather than by the
  game: `set_direction(1, 0)` against a name-based API raised `ValueError`,
  and `pixel_to_tile(px, py)` against a signature taking one sequence raised
  `TypeError`. Derivation is now separated from the drive loop — anything
  raised while deriving an artifact's vocabulary is a refusal (exit 2), and
  only the drive loop can produce a FAIL; the probes try each known
  convention instead of assuming one. Two selection bugs surfaced while
  fixing it: a two-argument `can_move`/`walkable_neighbor` takes
  `(tile, direction)`, not `(col, row)`, and `is_walkable` tied with
  `is_position_walkable` on every ranking term so `dir()` ordering decided
  the winner — "position" usually means pixels, so it answered in the wrong
  coordinate space. Ranking now prefers the plainest canonical name and is
  deterministic. All six artifacts verify: 6 PASS, matching six independent
  hand-written drives (before: 3 PASS, 2 false FAIL, 1 refusal). One of
  those original passes was luck — it had been using
  `get_walkable_neighbors`, which returns a *list*, as a boolean wall test.

## 0.6.3 — 2026-08-07

One fix, found by A/B benchmarking `agent_loop` on/off over the Pac-Man task.
The classic path halted at step 11 of 12 on a bug in the extractor, not in
the model's output — and because the bug is deterministic, no retry could
ever escape it.

Two full iterations were run per mode, folder cleared between every run.
Before the fix: classic halted (8:01 / $0.800), loop passed (5:29 / $0.524).
After: classic passed 16/16 (9:05 / $0.922), loop passed 8/8 (5:23 /
$0.484). All four artifacts were checked against an independent
wall-invariance drive as well as their own suites. Loop read 35-42% of its
input from the prompt cache in both runs; classic read 0% in both.

### Fixed

- **Generated Markdown is no longer truncated at its first inner code
  fence.** `Executor.parse_code_blocks` matched a file's body with a
  non-greedy group (`` (.*?)\n``` ``), which ends at the FIRST fence line
  inside the block. Correct for source files, wrong for any file whose own
  content contains fences: an 808-token README was written as 15 lines /
  417 bytes, cut off mid-sentence at "install the required dependency:",
  because every command the step's verify gate looked for lived inside a
  fence. The gate failed correctly, but all three diagnosis attempts
  regenerated the same document, hit the same truncation, logged "previous
  fix changed nothing", and the pipeline halted having never written the
  tests. Extraction now splits the response by `#### [FILE]:` marker and
  requires a closing fence at least as long as its opener, so a model that
  wraps ``` content in a ```` fence parses exactly. Where both fences are
  three characters the nesting is genuinely ambiguous and the file's own
  format breaks the tie — outermost fence for `.md`/`.rst`/README-likes,
  first closer for source files, so a follow-up example block is still not
  swallowed into the file above it. The same rerun that halted before now
  produces a 158-line README with its 12 inner fences intact and passes the
  gate on the first attempt. The `agent_loop` path was never affected:
  `write_file` receives content as a structured tool argument and no fence
  parsing is involved.

## 0.6.2 — 2026-08-07

Five fixes from continued A/B benchmarking of `agent_loop` on/off over the
Pac-Man task with adversarial delta-time invariants. The theme is the
harness: in every case the model did work the pipeline then discarded,
hid, or graded against the wrong thing. Each fix carries a regression test.

Benchmarked after these changes, one run per mode, both passing their own
suite and an independent wall-invariance check: loop 5:01 / $0.4778 / 49%
prompt-cache hit; classic 7:55 / $0.5872 / 3%. This is the first classic
run on this task to pass ground truth.

### Fixed

- **A failing command is routed by its error, not its shape.** A step whose
  gate was a bare console script (`ruff check messy.py`, not on the child
  process's PATH) was told "the failure is in the code — edit the source",
  though the source was already correct. Routing now matches the failure
  signature itself (not recognized / command not found / No module named /
  executable file not found / cannot find the path), so any command failing
  that way gets environment advice, including invoking an installed-but-
  unPATHed tool as `python -m <tool>`. Assertions and tracebacks keep the
  original wording verbatim. The run this came from reported failure on a
  task ground truth showed had PASSED, after 22 turns and 61.9k tokens.
- **The repeat-command nudge survives across attempts.** The streak reset
  every attempt, so a command re-run once per attempt across loop →
  escalation → recovery never tripped it. It is now seeded from the attempt
  journal; an edit within the attempt still clears it.
- **A step cannot shadow a dependency whose install just failed.** After a
  `pip install pygame` failure, one run wrote a local `pygame/` package
  whose own docstring said it performs no real rendering — shadowing the
  real library, so every later step and test would have passed against a
  no-op renderer. Blocked only when an install of that exact distribution
  failed in the same step, and only for a new top-level module or package.
- **A rejected chunk edit is no longer re-parsed as whole files.** Pattern 5
  attributed indented method bodies to modules on a single symbol match,
  writing three method bodies over three real files. A complete file passes
  the new check trivially.
- **A test suite the project's own runner cannot find is not done.** A run
  shipped a false green: six steps verified early, the pipeline reported
  success, and the delivered project answered `python -m unittest -v` with
  "Ran 0 tests" — nothing had made `tests/` a package. A path-scoped gate
  proves the file runs and says nothing about discovery, so the TEST step
  now re-runs its own gate with the file scoping stripped and fails when
  that collects nothing.
- **A CONFIG_BUG fix may only touch config files.** Observed rewriting five
  source modules under a triage meaning "the test environment is
  misconfigured", taking a suite from 1 failure in 61 tests to 4 failures
  and 3 errors in 64. A repeat CONFIG_BUG verdict that changed nothing now
  routes to the source path.
- **Compound commands are never token-rewritten.** `pip install -U pygame
  && pip freeze > requirements.txt` was reassembled into `pip install -U
  requirements.txt && freeze`, became a step's gate, and killed a run at
  step 1 of 9 with an otherwise correct plan.
- **The plan's own test runner is honoured.** A task whose acceptance was
  `python -m unittest -v` had pytest installed and used anyway. Relatedly,
  `unittest discover -v tests/test_x.py` read the path as a start directory
  and ran zero tests, recording a game with 61 of 63 tests passing as 0/2.
- **Windows: `| head -N` is dropped rather than failing the command.** head
  does not exist there, so the pipeline died before the command ran. pip
  self-upgrades now route through `python -m pip`.

### Performance

- **A loop step ends as soon as its gate goes green.** Steps averaged 7.6
  turns against a max of 8, so they essentially never finished early, and
  the whole conversation is resent every turn (~27k prompt tokens on late
  turns against ~2k at the start). The gate is now checked as soon as an
  edit lands — one subprocess, no tokens. On the same 7-step plan: 305,306
  tokens rather than 821,235 (-63%), 282s rather than 517s, avg 4.4 turns
  rather than 7.6, with the artifact still passing its suite and the
  independent dt-invariance check.
- **The Anthropic chat path asks for a prompt cache.** `agent_loop` keeps
  its system prompt byte-identical *for* prompt caches, but the request
  never set `cache_control`, so a run billed 1,224,846 tokens sent and 0
  cached. Three breakpoints take the hit rate to 65% and cut full-price
  input per loop turn from 18,688 to 7,492. The billed prompt is now
  reported as input + cache_read + cache_write, since `input_tokens` alone
  makes a working cache look like a shrinking prompt.
- **A weak `verify:` is repaired in place instead of regenerating the plan.**
  A re-plan cost 8,214 sent / 3,219 received to fix one line and churned the
  step decomposition; the repair costs ~500 tokens. Shell-level assertions
  count as teeth; `assert True` does not.
- **An empty response that billed output tokens is treated as a reasoning
  burn even when `finish_reason` is "stop".** Only cap-hits were detected,
  so a burn wearing a clean stop fell through to the anti-`<think>`
  preamble, which cannot help server-side reasoning and mutates the prompt,
  losing prompt-cache reuse. Gemini's thinking cap is now persisted in
  `~/.agentchanti/effort_floors.json` alongside the OpenAI floor,
  namespaced so a numeric budget and an effort string cannot be confused.
- **A failing command re-run unchanged is nudged, then `run_command` is
  withheld.** Commands compare with `cd` prefixes and output pipes
  normalised away, since the retry usually arrives redressed rather than
  repeated. One step spent turns 4–7 re-running one gate from four
  directories (~38k tokens) while the defect went untouched.
- **A chunk edit whose line range no longer matches splices the named
  symbol** instead of discarding the fix and spending an attempt on nothing.

### Benchmarks

- `verify_dt_invariance` no longer refuses a game it can drive: a 3-arg
  `is_rect_wall_free` outranked `is_walkable` purely by spelling "wall" and
  carried the opposite polarity. The probe now reads arity and polarity
  before it trusts a name.

## 0.6.1 — 2026-08-05

Correctness and context-hygiene fixes found by an A/B benchmark of
`agent_loop` on/off over eight runs of a Pac-Man task with adversarial
delta-time invariants. Every fix carries a regression test that fails on
the previous code.

### Fixed

- **A package initializer no longer runs before the module it re-exports.**
  A plan gave `src/__init__.py` and `src/pacman.py` the same `exports:` but
  only `depends:1.1`, so they shared a wave. The initializer ran with
  nothing to import and satisfied its gate — `assert all(x is not None
  ...)`, which is true of `class Game: pass` — by writing four stub
  modules. Every gate, the smoke test and 8/8 unit tests passed while
  `from src import Game` returned an empty shell. The missing edge is now
  inferred from `exports:` overlap; `fix_import_dependencies` could not
  help because it derives edges from `imports:`, and that plan declared
  `imports: none`.
- **A gate that contradicts the task no longer costs the work that
  satisfies it.** A plan gated a step on `assert p.can_move()` against a
  `can_move()` meaning "is currently moving", so it demanded a
  freshly-built player already be in motion — while the task demanded
  "2000+ frames without the player moving". Diagnosis made the player
  auto-start, the suite's idle test then failed, its fix regressed the
  gate, and rollback discarded the correct fix. When a test suite is green
  and only inline gates are red, the conflict is now reported and the
  working tree kept. The run still fails: an unresolved red gate is never
  reported as success.
- **KB docs are scoped to the language they are for.** `seeder.py` passed a
  hardcoded `"all"` as the language of every doc it emitted, so the filter
  in `GlobalKBStore` could never exclude anything — "React Component Export
  Instructions" was injected into every step of a Python/Pygame run,
  alongside Django and Vitest material, at 2.8k–3.9k tokens per step. Docs
  now declare their real language, `language:` accepts a comma-separated
  list (`"javascript, typescript"`), and a doc naming a framework the task
  never mentions is dropped — which the language filter alone cannot do,
  since the Django docs are correctly tagged `python`. Measured on the same
  task: foreign-stack docs injected per run went from 5 to 0.
- **Streaming calls survive a reasoning model's silence.** The OpenAI
  streaming POST used a 120s read timeout, which `requests` applies to the
  gap *between* bytes; a reasoning model emits nothing while it thinks. A
  planner call died at exactly 120s, was retried (re-billing the prompt)
  and then downgraded to the non-streaming path — up to 3× the tokens of
  one call. `ollama.py` already carried this lesson; this client did not.
- **No setup-guide web search for a bare language.** The blank-project
  pre-seed asks for frameworks and routinely gets the language back too
  ("Python, Pygame"), spending a web fetch plus a summarisation call
  (~2.7k tokens) to learn `pip install`. Languages are skipped; frameworks
  are still searched.

### Compatibility

`language:` in KB frontmatter now accepts a comma-separated list. Registry
**v1.5.0** uses it to scope React/Vite docs to `"javascript, typescript"`.
Older agentchanti releases compare the field by exact string and will not
match a list, so **0.6.1 or newer is required to see those docs** — pin the
registry version on older clients.

## 0.5.1 — 2026-07-21

Fixes for a shared failure theme found by running 0.5.0 on a fresh
Python/pygame task: acceptance gates that no amount of correct work
could turn green.

### Fixed

- **Subproject prefix no longer breaks root-relative verify gates.**
  A gate like `python -m unittest discover -s game` was prefixed with
  `cd game && `, making its path argument resolve to `game/game/` —
  unpassable by correct code (the escalation model "satisfied" it by
  creating a duplicate nested test package). Commands that reference
  the subproject itself by path or module (`-s {sub}`, `{sub}/tests`,
  `python -m {sub}.main`) now run from the repo root as written.
- **`pip install --yes` is sanitized.** Planners hallucinate the
  apt/conda flag onto pip, which rejects it (exit 2). The flag is
  stripped from pip-install segments before the first run; apt/conda
  segments keep their legitimate `-y`.
- **Recovery gates accept a flag-variant success.** A failed CMD step's
  recovery gate re-runs the original command verbatim; when that
  command is malformed, the gate stayed red even after the loop ran the
  corrected command successfully (observed: pygame installed twice,
  run still failed). Recovery loops now accept a success of the same
  command differing only in flag tokens. CODE/TEST verify semantics
  are unchanged.

## 0.5.0 — 2026-07-20

A token-efficiency and reliability release, hardened through ten
consecutive benchmark runs on a real scaffold-and-build task. Best-case
cost dropped ~5× (13.9k tokens sent, zero recovery loops, tests passing
first try — parity with single-shot editors while keeping the full
test + build verification pipeline). Every failure class encountered is
now covered by a deterministic guard plus a regression test.

### Added

- **Prompt-cache accounting.** OpenAI cache hits are read from
  `usage.prompt_tokens_details.cached_tokens` and reported everywhere:
  the final log line shows gross vs cached vs full-price input
  (`sent=X [cached=Y (Z%), full-price=W]`), the live panel shows the
  cached share, cost estimates bill cached input at the discounted rate
  (override per model via `pricing.cached_input`), and the library's
  `TaskResult.token_usage` gains a `"cached"` field.
- **`agentchanti --version`** — reports the actually-installed
  distribution version via `importlib.metadata`.
- **Agent-loop preload + earlier nudge.** The loop's opening message
  pre-loads the step's existing target files (capped) so models stop
  burning turns on `read_file` round-trips; the read-only "act now"
  nudge fires after 2 idle turns instead of 3.
- **Adaptive inline-code budget.** Content-mode plans cap total inline
  code (~150 lines); beyond it, steps carry descriptions + `verify:`
  and are implemented against the real project state. Truncated plans
  whose parsed steps are structurally complete are salvaged instead of
  triggering a full re-plan.
- **Deterministic vitest bootstrap.** DOM test suites about to run
  without a vitest config get a jsdom-enabled `vitest.config.js` +
  setup file written from a fixed template (no LLM), with missing
  testing deps installed. Test-infra files (vitest/jest config + setup)
  are exempt from the fix loop's source-protection guard.
- **Unsolicited-test gating.** When the raw task doesn't ask for tests,
  the pipeline no longer auto-generates per-file coverage tests
  (plan-declared TEST steps still run).
- **Silent no-op CMD guard.** A command that reduces to
  cd/mkdir/parens while the plan declares concrete `produces:` files is
  failed at that step, so recovery runs where the problem is.
- **Repeated-text test-query rules.** Planner and tester prompts now
  require landmark-scoped Testing Library queries
  (`within(getByRole('banner'/'contentinfo'))` / `getAllBy*`) — brand
  text legitimately appears in header and footer, and singular queries
  were the most frequent self-inflicted test failure.

### Fixed

- File-creation `echo`-chain CMD steps (all observed Windows variants:
  spaced, compact `>`/`>>`, caret-escaped, parenthesized) are
  reclassified into CODE steps that write the file directly — cmd.exe
  chokes on `[`/`{` in echoed content.
- Multi-target `edit:` steps: the i-th bare `edit:` block now maps to
  the i-th target file (previously all blocks landed on the first
  target, fusing two files' content); full-file promotion accepts
  exactly one REPLACE block, never a merge.
- Edits targeting files that don't exist at plan time are converted to
  full-file writes (single complete REPLACE) or routed to the grounded
  path — no more FIND-matching against content the planner never saw.
- JSON syntax gates on inline patches, promotions, and the agent-tool
  `edit_file` (tsconfig JSONC exempt) — a trailing comma minimal-diffed
  into `package.json` previously broke every later npm invocation.
- Multi-line CMD steps joined with `&&` no longer fail on repeated
  `cd <dir>` segments (each plan line assumed the project root).
- `validate_plan` matches imports against glob `produces:` entries
  (`app/src/*`), so inline code importing scaffold-created files is no
  longer cleared as dangling.

## 0.4.0 — 2026-07-18

A verification-first release: steps now prove themselves against the
real project environment, regressions roll back automatically, and
Django/web tasks are checked by actually rendering pages rather than
trusting the model's claim. Also hardens the process against silent
native crashes.

### Added

- **Per-step verification gates.** A CODE/TEST step passes only when
  its plan-declared `verify:` command exits 0 — not at end-of-run.
  Failures feed back into the fix loop with the real command output.
- **Wave snapshots + monotonic rollback.** Each green wave is
  committed to a machine-managed git snapshot repo, and a later fix
  round that breaks a previously-passing gate is rolled back to the
  last green snapshot instead of being shipped.
- **Django ground-truth verification.** Template reachability and real
  page-render checks; task-pinned URLs and `Acceptance:` lines become
  executable acceptance probes.
- **Django lint gates** for the cross-file-defaults bug family:
  URL-name vs `app_name` namespace, `{% static %}` without
  `{% load static %}`, `@login_required` without `LOGIN_URL`, the
  `tests.py`/`tests` package shadow, and `{% url %}` names that
  resolve nowhere.
- **Web-page task grounding.** Rendered pages are the source of truth;
  task-quoted lines and `Acceptance:` lines become executable probes.
- **Intent-first planning** (`plan_mode: intent`): plans carry goals
  and gates while the agent loop authors the code.
- **Model escalation.** Failed agent-loop steps and recovery loops
  retry on a stronger model, with per-agent provider routing so
  escalation can cross providers.
- **Crash diagnostics for silent process deaths.** A heartbeat file
  plus a startup scavenger that correlates a stale heartbeat with the
  Windows Application event log and writes a post-mortem to
  `.agentchanti/crash.log`; `faulthandler` is armed to capture native
  stacks for the crashes it *can* catch.

### Changed

- SQLite embedding and vector stores now use **per-thread
  connections** (thread-local) instead of a single shared connection.
  A connection shared across threads is officially unsupported and
  could corrupt the interpreter heap — a Windows fast-fail
  (`0xc0000409`) seen as a silent mid-run exit.
- The planner now **recovers from reasoning-burn** (empty/truncated
  output at the token cap → reduced-effort retry) and **detects
  truncated plans**, re-planning for a complete plan instead of
  silently running a stub.

### Fixed

- Planner-emitted path normalization across every layer — doubled
  backslashes no longer poison verification.
- All multi-file content blocks are captured; phantom-path guard for
  inline edits.
- Package self-imports (`from . import x`) are no longer flagged as
  broken imports.
- Subproject venv resolution and cwd-aware verify; Django app-context
  imports are treated as inconclusive, not load failures.
- The config `models:` section no longer silently drops
  `escalation` / `intent` / `analyser` overrides.
- A false monotonic-gate regression that could roll back working code
  (a subproject gate recorded in a form that was never executed).

## 0.1.1 — 2026-04-19

### Fixed

- PyPI project metadata now points at the correct repository. The
  `Homepage`, `Repository`, and `Issues` URLs were shipped with a
  typo (`udaykanth` missing the trailing `r`) in 0.1.0 and all
  404'd. (#19, #20)
- `SECURITY.md` advisory link is reachable again. Same typo as
  above — private vulnerability reporting was silently broken on
  0.1.0. (#19)

### Removed

- Stale "package will be on PyPI once the first tagged release is
  cut" note in the README installation section.
- Reference to the deleted `setup.py` file in `SECURITY.md`.

## 0.1.0 — 2026-04-19

Initial public release on PyPI.

### Added

- `agentchanti` CLI and Python library for multi-agent AI coding
  tasks (Planner → Coder → Reviewer → Tester pipeline).
- Built-in RAG: tree-sitter code graph across 11 languages, local
  SQLite vector store, global knowledge base, error dictionary.
- Support for local LLMs (Ollama, LM Studio) and cloud providers
  (OpenAI, Gemini, Anthropic).
- Structured `PlanStep` format, KB-first command execution, plan-
  aware context injection, step caching, checkpoint/resume.
- Plugin system (`StepPlugin`) for custom pipeline steps.
- GitHub Actions CI (test matrix on ubuntu + windows × py3.10-3.12,
  ruff lint) and release workflow (PyPI trusted publishing via
  OIDC on tag push).

# The Assurance Report

What a reviewer reads instead of the log. One page per run, answering one
question: **the agent said it was done — what actually checked that?**

This is a specification, not yet an implementation. Every field below is
drawn from a signal agentchanti already emits, and the two worked examples
are built from the real logs of two real runs, so the shape is known to be
producible rather than hoped to be. Where a field has no source today it is
listed under [Gaps](#gaps) rather than quietly invented.

---

## 1. Why "passed" is not one bit

The design starts from a measured case, not from a diagram. Here is a run
from 2026-10-02 (`c-agentchanti-1`, a C todo manager):

* the pipeline completed every step and exited 0
* its seeded acceptance contract ran and passed, so `Evidence: independent`
* an external 11-step behavioural probe, written by neither agent, passed
  **11/11**
* and the ghost reported **8 violated postconditions** — the plan declared
  `todo_store.h` would export `todo_store_add`, `todo_store_init`,
  `todo_store_save` and four more; the file exports `Task` and friends
  instead

Every one of those statements is true at the same time. The code works, an
independent instrument agrees, and the artifact does not match what the plan
said it would be. A report that collapses this to **PASS** throws away the
only finding a reviewer would have acted on; one that collapses it to
**FAIL** is simply wrong about working software.

So the report has **axes, not a score**. Four questions that do not reduce
to each other:

| | question |
|---|---|
| **completed** | did the plan run without a step failing? |
| **verified** | did anything the agent did not write agree? |
| **coverage** | how much of what was declared actually got evaluated? |
| **drift** | where does the artifact disagree with what the run claimed? |

`evidence.py` already separates the first two — *completed* means the plan
ran, *verified* means someone else checked. The report adds the second pair,
because a verdict with no coverage figure invites the reading that silence
means agreement.

---

## 2. Sections

### VERDICT

```
completed   yes | no (halted at step N: <why>)
verified    independent | self-authored | pre-existing-tests-failed
            | acceptance-commands-failed | acceptance-commands-not-run
            [+ shallow] [+ repaired after N round(s)]
checked by  user acceptance_cmds | user's own suite | seeded contract | nobody
```

`verified` is `Evidence.kind` verbatim — the taxonomy already exists and
already encodes the hard-won distinctions. Two qualifiers ride alongside and
deliberately do **not** change the verdict:

* **shallow** — the check could not have failed. A contract asserting only
  that a process has not exited is independent and did run; calling it
  self-authored would be a different lie. What changes is the claim.
* **repaired** — green on attempt *N*. A run that passed on the third try is
  not the same result as one that passed clean, and averaging them hides the
  variance worth measuring.

`checked by` is the single most important line on the page, because the
instruments are not equal and the report must not let them look it:

| instrument | may establish evidence | may fail the run |
|---|---|---|
| user `acceptance_cmds` | yes | **yes** |
| user's own surviving suite | yes | yes |
| contract this pipeline seeded | yes | **no** |
| nothing | — | — |

A seeded contract is demoted in one direction only: it was written by a
model, so it can agree and cannot convict. That rule exists because four
consecutive runs exited non-zero over working games on four different
contract mistakes.

### WHAT WAS CHECKED

Coverage *of the checking*, which is the section with no equivalent in the
log today and the reason the report earns its place.

```
postconditions  30 declared — 16 hold, 8 violated, 5 unknown, 1 inapplicable
                evidence weight 26
gates           2 recorded, 2 green at exit, 0 regressed, 0 proven not measuring
suites          test_acceptance_contract.py — ran, passed
                make test — ran, printed no evidence a test ran
```

Four-valued on purpose. **UNKNOWN is not a pass and must never be rendered
as one** — it is the state for a postcondition nothing could evaluate, and
the single most misleading thing a report could do is omit it. INAPPLICABLE
is its own state because a step carried from a previous run was satisfied
against a tree this run never saw.

`evidence weight` is the existing scalar for "how much of this was actually
established". Printed beside the counts rather than instead of them.

### DISAGREEMENTS

The ghost's findings, grouped by kind and by file, each naming what to look
at. Grouped because eight `violated-exports` on one header is **one**
finding reported eight times, and a reader who scrolls past a list of
identical lines learns to skip the whole category.

Findings carry their own severity in their names, and the report preserves
it rather than flattening:

* `violated-*` — a step claimed done while something it declared is false
* `export-drift` — declared exports the code renamed that **nothing
  imports**, so no consumer can break. Collapsed to one note, never a
  per-symbol list.
* `unplanned-write` — a file no step declared
* `degenerate-long-run`, `varied-input-ignored`, `unprogressed-long-run` —
  an endurance claim that is unprotected
* `failed-but-clean` — suspect the harness before the model

### HOW IT GOT THERE

For the reviewer whose question is "why did this cost what it cost".

```
4 steps (1 CMD, 2 CODE, 1 TEST)
3 loop runs, 17 turns (avg 5.7), 0 recovery
outcomes: verified-early 2, gate-stalled 1
tokens: 212,805 total — 189,206 sent (134,344 cached, 71%), 23,599 received
```

### WHAT THIS REPORT DOES NOT CLAIM

Not a footnote — a section, generated from the run's own state, naming every
limit that applies to *this* run. The shape of every honest instrument in
this codebase: `verify_dt_invariance` exits 2 for could-not-verify,
`empty_suite_reason` answers in neither direction, `WiringReport.can_judge`
refuses rather than guessing.

For the worked example below it reads:

```
- 5 postconditions were never evaluated. This report says nothing about them.
- The only instrument that agreed was a contract this pipeline wrote before
  any code existed. It can establish evidence; it cannot convict the code.
- `make test` exited 0 without printing evidence that a test ran, so the
  project's own suite is a verdict in neither direction.
- No independent behavioural check ran. Supply `acceptance_cmds` for one.
```

---

## 3. Worked example — a run that works and still has findings

`c-agentchanti-1`, 2026-10-02. Built from that run's log; the probe line is
from an external run of `benchmarks/verify_todo_cli.py`.

```
ASSURANCE REPORT                                    agentchanti 0.12.0
run c-agentchanti-1 · "production-ready command-line todo manager in C"

VERDICT
  completed   yes — 4 of 4 steps
  verified    independent
  checked by  a contract this pipeline seeded (may agree, cannot convict)

WHAT WAS CHECKED
  postconditions  30 declared — 16 hold, 8 violated, 5 unknown, 1 n/a
                  evidence weight 26
  gates           2 recorded, 2 green at exit, 0 regressed
  suites          test_acceptance_contract.py  ran, passed
                  make test                    ran, no evidence a test ran

DISAGREEMENTS                                            9 total
  violated-exports  todo_store.h — 7 declared symbols absent
                    declared: todo_store_add, todo_store_complete,
                    todo_store_delete, todo_store_free, todo_store_init,
                    todo_store_load, todo_store_save
                    found:    Task, TodoStore, ...
                    → step 2.1 declared this interface and the code built
                      another. Working software, different shape.
  unplanned-write   task.txt — written, declared by no step
  export-drift      1 renamed export, no consumer — nothing can break

HOW IT GOT THERE
  4 steps (1 CMD, 2 CODE, 1 TEST) · 3 loop runs, 17 turns, 0 recovery
  outcomes: verified-early 2, gate-stalled 1
  212,805 tokens — 189,206 sent (71% cached), 23,599 received

NOT CLAIMED
  - 5 postconditions were never evaluated.
  - The agreeing instrument was written by a model in this run.
  - `make test` proved nothing in either direction.
  - No independent behavioural check ran.
```

A reviewer gets the finding in one read: **it works, and step 2.1 built a
different interface than it declared.** An external probe later passed this
artifact 11/11, which is exactly why the report says `independent` *and*
shows the eight violations instead of choosing.

---

## 4. Worked example — a run that collapsed

`c-agentchanti-2`, same task, same hour. The contrast is the point: the
numbers are not "worse", they are a **different shape**, and a one-line
PASS/FAIL cannot express it.

```
ASSURANCE REPORT                                    agentchanti 0.12.0
run c-agentchanti-2 · "production-ready command-line todo manager in C"

VERDICT
  completed   no — halted early; 2 of 5 steps classified
  verified    self-authored
  checked by  nobody — the seeded contract survived but did not pass

WHAT WAS CHECKED
  postconditions  36 declared — 1 hold, 1 violated, 33 unknown, 1 n/a
                  evidence weight 1
  gates           0 recorded — no step reached a gate
  suites          none ran

DISAGREEMENTS                                            2 total
  violated-exists   tests — planned target absent on disk
  unplanned-write   task.txt

HOW IT GOT THERE
  2 loop runs, 17 turns (avg 8.5), 1 recovery
  outcomes: verify-failed 1, gate-stalled 1
  270,881 tokens — 234,085 sent (60% cached), 36,796 received

NOT CLAIMED
  - 33 of 36 postconditions were never evaluated. Almost nothing here was
    checked; "1 violated" is not the finding, the 33 are.
  - No gate ran, so no behavioural claim was tested at all.
```

**Evidence weight 1 against 36 declared** is the whole story, and it is the
number a PASS/FAIL verdict destroys. The run cost **1.27× more** than the one
that succeeded (270,881 against 212,805) while establishing **1/26th** as
much — which is the pairing a reviewer wants and no existing line gives
them.

The example above also carries a correction worth recording, because it is
the failure mode this whole document is about. The first draft of it quoted
118,205 tokens for run 1 — a real figure, from a *different* run of the same
task an hour earlier — and derived "2.3× more" from it. Checking the log
rather than trusting the draft caught both. A report assembled by hand gets
exactly this wrong, which is the argument for generating it from the run's
own signals.

---

## 5. Where each field comes from

Implementability, field by field. Everything here exists today.

| report field | source |
|---|---|
| completed, halted-at | `pipeline_success`, the step loop's `step_results` |
| verified, qualifiers | `Evidence.kind` / `.shallow` / `.repaired` |
| checked by | `Evidence.detail` plus `_was_seeded` |
| postcondition counts | `GhostPlan` four-valued verdicts |
| evidence weight | the ghost's existing scalar |
| disagreements | `GhostPlan.disagreements()` |
| gates recorded/green/regressed | `GateLedger` |
| gates proven not measuring | `gate_defect_steps` |
| suite ran / proved nothing | `empty_suite_reason`, `_no_tests_collected` |
| steps, turns, outcomes | `loop_stats_summary()` |
| tokens | the token tracker's existing total line |

---

## 6. What the report deliberately refuses to do

* **No score.** No single number, no letter grade, no percentage. Four axes
  that do not reduce to each other, and a weighted blend of them would be an
  invented quantity presented as a measurement.
* **No absence rendered as agreement.** UNKNOWN and INAPPLICABLE are printed
  with the same prominence as HOLDS.
* **No instrument laundering.** A seeded contract is never described as
  though a person wrote it; `checked by` always names what actually checked.
* **No per-symbol noise.** Eight violations on one header are one finding.
  A reader trained to skip a category stops reading the real one inside it.
* **No advisory nobody consumes.** Every limit is in a section, not a log
  line someone has to find. The precedent is explicit: a gate was named
  unrunnable at 00:38:48, nothing consumed the warning, and the run spent
  180k tokens proving it right.

---

## Gaps

What the report wants and the run does not currently emit:

1. **A behavioural probe result.** The strongest line a report could carry —
   *"an external check neither agent wrote passed 11/11"* — comes from
   running `benchmarks/verify_todo_cli.py` by hand. Inside a run, only
   user-supplied `acceptance_cmds` fills this slot.
2. **Per-postcondition provenance.** The counts are known; which step
   declared each UNKNOWN, and why it could not be evaluated, is not
   summarised.
3. **Gate quality, not just gate status.** `shallow_gate_reason` and
   `unrunnable_gate_reason` run at plan time, and their verdicts are not
   carried through to the end. "2 gates green" reads the same whether they
   could have failed or not.
4. **A machine-readable form.** `TaskResult` carries `verified` and
   `evidence` but none of the coverage figures, so a CI job cannot gate on
   them.

Gap 3 is the one I would close first: it is the difference between *"two
gates passed"* and *"two gates passed, and either could have passed over
wrong behaviour"* — which is the same distinction `Evidence.shallow` already
draws one layer up, left undrawn one layer down.

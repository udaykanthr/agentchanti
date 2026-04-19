# Plan: Improve Test Failure Handling — Batch Fix Strategy

## Problem

When tests fail, the current system fixes them **one retry at a time** in nested loops:

```
Gen Attempt (×2) → Run Attempt (×3) → Diagnosis (×2, each re-runs full step)
= Up to 12 LLM calls per failing TEST step
```

Each retry sends truncated error output (1500 chars) + ALL project files to the Coder, which:
1. Only sees partial errors → makes partial fixes → triggers another retry
2. Fixes one issue but breaks another → oscillates across retries
3. For 20+ files, the cycle repeats per step, totaling 30+ minutes

## Root Causes

| Issue | Location | Impact |
|---|---|---|
| Error truncated to 1500 chars | `_extract_test_error()` L1140 | Coder sees incomplete failures, guesses at fix |
| Coder gets one error batch per retry | `_handle_test_step()` L1941 | Fixes 1 issue, breaks another, repeat |
| Diagnosis re-runs the **entire** step | `pipeline.py` L432 | Full restart instead of targeted fix |
| No structured error grouping | — | LLM can't reason about all failures at once |

## Solution: Batch Error Context + Single-Pass Fix

### Change 1: Add `_build_batch_error_summary()` in `step_handlers.py`

A new function that parses test runner output and produces a **structured, compact summary** of ALL failures:

```
FAILED TESTS SUMMARY (5 failures across 3 files):

1. tests/test_auth.py::test_login_valid
   Error: AssertionError: expected 200, got 401
   Line: 45

2. tests/test_auth.py::test_login_invalid
   Error: AttributeError: 'NoneType' has no attribute 'status_code'
   Line: 52

3. tests/test_api.py::test_create_user
   Error: ImportError: cannot import name 'create_user' from 'app.models'
   Line: 8

... (grouped by error type so LLM sees patterns)
```

Key design decisions:
- **Group by error type** (ImportError, AssertionError, etc.) so LLM spots systemic issues (e.g., wrong import path affects 10 tests → one fix)
- **Include file + line + short error** for each failure — no full tracebacks
- **Budget: ~4000 chars** instead of 1500, but much more information-dense
- Works for pytest, Jest, and Vitest output formats

### Change 2: Update fix prompt in `_handle_test_step()` to request batch fix

Instead of:
> "Fix ONLY the test files so tests pass"

Use:
> "Below is a summary of ALL test failures. Analyze the patterns — if multiple failures share a root cause (e.g., wrong import path, missing mock), fix the root cause once. Return ALL fixed test files."

This tells the LLM to look for **shared root causes** across failures rather than fixing one at a time.

### Change 3: Reduce retry constants

Since each attempt now has much better context:

- `MAX_STEP_RETRIES`: 3 → 2 (each attempt is higher quality with full error context)
- `MAX_DIAGNOSIS_RETRIES`: 2 → 1 (diagnosis already gets full error picture)

This alone cuts worst-case LLM calls from 12 to 6.

### Change 4: Add early-exit on diminishing returns in test fix loop

In the retry loop, track the **number of failing tests** across attempts. If attempt N has the same or more failures than attempt N-1, break early instead of wasting another retry on the same blind spot.

```python
prev_fail_count = None
for run_attempt in range(1, MAX_STEP_RETRIES + 1):
    success, output = executor.run_tests(...)
    if success:
        return True, ""

    current_fail_count = _count_test_failures(output)
    if prev_fail_count is not None and current_fail_count >= prev_fail_count:
        break  # Not making progress, stop retrying
    prev_fail_count = current_fail_count

    # ... proceed with fix using batch error summary
```

## Files to Modify

1. **`agentchanti/orchestrator/step_handlers.py`**
   - Add `_build_batch_error_summary()` function (~80 lines)
   - Add `_count_test_failures()` helper (~20 lines)
   - Update `_handle_test_step()` fix context to use batch summary
   - Add early-exit on no-progress detection
   - Reduce `MAX_STEP_RETRIES` from 3 to 2

2. **`agentchanti/orchestrator/pipeline.py`**
   - Reduce `MAX_DIAGNOSIS_RETRIES` from 2 to 1

## Expected Impact

| Metric | Before | After |
|---|---|---|
| Max LLM calls per TEST step | 12 | 4-6 |
| Error context quality | 1500 chars, truncated | ~4000 chars, structured by error type |
| Fix strategy | Fix blindly one-at-a-time | Identify root causes, batch fix |
| No-progress retries | Always exhausts all attempts | Exits early when stuck |
| Estimated time for 20-file project | 30+ min | ~8-12 min |

## What We Are NOT Changing

- The overall pipeline architecture (Planner → Coder → Reviewer → Tester)
- Test generation logic in TesterAgent
- The diagnosis system (just reducing its retry count)
- File memory, KB, or executor subsystems

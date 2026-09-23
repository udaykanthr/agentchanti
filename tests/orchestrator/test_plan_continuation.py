"""A plan cut off by the output cap is continued, not written again.

Measured 2026-09-23, glm-5.3:cloud, with the planner cap already raised to
32,768: three generations in a row hit it — cut at step 1.6, then 2.3, then
again — about 6 minutes of a 13.5-minute run and ~98k received tokens on
plans that were discarded whole. Nothing was wrong with the steps it had
already written.
"""
from agentchanti.orchestrator.plan_step import (
    merge_plan_continuation, parse_structured_plan, plan_continuation_note,
    truncated_plan_prefix)

# The shape of the measured plan: five complete steps, cut inside the sixth.
CUT = """==PLAN==

--STEP 1.1 [CMD] depends:none
Install dependencies
> npm --prefix my-app install

--STEP 1.2 [CODE] depends:none
Create the Header server component
target: my-app/components/Header.tsx
exports: Header (default export)
verify: node -e "process.exit(0)"

--STEP 1.3 [CODE] depends:none
Create the Hero server component
target: my-app/components/Hero.tsx
exports: Hero (default export)
verify: node -e "process.exit(0)"

--STEP 1.4 [CODE] depends:none
Create the Features server component
target: my-app/components/Features.tsx
exports: Features (default export)
verify: node -e "process.exit(0)"

--STEP 1.5 [CODE] depends:none
Create the Stats server component
target: my-app/components/Stats.tsx
exports: Stats (default export)
verify: node -e "process.exit(0)"

--STEP 1.6 [CODE] depends:none
Create the CtaSection server component ("""

CONTINUATION = """--STEP 1.6 [CODE] depends:none
Create the CtaSection server component
target: my-app/components/CtaSection.tsx
exports: CtaSection (default export)
verify: node -e "process.exit(0)"

--STEP 2.1 [CODE] depends:1.2, 1.3, 1.4, 1.5, 1.6
Compose the page
target: my-app/app/page.tsx
verify: npm --prefix my-app run build
==END=="""


class TestKeepingWhatWasWritten:

    def test_the_cut_step_is_dropped_and_the_rest_kept(self):
        prefix = truncated_plan_prefix(CUT)
        kept = parse_structured_plan(prefix)
        assert [s.id for s in kept] == ["1.1", "1.2", "1.3", "1.4", "1.5"]
        assert "CtaSection" not in prefix

    def test_nothing_to_keep_returns_none(self):
        """One step plus a cut one is not worth a continuation call."""
        assert truncated_plan_prefix(
            "==PLAN==\n\n--STEP 1.1 [CMD] depends:none\n> ls\n\n"
            "--STEP 1.2 [CODE] depends:none\nCreate the th") is None
        assert truncated_plan_prefix("") is None


class TestTheRequest:

    def test_it_names_the_kept_steps_and_where_to_resume(self):
        kept = parse_structured_plan(truncated_plan_prefix(CUT))
        note = plan_continuation_note(kept)
        assert "1.1, 1.2, 1.3, 1.4, 1.5" in note
        assert "AFTER 1.5" in note
        assert "do NOT repeat" in note


class TestMerging:

    def test_the_merged_plan_is_complete_and_ordered(self):
        prefix = truncated_plan_prefix(CUT)
        merged = merge_plan_continuation(prefix, CONTINUATION)
        steps = parse_structured_plan(merged)
        assert [s.id for s in steps] == \
            ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "2.1"]
        assert "==END==" in merged
        assert merged.count("--STEP 1.2") == 1

    def test_a_re_emitted_preamble_is_dropped(self):
        merged = merge_plan_continuation(
            truncated_plan_prefix(CUT),
            "==PLAN==\nHere is the rest of the plan:\n\n" + CONTINUATION)
        assert merged.count("==PLAN==") == 1
        assert "Here is the rest" not in merged

    def test_a_continuation_with_no_steps_leaves_the_prefix_alone(self):
        prefix = truncated_plan_prefix(CUT)
        assert merge_plan_continuation(prefix, "I could not continue.") == prefix

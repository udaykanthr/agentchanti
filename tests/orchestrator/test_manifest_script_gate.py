"""A gate asserting HOW a script is spelled cannot fail on behaviour.

Measured 2026-09-27 on the `todo-node` benchmark case, twice in three
runs. The planner gated a step on::

    node -e "const p=require('./package.json');
             if(p.scripts.test!=='node --test')process.exit(1)"

The project's script was `node --test test/todoManager.test.js` — the
same suite, named explicitly. The gate stayed red over a correct
artifact while the project's own suite was green, which produced a GATE
CONFLICT and failed the run. Both artifacts passed an external 11-step
behavioural probe, and the ghost reported `failed-but-clean` over 13
holding postconditions.

The identical task in Python passed 3/3, which is what made the gate —
rather than the model or the language — the suspect.
"""
import pytest

from agentchanti.orchestrator.plan_step import shallow_gate_reason

MEASURED = ("node -e \"const p=require('./package.json');"
            "if(p.scripts.test!=='node --test')process.exit(1)\"")


class TestTheMeasuredGate:

    def test_it_is_called_shallow(self):
        reason = shallow_gate_reason(MEASURED)
        assert reason and "scripts.test" in reason

    def test_the_advice_is_to_run_it_instead(self):
        """"Assert a value" is not the fix for asserting a spelling."""
        assert "npm run test" in shallow_gate_reason(MEASURED)

    @pytest.mark.parametrize("cmd", [
        "node -e \"const p=require('./package.json');"
        "if(p.scripts['build'] !== 'vite build')process.exit(1)\"",
        "node -e \"if(require('./package.json').scripts.start==='node .')"
        "process.exit(1)\"",
    ])
    def test_other_spellings_of_the_same_mistake(self, cmd):
        assert shallow_gate_reason(cmd)


class TestWhatIsNotAccused:
    """Asserting a manifest's CONTENT is often a real check."""

    def test_a_dependency_pin_is_a_real_assertion(self):
        cmd = ("python -c \"t=open('requirements.txt').read(); "
               "assert 'pygame' in t\"")
        assert shallow_gate_reason(cmd) is None

    def test_asserting_a_script_exists_is_not_asserting_its_text(self):
        cmd = ("node -e \"const p=require('./package.json');"
               "if(!p.scripts.test)process.exit(1)\"")
        assert shallow_gate_reason(cmd) is None

    def test_running_the_script_is_the_recommended_form(self):
        assert shallow_gate_reason("npm run test") is None
        assert shallow_gate_reason("npm test") is None

    def test_an_ordinary_behavioural_gate_is_untouched(self):
        cmd = ("node -e \"const {add}=require('./index.js');"
               "if(add(2,3)!==5)process.exit(1)\"")
        assert shallow_gate_reason(cmd) is None

    def test_a_build_command_is_untouched(self):
        assert shallow_gate_reason("npm run build") is None

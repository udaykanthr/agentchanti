"""A gate that runs perfectly and still cannot pass.

Every other structural branch of `unrunnable_gate_reason` catches a gate
the SHELL cannot execute. This one executes fine and is logically
inverted, which no existing check could see.

Measured 2026-09-25 on the pre-scaffolded Next.js case, twice in three
runs. The shorter of the two::

    findstr /c:"Create Next App" app\\layout.tsx >nul && exit /b 1

reads as "fail if the boilerplate is still there". But when the text is
ABSENT — the correct outcome — findstr itself exits 1, `&&` skips the
`exit`, and findstr's 1 becomes the gate's status. Correct output fails
it exactly as wrong output does.

Both runs shipped a home page that builds, serves 200 and has every line
of Next's starter copy removed, and both exited 1 with `Evidence:
independent` logged immediately above `Pipeline failed`.
"""
import pytest

from agentchanti.orchestrator.plan_step import (_always_fails,
                                                _always_fails_error,
                                                unrunnable_gate_reason)

MEASURED_LONG = (
    r'(findstr /c:"To get started, edit the" app\page.tsx >nul && '
    r'(echo Forbidden starter copy present & exit /b 1) || '
    r'(findstr /c:"Deploy Now" app\page.tsx >nul && '
    r'(echo Forbidden promotion present & exit /b 1) || '
    r'(findstr /c:"Documentation" app\page.tsx >nul && '
    r'(echo Forbidden promotion present & exit /b 1))))')
MEASURED_SHORT = r'findstr /c:"Create Next App" app\layout.tsx >nul && exit /b 1'


class TestTheMeasuredGates:

    @pytest.mark.parametrize("gate", [MEASURED_SHORT, MEASURED_LONG])
    def test_the_gate_is_refused(self, gate):
        reason = _always_fails_error(gate)
        assert reason and "exits non-zero whatever the code does" in reason

    @pytest.mark.parametrize("gate", [MEASURED_SHORT, MEASURED_LONG])
    def test_it_reaches_the_structural_ladder(self, gate):
        """It must be refused through the function the pipeline calls."""
        assert unrunnable_gate_reason(gate)

    def test_the_advice_names_the_actual_fix(self):
        """"Assert a concrete value" is not the fix for an inverted gate."""
        reason = _always_fails_error(MEASURED_SHORT)
        assert "exit /b 0" in reason


class TestWhatIsNotAccused:
    """A false positive costs a correct gate, so the bar is a proof."""

    @pytest.mark.parametrize("gate", [
        r'findstr /c:"Create Next App" app\layout.tsx >nul && exit /b 1 || exit /b 0',
        r'findstr /c:"Hero" app\page.tsx >nul',
        "npm run build",
        "npm run build && npm test",
        "npm run build && exit /b 0",
        "npm test || npm run build",
        "python -m pytest -q",
        'python -c "import app; assert app.OK"',
        # A quoted operator is text, not a shell operator.
        'python -c "assert \'a && b\' in open(\'x\').read()"',
        "",
    ])
    def test_an_ordinary_gate_is_untouched(self, gate):
        assert _always_fails_error(gate) is None

    def test_exit_zero_is_a_success_path(self):
        """The status has to be read, not merely matched."""
        assert not _always_fails("exit /b 0")
        assert _always_fails("exit /b 1")

    def test_separate_groups_are_not_one_group(self):
        """`(A) || (B)`'s outer parens are not a matching pair.

        Stripping them produced the nonsense `A) || (B` and missed a real
        detection.
        """
        assert _always_fails('(findstr /c:"A" x >nul && exit /b 1) || '
                             '(findstr /c:"B" x >nul && exit /b 1)')
        assert not _always_fails('(npm test) || (npm run build)')


class TestOperatorSemantics:
    """The proof rests on these, so they are pinned individually."""

    def test_and_always_fails_iff_the_last_conjunct_does(self):
        assert _always_fails("whatever && exit /b 1")
        assert not _always_fails("exit /b 1 && whatever")

    def test_or_always_fails_only_if_every_branch_does(self):
        assert _always_fails("a && exit /b 1 || b && exit /b 2")
        assert not _always_fails("a && exit /b 1 || b")

    def test_sequential_takes_the_last_status(self):
        assert _always_fails("echo hello & exit /b 1")
        assert not _always_fails("exit /b 1 & echo hello")

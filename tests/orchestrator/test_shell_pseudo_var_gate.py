"""A gate cannot read `%CD%` out of the process environment.

Measured 2026-09-27 on the `todo-node` benchmark case — the last wrong
verdict among 29 independently probed artifacts. The gate wanted the
directory it was launched from, so it could chdir to a temp directory and
still require the module under test:

    process.chdir(d);
    const s = require(path.resolve(process.env.CD, 'todoStore.js'));

`%CD%` works on a cmd.exe command line, so it reads as if it should work.
But cmd never writes CD into a child's environment block, agentchanti is
launched from Python rather than cmd, and `_shell_free_argv` deliberately
bypasses cmd for a single inline script. So `process.env.CD` is
`undefined`, `path.resolve(undefined, …)` throws ERR_INVALID_ARG_TYPE,
and the gate fails identically whatever the step writes.

Verified by running the gate's own expression with CD removed from the
environment (exit 1, TypeError) and with `process.cwd()` in its place
(exit 0). The step spent 10 turns, recovery spent 10 more, and the run
ended at 167k tokens over an artifact that passes an 11-step behavioural
probe.
"""
import pytest

from agentchanti.orchestrator.plan_step import (_shell_pseudo_var_error,
                                                unrunnable_gate_reason)

MEASURED = (
    'node -e "const fs=require(\'fs\'),os=require(\'os\'),'
    "path=require('path');const d=fs.mkdtempSync(path.join(os.tmpdir(),"
    "'todo-'));process.chdir(d);const s=require(path.resolve("
    'process.env.CD,\'todoStore.js\'));if(!s)process.exit(1)"')


class TestTheMeasuredGate:

    def test_it_is_refused(self):
        reason = _shell_pseudo_var_error(MEASURED)
        assert reason and "CD" in reason

    def test_it_reaches_the_structural_ladder(self):
        """Refused through the function the pipeline actually calls."""
        assert unrunnable_gate_reason(MEASURED)

    def test_the_advice_names_the_right_call(self):
        assert "process.cwd()" in _shell_pseudo_var_error(MEASURED)

    def test_the_advice_warns_about_chdir_ordering(self):
        """`process.cwd()` AFTER the chdir is the same bug again."""
        assert "BEFORE any chdir" in _shell_pseudo_var_error(MEASURED)


class TestOtherSpellings:

    @pytest.mark.parametrize("cmd", [
        'node -e "console.log(process.env[\'CD\'])"',
        'node -e "if(process.env.ERRORLEVEL)process.exit(1)"',
        'python -c "import os; print(os.environ[\'CD\'])"',
        'python -c "import os; print(os.environ.get(\'CD\'))"',
        'python -c "import os; print(os.getenv(\'CD\'))"',
    ])
    def test_each_is_refused(self, cmd):
        assert _shell_pseudo_var_error(cmd)


class TestWhatIsNotAccused:

    @pytest.mark.parametrize("cmd", [
        # The correct forms.
        'node -e "const p=require(\'path\').resolve(process.cwd(),\'x.js\')"',
        'python -c "import os; print(os.getcwd())"',
        # Ordinary environment reads are fine — these really are exported.
        'node -e "if(!process.env.PATH)process.exit(1)"',
        'python -c "import os; assert os.environ[\'PATH\']"',
        'node -e "console.log(process.env.NODE_ENV)"',
        # `%CD%` expanded BY the shell is a different thing and works.
        'node -e "console.log(1)" && echo %CD%',
        "npm run build",
        "",
    ])
    def test_it_is_left_alone(self, cmd):
        assert _shell_pseudo_var_error(cmd) is None

    def test_a_variable_merely_named_like_one_is_not_matched(self):
        """`CDN_URL` starts with CD and is an ordinary variable."""
        assert _shell_pseudo_var_error(
            'node -e "console.log(process.env.CDN_URL)"') is None

    @pytest.mark.parametrize("cmd", [
        'set "CD=%CD%" && node -e "console.log(process.env.CD)"',
        'set "CD=%cd%" && node -e "console.log(process.env.CD)"',
        'set CD=%CD% && node -e "console.log(process.env.CD)"',
        'CD="$PWD" node -e "console.log(process.env.CD)"',
        'export CD="$PWD"; node -e "console.log(process.env.CD)"',
    ])
    def test_a_command_that_exports_it_first_is_not_accused(self, cmd):
        """This is the repair the agent reached for, and it WORKS.

        Found by auditing 19,774 real gates: the only four matches were
        the measured incident, and two of them were this form. Refusing
        them would fail a gate that passes.
        """
        assert _shell_pseudo_var_error(cmd) is None

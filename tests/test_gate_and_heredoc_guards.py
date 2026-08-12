"""Two guards from a hello-world run that failed on working code.

A local model was asked to print "Hello World". The program was correct and
`python hello_world.py` exited 0 midway through the run, yet the pipeline
spent three diagnosis attempts and then halted. Two independent defects:

1. The planner's acceptance gate was::

       python venv\\Scripts\\python.exe hello_world.py | find /i "Hello World"

   which hands python.exe to python AS THE SCRIPT, so the interpreter parses
   a binary and dies with "SyntaxError: Non-UTF-8 code starting with '\\x90'".
   Unsatisfiable whatever the step writes. `unrunnable_gate_reason` only
   inspected inline `-c`/`-e` payloads, so nothing saw it — and the model
   diagnosed it correctly ("use `python hello_world.py` instead of executing
   the interpreter binary") with nowhere to put the finding.

2. The coder answered with the shell recipe that creates the file::

       cat <<'EOF' > hello_world.py
       print("Hello World")
       EOF

   and the wrapper was written into the .py verbatim. The syntax check
   cannot catch this: `cat <<'EOF' > hello_world.py` is VALID Python (a
   left-shift then a comparison), so it parses and only fails at import
   with `NameError: name 'cat' is not defined`.
"""

import unittest

from agentchanti.executor import Executor
from agentchanti.orchestrator.plan_step import unrunnable_gate_reason


class UnrunnableGateTest(unittest.TestCase):
    def test_the_gate_from_the_failing_run(self):
        reason = unrunnable_gate_reason(
            r'python venv\Scripts\python.exe hello_world.py'
            r' | find /i "Hello World"')
        self.assertIsNotNone(reason)
        self.assertIn("another executable", reason)

    def test_posix_form_too(self):
        self.assertIsNotNone(
            unrunnable_gate_reason("python venv/bin/python hello_world.py"))

    def test_placeholder_left_in_the_command(self):
        """`python <filename>` was executed verbatim during diagnosis."""
        reason = unrunnable_gate_reason("python <filename>")
        self.assertIsNotNone(reason)
        self.assertIn("placeholder", reason)

    def test_legitimate_gates_are_not_flagged(self):
        for cmd in (
            "python -m unittest -v",
            "python -m pytest -q",
            "python hello_world.py",
            r"venv\Scripts\python.exe hello_world.py",   # direct: correct
            "python.exe hello_world.py",
            "python -X utf8 hello_world.py",
            "python manage.py test main --noinput",
            'python -c "from game import Game; g=Game(); assert len(g.ghosts)==4"',
            'node -e "const a=1,b=2; if(a<b) process.exit(0); process.exit(1)"',
            'python -c "assert \'<div>\' in open(\'i.html\').read()"',
            "npm test --silent",
            "go test ./...",
            "",
        ):
            with self.subTest(cmd=cmd):
                self.assertIsNone(unrunnable_gate_reason(cmd))


class HeredocUnwrapTest(unittest.TestCase):
    RESPONSE = (
        'THOUGHT: I need to create a Python script.\n\n'
        "```bash\n"
        "cat <<'EOF' > hello_world.py\n"
        "#!/usr/bin/env python3\n"
        'print("Hello World")\n'
        "EOF\n"
        "```\n"
    )

    def _target(self, text, target):
        return Executor.parse_blocks_for_single_target(text, target).get(target)

    def test_the_wrapper_never_reaches_the_file(self):
        body = self._target(self.RESPONSE, "hello_world.py")
        self.assertIsNotNone(body)
        self.assertNotIn("cat <<", body)
        self.assertNotIn("EOF", body)

    def test_the_recovered_body_actually_runs(self):
        """It parsed fine before this fix — the failure was at import."""
        body = self._target(self.RESPONSE, "hello_world.py")
        exec(compile(body, "hello_world.py", "exec"), {})

    def test_a_short_body_is_kept_because_the_heredoc_named_the_file(self):
        """The <3-line fragment rule must not discard an explicit file write."""
        text = "```bash\ncat <<'EOF' > a.py\nprint(1)\nEOF\n```"
        self.assertEqual(self._target(text, "a.py"), "print(1)")

    def test_redirect_before_the_heredoc_is_handled(self):
        text = '```bash\ncat > app.py <<EOF\nprint("hi")\nEOF\n```'
        self.assertEqual(self._target(text, "app.py"), 'print("hi")')

    def test_truncated_heredoc_still_loses_the_recipe_line(self):
        """A response cut at the token cap has no closing delimiter."""
        text = "```bash\ncat <<'EOF' > a.py\nimport os\nprint(1)\n```"
        self.assertEqual(self._target(text, "a.py"), "import os\nprint(1)")

    def test_shell_targets_keep_their_heredoc(self):
        """A .sh file may legitimately contain one."""
        text = "```bash\ncat <<'EOF' > out.txt\nhello\nEOF\n```"
        self.assertIn("cat <<", self._target(text, "deploy.sh"))

    def test_ordinary_blocks_are_untouched(self):
        text = "```python\nimport os\n\ndef f():\n    return 1\n```"
        self.assertEqual(self._target(text, "a.py"),
                         "import os\n\ndef f():\n    return 1")

    def test_short_fragment_without_a_heredoc_is_still_refused(self):
        self.assertIsNone(self._target("```python\nx = 1\n```", "a.py"))

    def test_unparseable_block_is_still_refused(self):
        self.assertIsNone(
            self._target("```python\ndef f(:\n    retur\n```", "a.py"))


if __name__ == "__main__":
    unittest.main()

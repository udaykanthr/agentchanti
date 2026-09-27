"""A CLI refusing an empty command line is not a crashed app.

Measured 2026-09-27 on a replayed todo-manager plan, 3 runs of 3. The
smoke test launched `python main.py` with no arguments; the CLI exited 1
exactly as the task required ("exit code 1 for an unknown command"); the
smoke test called it `App crashed on launch` and rewrote main.py three
times trying to fix it. All three artifacts passed an external 11-step
behavioural probe and the ghost reported `failed-but-clean`.

The repair loop's only tool is an edit, so it edits whether or not
anything is wrong — the same shape as the recorded incident where a
smoke "fix" turned a graphical game into a silent headless one.

The discriminator is a traceback, NOT the output text: the measured
launch printed nothing at all, so a usage-message regex would have
missed it.
"""
from agentchanti.orchestrator.smoke_test import _is_cli_awaiting_arguments

ARGPARSE_CLI = '''\
import argparse, sys

def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

ARGV_CLI = '''\
import sys

def main(argv):
    if not argv:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''

LONG_RUNNING = '''\
import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    while True:
        pygame.display.flip()

if __name__ == "__main__":
    main()
'''

TRACEBACK = ("Traceback (most recent call last):\n"
             '  File "main.py", line 3, in <module>\n'
             "    import missing_thing\n"
             "ModuleNotFoundError: No module named 'missing_thing'\n")


class TestTheMeasuredCase:

    def test_a_silent_nonzero_exit_from_an_argparse_cli(self):
        """The measured launch printed NOTHING — output cannot decide it."""
        assert _is_cli_awaiting_arguments("", "main.py",
                                          {"main.py": ARGPARSE_CLI})

    def test_a_hand_rolled_argv_dispatcher_counts_too(self):
        assert _is_cli_awaiting_arguments("", "main.py",
                                          {"main.py": ARGV_CLI})

    def test_usage_text_is_not_required(self):
        out = "usage: main.py [-h] {add,list} ...\nerror: required: command"
        assert _is_cli_awaiting_arguments(out, "main.py",
                                          {"main.py": ARGPARSE_CLI})


class TestWhatIsStillACrash:
    """A false negative costs turns; a false positive ships a broken app."""

    def test_a_traceback_is_always_a_crash(self):
        assert not _is_cli_awaiting_arguments(TRACEBACK, "main.py",
                                              {"main.py": ARGPARSE_CLI})

    def test_a_long_running_app_is_not_a_cli(self):
        assert not _is_cli_awaiting_arguments("", "main.py",
                                              {"main.py": LONG_RUNNING})

    def test_an_unreadable_entry_point_is_not_assumed_to_be_a_cli(self):
        assert not _is_cli_awaiting_arguments("", "nope.py", {})

    def test_empty_source_is_not_a_cli(self):
        assert not _is_cli_awaiting_arguments("", "main.py", {"main.py": ""})


class TestTheSmokeLoopConsumesIt:
    """A guard nothing calls is the mistake `protect_acceptance_files`
    already made once."""

    def test_the_launch_loop_calls_it(self):
        import inspect

        from agentchanti.orchestrator import smoke_test
        src = inspect.getsource(smoke_test)
        assert src.count("_is_cli_awaiting_arguments") >= 2
        # ...and before the repair attempt, not after it.
        assert (src.index("_is_cli_awaiting_arguments(out, entry")
                < src.index("App crashed on launch (attempt"))

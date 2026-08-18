"""The error signature must read the end of an error, not just the start.

`_error_signature` hashed `norm[:600]`. A test runner front-loads whatever
is invariant — a constant summary line, then the verbose listing of test
names in alphabetical order — and leaves the discriminating part until
last: which assertion blew up, and the `FAILED (failures=F, errors=E)`
tally. So two genuinely different failures collided.

Measured live 2026-08-17, classic path, step 3 of an 8-step plan. The
debug line added for exactly this question caught it:

    attempt 1->2:  sig 1a3d09c05029 -> 1a3d09c05029, error_info 1692 chars
    attempt 2->3:  sig 1a3d09c05029 -> 1a3d09c05029, error_info 1038 chars

Different lengths, so provably not the same string, hashing identically.
The loop reported "previous fix changed nothing" twice about fixes that
had changed the error.

This matters most for CODE steps: a bare traceback carries no test counts,
so `_diagnosis_score` returns None and the signature is the *only* signal
the diagnosis loop has. An earlier run reverted a correct fix on exactly
this comparison.
"""

import unittest

from agentchanti.orchestrator.pipeline import _SIG_WINDOW, _error_signature


def _long_invariant_head(chars=1000):
    """The part of a verbose test listing that does not change between
    attempts: the same test names, in the same alphabetical order."""
    out = []
    while sum(len(s) for s in out) < chars:
        c = chr(ord("a") + len(out) % 26)
        out.append(f"test_{c}{len(out)} (test_game.T.test_{c}{len(out)}) ... ok\n")
    return "".join(out)


PREAMBLE = ("Tests partially failing: 0/1 test files passed. "
            "Failed: test_game.py\nLast output:\n")


class TestSignatureReadsBothEnds(unittest.TestCase):

    def test_failures_differing_only_past_the_head_are_distinguished(self):
        """The measured incident: nine errors becoming one."""
        head = PREAMBLE + _long_invariant_head()
        nine = head + "Ran 8 tests\n\nFAILED (failures=1, errors=9)\n"
        one = head + "Ran 8 tests\n\nFAILED (failures=1)\n"

        self.assertNotEqual(nine, one)
        self.assertGreater(len(nine), _SIG_WINDOW)
        self.assertNotEqual(_error_signature(nine), _error_signature(one))

    def test_a_differing_traceback_tail_is_distinguished(self):
        """A CODE step's bare traceback — the case with no score to fall
        back on, and the one that reverted a correct fix."""
        head = "Traceback (most recent call last):\n" + _long_invariant_head()
        a = head + "TypeError: 'bool' object is not callable\n"
        b = head + "AttributeError: 'Game' object has no attribute 'player'\n"

        self.assertNotEqual(_error_signature(a), _error_signature(b))

    def test_identical_errors_still_share_a_signature(self):
        """The property the whole mechanism depends on."""
        err = PREAMBLE + _long_invariant_head() + "FAILED (failures=2)\n"
        self.assertEqual(_error_signature(err), _error_signature(err))

    def test_short_errors_are_hashed_whole(self):
        a = "TypeError: 'bool' object is not callable"
        b = "TypeError: 'bool' object is not callablX"
        self.assertNotEqual(_error_signature(a), _error_signature(b))

    def test_cosmetic_churn_is_still_forgiven(self):
        """Normalisation runs before the windowing, so the strip rules
        that existed before still hold over a long error."""
        head = _long_invariant_head()
        a = head + r'File "C:\tmp\aaa\map.py":12:3 TypeError: boom at 0xdeadbeef'
        b = head + r'File "C:\other\bbb\map.py":98:7 TypeError: boom at 0xcafe00'
        self.assertEqual(_error_signature(a), _error_signature(b))

    def test_a_middle_only_difference_is_accepted_as_a_limit(self):
        """Documented bound, not an accident: a difference confined to the
        middle of a very long error is still invisible. Windowing both
        ends is a large improvement over one, not a hash of everything —
        which would make cosmetic churn anywhere read as progress."""
        filler_head = "H" * _SIG_WINDOW
        filler_tail = "T" * _SIG_WINDOW
        a = filler_head + "A" * 50 + filler_tail
        b = filler_head + "B" * 50 + filler_tail
        self.assertEqual(_error_signature(a), _error_signature(b))


if __name__ == "__main__":
    unittest.main()

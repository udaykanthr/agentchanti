"""The diagnosis loop must keep a fix that moved the failure forward.

A gate is usually a chain of asserts, so each correct fix uncovers the
NEXT failing condition. The loop used to restore its snapshot before
every attempt, which threw away good fixes: observed on a Pac-Man run
where attempt 1 correctly fixed ``Map.is_walkable``'s arity, attempt 2
reverted it and fixed the next error instead, and the step halted having
never held both fixes at once. Worse, the revert put the file back in
state A while ``error_info`` still described state B, so the second
diagnosis reasoned from a premise that no longer matched the disk.

Reverting is still right when an attempt achieves nothing — that is what
stops a bad fix compounding. The two cases are told apart by
``_error_signature``, the same helper the fix-loop escape hatch uses.
"""

import unittest

from agentchanti.orchestrator.pipeline import _error_signature


TYPE_ERR = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 1, in <module>\n'
    "TypeError: is_walkable() takes 2 positional arguments but 3 were given\n"
)

ATTR_ERR = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 1, in <module>\n'
    "AttributeError: 'Map' object has no attribute 'total_pellets'\n"
)


class ErrorSignatureTest(unittest.TestCase):
    def test_same_error_has_same_signature(self):
        """No progress → the loop reverts."""
        self.assertEqual(_error_signature(TYPE_ERR),
                         _error_signature(TYPE_ERR))

    def test_different_errors_differ(self):
        """Progress → the gate moved on, so the fix is kept."""
        self.assertNotEqual(_error_signature(TYPE_ERR),
                            _error_signature(ATTR_ERR))

    def test_column_and_line_suffixes_normalised(self):
        a = _error_signature("src/map.py:12:3 TypeError: boom")
        b = _error_signature("src/map.py:98:7 TypeError: boom")
        self.assertEqual(a, b)

    def test_memory_addresses_normalised(self):
        a = _error_signature("ValueError: bad object at 0x7f0011223344")
        b = _error_signature("ValueError: bad object at 0xdeadbeefcafe")
        self.assertEqual(a, b)

    def test_absolute_paths_normalised(self):
        a = _error_signature(r'File "C:\tmp\aaa\map.py" TypeError: boom')
        b = _error_signature(r'File "C:\other\bbb\map.py" TypeError: boom')
        self.assertEqual(a, b)

    def test_empty_is_stable(self):
        self.assertEqual(_error_signature(""), "")

    def test_whitespace_only_differences_ignored(self):
        self.assertEqual(_error_signature("TypeError:  boom\n\n"),
                         _error_signature("TypeError: boom"))

    def test_signature_is_short_and_stable(self):
        sig = _error_signature(TYPE_ERR)
        self.assertEqual(len(sig), 12)
        self.assertEqual(sig, _error_signature(TYPE_ERR))


if __name__ == "__main__":
    unittest.main()

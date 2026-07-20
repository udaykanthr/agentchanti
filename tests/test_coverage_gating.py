"""Tests for task_requests_tests — the raw-task predicate that gates
unsolicited auto-generated coverage tests (and testing-guide force-includes)."""

import unittest

from agentchanti.language import task_requests_tests


class TestTaskRequestsTests(unittest.TestCase):

    def test_plain_feature_tasks_do_not_request_tests(self):
        for task in (
            "using bootstrap and react project create a simple homepage "
            "with responsive header, herbanner, large footer",
            "build a REST API with express",
            "create a landing page",
        ):
            self.assertFalse(task_requests_tests(task), task)

    def test_explicit_test_requests(self):
        for task in (
            "add unit tests for the auth module",
            "write a test case for login",
            "run pytest and fix failures",
            "set up vitest for the project",
            "add jest coverage reporting",
            "improve testing of the parser",
        ):
            self.assertTrue(task_requests_tests(task), task)

    def test_substring_false_positives_rejected(self):
        # \b boundaries: "latest" and "contest" contain the letters 'test'
        # but are not requests for tests.
        for task in (
            "use the latest bootstrap version",
            "build a coding contest leaderboard",
            "install the latest docs guide",
        ):
            self.assertFalse(task_requests_tests(task), task)

    def test_empty_and_none_safe(self):
        self.assertFalse(task_requests_tests(""))
        self.assertFalse(task_requests_tests(None))


if __name__ == "__main__":
    unittest.main()

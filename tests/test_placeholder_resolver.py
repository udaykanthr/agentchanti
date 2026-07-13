"""Tests for <placeholder> resolution in planned commands.

Regression for the run where an IntentAgent-enriched Django task
mentioned React incidentally (KB doc titles), and the substring table
with 'react' before 'django' resolved <project_name> to 'react-app' —
a directory nothing created, failing every CMD step.
"""

import unittest

from agentchanti.orchestrator.classification import resolve_cmd_placeholders


class TestPlaceholderResolution(unittest.TestCase):

    def test_django_wins_over_incidental_react_mention(self):
        task = ("create a django application with a responsive homepage; "
                "context mentions React Component Patterns docs")
        cmd = resolve_cmd_placeholders(
            "cd <project_name> && python manage.py check", task=task)
        self.assertEqual(cmd, "cd django-project && python manage.py check")

    def test_python_language_never_gets_js_name(self):
        # No framework named at all — language decides
        cmd = resolve_cmd_placeholders(
            "cd <project_name> && pytest", task="build a web thing",
            language="python")
        self.assertEqual(cmd, "cd python-app && pytest")

    def test_react_task_still_react(self):
        cmd = resolve_cmd_placeholders(
            "cd <app-name> && npm test", task="build a react dashboard")
        self.assertEqual(cmd, "cd react-app && npm test")

    def test_word_boundary_no_substring_hijack(self):
        # 'nextcloud' must not match 'next'
        cmd = resolve_cmd_placeholders(
            "cd <project_name> && ls", task="sync files to nextcloud with vue")
        self.assertEqual(cmd, "cd vue-app && ls")

    def test_explicit_name_beats_framework_table(self):
        cmd = resolve_cmd_placeholders(
            "cd <project_name> && npm test",
            task='create a react app named "storefront"')
        self.assertEqual(cmd, "cd storefront && npm test")

    def test_no_placeholders_untouched(self):
        cmd = "python -c \"print(1 < 2)\""
        self.assertEqual(resolve_cmd_placeholders(cmd, task="django app"), cmd)


if __name__ == "__main__":
    unittest.main()

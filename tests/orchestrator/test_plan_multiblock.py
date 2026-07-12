"""Multi-file content blocks: every fenced block reaches inline_code.

A live plan emitted nine templates in one step, closing each block with
---file-content-end--- but opening the next with a bare ``` fence (no
repeated `content:` line). The parser captured only the first block and
silently leaked the other eight into the step description — the files
were never written, and the app failed at runtime.
"""

import unittest

from agentchanti.orchestrator.plan_step import parse_structured_plan


PLAN = """\
==PLAN==

--STEP 1.1 [CODE] depends:none
Create all templates
target: accounts/templates/accounts/base.html, accounts/templates/accounts/home.html, accounts/templates/accounts/login.html
content:
```html
<!-- accounts/templates/accounts/base.html -->
<!doctype html>
<html><body>{% block content %}{% endblock %}</body></html>
---file-content-end---

```html
<!-- accounts/templates/accounts/home.html -->
{% extends "accounts/base.html" %}
{% block content %}<h1>Home</h1>{% endblock %}
---file-content-end---

```html
<!-- accounts/templates/accounts/login.html -->
{% extends "accounts/base.html" %}
{% block content %}<h2>Login</h2>{% endblock %}
```
---file-content-end---

==END==
"""


class TestTargetPathNormalization(unittest.TestCase):
    """Planner-emitted doubled backslashes must not survive parsing."""

    def test_doubled_backslash_targets_normalized(self):
        plan = (
            "==PLAN==\n\n"
            "--STEP 1.1 [CODE] depends:none\n"
            "Create template\n"
            "target: main\\\\templates\\\\main\\\\base.html, spacious_site\\\\settings.py\n"
            "content:\n"
            "```html\n<p>x</p>\n```\n"
            "---file-content-end---\n\n"
            "==END==\n"
        )
        from agentchanti.orchestrator.plan_step import parse_structured_plan
        steps = parse_structured_plan(plan)
        self.assertEqual(steps[0].target_files,
                         ["main/templates/main/base.html",
                          "spacious_site/settings.py"])
        # inline code keys derive from targets — no backslashes anywhere
        for key in steps[0].inline_code:
            self.assertNotIn("\\", key)
            self.assertNotIn("//", key)


class TestMultiBlockContentCapture(unittest.TestCase):

    def test_all_blocks_captured(self):
        steps = parse_structured_plan(PLAN)
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertEqual(len(step.inline_code), 3,
                         f"captured: {list(step.inline_code)}")
        joined = " ".join(step.inline_code)
        for name in ("base.html", "home.html", "login.html"):
            self.assertIn(name, joined)
        # Content routed to the right files
        home = next(v for k, v in step.inline_code.items()
                    if "home.html" in k)
        self.assertIn("<h1>Home</h1>", home)
        self.assertNotIn("Login", home)

    def test_prose_fences_not_captured_when_targets_full(self):
        plan = PLAN.replace(
            "==END==",
            "--STEP 1.2 [CMD] depends:1.1\n"
            "Run it. Example output:\n"
            "```\nsome console output\n```\n"
            "> echo done\n\n==END==")
        steps = parse_structured_plan(plan)
        # The CMD step has no target files, so the prose fence is ignored
        self.assertEqual(steps[1].inline_code, {})


if __name__ == "__main__":
    unittest.main()

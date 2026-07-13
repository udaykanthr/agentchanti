"""Tests for the deterministic Django wiring lints.

Both bug classes here killed real benchmark runs: run 1 died on
{% static %} without {% load static %}, runs 1 and 2 both died on
redirect('name') against a namespaced urls.py.
"""

import unittest
from unittest.mock import MagicMock, patch

from agentchanti.orchestrator.django_lint import check_django_project


URLS_NAMESPACED = """\
from django.urls import path
from .views import home, dashboard, login_view

app_name = 'sitepages'

urlpatterns = [
    path('', home, name='home'),
    path('dashboard/', dashboard, name='dashboard'),
    path('login/', login_view, name='login'),
]
"""

URLS_PLAIN = """\
from django.urls import path
from .views import health

urlpatterns = [
    path('health/', health, name='health'),
]
"""


class TestNamespaceLint(unittest.TestCase):

    def test_run2_repro_bare_redirect_flagged(self):
        views = "def home(request):\n    return redirect('dashboard')\n"
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/views.py": views,
        })
        self.assertEqual(len(errors), 1)
        self.assertIn("sitepages/views.py", errors[0])
        self.assertIn("'sitepages:dashboard'", errors[0])

    def test_namespaced_redirect_clean(self):
        views = "def home(request):\n    return redirect('sitepages:dashboard')\n"
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/views.py": views,
        })
        self.assertEqual(errors, [])

    def test_plain_route_not_flagged(self):
        # 'health' is reachable unnamespaced — bare reverse is correct
        views = "def go(request):\n    return redirect('health')\n"
        errors = check_django_project({
            "api/urls.py": URLS_PLAIN,
            "api/views.py": views,
        })
        self.assertEqual(errors, [])

    def test_unknown_name_not_flagged(self):
        # Names we can't prove wrong (third-party, admin) stay silent
        views = "def go(request):\n    return redirect('admin:index')\n" \
                "def go2(request):\n    return redirect('password_reset')\n"
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/views.py": views,
        })
        self.assertEqual(errors, [])

    def test_url_path_literal_not_flagged(self):
        views = "def go(request):\n    return redirect('/dashboard/')\n"
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/views.py": views,
        })
        self.assertEqual(errors, [])

    def test_template_url_tag_flagged(self):
        tpl = "<a href=\"{% url 'dashboard' %}\">Go</a>"
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/templates/sitepages/base.html": tpl,
        })
        self.assertEqual(len(errors), 1)
        self.assertIn("'sitepages:dashboard'", errors[0])

    def test_template_namespaced_url_clean(self):
        tpl = "<a href=\"{% url 'sitepages:dashboard' %}\">Go</a>"
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/templates/sitepages/base.html": tpl,
        })
        self.assertEqual(errors, [])

    def test_reverse_and_reverse_lazy_flagged(self):
        code = ("LOGIN = reverse_lazy('login')\n"
                "def t(self):\n    self.client.get(reverse('home'))\n")
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/tests.py": code,
        })
        self.assertEqual(len(errors), 2)


class TestLoadStaticLint(unittest.TestCase):

    def test_run1_repro_static_without_load(self):
        tpl = ('<link href="{% static \'css/styles.css\' %}" '
               'rel="stylesheet">')
        errors = check_django_project({"main/templates/base.html": tpl})
        self.assertEqual(len(errors), 1)
        self.assertIn("{% load static %}", errors[0])

    def test_static_with_load_clean(self):
        tpl = ("{% load static %}\n"
               '<link href="{% static \'css/styles.css\' %}">')
        errors = check_django_project({"main/templates/base.html": tpl})
        self.assertEqual(errors, [])

    def test_load_with_multiple_libraries(self):
        tpl = ("{% load i18n static %}\n"
               '<img src="{% static \'logo.png\' %}">')
        errors = check_django_project({"t.html": tpl})
        self.assertEqual(errors, [])


class TestLintScope(unittest.TestCase):

    def test_non_django_project_is_clean(self):
        errors = check_django_project({
            "src/App.jsx": "export default function App() {}",
            "index.html": "<div id=\"root\"></div>",
        })
        self.assertEqual(errors, [])

    def test_cmd_output_entries_skipped(self):
        errors = check_django_project({
            "sitepages/urls.py": URLS_NAMESPACED,
            "_cmd_output/step_1.txt": "redirect('dashboard')",
        })
        self.assertEqual(errors, [])

    def test_empty_files_dict(self):
        self.assertEqual(check_django_project({}), [])


class TestLintGateIntegration(unittest.TestCase):
    """A 'successful' step is gated when the project carries lint bugs."""

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "loop says done"))
    def test_loop_success_gated_by_lint(self, mock_loop):
        from agentchanti.orchestrator.step_handlers import _handle_code_step
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        coder = MagicMock()
        coder.llm_client.supports_tools.return_value = True
        coder.escalation_client = None
        memory = MagicMock()
        memory.summary.return_value = "files"
        memory._scaffolded_subproject = None
        memory.all_files.return_value = {
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/views.py":
                "def home(r):\n    return redirect('dashboard')\n",
        }
        success, error = _handle_code_step(
            "step", coder, MagicMock(), MagicMock(), "task",
            memory, MagicMock(), 0, cfg=cfg)
        self.assertFalse(success)
        self.assertIn("sitepages:dashboard", error)

    @patch("agentchanti.orchestrator.agent_loop.run_agent_loop",
           return_value=(True, "loop says done"))
    def test_clean_project_passes(self, mock_loop):
        from agentchanti.orchestrator.step_handlers import _handle_code_step
        cfg = MagicMock()
        cfg.AGENT_LOOP = True
        cfg.AGENT_LOOP_MAX_TURNS = 8
        coder = MagicMock()
        coder.llm_client.supports_tools.return_value = True
        coder.escalation_client = None
        memory = MagicMock()
        memory.summary.return_value = "files"
        memory._scaffolded_subproject = None
        memory.all_files.return_value = {
            "sitepages/urls.py": URLS_NAMESPACED,
            "sitepages/views.py":
                "def home(r):\n    return redirect('sitepages:dashboard')\n",
        }
        success, _ = _handle_code_step(
            "step", coder, MagicMock(), MagicMock(), "task",
            memory, MagicMock(), 0, cfg=cfg)
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()

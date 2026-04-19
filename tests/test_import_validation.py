"""
Tests for _validate_import_paths and _extract_npm_package_name — catches
broken relative imports and detects package imports for auto-install.
"""
import pytest
from unittest.mock import MagicMock

from agentchanti.orchestrator.step_handlers import (
    _validate_import_paths,
    _extract_npm_package_name,
)


def _make_memory(files: dict[str, str]) -> MagicMock:
    """Create a mock FileMemory with the given files."""
    mem = MagicMock()
    mem.all_files.return_value = dict(files)
    return mem


# ── JS/JSX import tests ──────────────────────────────────────


def test_valid_import_no_errors():
    """Valid imports should produce no errors."""
    memory = _make_memory({
        "my-app/src/App.jsx": "export default function App() {}",
    })
    files = {
        "my-app/src/main.jsx": "import App from './App';\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_wrong_nested_path_detected():
    """import './App/App' when App.jsx is at ./App.jsx should error."""
    memory = _make_memory({
        "my-app/src/App.jsx": "export default function App() {}",
    })
    files = {
        "my-app/src/main.jsx": "import App from './App/App';\n",
    }
    result = _validate_import_paths(files, memory)
    assert "IMPORT ERRORS" in result
    assert "./App/App" in result


def test_suggests_correct_path():
    """Error message should suggest the correct import path."""
    memory = _make_memory({
        "my-app/src/App.jsx": "export default function App() {}",
    })
    files = {
        "my-app/src/main.jsx": "import App from './App/App';\n",
    }
    result = _validate_import_paths(files, memory)
    assert "./App" in result


def test_import_with_extension_resolves():
    """import './App.jsx' should resolve to App.jsx."""
    memory = _make_memory({
        "my-app/src/App.jsx": "export default function App() {}",
    })
    files = {
        "my-app/src/main.jsx": "import App from './App.jsx';\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_import_from_subdirectory():
    """import './components/Header' should resolve correctly."""
    memory = _make_memory({
        "my-app/src/components/Header.jsx": "export default function Header() {}",
    })
    files = {
        "my-app/src/App.jsx": "import Header from './components/Header';\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_import_parent_directory():
    """import '../App' from a subdirectory should resolve."""
    memory = _make_memory({
        "my-app/src/App.jsx": "export default function App() {}",
    })
    files = {
        "my-app/src/pages/Home.jsx": "import App from '../App';\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_import_index_file():
    """import './components' should resolve to components/index.jsx."""
    memory = _make_memory({
        "my-app/src/components/index.jsx": "export { Header } from './Header';",
    })
    files = {
        "my-app/src/App.jsx": "import { Header } from './components';\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_css_import_not_validated():
    """CSS imports like './index.css' should not trigger JS extension checks."""
    memory = _make_memory({
        "my-app/src/index.css": "@import 'tailwindcss';",
    })
    files = {
        "my-app/src/main.jsx": "import './index.css';\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_newly_generated_files_resolve():
    """Files in the current batch (not yet in memory) should resolve."""
    memory = _make_memory({})  # empty memory
    files = {
        "my-app/src/App.jsx": "export default function App() {}",
        "my-app/src/main.jsx": "import App from './App';\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_multiple_broken_imports():
    """Multiple broken imports in one file should all be reported."""
    memory = _make_memory({
        "my-app/src/App.jsx": "code",
    })
    files = {
        "my-app/src/main.jsx": (
            "import App from './App/App';\n"
            "import Foo from './Foo/Bar';\n"
        ),
    }
    result = _validate_import_paths(files, memory)
    assert "IMPORT ERRORS" in result
    assert "./App/App" in result
    assert "./Foo/Bar" in result


def test_non_relative_imports_ignored():
    """Package imports like 'react' should not be validated."""
    memory = _make_memory({})
    files = {
        "my-app/src/main.jsx": (
            "import React from 'react';\n"
            "import { BrowserRouter } from 'react-router-dom';\n"
        ),
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_export_from_validated():
    """export { X } from './path' should also be validated."""
    memory = _make_memory({})
    files = {
        "my-app/src/index.jsx": "export { Header } from './Header/Header';\n",
    }
    result = _validate_import_paths(files, memory)
    assert "IMPORT ERRORS" in result
    assert "./Header/Header" in result


# ── Python import tests ──────────────────────────────────────


def test_python_valid_relative_import():
    """Valid Python relative import should produce no errors."""
    memory = _make_memory({
        "src/utils.py": "def helper(): pass",
    })
    files = {
        "src/main.py": "from .utils import helper\n",
    }
    result = _validate_import_paths(files, memory)
    assert result == ""


def test_python_broken_relative_import():
    """Broken Python relative import should be detected."""
    memory = _make_memory({})
    files = {
        "src/main.py": "from .nonexistent import something\n",
    }
    result = _validate_import_paths(files, memory)
    assert "IMPORT ERRORS" in result
    assert ".nonexistent" in result


# ── Package name extraction tests ────────────────────────────


def test_extract_scoped_package():
    """@heroicons/react/outline → @heroicons/react"""
    assert _extract_npm_package_name("@heroicons/react/outline") == "@heroicons/react"


def test_extract_scoped_package_no_subpath():
    """@testing-library/react → @testing-library/react"""
    assert _extract_npm_package_name("@testing-library/react") == "@testing-library/react"


def test_extract_regular_package():
    """react-router-dom → react-router-dom"""
    assert _extract_npm_package_name("react-router-dom") == "react-router-dom"


def test_extract_package_with_subpath():
    """lucide-react/icons → lucide-react"""
    assert _extract_npm_package_name("lucide-react/icons") == "lucide-react"


def test_extract_simple_package():
    """react → react"""
    assert _extract_npm_package_name("react") == "react"


def test_relative_import_returns_none():
    """Relative imports should return None."""
    assert _extract_npm_package_name("./App") is None
    assert _extract_npm_package_name("../utils") is None


def test_node_builtin_returns_none():
    """Node built-in modules should return None."""
    assert _extract_npm_package_name("fs") is None
    assert _extract_npm_package_name("path") is None
    assert _extract_npm_package_name("crypto") is None
    assert _extract_npm_package_name("http") is None


def test_node_protocol_returns_none():
    """node: protocol imports should return None."""
    assert _extract_npm_package_name("node:fs") is None
    assert _extract_npm_package_name("node:path") is None


def test_empty_returns_none():
    assert _extract_npm_package_name("") is None
    assert _extract_npm_package_name(None) is None


def test_deeply_nested_scoped_path():
    """@mui/material/styles/createTheme → @mui/material"""
    assert _extract_npm_package_name("@mui/material/styles/createTheme") == "@mui/material"

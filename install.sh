#!/bin/bash

set -e

# Check python3 is available
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python 3.10+ from https://python.org"
    exit 1
fi

# Check minimum Python version (3.10)
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null || {
    echo "Python 3.10+ is required. Found: $(python3 --version)"
    exit 1
}

echo "Setting up virtual environment..."
python3 -m venv .venv

echo "Installing agentchanti..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install -e .

echo ""
echo "Installation successful!"
echo ""
echo "Activate the environment before use:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  agentchanti \"your task\" --provider ollama --model deepseek-coder-v2:16b"

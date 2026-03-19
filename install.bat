@echo off

REM Check python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo python could not be found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo Setting up virtual environment...
python -m venv .venv

echo Installing agentchanti...
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install -e .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation successful!
    echo.
    echo Activate the environment before use:
    echo   .venv\Scripts\activate
    echo.
    echo Then run:
    echo   agentchanti "your task" --provider ollama --model deepseek-coder-v2:16b
) else (
    echo Installation failed.
)
pause

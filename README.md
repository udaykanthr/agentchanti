
```text
     _                    _      ____ _                 _   _ 
    / \   __ _  ___ _ __ | |_   / ___| |__   __ _ _ __ | |_(_)
   / _ \ / _` |/ _ \ '_ \| __| | |   | '_ \ / _` | '_ \| __| |
  / ___ \ (_| |  __/ | | | |_  | |___| | | | (_| | | | | |_| |
 /_/   \_\__, |\___|_| |_|\__|  \____|_| |_|\__,_|_| |_|\__|_|
         |___/                                                 
                    ━━  L o c a l   C o d e r  ━━              
```

**AgentChanti** is a modular, multi-agent coding assistant designed to run entirely offline with local LLMs. It uses a specialized team of AI agents (Planner, Coder, Reviewer, Tester) to build, refine, and verify software automatically.

---

## 🚀 Features

-   **100% Local**: Works with Ollama and LM Studio. No API keys required.
-   **Multi-Agent Workflow**:
    -   **Planner**: Breaks down complex tasks into actionable steps.
    -   **Coder**: Writes clean, functional code.
    -   **Reviewer**: Analyzes code for bugs and improvements.
    -   **Tester**: Generates and runs unit tests to ensure reliability.
-   **Live Progress**: Beautiful CLI display with step-by-step progress and token tracking.
-   **Robust**: Auto-fixes code based on test failures and review feedback.

---

## 📦 Installation

### Option 1: Automatic Installer (Recommended)

**Linux / macOS:**
```bash
./install.sh
```

**Windows:**
```cmd
install.bat
```

### Option 2: Manual Installation
```bash
pip install -e .
```

---

## 🤖 LLM Setup

AgentChanti requires a local LLM server running. You can use either **Ollama** or **LM Studio**.

### Option A: Ollama (Recommended)
1.  **Download & Install**: [ollama.com](https://ollama.com)
2.  **Pull a Model**:
    ```bash
    ollama pull qwen3-coder:480b-cloud  # Recommended for coding
    # OR
    ollama pull llama3
    ```
3.  **Ensure Server is Running**:
    ```bash
    ollama serve
    ```
    *Default URL: `http://localhost:11434`*

### Option B: LM Studio
1.  **Download & Install**: [lmstudio.ai](https://lmstudio.ai)
2.  **Load a Model**: Search for and load a coding model (e.g., `DeepSeek Coder`, `Llama 3`, `Mistral`).
3.  **Start Server**:
    -   Go to the "Local Server" tab.
    -   Start the server on port `1234`.
    -   Ensure "Cross-Origin-Resource-Sharing (CORS)" is enabled if needed.
    *Default URL: `http://localhost:1234/v1`*

---

## 💻 Usage

Once installed, you can use the `agentchanti` command from anywhere in your terminal.

```bash
agentchanti "<Task Description>" [options]
```

### Examples

**1. Basic Usage (Ollama)**
```bash
agentchanti "Create a python script to solve the knapsack problem"
```

**2. Using LM Studio**
```bash
agentchanti "Build a simple snake game using pygame" --provider lm_studio
```

**3.  Specifying a Model**
```bash
agentchanti "Resize all images in a folder" --model llama3
```

---

## 📂 Project Structure

-   `multi_agent_coder/`: Core package source.
    -   `agents/`: Definitions for Planner, Coder, Reviewer, and Tester agents.
    -   `llm/`: Clients for Ollama and LM Studio.
    -   `orchestrator.py`: Main logic tying everything together.
-   `setup.py`: Package configuration.
-   `install.sh` / `install.bat`: Installation scripts.

---

## ✅ Verification

The system includes built-in verification mock tests to ensure agent orchestration works correctly.

```bash
python3 -m unittest tests/test_flow.py
```
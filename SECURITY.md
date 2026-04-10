# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AgentChanti, please report it
**privately** rather than opening a public GitHub issue.

Use one of these channels:

- **GitHub Security Advisories**: open a draft advisory at
  <https://github.com/udaykanth/agentchanti/security/advisories/new>
- **Email**: contact the maintainer listed in `setup.py` /
  `pyproject.toml`

Please include:

- A description of the issue and its impact
- Steps to reproduce (or a proof-of-concept)
- The affected version(s) / commit SHA
- Any suggested mitigation

You should expect an initial response within a few business days. We
will work with you on a coordinated disclosure timeline before any
public announcement.

## Supported Versions

AgentChanti is currently in pre-release (`0.x`). Only the latest tagged
release on `main` receives security fixes. Once we cut `1.0`, this
section will list the supported version range.

## Scope

In-scope:

- The `agentchanti` CLI and `agentchanti` Python package
- Default configuration files shipped in this repository
- The KB ingestion / code-graph subsystem

Out of scope:

- Vulnerabilities in third-party LLM providers (Ollama, LM Studio,
  OpenAI, Gemini, Anthropic) — please report those upstream
- Bugs in user-supplied prompts or generated code

## Hardening Notes for Operators

AgentChanti executes shell commands and writes files on the host
machine as part of normal operation. When running it in shared or
production environments:

- Run the agent in an isolated working directory or container
- Review the generated plan before approving execution (`--auto` skips
  this check — use it only on trusted tasks)
- Treat any LLM output as untrusted input until reviewed

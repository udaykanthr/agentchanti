# Changelog

All notable user-facing changes to `agentchanti` land here. This
project follows [Semantic Versioning](https://semver.org): breaking
changes bump the minor (until 1.0), bugfixes bump the patch.

## 0.1.1 — 2026-04-19

### Fixed

- PyPI project metadata now points at the correct repository. The
  `Homepage`, `Repository`, and `Issues` URLs were shipped with a
  typo (`udaykanth` missing the trailing `r`) in 0.1.0 and all
  404'd. (#19, #20)
- `SECURITY.md` advisory link is reachable again. Same typo as
  above — private vulnerability reporting was silently broken on
  0.1.0. (#19)

### Removed

- Stale "package will be on PyPI once the first tagged release is
  cut" note in the README installation section.
- Reference to the deleted `setup.py` file in `SECURITY.md`.

## 0.1.0 — 2026-04-19

Initial public release on PyPI.

### Added

- `agentchanti` CLI and Python library for multi-agent AI coding
  tasks (Planner → Coder → Reviewer → Tester pipeline).
- Built-in RAG: tree-sitter code graph across 11 languages, local
  SQLite vector store, global knowledge base, error dictionary.
- Support for local LLMs (Ollama, LM Studio) and cloud providers
  (OpenAI, Gemini, Anthropic).
- Structured `PlanStep` format, KB-first command execution, plan-
  aware context injection, step caching, checkpoint/resume.
- Plugin system (`StepPlugin`) for custom pipeline steps.
- GitHub Actions CI (test matrix on ubuntu + windows × py3.10-3.12,
  ruff lint) and release workflow (PyPI trusted publishing via
  OIDC on tag push).

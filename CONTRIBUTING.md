# Contributing to AgentChanti

Thanks for your interest in contributing! This document covers the
day-to-day workflow for proposing changes.

## Development Setup

```bash
git clone https://github.com/udaykanthr/agentchanti.git
cd agentchanti
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

For convenience, the repository ships `install.sh` (POSIX) and
`install.bat` (Windows) which run the same steps.

## Branch & PR Workflow

`main` is protected — all changes land via pull request.

1. Create a topic branch off the latest `main`:
   ```bash
   git switch main && git pull
   git switch -c <type>/<short-description>
   ```
   Use a conventional prefix: `feat/`, `fix/`, `chore/`, `docs/`,
   `refactor/`, `test/`, `ci/`.
2. Make focused commits. Prefer many small commits over one large
   commit; the commit message should explain *why*, not *what*.
3. Push your branch and open a PR against `main`. Fill in the PR
   description: what changed, why, how it was tested.
4. CI must be green before merging:
   - `test` job (pytest on Linux + Windows, Python 3.10 / 3.11 / 3.12)
   - `lint` job (ruff, errors-only ruleset)
5. At least one approving review is required. Squash-and-merge is the
   default to keep `main` linear.

## Running Tests Locally

```bash
python -m pytest tests/ -v
```

To match the CI lint check:

```bash
pip install ruff
ruff check --select=E9,F63,F7,F82 agentchanti/ tests/
```

## Commit Message Style

Follow Conventional Commits:

```
<type>: <imperative summary, ~70 chars>

<optional body — wrap at 72 chars, explain motivation and any
trade-offs the reviewer should know about>
```

Common types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`,
`ci`, `perf`.

## Code Style

- Python 3.10+ syntax is fine (`match`, `X | None`, etc.)
- Run `ruff check --select=E9,F63,F7,F82` before pushing — these are
  the rules CI enforces today and they catch real bugs
- New code should include tests. Bug fixes should include a
  regression test that fails before the fix.

## Reporting Bugs

Open a GitHub issue with:

- AgentChanti version (`agentchanti --version` or commit SHA)
- Python version and OS
- Minimal reproduction (the task description, provider, model)
- Expected vs. actual behavior
- Relevant log output (with secrets redacted)

For security issues, see [`SECURITY.md`](SECURITY.md) instead.

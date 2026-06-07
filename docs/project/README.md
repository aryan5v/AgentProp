# Project meta

Governance, history, and contributor docs.

| File | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Map for coding agents working in the repo |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, quality gates, artifact policy |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

Related:

- [Security policy](../../SECURITY.md) — GitHub security policy
- [Environment template](../../dev/configs/.env.example) — copy to `.env` locally

## Root files (tooling requirements)

| File | Why it stays at the repo root |
| --- | --- |
| `README.md` | GitHub and PyPI entry point |
| `pyproject.toml` / `uv.lock` | Python packaging (`uv` expects the lockfile here) |
| `LICENSE` | License discovery |
| `Makefile` | Thin wrapper → [dev/scripts/Makefile](../../dev/scripts/Makefile) |
| `CONTRIBUTING.md` | Short pointer for GitHub’s contributing link |
| `AGENTS.md` | Short pointer for coding-agent tools |

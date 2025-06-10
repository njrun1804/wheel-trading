# Codex Usage Guide

This directory contains configuration files and guidelines for using OpenAI Codex with the Unity Wheel Trading Bot repository.

- `config.yaml` sets project conventions like absolute imports, error handling patterns, and testing requirements.
- Codex-generated code must conform to `pyproject.toml` settings and the guard-rails documented in `CONTRIBUTING.md`.
- Run `ruff format . && black .` followed by `ruff check --fix .` and `mypy --strict` before committing any Codex changes.
- Integration tests under `tests/integration/` will skip automatically if credentials are missing.

See [CODEX_GUIDE.md](../CODEX_GUIDE.md) for full instructions on collaborating with Codex.

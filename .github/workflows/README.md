# GitHub Actions Workflows

## Simple CI for Private Bot

This repository uses a single, simple CI workflow appropriate for a private recommendation bot.

### ci.yml
- Runs tests with pytest
- Checks for accidentally committed secrets
- Scans dependencies for vulnerabilities
- That's it!

## Why So Simple?

This is a private, locally-run recommendation bot that:
- Never executes trades
- Is used by one person
- Only fetches data from FRED & Databento

We don't need:
- ❌ CodeQL (not a web service)
- ❌ Complex security scanning (no attack surface)
- ❌ Performance tracking (can profile locally)
- ❌ Release workflows (not publishing)
- ❌ Multiple test matrices (one environment)

## Running Tests Locally

```bash
# Same as CI
poetry install
poetry run pytest -v
```

That's all you need!

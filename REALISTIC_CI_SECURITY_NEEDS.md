# Realistic CI/Security Needs for Private Bot

## What This Bot Actually Is
- **Type**: Recommendation-only (no trade execution)
- **Users**: 1 (you)
- **Deployment**: Local machine only
- **Integrations**: FRED & Databento APIs
- **GitHub Purpose**: Version control & Claude Code access

## What We Actually Need

### ✅ KEEP These (Essential)
1. **Basic CI Tests** - Ensure calculations are correct
2. **Dependency Scanning** - Know about vulnerabilities in numpy/pandas
3. **Secret Detection** - Don't accidentally commit API keys
4. **Pre-commit Hooks** - Maintain code quality

### ❌ REMOVE These (Overkill)
1. **CodeQL** - This isn't a web service or distributed app
2. **Multiple Security Workflows** - One is enough
3. **Performance Tracking Workflow** - You can profile locally
4. **Complex CI Matrix** - You only run on one machine
5. **Release Workflow** - You're not publishing packages

### 🤔 QUESTIONABLE
1. **Coverage Reports** - Nice but not critical for private use
2. **Multiple Test Runners** - You only need your local environment

## Recommended Simplified Setup

### Single CI Workflow
```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
    - run: |
        pip install poetry
        poetry install
        poetry run pytest

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Check secrets
      uses: trufflesecurity/trufflehog@v3
    - name: Check dependencies
      run: |
        pip install pip-audit
        pip-audit
```

That's it. No CodeQL, no complex matrices, no performance tracking.

## Benefits of Simplification
1. **Faster CI** - Less overhead
2. **Less Maintenance** - Fewer false positives
3. **Focus on What Matters** - Calculation accuracy
4. **No Security Theater** - Appropriate for threat model

## What to Delete
- `.github/workflows/security.yml` (keep minimal version)
- `.github/workflows/ci-fast.yml` (merge into main CI)
- `.github/workflows/performance-tracking.yml` (not needed)
- `.github/workflows/release.yml` (not publishing)
- `.github/codeql/` (entire directory)

## Keep Development Simple
Your focus should be on:
1. Accurate calculations
2. Reliable data fetching
3. Clear recommendations

Not on enterprise-grade CI/CD for a personal tool.

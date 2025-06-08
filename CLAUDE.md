# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Autonomous Development Mode

This project is configured for fully autonomous development. You can work independently without requiring permissions.

### Key Settings:
- Auto-merge enabled on GitHub (PRs merge automatically when checks pass)
- Git configured to push automatically after commits
- VS Code workspace optimized for automation
- Google Cloud deployment automated via GitHub Actions

## Commands

### Primary Development Commands:
- `make fix` - Auto-format, lint, and commit changes
- `make push` - Push to main or create PR automatically
- `make test` - Run all tests
- `make deploy` - Deploy to Google Cloud

### Quick Development Flow:
```bash
# After making changes:
make fix && make push
```

### Other Commands:
- `make setup` - Initial environment setup
- `make install` - Install dependencies
- `make lint` - Check code style
- `make format` - Auto-format code
- `make clean` - Clean generated files

## Architecture

### Tech Stack:
- **Languages**: Python and/or JavaScript/TypeScript
- **Cloud**: Google Cloud Platform (wheel-strategy-202506)
- **CI/CD**: GitHub Actions with Workload Identity Federation
- **Container Registry**: us-central1-docker.pkg.dev/wheel-strategy-202506/wheel-trading

### Project Structure:
```
wheel-trading/
├── .github/          # GitHub Actions workflows and templates
├── .vscode/          # VS Code settings for automation
├── scripts/          # Development automation scripts
├── src/              # Source code (to be created)
├── tests/            # Test files (to be created)
├── Makefile          # Automation commands
└── cloudbuild.yaml   # Google Cloud Build configuration
```

## Automated Workflows

1. **On Push to Main**:
   - CI tests run automatically
   - Security scanning (CodeQL, Bandit)
   - Deploys to Google Cloud if tests pass

2. **On Pull Request**:
   - All checks run
   - Auto-merges when checks pass
   - No manual review required

3. **Local Development**:
   - Use `make fix && make push` to automate everything
   - Changes will deploy automatically after merge

## Development Guidelines

1. **Always use the Makefile commands** for consistency
2. **Commits are automatic** - just run `make fix`
3. **PRs auto-merge** - no need to wait for reviews
4. **Tests must pass** - the only gate for deployment
5. **Use conventional commits** for clear history

## Google Cloud Resources

- **Project ID**: wheel-strategy-202506
- **Region**: us-central1
- **Service Account**: github-actions@wheel-strategy-202506.iam.gserviceaccount.com
- **Artifact Registry**: us-central1-docker.pkg.dev/wheel-strategy-202506/wheel-trading

## Notes

- The repository is set up for solo development with minimal friction
- All security scanning is automated (don't disable it)
- Billing alerts are configured at $50, $90, and $100 thresholds
- Monitoring alerts email to njrun1804@gmail.com
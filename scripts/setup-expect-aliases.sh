#!/bin/bash
# Add these aliases to your ~/.zshrc or ~/.bashrc

# Make expect scripts executable
chmod +x scripts/auto-*.exp

# Claude Code auto-approval for safe operations
alias claude-auto='expect scripts/auto-approve-claude.exp'

# Git operations with auto-handling
alias git-merge-auto='expect scripts/auto-git-operations.exp merge'
alias git-rebase-auto='expect scripts/auto-git-operations.exp rebase'
alias git-commit-auto='expect scripts/auto-git-operations.exp commit'

# NPM with auto-handling
alias npm-install-auto='expect scripts/auto-npm-install.exp'

# Python virtual env activation (no prompts)
alias venv-auto='source venv/bin/activate 2>/dev/null || python -m venv venv && source venv/bin/activate'

echo "Expect automation aliases configured!"
echo "Usage examples:"
echo "  claude-auto 'search for WheelStrategy'"
echo "  git-merge-auto"
echo "  npm-install-auto"
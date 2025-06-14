#!/bin/bash
# WezTerm initialization script for Wheel Trading project
# This runs automatically when WezTerm starts in this directory

# Source expect aliases
if [ -f scripts/setup-expect-aliases.sh ]; then
    source scripts/setup-expect-aliases.sh
fi

# Claude-specific aliases with auto-approval
alias claude-search='expect scripts/auto-approve-claude.exp "search"'
alias claude-read='expect scripts/auto-approve-claude.exp "read"'
alias claude-test='expect scripts/auto-approve-claude.exp "run tests"'

# Project-specific Claude commands
alias wheel-analyze='claude-auto "analyze trading performance"'
alias wheel-optimize='claude-auto "optimize wheel strategy"'
alias wheel-backtest='claude-auto "run backtests"'

# Git workflow with expect
alias gpush='git add . && expect scripts/auto-git-operations.exp commit "Update" && git push'
alias gcommit='expect scripts/auto-git-operations.exp commit'

# Auto-activate Python environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Set up environment for hardware acceleration
export CLAUDE_HARDWARE_ACCEL=1
export USE_TURBO_MODE=1

# Display initialization message
echo "🚀 Wheel Trading Environment Initialized"
echo "📊 Hardware acceleration: ENABLED"
echo "🤖 Claude auto-approval: CONFIGURED"
echo ""
echo "Quick commands:"
echo "  claude-search 'pattern'  - Auto-approved search"
echo "  claude-read 'file'       - Auto-approved file read"
echo "  wheel-analyze            - Trading analysis"
echo "  wheel-optimize           - Strategy optimization"
echo ""
echo "Git shortcuts:"
echo "  gpush     - Add, commit, push with auto-handling"
echo "  gcommit   - Commit with auto-handling"
echo ""
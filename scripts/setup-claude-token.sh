#!/bin/bash

echo "Claude VS Code Extension - GitHub Token Setup"
echo "============================================"
echo ""
echo "This script will help you set up your GitHub token for the Claude VS Code extension."
echo ""

# Check if token is already set
if [ -n "$GITHUB_TOKEN" ]; then
    echo "✓ GitHub token is already set in your environment."
    echo "  Token preview: ${GITHUB_TOKEN:0:10}..."
else
    echo "⚠️  GitHub token is not set in your environment."
    echo ""
    echo "To set it up:"
    echo ""
    echo "1. Create a GitHub personal access token with 'repo' scope:"
    echo "   https://github.com/settings/tokens/new"
    echo ""
    echo "2. Add to your shell profile (~/.zshrc or ~/.bash_profile):"
    echo "   export GITHUB_TOKEN='your_token_here'"
    echo ""
    echo "3. Reload your shell configuration:"
    echo "   source ~/.zshrc"
    echo ""
fi

echo ""
echo "VS Code Configuration:"
echo "====================="
echo "✓ Claude extension (anthropic.claude-code) is installed"
echo "✓ VS Code settings.json is configured to use \${env:GITHUB_TOKEN}"
echo ""
echo "Next steps:"
echo "1. Ensure GITHUB_TOKEN is set in your environment"
echo "2. Restart VS Code to pick up the environment variable"
echo "3. Use Command Palette (Cmd+Shift+P) > 'Claude: Authenticate' if needed"
echo ""

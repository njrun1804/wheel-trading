#!/bin/bash
# Weekly maintenance script

echo "🧹 Running maintenance..."

# Clean package caches
brew cleanup 2>/dev/null || echo "Homebrew not installed"
pip cache purge 2>/dev/null || echo "No pip cache to clean"
npm cache clean --force 2>/dev/null || echo "No npm cache to clean"

# Clean VS Code workspace storage
rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage/*

# Optimize git
git gc --aggressive --prune=now

echo "✅ Maintenance complete!"

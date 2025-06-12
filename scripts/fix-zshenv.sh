#!/bin/bash

echo "Fixing .zshenv ulimit errors..."

# Create backup
cp ~/.zshenv ~/.zshenv.backup

# Fix the issues
sed -i '' -e 's/ulimit -n 524288/ulimit -n 10240/' ~/.zshenv
sed -i '' -e 's/^ulimit -m unlimited/# ulimit -m unlimited/' ~/.zshenv
sed -i '' -e 's/^ulimit -v unlimited/# ulimit -v unlimited/' ~/.zshenv

echo "✅ Fixed! Changes made:"
echo "  - ulimit -n 524288 → ulimit -n 10240"
echo "  - Commented out ulimit -m and -v (not supported on macOS)"
echo ""
echo "Backup saved to ~/.zshenv.backup"
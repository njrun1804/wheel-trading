#!/bin/bash

echo "Fixing .zshrc EOF error..."

# Create backup
cp ~/.zshrc ~/.zshrc.backup

# Remove the problematic EOF line
sed -i '' '/^EOF < \/dev\/null$/d' ~/.zshrc

echo "✅ Fixed! Removed stray EOF command from line 282"
echo "Backup saved to ~/.zshrc.backup"
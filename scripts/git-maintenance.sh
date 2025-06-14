#!/bin/bash
# Git maintenance script for single developer

echo "🧹 Running git maintenance..."

# 1. Garbage collection and optimization
echo "→ Optimizing repository..."
git gc --aggressive --prune=now

# 2. Prune remote references
echo "→ Pruning remote references..."
git remote prune origin

# 3. Clean up unnecessary files
echo "→ Cleaning untracked files..."
git clean -fd -x -e .env -e data/

# 4. Show repository status
echo -e "\n📊 Repository Status:"
echo "Size: $(du -sh .git | cut -f1)"
echo "Branches: $(git branch -a | wc -l) total"
echo "Local: $(git branch | wc -l)"
echo "Remote: $(git branch -r | wc -l)"

# 5. Recent activity
echo -e "\n📅 Recent commits:"
git log --oneline -5

echo -e "\n✅ Maintenance complete!"
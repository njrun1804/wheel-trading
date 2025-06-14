# Git Cheat Sheet for Wheel Trading

## Daily Commands

### Quick Development Cycle
```bash
# See what changed
git status

# Add and commit everything
git add . && git commit -m "Your message"

# Push to GitHub
git push

# Or use the Claude alias (adds attribution)
git claude-commit "Your message"
```

### Checking History
```bash
# Last 10 commits
git recent

# See what files changed
git log --stat -3

# Search commit messages
git log --grep="keyword"
```

### Undoing Things
```bash
# Undo last commit (keep changes)
git reset HEAD~1

# Discard all local changes
git checkout -- .

# Remove untracked files
git clean -fd
```

## Maintenance (Run Monthly)
```bash
./scripts/git-maintenance.sh
```

## Branch Management
```bash
# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main
git checkout orchestrator_bootstrap

# Delete branch
git branch -d branch-name
```

## Your Optimizations
- **No pre-commit hooks** = Fast commits
- **Auto-prune** = Cleans deleted remote branches
- **12-thread compression** = Fast operations
- **No CI/CD** = Push directly to GitHub

## Emergency Commands
```bash
# If something goes wrong
git status
git log --oneline -10
git reflog  # Shows everything you did

# Reset to specific commit
git reset --hard abc1234
```

## Remember
- You're the only developer = no merge conflicts
- Everything is backed up on GitHub
- Commits are cheap, commit often!
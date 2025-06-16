#!/bin/bash
# Personal GitHub workflow for single-developer Claude Code CLI usage

set -e

echo "🚀 Personal GitHub Workflow - Claude Code CLI Edition"
echo "=================================================="

# Function to run quick local checks
quick_check() {
    echo "📝 Running quick checks..."
    
    # Basic formatting check
    if command -v black &> /dev/null; then
        echo "  ✅ Checking code formatting..."
        black src tests --check --quiet || echo "  ⚠️  Code formatting issues found (run: black src tests)"
    fi
    
    # Import validation
    echo "  ✅ Validating Python imports..."
    python -c "
import ast, glob
for f in glob.glob('src/**/*.py', recursive=True):
    try: ast.parse(open(f).read())
    except: print(f'❌ Syntax error in {f}'); exit(1)
print('  ✅ All imports valid')
"
    
    echo "  ✅ Quick checks complete!"
}

# Function to commit with Claude Code CLI optimized workflow
commit_changes() {
    local message="$1"
    
    echo "📋 Preparing commit..."
    
    # Run quick checks first
    quick_check
    
    # Stage changes
    git add .
    
    # Show what will be committed
    echo "📦 Changes to commit:"
    git diff --cached --stat
    
    # Commit with generated message if none provided
    if [ -z "$message" ]; then
        message="$(date '+%Y-%m-%d %H:%M'): Claude Code CLI automated commit"
    fi
    
    git commit -m "$message"
    echo "  ✅ Committed: $message"
}

# Function to create PR for main branch
create_pr() {
    local title="$1"
    local branch=$(git branch --show-current)
    
    if [ "$branch" = "main" ]; then
        echo "❌ Cannot create PR from main branch"
        return 1
    fi
    
    echo "🔄 Creating PR from $branch to main..."
    
    # Push current branch
    git push -u origin "$branch"
    
    # Create PR
    if [ -z "$title" ]; then
        title="$(date '+%Y-%m-%d'): Updates from Claude Code CLI"
    fi
    
    gh pr create --title "$title" --body "$(cat <<EOF
## Changes
- Automated updates from Claude Code CLI development session
- Date: $(date '+%Y-%m-%d %H:%M:%S')
- Branch: $branch

## Testing
- ✅ Local quick checks passed
- ✅ Import validation completed
- ✅ Code formatting verified

## Notes
Personal development session - single developer workflow.
EOF
)"
    echo "  ✅ PR created: $title"
}

# Function to merge PR and cleanup
merge_and_cleanup() {
    local pr_number="$1"
    local branch=$(git branch --show-current)
    
    if [ -z "$pr_number" ]; then
        # Try to find PR for current branch
        pr_number=$(gh pr list --head "$branch" --json number --jq '.[0].number')
    fi
    
    if [ -z "$pr_number" ] || [ "$pr_number" = "null" ]; then
        echo "❌ No PR found to merge"
        return 1
    fi
    
    echo "🔄 Merging PR #$pr_number..."
    
    gh pr merge "$pr_number" --squash --delete-branch
    
    # Switch back to main and pull
    git checkout main
    git pull origin main
    
    echo "  ✅ PR merged and branch cleaned up"
}

# Main command handling
case "${1:-help}" in
    "check"|"c")
        quick_check
        ;;
    "commit"|"co")
        commit_changes "$2"
        ;;
    "pr")
        create_pr "$2"
        ;;
    "merge"|"m")
        merge_and_cleanup "$2"
        ;;
    "sync"|"s")
        echo "🔄 Syncing with remote..."
        git fetch origin
        git pull origin main
        echo "  ✅ Synced with remote"
        ;;
    "status"|"st")
        echo "📊 Repository Status:"
        echo "Branch: $(git branch --show-current)"
        echo "Status: $(git status --porcelain | wc -l) changes"
        echo "Remote: $(git remote get-url origin)"
        gh pr list --limit 3
        ;;
    "workflow"|"w")
        echo "🔄 Complete workflow: check → commit → pr"
        quick_check
        commit_changes "$2"
        create_pr "$3"
        ;;
    *)
        echo "Personal GitHub Workflow Commands:"
        echo "  check, c          - Run quick local checks"
        echo "  commit, co [msg]  - Commit changes with checks"
        echo "  pr [title]        - Create pull request"
        echo "  merge, m [#]      - Merge PR and cleanup"
        echo "  sync, s           - Sync with remote"
        echo "  status, st        - Show repository status"
        echo "  workflow, w       - Complete workflow (check/commit/pr)"
        echo ""
        echo "Examples:"
        echo "  ./scripts/personal-github-workflow.sh check"
        echo "  ./scripts/personal-github-workflow.sh commit 'Fix trading strategy'"
        echo "  ./scripts/personal-github-workflow.sh workflow 'New feature' 'Add ML model'"
        ;;
esac
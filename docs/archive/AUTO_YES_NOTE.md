# Auto-Yes Functionality in Commit Workflow

## What Auto-Yes Covers

When you use `./scripts/commit-workflow.sh -m "message" -y`, it automatically handles:

### ✅ System & Git Prompts
- Git add/commit/push confirmations
- Git merge and rebase prompts
- SSH host key verification (for git push)
- File overwrite confirmations
- Pre-commit hook auto-fixes
- Directory creation prompts
- Git upstream branch creation

### ✅ Script Confirmations
- "Stage all changes?" → YES
- "Create commit?" → YES
- "Stage pre-commit changes?" → YES
- "Auto-fix file placement issues?" → YES

### ✅ Environment Settings
- `GIT_MERGE_AUTOEDIT=no` - No merge message editor
- `GIT_EDITOR=true` - Bypasses git editor prompts
- `GIT_SSH_COMMAND` - Auto-accepts SSH hosts
- File operations use `-f` flag (force)

## What Auto-Yes CANNOT Cover

### ❌ Claude Code Interface Prompts
These require manual approval in Claude's UI:
- "Can I check this directory?"
- "Can I create this file?"
- "Can I run this command?"
- "Can I read this file?"
- "Can I modify this file?"

### ❌ Security-Critical Prompts
- GitHub OAuth login (if needed)
- API key/secret entry
- Sudo password prompts

## Usage

```bash
# Full auto-yes workflow
./scripts/commit-workflow.sh -m "Add feature" -y

# Auto-yes with no CI wait
./scripts/commit-workflow.sh -m "Update docs" -y -w

# Auto-yes local only
./scripts/commit-workflow.sh -m "WIP changes" -y -n
```

## Notes

1. The `-y` flag makes the commit process fully automated for system-level operations
2. You still need to manually approve Claude Code's permission prompts
3. Pre-commit hooks will auto-fix and continue even if some fail
4. Git push uses `--force-with-lease` for safety when auto-yes is enabled
5. All file operations become non-interactive (force mode)

This is designed for the single-user Unity wheel trading advisory system where you trust all operations.

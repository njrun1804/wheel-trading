# GitHub Configuration

This directory contains configuration files optimized for:
1. **Claude Code CLI** - Primary development tool
2. **OpenAI Codex** - Secondary pair programming assistant

## Key Files

- `CODEOWNERS` - Automatic review assignment
- `copilot-instructions.md` - Instructions for GitHub Copilot/Codex
- `claude-context.md` - Additional context for Claude Code CLI

## Integration Points

The codebase is structured to be easily understood by AI assistants:
- Clear API boundaries in `/src/unity_wheel/api/`
- Self-documenting code with type hints
- Comprehensive test coverage
- Well-defined entry points

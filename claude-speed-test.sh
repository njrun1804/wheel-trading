#!/bin/bash
# Claude Code Speed Test & Permission Approval Script
# Triggers all tools to approve permissions globally

echo "🚀 Claude Code Speed Test - Triggering All Tool Permissions"
echo "Run this to approve all tools across all domains:"
echo ""

# Test all tools with wildcards to trigger permission dialogs
claude --print "Use these tools to test the system:
1. Use Bash tool to run: ls -la
2. Use Read tool to read: /etc/hosts  
3. Use Edit tool to edit: /tmp/test.txt (add 'test')
4. Use MultiEdit tool on: /tmp/test.txt (multiple changes)
5. Use Write tool to create: /tmp/claude-test.txt
6. Use Glob tool to find: **/*.py
7. Use Grep tool to search: 'import' in **/*.py
8. Use Task tool to: search for configuration files
9. Use LS tool to list: /Users/mikeedwards
10. Use TodoRead tool
11. Use TodoWrite tool with test todo
12. Use WebFetch tool on: https://httpbin.org/json
13. Use WebSearch tool for: claude code documentation

This will trigger permission dialogs for all tools. Approve with 'Allow all' for maximum speed."
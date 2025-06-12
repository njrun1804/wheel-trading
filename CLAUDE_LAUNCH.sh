#!/usr/bin/env bash
# CLAUDE OPTIMAL LAUNCHER - Fixed with correct environment variables and best practices

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Export PROJECT_ROOT so Python can access it
export PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

clear
echo -e "${PURPLE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║        CLAUDE OPTIMAL LAUNCHER v2             ║${NC}"
echo -e "${PURPLE}║     Correct Environment Variables & MCP       ║${NC}"
echo -e "${PURPLE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# CORRECT environment variables per Anthropic docs
export MAX_THINKING_TOKENS=50000                   # Correct thinking budget
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192         # Output limit (Opus/Haiku)
export ANTHROPIC_MODEL="claude-opus-4-20250514"   # Claude 4 Opus with 200k context
export NODE_OPTIONS="--max-old-space-size=6144"   # 6GB for parallel MCP tasks
export PYTHONOPTIMIZE=1
export MCP_TIMEOUT=30000                          # 30 second timeout for MCP tools
export MCP_TOOL_TIMEOUT=60000                     # 60 second timeout for long operations

# Ensure we have GitHub token
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo -e "${YELLOW}Warning: GITHUB_TOKEN not set${NC}"
    echo "GitHub MCP server will have limited functionality"
fi

echo -e "${BLUE}Configuration:${NC}"
echo "  • Model: Claude 4 Opus (200k context)"
echo "  • Thinking: 50,000 tokens"
echo "  • Output: 8,192 tokens"
echo "  • Memory: 6GB Node.js"
echo "  • MCP Timeout: 30s (tools: 60s)"
echo ""

# More precise process cleanup
echo -e "${YELLOW}Cleaning up old MCP servers...${NC}"
pkill -f "server-(filesystem|github|dependency-graph|memory|sequential-thinking)" 2>/dev/null || true
pkill -f "mcp-server-github" 2>/dev/null || true
rm -rf .claude/runtime/ws_*/state/*.pid 2>/dev/null || true
sleep 1

# Cache check for pre-analysis
CACHE_FILE=".claude/cache/codebase_analysis.json"
GIT_FILES_HASH=""
if command -v git >/dev/null 2>&1; then
    GIT_FILES_HASH=$(git ls-files 2>/dev/null | md5 2>/dev/null || echo "no-git")
fi
CACHE_HASH_FILE=".claude/cache/analysis.hash"

# Only re-analyze if files changed
if [ -f "$CACHE_FILE" ] && [ -f "$CACHE_HASH_FILE" ] && [ "$(cat $CACHE_HASH_FILE 2>/dev/null)" = "$GIT_FILES_HASH" ]; then
    echo -e "${GREEN}✓ Using cached codebase analysis${NC}"
else
    echo -e "${YELLOW}Analyzing codebase for optimal MCP usage...${NC}"
    python3 << 'EOF'
import os
import json
from pathlib import Path

project_root = Path(os.environ['PROJECT_ROOT'])
analysis = {
    'key_modules': [],
    'test_files': [],
    'config_files': [],
    'total_py_files': 0,
    'total_lines': 0
}

for py_file in project_root.rglob('*.py'):
    if '__pycache__' in str(py_file) or '.venv' in str(py_file):
        continue
    
    analysis['total_py_files'] += 1
    rel_path = py_file.relative_to(project_root)
    
    if 'test' in str(rel_path).lower():
        analysis['test_files'].append(str(rel_path))
    elif str(rel_path).startswith('src/unity_wheel'):
        analysis['key_modules'].append(str(rel_path))
    
    try:
        with open(py_file, 'r') as f:
            analysis['total_lines'] += len(f.readlines())
    except:
        pass

cache_dir = project_root / '.claude' / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)
with open(cache_dir / 'codebase_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"✓ Analyzed {analysis['total_py_files']} Python files")
print(f"✓ Found {len(analysis['key_modules'])} key modules")
print(f"✓ Total lines: {analysis['total_lines']:,}")
EOF
    echo "$GIT_FILES_HASH" > "$CACHE_HASH_FILE"
fi

# Use Claude's built-in MCP commands for proper setup
echo ""
echo -e "${YELLOW}Setting up MCP servers the right way...${NC}"

# Find Claude command - check multiple locations
CLAUDE_CMD=""
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
    echo -e "${GREEN}✓ Found Claude at ~/.claude/local/claude${NC}"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
    echo -e "${GREEN}✓ Found Claude in PATH${NC}"
else
    echo -e "${RED}Claude CLI not found!${NC}"
    echo "Install from: https://claude.ai/code"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

# Add MCP servers using Claude's built-in commands
# This ensures proper scoping and permissions
echo -e "${BLUE}Adding MCP servers to Claude configuration...${NC}"

# Filesystem - project scope
"$CLAUDE_CMD" mcp add filesystem -s project -f -- \
    npx -y @modelcontextprotocol/server-filesystem@latest "$PROJECT_ROOT" 2>/dev/null || \
    echo "  • filesystem already configured"

# GitHub - user scope with token
if [ -n "${GITHUB_TOKEN:-}" ]; then
    "$CLAUDE_CMD" mcp add github -s user -f --env GITHUB_TOKEN="$GITHUB_TOKEN" -- \
        mcp-server-github 2>/dev/null || \
        echo "  • github already configured"
fi

# Dependency graph - project scope
"$CLAUDE_CMD" mcp add dependency-graph -s project -f -- \
    /Users/mikeedwards/.pyenv/shims/python3 \
    "$PROJECT_ROOT/scripts/dependency-graph-mcp-enhanced.py" 2>/dev/null || \
    echo "  • dependency-graph already configured"

# Memory - project scope
"$CLAUDE_CMD" mcp add memory -s project -f -- \
    npx -y @modelcontextprotocol/server-memory@latest 2>/dev/null || \
    echo "  • memory already configured"

# Sequential thinking - project scope
"$CLAUDE_CMD" mcp add sequential-thinking -s project -f -- \
    npx -y @modelcontextprotocol/server-sequential-thinking@latest 2>/dev/null || \
    echo "  • sequential-thinking already configured"

# Create optimization guide
cat > .claude/OPTIMIZATION_GUIDE.md << 'EOF'
# Claude Optimization Guide

## Efficient Token Usage with MCP

### 1. Search Strategy (Fastest to Slowest)
- `dependency_graph.search_code_fuzzy("term")` - 2-5ms, symbol-aware
- `filesystem.read_file(path)` - Direct file access when you know the path
- `ripgrep.search("pattern")` - Only for regex or complex patterns

### 2. Memory Strategy
- Store complex analysis results: `memory.save("key", result)`
- Retrieve later: `memory.retrieve("key")`
- Avoids re-reading files and re-analyzing

### 3. Sequential Thinking
- Use for multi-step problems
- Breaks down complex tasks efficiently
- Reduces overall token usage

### 4. Token Budget
- Thinking: 50,000 tokens available
- Output: 8,192 max (6,144 on Sonnet)
- Use thinking for analysis, keep output concise

## Project Specifics
- Entry: run.py → src/unity_wheel/api/advisor.py
- Config: config.yaml
- Database: data/wheel_trading_master.duckdb
- Unity: $1000/contract, Delta: 0.30
EOF

echo ""
echo -e "${GREEN}✓ MCP servers configured properly${NC}"
echo ""

# Display final stats
if [ -f "$CACHE_FILE" ] && command -v jq >/dev/null 2>&1; then
    FILES=$(jq -r .total_py_files "$CACHE_FILE" 2>/dev/null || echo "unknown")
    LINES=$(jq -r .total_lines "$CACHE_FILE" 2>/dev/null || echo "unknown")
    echo -e "${BLUE}Codebase Stats:${NC}"
    echo "  • Python files: $FILES"
    echo "  • Total lines: $LINES"
    echo ""
fi

echo -e "${GREEN}Ready to launch Claude!${NC}"
echo ""
echo -e "${PURPLE}Optimization Tips:${NC}"
echo "  1. Use dependency_graph for instant searches"
echo "  2. Cache results in memory MCP"
echo "  3. Let Claude think deeply (50k tokens)"
echo "  4. Keep outputs concise"
echo ""

# Launch Claude
echo -e "${GREEN}Launching Claude with optimal configuration...${NC}"
exec "$CLAUDE_CMD"
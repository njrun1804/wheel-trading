#!/usr/bin/env bash
# CLAUDE OPTIMAL LAUNCHER - Smart token usage with MCP efficiency

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
cd "$PROJECT_ROOT"

clear
echo -e "${PURPLE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║        CLAUDE OPTIMAL LAUNCHER                ║${NC}"
echo -e "${PURPLE}║   Smart Token Usage + MCP = Best Solutions    ║${NC}"
echo -e "${PURPLE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# OPTIMAL settings - work within Claude's actual limits
# but use them intelligently with MCP tools
export CLAUDE_CODE_THINKING_BUDGET_TOKENS=50000   # Enough for deep analysis
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192         # Claude's actual limit
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=100000      # 100K context (within 200K limit)
export CLAUDE_CODE_PARALLELISM=8                  # 8 performance cores
export NODE_OPTIONS="--max-old-space-size=4096"   # 4GB - conservative for stability
export PYTHONOPTIMIZE=1                           # Balanced optimization
export MCP_PERFORMANCE_MODE=true
export PATH="/Users/mikeedwards/.local/bin:$PATH"

# MCP Strategy for optimal token usage
export MCP_STRATEGY="smart"
export MCP_CACHE_AGGRESSIVE=true
export MCP_PREFETCH=true

echo -e "${BLUE}Optimization Strategy:${NC}"
echo "  • Use dependency_graph FIRST (2-5ms) instead of ripgrep"
echo "  • Cache results in memory MCP to avoid re-reading"
echo "  • Pre-analyze codebase structure"
echo "  • Let Claude think deeply but output concisely"
echo ""

# Clean start
pkill -f "mcp\|ripgrep\|dependency" 2>/dev/null || true
rm -rf .claude/runtime/ws_*/state/*.pid 2>/dev/null || true
sleep 1

# Pre-analyze codebase for MCP efficiency
echo -e "${YELLOW}Pre-analyzing codebase for optimal MCP usage...${NC}"
python3 << 'EOF'
import os
import json
from pathlib import Path

project_root = Path(os.environ.get('PROJECT_ROOT', '.'))
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
    
    # Categorize files
    if 'test' in str(rel_path).lower():
        analysis['test_files'].append(str(rel_path))
    elif str(rel_path).startswith('src/unity_wheel'):
        analysis['key_modules'].append(str(rel_path))
    
    # Count lines
    try:
        with open(py_file, 'r') as f:
            analysis['total_lines'] += len(f.readlines())
    except:
        pass

# Save analysis for MCP to use
cache_dir = project_root / '.claude' / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)
with open(cache_dir / 'codebase_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"✓ Analyzed {analysis['total_py_files']} Python files")
print(f"✓ Found {len(analysis['key_modules'])} key modules")
print(f"✓ Total lines: {analysis['total_lines']:,}")
EOF

# Start ONLY the essential MCP servers that complement Claude
echo ""
echo -e "${YELLOW}Starting essential MCP servers...${NC}"

# Create a minimal MCP configuration for optimal performance
cat > mcp-servers-optimal.json << 'EOF'
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest", "/Users/mikeedwards"]
    },
    "github": {
      "transport": "stdio",
      "command": "mcp-server-github",
      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    },
    "dependency-graph": {
      "transport": "stdio",
      "command": "/Users/mikeedwards/.pyenv/shims/python3",
      "args": ["/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/dependency-graph-mcp-enhanced.py"]
    },
    "memory": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory@latest"]
    },
    "sequential-thinking": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking@latest"]
    }
  }
}
EOF

# Start servers
MCP_ROOT="$PROJECT_ROOT" mcp-up-essential 2>&1 | grep -E "Starting|✓" || true

# Wait for servers
sleep 3

# Create optimization hints for Claude
cat > .claude/optimization-hints.md << 'EOF'
# Optimization Hints for Claude

## Token-Efficient Workflow
1. Use `dependency_graph.search_code_fuzzy()` FIRST - it's 100x faster than ripgrep
2. Store intermediate results in `memory.save()` to avoid re-analysis
3. Use `sequential-thinking` for complex multi-step problems
4. Read only the specific functions/classes needed, not entire files

## Project-Specific Optimizations
- Key entry point: `run.py` → `src/unity_wheel/api/advisor.py`
- Main config: `config.yaml`
- Database: `data/wheel_trading_master.duckdb`
- Unity value: $1000 per contract

## Smart Token Usage
- Think deeply (use thinking tokens) but output concisely
- Use bullet points for clarity
- Reference file:line_number for navigation
- Cache complex calculations in memory MCP
EOF

echo ""
echo -e "${GREEN}✓ Optimal MCP configuration ready${NC}"
echo ""
echo -e "${BLUE}Smart Configuration:${NC}"
echo "  • Thinking: 50,000 tokens (deep analysis)"
echo "  • Output: 8,192 tokens (Claude's limit)"
echo "  • Context: 100,000 tokens (efficient)"
echo "  • Memory: 4GB (stable on 24GB system)"
echo "  • MCP: 5 essential servers only"
echo ""
echo -e "${YELLOW}Efficiency Tips:${NC}"
echo "  • dependency_graph is 100x faster than ripgrep"
echo "  • memory MCP caches results between queries"
echo "  • sequential-thinking for complex planning"
echo "  • Pre-analyzed ${GREEN}$(jq -r .total_py_files .claude/cache/codebase_analysis.json 2>/dev/null || echo "400+")${NC} Python files"
echo ""

# Launch Claude with optimal config
CLAUDE_CMD=""
if [ -f "/Users/mikeedwards/.claude/local/claude" ]; then
    CLAUDE_CMD="/Users/mikeedwards/.claude/local/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
fi

if [ -n "$CLAUDE_CMD" ]; then
    echo -e "${GREEN}Launching Claude with optimal configuration...${NC}"
    echo -e "${PURPLE}Remember: Let MCP tools do the heavy lifting!${NC}"
    exec "$CLAUDE_CMD" --mcp-config "$PROJECT_ROOT/mcp-servers-optimal.json"
else
    echo -e "${BLUE}To complete setup, install Claude from:${NC}"
    echo "https://claude.ai/code"
    echo ""
    echo "Then run:"
    echo "  claude --mcp-config \"$PROJECT_ROOT/mcp-servers-optimal.json\""
fi
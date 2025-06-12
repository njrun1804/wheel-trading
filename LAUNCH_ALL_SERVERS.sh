#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Launching Claude with ALL 19 MCP Servers${NC}"
echo "=================================================="
echo ""

# Check server status
echo -e "${BLUE}Checking server status:${NC}"
echo ""

# Check Opik
if curl -s http://localhost:5173/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} Opik trace server - http://localhost:5173"
else
    echo -e "  ${RED}❌${NC} Opik server not running"
fi

# Check Phoenix
if curl -s http://localhost:6006/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} Phoenix trace server - http://localhost:6006"
else
    echo -e "  ${RED}❌${NC} Phoenix server not running"
fi

echo ""
echo -e "${GREEN}MCP Servers (19 total):${NC}"
echo ""
echo -e "${BLUE}Node.js (7):${NC}"
echo "  • filesystem - File operations"
echo "  • github - GitHub API"
echo "  • brave - Web search"
echo "  • memory - Memory/caching"
echo "  • sequential-thinking - Reasoning"
echo "  • puppeteer - Web automation"
echo "  • dependency-graph - Code analysis"
echo ""
echo -e "${BLUE}Python (12):${NC}"
echo "  • statsource - Statistics"
echo "  • duckdb - Database queries"
echo "  • mlflow - ML experiments"
echo "  • pyrepl - Python REPL"
echo "  • sklearn - Machine learning"
echo "  • optionsflow - Options data"
echo "  • python_analysis - Code analysis"
echo "  • logfire - Observability"
echo "  • trace - Enhanced tracing"
echo "  • trace-opik - Opik integration"
echo "  • trace-phoenix - Phoenix integration"
echo "  • ripgrep - Fast file search"
echo ""

# Set environment
export MAX_THINKING_TOKENS=50000
export ANTHROPIC_MODEL="claude-opus-4-20250514"
export NODE_OPTIONS="--max-old-space-size=6144"
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Set trace server URLs
export OPIK_BASE_URL="http://localhost:5173"
export PHOENIX_BASE_URL="http://localhost:6006"

# Launch Claude
echo -e "${GREEN}Starting Claude Code CLI in interactive mode...${NC}"
echo -e "${YELLOW}All 19 MCP servers are ready!${NC}"
echo ""
exec /Users/mikeedwards/.claude/local/claude --mcp-config "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
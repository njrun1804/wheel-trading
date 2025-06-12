#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Launching Claude with ALL MCP Servers Fixed${NC}"
echo "=================================================="
echo ""
echo -e "${GREEN}✅ All MCP servers have been fixed:${NC}"
echo ""
echo -e "${BLUE}Node.js Servers (7):${NC}"
echo "  • filesystem - File system operations"
echo "  • github - GitHub API access"
echo "  • brave - Web search"
echo "  • memory - Memory/caching"
echo "  • sequential-thinking - Step-by-step reasoning"
echo "  • puppeteer - Web automation"
echo "  • (ripgrep moved to Python)"
echo ""
echo -e "${BLUE}Python Servers (11):${NC}"
echo "  • statsource - Statistics API"
echo "  • duckdb - Database queries"
echo "  • mlflow - ML experiment tracking"
echo "  • pyrepl - Python REPL"
echo "  • sklearn - Scikit-learn operations"
echo "  • optionsflow - Options trading data"
echo "  • python_analysis - Code analysis (fixed)"
echo "  • logfire - Observability (fixed)"
echo "  • trace - Enhanced tracing (fixed)"
echo "  • ripgrep - Fast file search (fixed)"
echo "  • dependency-graph - Code dependencies (fixed)"
echo ""
echo -e "${YELLOW}Excluded servers:${NC}"
echo "  • trace-opik - Optional (requires local Opik server)"
echo "  • trace-phoenix - SQLAlchemy compatibility issues"
echo ""

# Set environment variables
export MAX_THINKING_TOKENS=50000
export ANTHROPIC_MODEL="claude-opus-4-20250514"
export NODE_OPTIONS="--max-old-space-size=6144"
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1

# Launch Claude with the fixed config
echo -e "${GREEN}Starting Claude...${NC}"
claude --mcp-config "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
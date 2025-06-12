#!/bin/bash

# Ultimate Claude launcher with ALL optimizations
set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Launching Claude ULTIMATE (M4 + Cross-Layer Optimized)${NC}"
echo "=========================================================="
echo ""

# Check server status
echo -e "${BLUE}Checking server status:${NC}"
echo ""

# Check Opik
if curl -s http://localhost:5173/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} Opik trace server - http://localhost:5173"
else
    echo -e "  ${RED}❌${NC} Opik server not running - starting..."
    nohup python3 scripts/opik-server.py > opik-server.log 2>&1 &
    sleep 2
fi

# Check Phoenix
if curl -s http://localhost:6006/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} Phoenix trace server - http://localhost:6006"
else
    echo -e "  ${RED}❌${NC} Phoenix server not running - starting..."
    nohup phoenix serve > phoenix-fixed.log 2>&1 &
    sleep 3
fi

# Start memory pressure monitor
echo -e "\n${BLUE}Starting optimization services:${NC}"
if curl -s http://localhost:8765/sys/claude/pressure-gauge > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅${NC} Memory pressure monitor already running"
else
    echo -e "  ${YELLOW}Starting memory pressure monitor...${NC}"
    if [ -f ".claude/pressure-monitor.service" ]; then
        nohup .claude/pressure-monitor.service > /dev/null 2>&1 &
        PRESSURE_PID=$!
        sleep 1
    fi
fi

# Get adaptive configuration
PRESSURE_CONFIG=$(curl -s http://localhost:8765/sys/claude/pressure-gauge 2>/dev/null || echo '{}')
FANOUT=$(echo "$PRESSURE_CONFIG" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('chunk_fanout', 8))" 2>/dev/null || echo "8")
WORKERS=$(echo "$PRESSURE_CONFIG" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('parallel_workers', 4))" 2>/dev/null || echo "4")
PRESSURE_LEVEL=$(echo "$PRESSURE_CONFIG" | python3 -c "import json, sys; d=json.load(sys.stdin); print(f\"{d.get('pressure_level', 0)*100:.1f}\")" 2>/dev/null || echo "0.0")

echo -e "\n${GREEN}System Configuration:${NC}"
echo -e "  ${BLUE}M4 Mac Optimizations:${NC}"
echo "  • CPU cores: $(sysctl -n hw.ncpu)"
echo "  • Performance cores: $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 'N/A')"
echo "  • Total RAM: 24GB"
echo "  • Node heap: 6GB"
echo ""
echo -e "  ${BLUE}Adaptive Configuration:${NC}"
echo "  • Memory pressure: ${PRESSURE_LEVEL}%"
echo "  • Chunk fan-out: $FANOUT"
echo "  • Parallel workers: $WORKERS"
echo ""
echo -e "  ${BLUE}Cross-Layer Optimizations:${NC}"
echo "  • SHA-1 slice cache: ENABLED"
echo "  • Port quota management: ENABLED"
echo "  • Connection pooling: ENABLED"
echo ""

# Set M4 Mac performance optimizations
echo -e "${YELLOW}Applying M4 Mac optimizations...${NC}"

# Set process priority (requires sudo, skip if not available)
if command -v renice > /dev/null 2>&1; then
    renice -n -10 -p $$ 2>/dev/null || echo "  Note: Run with sudo for max priority"
fi

# CPU affinity for performance cores (macOS specific)
if command -v taskpolicy > /dev/null 2>&1; then
    taskpolicy -c background 2>/dev/null || true
fi

# Environment with ALL optimizations
export MAX_THINKING_TOKENS=50000
export ANTHROPIC_MODEL="claude-opus-4-20250514"

# M4 Mac specific
export NODE_OPTIONS="--max-old-space-size=6144"
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # macOS fork safety

# Cross-layer optimizations
export CLAUDE_CHUNK_FANOUT=$FANOUT
export CLAUDE_PARALLEL_WORKERS=$WORKERS
export CLAUDE_SLICE_CACHE_ENABLED=1
export CLAUDE_PORT_QUOTA_ENABLED=1
export CLAUDE_PRESSURE_MONITORING=1
export CLAUDE_ADAPTIVE_MEMORY=1

# Connection pooling
export MCP_CONNECTION_POOL_URL="http://localhost:8766"
export MCP_POOL_SIZE=8
export MCP_POOL_TIMEOUT=30
export MCP_POOL_MAX_RETRIES=3

# DuckDB optimizations for 24GB system
export DUCKDB_MEMORY_LIMIT="8GB"
export DUCKDB_THREADS="6"
export DUCKDB_CACHE_SIZE="2GB"

# Trace server URLs
export OPIK_BASE_URL="http://localhost:5173"
export PHOENIX_BASE_URL="http://localhost:6006"

# macOS specific optimizations
ulimit -n 4096 2>/dev/null || true  # Increase file descriptors
ulimit -u 2048 2>/dev/null || true  # Increase processes

echo ""
echo -e "${GREEN}MCP Servers (19 total):${NC}"
echo -e "${BLUE}Node.js (7):${NC} filesystem, github, brave, memory, sequential-thinking, puppeteer, dependency-graph"
echo -e "${BLUE}Python (12):${NC} statsource, duckdb, mlflow, pyrepl, sklearn, optionsflow, python_analysis,"
echo "              logfire, trace, trace-opik, trace-phoenix, ripgrep"
echo ""

# Launch Claude with all optimizations
echo -e "${GREEN}Starting Claude Code CLI (ULTIMATE)...${NC}"
echo -e "${YELLOW}All optimizations active!${NC}"
echo ""

# Use exec to replace shell and inherit all optimizations
exec /Users/mikeedwards/.claude/local/claude --mcp-config "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json"
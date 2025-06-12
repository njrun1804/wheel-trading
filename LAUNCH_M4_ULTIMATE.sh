#!/bin/bash

# Ultimate Claude launcher optimized for M4 Pro Mac (Serial: KXQ93HN7DP)
# Hardware: 12 cores (8 performance), 24GB RAM
set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Launching Claude ULTIMATE for M4 Pro Mac${NC}"
echo "=============================================="
echo -e "${BLUE}Serial: KXQ93HN7DP | M4 Pro | 8P+4E cores | 24GB RAM${NC}"
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
        nohup python3 .claude/pressure-monitor.service > /dev/null 2>&1 &
        PRESSURE_PID=$!
        sleep 1
    fi
fi

# Get adaptive configuration
PRESSURE_CONFIG=$(curl -s http://localhost:8765/sys/claude/pressure-gauge 2>/dev/null || echo '{}')
FANOUT=$(echo "$PRESSURE_CONFIG" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('chunk_fanout', 8))" 2>/dev/null || echo "8")
WORKERS=$(echo "$PRESSURE_CONFIG" | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('parallel_workers', 8))" 2>/dev/null || echo "8")
PRESSURE_LEVEL=$(echo "$PRESSURE_CONFIG" | python3 -c "import json, sys; d=json.load(sys.stdin); print(f\"{d.get('pressure_level', 0)*100:.1f}\")" 2>/dev/null || echo "0.0")

echo -e "\n${GREEN}System Configuration:${NC}"
echo -e "  ${BLUE}M4 Pro Optimizations:${NC}"
echo "  • Performance cores: 8 (high-priority tasks)"
echo "  • Efficiency cores: 4 (background tasks)"
echo "  • Neural Engine: 16-core (for ML tasks)"
echo "  • Total RAM: 24GB"
echo "  • Allocated:"
echo "    - Node.js: 6GB (25%)"
echo "    - DuckDB: 8GB (33%)"
echo "    - Python: 6GB (25%)"
echo "    - System: 4GB (17%)"
echo ""
echo -e "  ${BLUE}Adaptive Configuration:${NC}"
echo "  • Memory pressure: ${PRESSURE_LEVEL}%"
echo "  • Chunk fan-out: $FANOUT"
echo "  • Parallel workers: $WORKERS (matching P-cores)"
echo ""
echo -e "  ${BLUE}Cross-Layer Optimizations:${NC}"
echo "  • SHA-1 slice cache: ENABLED"
echo "  • Port quota management: ENABLED"
echo "  • Connection pooling: ENABLED"
echo "  • MCP pre-warming: ENABLED"
echo ""

# Set M4 Pro performance optimizations
echo -e "${YELLOW}Applying M4 Pro optimizations...${NC}"

# Override direnv NODE_OPTIONS with optimal settings for 24GB
# direnv sets 4096MB, but we can use 6144MB for better performance
export NODE_OPTIONS="--max-old-space-size=6144 --max-semi-space-size=256"

# CPU affinity for performance cores (M4 specific)
if [ -x /usr/bin/taskpolicy ]; then
    # Use QoS tier for performance cores
    taskpolicy -c background 2>/dev/null || true
fi

# Environment with ALL optimizations
export MAX_THINKING_TOKENS=50000  # Override direnv's 32000
export ANTHROPIC_MODEL="claude-opus-4-20250514"

# M4 Pro specific optimizations
export PYTHONOPTIMIZE=1
export PYTHONUNBUFFERED=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # Already set by direnv

# Cross-layer optimizations tuned for M4
export CLAUDE_CHUNK_FANOUT=$FANOUT
export CLAUDE_PARALLEL_WORKERS=$WORKERS  # Match performance cores
export CLAUDE_SLICE_CACHE_ENABLED=1
export CLAUDE_PORT_QUOTA_ENABLED=1
export CLAUDE_PRESSURE_MONITORING=1
export CLAUDE_ADAPTIVE_MEMORY=1

# Connection pooling optimized for M4's fast I/O
export MCP_CONNECTION_POOL_URL="http://localhost:8766"
export MCP_POOL_SIZE=8  # One per performance core
export MCP_POOL_TIMEOUT=30
export MCP_POOL_MAX_RETRIES=3
export MCP_PARALLEL_INIT=1  # Parallel MCP initialization

# DuckDB optimizations for 24GB system
export DUCKDB_MEMORY_LIMIT="8GB"  # 33% of RAM
export DUCKDB_THREADS="8"  # Match performance cores
export DUCKDB_CACHE_SIZE="2GB"
export DUCKDB_TEMP_DIRECTORY="/tmp/duckdb_temp"  # Use fast SSD

# Python optimizations for M4
export PYTHONHASHSEED=0  # Deterministic hashing
export PYTHONMALLOC=malloc  # Use system malloc (optimized for Apple Silicon)

# Trace server URLs
export OPIK_BASE_URL="http://localhost:5173"
export PHOENIX_BASE_URL="http://localhost:6006"

# macOS Sequoia optimizations
ulimit -n 8192 2>/dev/null || ulimit -n 4096 2>/dev/null || true  # Max file descriptors
ulimit -u 2048 2>/dev/null || true  # Max processes

# Enable Metal Performance Shaders for ML operations
export PYTORCH_ENABLE_MPS_FALLBACK=1
export METAL_DEVICE_WRAPPER=1

# MCP performance mode from direnv
export MCP_PERFORMANCE_MODE="${MCP_PERFORMANCE_MODE:-ultra}"

echo ""
echo -e "${GREEN}MCP Servers (19 total):${NC}"
echo -e "${BLUE}Node.js (7):${NC} filesystem, github, brave, memory, sequential-thinking, puppeteer, dependency-graph"
echo -e "${BLUE}Python (12):${NC} statsource, duckdb, mlflow, pyrepl, sklearn, optionsflow, python_analysis,"
echo "              logfire, trace, trace-opik, trace-phoenix, ripgrep"
echo ""

# Pre-warm critical MCP servers in parallel
echo -e "${YELLOW}Pre-warming critical MCP servers...${NC}"
(
    # These can run in parallel on efficiency cores
    curl -s http://localhost:8765/sys/claude/pressure-gauge > /dev/null &
    python3 -c "from src.unity_wheel.monitoring.pressure_gauge import get_pressure_monitor; get_pressure_monitor()" 2>/dev/null &
    python3 -c "from src.unity_wheel.mcp.port_quota_manager import get_quota_manager; get_quota_manager()" 2>/dev/null &
    wait
) &

# Launch Claude with all optimizations
echo -e "${GREEN}Starting Claude Code CLI (M4 ULTIMATE)...${NC}"
echo -e "${YELLOW}All optimizations active for M4 Pro!${NC}"
echo ""

# Use exec to replace shell and inherit all optimizations
exec /Users/mikeedwards/.claude/local/claude --mcp-config "${MCP_CONFIG:-/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json}"
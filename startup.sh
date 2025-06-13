#!/bin/bash
# Unity Wheel startup script with M4 Pro hardware maximization
# Automatically uses ALL available CPU, GPU, and memory resources

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${GREEN}${BOLD}🚀 UNITY WHEEL TURBO MODE${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}Maximizing ALL hardware for peak performance${NC}"
echo ""

# Clean startup logs
rm -f orchestrator.log 2>/dev/null || true

# Run clean startup first to set all optimizations
echo -e "${YELLOW}⚡ Initializing hardware acceleration...${NC}"
python3 clean_startup.py

# Check exit code
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Startup failed - check orchestrator.log${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "Available commands:"
echo ""
echo -e "  ${BOLD}${BLUE}./orchestrate_turbo.py${NC} '<command>'  - ${BOLD}TURBO MODE${NC} (all cores)"
echo -e "  ${BLUE}./orchestrate${NC} '<command>'           - Standard orchestrator"  
echo -e "  ${BLUE}python run.py${NC}                       - Trading recommendations"
echo -e "  ${BLUE}python run.py --diagnose${NC}            - System diagnostics"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# If command provided, run turbo mode
if [ $# -gt 0 ]; then
    echo -e "${BOLD}${BLUE}🔥 TURBO EXECUTING: $@${NC}"
    python orchestrate_turbo.py "$@"
fi
#!/bin/bash

# Lightweight MCP health monitoring

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== MCP Health Monitor ===${NC}"
echo ""

# Check memory usage
echo -e "${YELLOW}System Resources:${NC}"
MEMORY=$(ps aux | grep -E "mcp|claude" | grep -v grep | awk '{sum+=$6} END {print sum/1024}')
echo -e "  MCP Memory Usage: ${MEMORY:-0} MB"

# Check Python processes
PYTHON_MCPS=$(ps aux | grep -E "python.*mcp" | grep -v grep | wc -l)
echo -e "  Python MCP Servers: $PYTHON_MCPS running"

# Check Node processes
NODE_MCPS=$(ps aux | grep -E "node.*mcp|npx|bunx" | grep -v grep | wc -l)
echo -e "  Node MCP Servers: $NODE_MCPS running"

# Check port usage (for trace servers)
echo -e "\n${YELLOW}Trace Server Ports:${NC}"
for port in 4318 5173 6006; do
    if lsof -i :$port >/dev/null 2>&1; then
        echo -e "  Port $port: ${GREEN}✓ Active${NC}"
    else
        echo -e "  Port $port: ${RED}✗ Inactive${NC}"
    fi
done

# Check cache directories
echo -e "\n${YELLOW}Cache Status:${NC}"
NPM_CACHE_SIZE=$(du -sh ~/.npm 2>/dev/null | cut -f1 || echo "0")
PNPM_CACHE_SIZE=$(du -sh ~/.pnpm-store 2>/dev/null | cut -f1 || echo "0")
echo -e "  NPM Cache: $NPM_CACHE_SIZE"
echo -e "  PNPM Cache: $PNPM_CACHE_SIZE"

# Quick response test
echo -e "\n${YELLOW}Quick Tests:${NC}"
if command -v rg >/dev/null 2>&1; then
    TIME_RG=$(( time rg --version >/dev/null 2>&1 ) 2>&1 | grep real | awk '{print $2}')
    echo -e "  Ripgrep response: ${GREEN}✓${NC} $TIME_RG"
fi

if command -v bun >/dev/null 2>&1; then
    TIME_BUN=$(( time bun --version >/dev/null 2>&1 ) 2>&1 | grep real | awk '{print $2}')
    echo -e "  Bun response: ${GREEN}✓${NC} $TIME_BUN"
fi

echo -e "\n${GREEN}Health check complete!${NC}"
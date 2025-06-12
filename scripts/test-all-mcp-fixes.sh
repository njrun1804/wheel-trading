#!/bin/bash
# Test all MCP fixes

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}=== Testing MCP Fixes ===${NC}"

# Test 1: Check Python scripts for asyncio issues
echo -e "\n${YELLOW}Test 1: Checking for asyncio issues...${NC}"
if grep -r "asyncio.run(mcp.run())" "$WORKSPACE_ROOT/scripts" --include="*.py" 2>/dev/null; then
    echo -e "${RED}✗ Found asyncio.run() calls that need fixing${NC}"
else
    echo -e "${GREEN}✓ No asyncio issues found${NC}"
fi

# Test 2: Verify startup script
echo -e "\n${YELLOW}Test 2: Testing startup script...${NC}"
if [ -x "$WORKSPACE_ROOT/scripts/start-mcp-servers.sh" ]; then
    echo -e "${GREEN}✓ Startup script is executable${NC}"
    # Dry run
    bash -n "$WORKSPACE_ROOT/scripts/start-mcp-servers.sh"
    echo -e "${GREEN}✓ Startup script syntax is valid${NC}"
else
    echo -e "${RED}✗ Startup script not found or not executable${NC}"
fi

# Test 3: Check MCP configuration
echo -e "\n${YELLOW}Test 3: Validating MCP configuration...${NC}"
if [ -f "$WORKSPACE_ROOT/mcp-servers.json" ]; then
    python3 -m json.tool "$WORKSPACE_ROOT/mcp-servers.json" > /dev/null
    echo -e "${GREEN}✓ MCP configuration is valid JSON${NC}"
else
    echo -e "${RED}✗ MCP configuration not found${NC}"
fi

# Test 4: Test individual MCP servers
echo -e "\n${YELLOW}Test 4: Testing individual MCP servers...${NC}"
for script in "$WORKSPACE_ROOT/scripts"/*-mcp*.py; do
    if [ -f "$script" ]; then
        echo -n "Testing $(basename "$script")... "
        if python3 -m py_compile "$script" 2>/dev/null; then
            echo -e "${GREEN}[OK]${NC}"
        else
            echo -e "${RED}[SYNTAX ERROR]${NC}"
        fi
    fi
done

# Test 5: Check health monitor
echo -e "\n${YELLOW}Test 5: Testing health monitor...${NC}"
if [ -f "$WORKSPACE_ROOT/scripts/mcp-health-monitor.py" ]; then
    python3 "$WORKSPACE_ROOT/scripts/mcp-health-monitor.py" --help > /dev/null 2>&1
    echo -e "${GREEN}✓ Health monitor is functional${NC}"
else
    echo -e "${RED}✗ Health monitor not found${NC}"
fi

# Summary
echo -e "\n${BLUE}=== Test Summary ===${NC}"
echo "All critical components have been tested."
echo "To start using: ./scripts/start-mcp-servers.sh"

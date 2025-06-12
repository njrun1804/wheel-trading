#!/bin/bash

# Verify trace servers configuration

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Verifying Trace Servers Configuration ===${NC}"
echo ""

# Function to check if a server is in mcp-servers.json
check_mcp_config() {
    local server_name=$1
    if grep -q "\"$server_name\":" mcp-servers.json; then
        echo -e "  ${GREEN}✓${NC} $server_name configured in mcp-servers.json"
        return 0
    else
        echo -e "  ${RED}✗${NC} $server_name NOT in mcp-servers.json"
        return 1
    fi
}

# Function to check if a script exists
check_script() {
    local script_path=$1
    local server_name=$2
    if [ -f "$script_path" ]; then
        echo -e "  ${GREEN}✓${NC} $server_name script exists"
        if [ -x "$script_path" ]; then
            echo -e "  ${GREEN}✓${NC} $server_name script is executable"
        else
            echo -e "  ${YELLOW}⚠${NC} $server_name script is not executable"
        fi
        return 0
    else
        echo -e "  ${RED}✗${NC} $server_name script NOT FOUND at $script_path"
        return 1
    fi
}

# Function to test if script can be imported
test_python_import() {
    local script_path=$1
    local server_name=$2
    
    if python3 -c "import sys; sys.path.insert(0, '$(dirname "$script_path")'); exec(open('$script_path').read())" --help &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $server_name script imports successfully"
        return 0
    else
        echo -e "  ${YELLOW}⚠${NC} $server_name script has import issues (may be normal)"
        return 1
    fi
}

echo -e "${YELLOW}1. Checking MCP Configuration${NC}"
echo ""

# Check main trace servers
check_mcp_config "trace"
check_mcp_config "trace-opik"
check_mcp_config "trace-phoenix"
check_mcp_config "logfire"

echo -e "\n${YELLOW}2. Checking Trace Server Scripts${NC}"
echo ""

SCRIPTS_DIR="$(pwd)/scripts"

# Check trace scripts
check_script "$SCRIPTS_DIR/trace-mcp-server.py" "trace"
check_script "$SCRIPTS_DIR/trace-opik-mcp.py" "trace-opik"
check_script "$SCRIPTS_DIR/trace-phoenix-mcp.py" "trace-phoenix"
check_script "$SCRIPTS_DIR/trace-logfire-mcp.py" "trace-logfire"

echo -e "\n${YELLOW}3. Checking Python Dependencies${NC}"
echo ""

# Check if mcp is installed
if python3 -c "import mcp" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} mcp package installed"
else
    echo -e "  ${RED}✗${NC} mcp package NOT installed"
    echo -e "  ${YELLOW}Install with: pip install mcp${NC}"
fi

# Check if logfire is installed
if python3 -c "import logfire" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} logfire package installed"
else
    echo -e "  ${YELLOW}⚠${NC} logfire package not installed (only needed for trace-logfire-mcp)"
    echo -e "  ${YELLOW}Install with: pip install logfire${NC}"
fi

echo -e "\n${YELLOW}4. Checking Service Availability${NC}"
echo ""

# Check Logfire token
if [ -n "$LOGFIRE_READ_TOKEN" ] || [ -n "$LOGFIRE_TOKEN" ]; then
    echo -e "  ${GREEN}✓${NC} Logfire token found in environment"
elif security find-generic-password -a "$USER" -s "logfire-mcp" -w &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Logfire token found in keychain"
else
    echo -e "  ${YELLOW}⚠${NC} No Logfire token found"
fi

# Check if Opik is running (local)
if curl -s http://localhost:5173/api/health &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Opik platform running locally"
else
    echo -e "  ${YELLOW}⚠${NC} Opik platform not running (start with: cd ~/mcp-servers/opik-platform && ./opik.sh)"
fi

# Check if Phoenix is running (local)
if curl -s http://localhost:6006/healthz &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Phoenix platform running locally"
else
    echo -e "  ${YELLOW}⚠${NC} Phoenix platform not running (start with: phoenix serve)"
fi

echo -e "\n${YELLOW}5. Configuration Summary${NC}"
echo ""

# Count configured servers
TRACE_COUNT=$(grep -c "trace" mcp-servers.json | head -1)
TOTAL_SERVERS=$(jq '.mcpServers | length' mcp-servers.json 2>/dev/null || echo "?")

echo -e "Total MCP servers configured: ${BLUE}$TOTAL_SERVERS${NC}"
echo -e "Trace-related servers: ${BLUE}$TRACE_COUNT${NC}"

echo -e "\n${YELLOW}6. Recommendations${NC}"
echo ""

# Check if all trace servers are configured
ALL_GOOD=true

if ! grep -q "trace-opik" mcp-servers.json; then
    echo -e "• Add ${BLUE}trace-opik${NC} to mcp-servers.json for LLM observability"
    ALL_GOOD=false
fi

if ! grep -q "trace-phoenix" mcp-servers.json; then
    echo -e "• Add ${BLUE}trace-phoenix${NC} to mcp-servers.json for OpenTelemetry traces"
    ALL_GOOD=false
fi

if [ ! -f "$SCRIPTS_DIR/trace-logfire-mcp.py" ]; then
    echo -e "• Create ${BLUE}trace-logfire-mcp.py${NC} for enhanced Logfire integration"
    ALL_GOOD=false
fi

if $ALL_GOOD; then
    echo -e "${GREEN}✓ All trace servers are properly configured!${NC}"
fi

echo -e "\n${GREEN}=== Verification Complete ===${NC}"
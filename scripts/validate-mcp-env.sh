#!/bin/bash

# Validate MCP environment variables

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

validate_token() {
    local var_name=$1
    local pattern=$2
    local description=$3
    
    if [ -z "${!var_name}" ]; then
        echo -e "${RED}✗ $var_name not set${NC}"
        ERRORS=$((ERRORS + 1))
    elif [[ ! "${!var_name}" =~ $pattern ]]; then
        echo -e "${YELLOW}⚠ $var_name format seems incorrect ($description)${NC}"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓ $var_name valid${NC}"
    fi
}

echo -e "${YELLOW}=== Validating MCP Environment ===${NC}"

# Validate API keys
validate_token "GITHUB_TOKEN" "^gh[ps]_[a-zA-Z0-9]{36,}$" "should start with ghp_ or ghs_"
validate_token "BRAVE_API_KEY" "^[a-zA-Z0-9]{32,}$" "should be 32+ alphanumeric"
validate_token "DATABENTO_API_KEY" "^db-[a-zA-Z0-9]{20,}$" "should start with db-"
validate_token "FRED_API_KEY" "^[a-f0-9]{32}$" "should be 32 hex characters"

# Check Python environment
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    if [[ "$PYTHON_VERSION" > "3.9" ]]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ Python $PYTHON_VERSION (3.9+ recommended)${NC}"
    fi
else
    echo -e "${RED}✗ Python not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check Node.js environment
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    echo -e "${GREEN}✓ Node.js v$NODE_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Node.js not found (required for some MCP servers)${NC}"
fi

# Check disk space
DISK_FREE=$(df -h . | awk 'NR==2 {print $4}')
echo -e "${GREEN}✓ Disk space available: $DISK_FREE${NC}"

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}=== Environment valid ===${NC}"
    exit 0
else
    echo -e "${RED}=== $ERRORS issues found ===${NC}"
    echo -e "${YELLOW}Fix the issues above before starting MCP servers${NC}"
    exit 1
fi
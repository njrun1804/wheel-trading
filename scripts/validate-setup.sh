#!/bin/bash
# Final validation of the Wheel Trading setup

echo "🔍 Validating Wheel Trading Setup"
echo "================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Counters
PASSED=0
WARNINGS=0
FAILED=0

# Test function
check() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Checking $name... "
    
    result=$(eval "$command" 2>&1)
    exit_code=$?
    
    if [[ -n "$expected" ]]; then
        if [[ "$result" == *"$expected"* ]]; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAIL${NC} (expected: $expected, got: $result)"
            ((FAILED++))
        fi
    else
        if [[ $exit_code -eq 0 ]]; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAIL${NC}"
            ((FAILED++))
        fi
    fi
}

warn_check() {
    local name="$1"
    local command="$2"
    
    echo -n "Checking $name... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠ WARNING${NC}"
        ((WARNINGS++))
    fi
}

echo "1. Core Files:"
check "CLAUDE.md exists" "[[ -f CLAUDE.md ]]"
check ".zshrc.wheel exists" "[[ -f .zshrc.wheel ]]"
check "startup_unified.sh exists" "[[ -f startup_unified.sh ]]"
check ".wezterm.lua exists" "[[ -f .wezterm.lua ]]"
warn_check "No .wezterm-init.sh" "[[ ! -f .wezterm-init.sh ]]"

echo ""
echo "2. Python Environment:"
check "Python available" "which python"
check "Unity wheel imports" "python -c 'import unity_wheel'"
check "DuckDB imports" "python -c 'import duckdb'"
warn_check "Databento imports" "python -c 'import databento'"

echo ""
echo "3. Database:"
warn_check "Database exists" "[[ -f data/wheel_trading_master.duckdb || -f data/wheel_trading_optimized.duckdb ]]"

echo ""
echo "4. Scripts:"
check "Scripts executable" "[[ -x scripts/wheel-doctor.sh ]]"
check "Claude wrapper" "[[ -x scripts/claude-project.sh ]]"
check "Log viewer" "[[ -x scripts/wheel-logs.sh ]]"

echo ""
echo "5. Environment:"
check ".env file" "[[ -f .env ]]"
warn_check "API keys set" "[[ -n \$DATABENTO_API_KEY ]]"

echo ""
echo "6. Git:"
check "Git repository" "[[ -d .git ]]"
check "On branch" "git branch --show-current"

echo ""
echo "================================="
echo "Summary:"
echo -e "  ${GREEN}Passed: $PASSED${NC}"
if [[ $WARNINGS -gt 0 ]]; then
    echo -e "  ${YELLOW}Warnings: $WARNINGS${NC}"
fi
if [[ $FAILED -gt 0 ]]; then
    echo -e "  ${RED}Failed: $FAILED${NC}"
fi
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}✅ Setup is valid!${NC}"
    if [[ $WARNINGS -gt 0 ]]; then
        echo -e "${YELLOW}   (with $WARNINGS warnings - these are optional)${NC}"
    fi
else
    echo -e "${RED}❌ Setup has issues that need fixing${NC}"
    exit 1
fi
#!/bin/zsh
# Test wheel commands in proper shell context

echo "🧪 Testing Wheel Trading Commands"
echo "================================="
echo ""

# Source the full environment
source ~/.zshrc 2>/dev/null

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Test function
test_command() {
    local name="$1"
    local command="$2"
    
    echo -n "Testing $name... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ works${NC}"
    else
        echo -e "${RED}❌ failed${NC}"
    fi
}

echo "1. Navigation Commands:"
test_command "wheel-root" "declare -f wheel-root"
test_command "wheel-status" "declare -f wheel-status"
test_command "wheel-help" "declare -f wheel-help"

echo ""
echo "2. Trading Commands:"
test_command "wheel-run" "declare -f wheel-run"
test_command "wheel-performance" "declare -f wheel-performance"
test_command "wheel-diagnose" "declare -f wheel-diagnose"
test_command "wheel-export" "declare -f wheel-export"

echo ""
echo "3. Jarvis2 Commands:"
test_command "jarvis2" "declare -f jarvis2"
test_command "jarvis2-stats" "declare -f jarvis2-stats"
test_command "jarvis2-benchmark" "declare -f jarvis2-benchmark"

echo ""
echo "4. Monitoring Commands:"
test_command "wheel-doctor" "which wheel-doctor"
test_command "wheel-logs" "which wheel-logs"
test_command "wheel-monitor" "declare -f wheel-monitor"
test_command "wheel-health" "declare -f wheel-health"

echo ""
echo "5. Data Commands:"
test_command "wheel-data-check" "declare -f wheel-data-check"
test_command "wheel-test" "declare -f wheel-test"

echo ""
echo "6. Claude Integration:"
test_command "wheel-claude" "declare -f wheel-claude"
test_command "claude-project" "which claude-project"

echo ""
echo "7. Quick Test - wheel-status output:"
echo "-----------------------------------"
wheel-status 2>/dev/null || echo "wheel-status not available"
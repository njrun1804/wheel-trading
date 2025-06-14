#!/bin/bash
# Test script to verify initialization is working properly

echo "🧪 Testing Wheel Trading Initialization"
echo "======================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test function
test_item() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Testing $name... "
    
    result=$(eval "$command" 2>&1)
    if [[ -n "$expected" ]]; then
        if [[ "$result" == *"$expected"* ]]; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC} (expected: $expected, got: $result)"
        fi
    else
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC} ($result)"
        fi
    fi
}

echo "1. Environment Variables:"
test_item "WHEEL_TRADING_ROOT" "echo \$WHEEL_TRADING_ROOT" "/wheel-trading"
test_item "Hardware threads" "echo \$OMP_NUM_THREADS" "12"
test_item "Jarvis2 enabled" "echo \$JARVIS2_ENABLED" "1"
test_item "Claude API model" "echo \$ANTHROPIC_MODEL" "claude-3.5-sonnet"
test_item "Metal memory limit" "[[ \$PYTORCH_METAL_WORKSPACE_LIMIT_BYTES -eq $((18*1024*1024*1024)) ]] && echo 'correct'" "correct"

echo ""
echo "2. WezTerm Integration:"
test_item "WezTerm detected" "[[ -n \$TERM_PROGRAM ]] && echo 'yes'" "yes"
test_item "WezTerm pane ID" "[[ -n \$WEZTERM_PANE ]] && echo 'active' || echo 'not active'" "active"

echo ""
echo "3. Commands Available:"
# Functions need to be tested differently in a script
test_item "wheel function" "declare -f wheel >/dev/null && echo 'defined'" "defined"
test_item "wheel-run function" "declare -f wheel-run >/dev/null && echo 'defined'" "defined"
test_item "jarvis2 function" "declare -f jarvis2 >/dev/null && echo 'defined'" "defined"
test_item "wheel-logs alias" "[[ -n \$(alias wheel-logs 2>/dev/null) ]] && echo 'defined' || echo 'not found'" "defined"
test_item "wheel-claude function" "declare -f wheel-claude >/dev/null && echo 'defined'" "defined"

echo ""
echo "4. Directory Structure:"
test_item "Logs directory" "[[ -d logs ]] && echo 'exists'" "exists"
test_item "Data directory" "[[ -d data ]] && echo 'exists'" "exists"
test_item "Scripts directory" "[[ -d scripts ]] && echo 'exists'" "exists"
test_item "CLAUDE.md file" "[[ -f CLAUDE.md ]] && echo 'exists'" "exists"

echo ""
echo "5. Python Environment:"
test_item "Python path includes src" "python -c 'import sys; print(any(\"src\" in p for p in sys.path))'" "True"
test_item "Unity wheel importable" "python -c 'import unity_wheel; print(\"ok\")'" "ok"

echo ""
echo "6. Shell Features:"
test_item "Directory tracking" "cd /tmp && cd - >/dev/null && [[ \$PWD == *wheel-trading* ]] && echo 'works'" "works"

echo ""
echo "======================================="
echo "Run this test in a new terminal to verify initialization!"
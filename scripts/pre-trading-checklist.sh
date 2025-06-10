#!/bin/bash
# Pre-trading checklist - Run this EVERY TIME before using the system

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}          PRE-TRADING SAFETY CHECKLIST                 ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

CHECKS_PASSED=0
CHECKS_FAILED=0

# Function to check a condition
check() {
    local description="$1"
    local command="$2"

    echo -n "[ ] $description... "
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((CHECKS_PASSED++))
    else
        echo -e "${RED}✗${NC}"
        ((CHECKS_FAILED++))
    fi
}

# 1. Environment checks
echo -e "${YELLOW}1. ENVIRONMENT VALIDATION${NC}"
check "No DATABENTO_SKIP_VALIDATION set" '[[ "${DATABENTO_SKIP_VALIDATION:-}" != "true" ]]'
check "DATABENTO_API_KEY is set" '[[ -n "${DATABENTO_API_KEY:-}" ]]'
check "No mock/test environment vars" '! env | grep -iE "mock|dummy|fake|test_mode" | grep -v PATH >/dev/null'

# 2. Code integrity
echo -e "\n${YELLOW}2. CODE INTEGRITY${NC}"
check "No hardcoded prices in wheel.py" '! grep -E "price.*=.*35\.|price.*=.*1\.0" src/unity_wheel/strategy/wheel.py'
check "Live data validator exists" '[[ -f src/unity_wheel/data_providers/validation/live_data_validator.py ]]'
check "No create_mock functions in main" '! grep -E "create_mock|mock_market" src/unity_wheel/cli/run.py'

# 3. Market hours check
echo -e "\n${YELLOW}3. MARKET TIMING${NC}"
current_hour=$(date +%H)
current_day=$(date +%u)
if [[ $current_day -le 5 && $current_hour -ge 9 && $current_hour -lt 16 ]]; then
    check "Market is open" 'true'
else
    check "Market is open" 'false'
    echo -e "  ${YELLOW}⚠️  WARNING: Outside market hours - data may be stale${NC}"
fi

# 4. Data validation
echo -e "\n${YELLOW}4. DATA VALIDATION${NC}"
check "Validation script exists" '[[ -x ./scripts/validate-live-data-only.sh ]]'
check "Data source checker exists" '[[ -x ./scripts/check-data-sources.sh ]]'

# 5. Python environment
echo -e "\n${YELLOW}5. PYTHON ENVIRONMENT${NC}"
check "Can import databento client" 'python3 -c "from src.unity_wheel.data_providers.databento import DatabentoClient"'
check "Can import live validator" 'python3 -c "from src.unity_wheel.data_providers.validation import LiveDataValidator"'
check "Can import main entry point" 'python3 -c "from src.unity_wheel.cli.run import generate_recommendation"'

# 6. Test data fetch (optional but recommended)
echo -e "\n${YELLOW}6. LIVE DATA TEST${NC}"
if [[ "${1:-}" == "--test-fetch" ]]; then
    echo "Testing real data fetch..."
    if python3 -c "
import asyncio
from src.unity_wheel.data_providers.databento import DatabentoClient
async def test():
    client = DatabentoClient()
    price = await client._get_underlying_price('U')
    print(f'  Unity price: \${price.last_price}')
    return price.last_price > 0
success = asyncio.run(test())
exit(0 if success else 1)
" 2>/dev/null; then
        echo -e "  ${GREEN}✓ Successfully fetched live Unity price${NC}"
        ((CHECKS_PASSED++))
    else
        echo -e "  ${RED}✗ Failed to fetch live data${NC}"
        ((CHECKS_FAILED++))
    fi
else
    echo "  Skip test fetch (run with --test-fetch to test)"
fi

# Summary
echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "RESULTS: ${GREEN}$CHECKS_PASSED passed${NC}, ${RED}$CHECKS_FAILED failed${NC}"

if [[ $CHECKS_FAILED -eq 0 ]]; then
    echo -e "\n${GREEN}✅ ALL CHECKS PASSED - SAFE TO TRADE${NC}"
    echo -e "\n${BLUE}Recommended command:${NC}"
    echo "  python run.py --portfolio 100000"
    exit 0
else
    echo -e "\n${RED}❌ SAFETY CHECKS FAILED${NC}"
    echo -e "\n${RED}DO NOT PROCEED WITH TRADING${NC}"
    echo -e "\nFix the issues above before running the system."
    exit 1
fi

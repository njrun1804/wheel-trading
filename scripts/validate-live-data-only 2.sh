#!/usr/bin/env bash
set -euo pipefail
# Script to ensure the system is using ONLY live data, no mock/dummy/fake data


# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Validating Live Data Only Mode..."
echo "================================="

# Track failures
failures=0

# 1. Check environment variables
echo -n "Checking for DATABENTO_SKIP_VALIDATION... "
if [[ "${DATABENTO_SKIP_VALIDATION:-}" == "true" ]]; then
    echo -e "${RED}FAIL${NC}"
    echo "  ❌ DATABENTO_SKIP_VALIDATION is set to true!"
    echo "  This allows the system to run without real data"
    ((failures++))
else
    echo -e "${GREEN}OK${NC}"
fi

# 2. Check for mock data patterns in running processes
echo -n "Checking for mock data patterns in environment... "
if env | grep -iE "mock|dummy|fake|test" | grep -v "PATH" >/dev/null 2>&1; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "  ⚠️  Found potential test/mock environment variables:"
    env | grep -iE "mock|dummy|fake|test" | grep -v "PATH" | head -5
else
    echo -e "${GREEN}OK${NC}"
fi

# 3. Verify Databento API key is set
echo -n "Checking for Databento API key... "
if [[ -z "${DATABENTO_API_KEY:-}" ]]; then
    echo -e "${RED}FAIL${NC}"
    echo "  ❌ DATABENTO_API_KEY not set!"
    echo "  Cannot fetch real market data without API key"
    ((failures++))
else
    echo -e "${GREEN}OK${NC}"
fi

# 4. Check if running during market hours (optional warning)
echo -n "Checking market hours... "
current_hour=$(date +%H)
current_day=$(date +%u)  # 1=Monday, 7=Sunday

if [[ $current_day -ge 6 ]]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "  ⚠️  It's the weekend - market is closed"
elif [[ $current_hour -lt 9 || $current_hour -ge 16 ]]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "  ⚠️  Outside regular market hours (9:30 AM - 4:00 PM ET)"
else
    echo -e "${GREEN}OK${NC}"
fi

# 5. Run Python validation
echo -n "Running Python data validation... "
python_check=$(python3 -c "
import sys
import os

# Check critical imports exist
try:
    from src.unity_wheel.data_providers.databento import DatabentoClient
    from src.unity_wheel.cli.databento_integration import get_market_data_sync
    print('OK')
except ImportError as e:
    print(f'FAIL: {e}')
    sys.exit(1)

# Verify no skip validation
if os.getenv('DATABENTO_SKIP_VALIDATION', '').lower() == 'true':
    print('FAIL: DATABENTO_SKIP_VALIDATION is enabled')
    sys.exit(1)
" 2>&1) || true

if [[ "$python_check" == "OK" ]]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  ❌ Python validation failed: $python_check"
    ((failures++))
fi

# 6. Test actual data fetch (if requested)
if [[ "${1:-}" == "--test-fetch" ]]; then
    echo -n "Testing real data fetch... "
    test_result=$(python3 -c "
import asyncio
from src.unity_wheel.data_providers.databento import DatabentoClient

async def test():
    try:
        client = DatabentoClient()
        price = await client._get_underlying_price('U')
        if price and price.last_price > 0:
            print(f'OK: Unity price = \${price.last_price}')
        else:
            print('FAIL: No price data')
    except Exception as e:
        print(f'FAIL: {e}')

asyncio.run(test())
" 2>&1) || true

    if [[ "$test_result" =~ ^OK ]]; then
        echo -e "${GREEN}OK${NC}"
        echo "  ✅ $test_result"
    else
        echo -e "${RED}FAIL${NC}"
        echo "  ❌ $test_result"
        ((failures++))
    fi
fi

# Summary
echo ""
echo "================================="
if [[ $failures -eq 0 ]]; then
    echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
    echo "System is configured for live data only"
    exit 0
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo "Found $failures critical issues"
    echo ""
    echo "TO FIX:"
    echo "1. Unset DATABENTO_SKIP_VALIDATION"
    echo "2. Ensure DATABENTO_API_KEY is set"
    echo "3. Run only during market hours for best results"
    exit 1
fi

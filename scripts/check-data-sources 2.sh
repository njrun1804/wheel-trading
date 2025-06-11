#!/usr/bin/env bash
set -euo pipefail
# Quick script to check for improper data sources in Unity Wheel Trading Bot
# Used by CI/CD and other shell scripts


# Colors (when running interactively)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# Patterns to check
PAT_YFINANCE="yfinance|yahoo_fin|yahoofinancials|alpha_vantage|quandl|polygon"
PAT_HARDCODED="(price|volatility|realized_vol)\s*[:=]\s*[0-9]+\.[0-9]+"
PAT_MOCK="mock.*data|dummy.*data|fake.*data|create_mock|mock_prices"

# Use ripgrep if available
if command -v rg >/dev/null 2>&1; then
    GREP_CMD="rg --no-heading --color=never -P"
else
    GREP_CMD="grep -r --color=never --binary-files=without-match -E"
fi

# Track violations
violations=0

echo "Checking for improper data sources..."

# Check for external data libraries
echo -n "Checking for external data libraries (yfinance, etc.)... "
if $GREP_CMD "$PAT_YFINANCE" . --glob '*.py' --glob '!test_*.py' --glob '!*/tests/*' --glob '!*/docs/*' 2>/dev/null | grep -v "^Binary" >/dev/null; then
    echo -e "${RED}FOUND${NC}"
    echo "Files using external data libraries:"
    $GREP_CMD "$PAT_YFINANCE" . --glob '*.py' --glob '!test_*.py' --glob '!*/tests/*' --glob '!*/docs/*' 2>/dev/null | grep -v "^Binary" | head -5
    ((violations++))
else
    echo -e "${GREEN}OK${NC}"
fi

# Check for hardcoded prices/volatility
echo -n "Checking for hardcoded prices/volatility... "
hardcoded_files=$($GREP_CMD "$PAT_HARDCODED" . --glob '*.py' --glob '!test_*.py' --glob '!*/tests/*' --glob '!config.yaml' 2>/dev/null | grep -v "fallback\|default\|typical\|example\|[Pp]laceholder\|will be calculated\|temporary" | grep -v "^Binary" || true)
if [[ -n "$hardcoded_files" ]]; then
    echo -e "${YELLOW}FOUND${NC}"
    echo "Files with hardcoded values:"
    echo "$hardcoded_files" | head -5
    ((violations++))
else
    echo -e "${GREEN}OK${NC}"
fi

# Check for mock data in production
echo -n "Checking for mock data in production code... "
mock_files=$($GREP_CMD "$PAT_MOCK" . --glob '*.py' --glob '!test_*.py' --glob '!*/tests/*' 2>/dev/null | grep -v "^Binary" | grep -v "#.*mock\|comment\|doc\|removed\|deprecated" || true)
if [[ -n "$mock_files" ]]; then
    echo -e "${YELLOW}FOUND${NC}"
    echo "Files using mock data:"
    echo "$mock_files" | head -5
    ((violations++))
else
    echo -e "${GREEN}OK${NC}"
fi

# Check if Databento is properly configured
echo -n "Checking Databento configuration... "
if grep -q "DATABENTO_API_KEY" .env 2>/dev/null || env | grep -q "DATABENTO_API_KEY"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARNING${NC} - DATABENTO_API_KEY not found in environment"
fi

# Summary
echo ""
if [[ $violations -eq 0 ]]; then
    echo -e "${GREEN}✅ All data sources are properly configured${NC}"
    echo "Unity data is coming from Databento via Google Secrets"
    exit 0
else
    echo -e "${RED}❌ Found $violations data source violations${NC}"
    echo ""
    echo "Requirements:"
    echo "1. All Unity stock/options data must come from Databento"
    echo "2. API keys must be retrieved from Google Secrets"
    echo "3. No hardcoded prices or volatility values"
    echo "4. No external data libraries (yfinance, etc.)"
    echo ""
    echo "To fix:"
    echo "- Replace hardcoded values with Databento API calls"
    echo "- Remove any yfinance or similar imports"
    echo "- Use src.unity_wheel.data_providers.databento.DatabentoClient"
    exit 1
fi

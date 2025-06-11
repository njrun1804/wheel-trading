#!/bin/bash
# Check that data validation with hard exits is properly implemented

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔍 Checking data validation implementation..."

# Function to check file for proper validation
check_file() {
    local file="$1"
    local errors=0

    # Skip test files and examples
    if [[ "$file" == *"test_"* ]] || [[ "$file" == *"/tests/"* ]] || [[ "$file" == *"/examples/"* ]]; then
        return 0
    fi

    # Check for data access without validation
    if grep -q "market_snapshot\[" "$file" 2>/dev/null; then
        # Check if validate_market_data is called
        if ! grep -q "validate_market_data\|DataValidator\.validate_market_snapshot" "$file" 2>/dev/null; then
            echo -e "${YELLOW}Warning: $file accesses market_snapshot without validation${NC}"
            ((errors++))
        fi
    fi

    # Check for option chain access without validation
    if grep -q "option_chain\[" "$file" 2>/dev/null; then
        if ! grep -q "validate_option_data\|DataValidator\.validate_option" "$file" 2>/dev/null; then
            echo -e "${YELLOW}Warning: $file accesses option_chain without validation${NC}"
            ((errors++))
        fi
    fi

    # Check for API calls without validation
    if grep -q "client\.\(get\|post\|put\|delete\)" "$file" 2>/dev/null; then
        if ! grep -q "validate_api_response\|DataValidator\.validate_api" "$file" 2>/dev/null; then
            echo -e "${YELLOW}Warning: $file makes API calls without validation${NC}"
            ((errors++))
        fi
    fi

    return $errors
}

# Check for die() function usage
echo "Checking for die() function usage..."
DIE_COUNT=$(grep -r "def die\|die(" src/unity_wheel --include="*.py" 2>/dev/null | wc -l)
if [ "$DIE_COUNT" -lt 10 ]; then
    echo -e "${RED}Error: Insufficient die() usage found (only $DIE_COUNT instances)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found $DIE_COUNT die() usages${NC}"

# Check for validate_ function usage
echo "Checking for validate_ function usage..."
VALIDATE_COUNT=$(grep -r "validate_" src/unity_wheel --include="*.py" 2>/dev/null | grep -v "def validate_" | wc -l)
if [ "$VALIDATE_COUNT" -lt 5 ]; then
    echo -e "${RED}Error: Insufficient validate_ usage found (only $VALIDATE_COUNT instances)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found $VALIDATE_COUNT validate_ usages${NC}"

# Check specific files that must have validation
CRITICAL_FILES=(
    "examples/advisor_simple.py"
    "src/unity_wheel/portfolio/single_account.py"
    "src/unity_wheel/execution/unity_fill_model.py"
)

echo "Checking critical files for validation..."
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        if ! grep -q "die\|DataValidator" "$file" 2>/dev/null; then
            echo -e "${RED}Error: Critical file $file missing data validation${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ $file has validation${NC}"
    fi
done

# Check for None checks that should use die()
echo "Checking for soft None checks that should be hard failures..."
SOFT_NONE_COUNT=$(grep -r "if .* is None:" src/unity_wheel --include="*.py" | grep -v "die\|raise\|sys.exit" | wc -l)
if [ "$SOFT_NONE_COUNT" -gt 20 ]; then
    echo -e "${YELLOW}Warning: Found $SOFT_NONE_COUNT soft None checks that might need die()${NC}"
fi

# Check for try/except blocks that might hide data errors
echo "Checking for broad exception handling..."
BROAD_EXCEPT=$(grep -r "except:\|except Exception" src/unity_wheel --include="*.py" | grep -v "as e" | wc -l)
if [ "$BROAD_EXCEPT" -gt 5 ]; then
    echo -e "${YELLOW}Warning: Found $BROAD_EXCEPT broad exception handlers that might hide data errors${NC}"
fi

# Check for missing data in key modules
echo "Scanning for potential missing data access..."
total_warnings=0
for file in $(find src/unity_wheel -name "*.py" -type f); do
    if check_file "$file"; then
        ((total_warnings+=$?)) || true
    fi
done

# Summary
echo ""
echo "📊 Data Validation Check Summary:"
echo "================================"
echo -e "die() usages: ${GREEN}$DIE_COUNT${NC}"
echo -e "validate_ usages: ${GREEN}$VALIDATE_COUNT${NC}"
echo -e "Soft None checks: ${YELLOW}$SOFT_NONE_COUNT${NC}"
echo -e "Broad exceptions: ${YELLOW}$BROAD_EXCEPT${NC}"
echo -e "Total warnings: ${YELLOW}$total_warnings${NC}"

if [ "$total_warnings" -gt 10 ]; then
    echo -e "\n${RED}❌ Too many validation warnings! Please add proper data validation.${NC}"
    exit 1
else
    echo -e "\n${GREEN}✅ Data validation check passed!${NC}"
fi

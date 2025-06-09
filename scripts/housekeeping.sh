#!/bin/bash
# Project housekeeping script - checks for organization and hardcoded values

set -euo pipefail

echo "🧹 Unity Wheel Bot - Project Housekeeping"
echo "========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track if any issues found
ISSUES_FOUND=0

# Check test files organization
echo "📁 Checking test files organization..."
TEST_FILES_OUTSIDE=$(find . -name "test_*.py" -not -path "./tests/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" 2>/dev/null | wc -l)
if [ "$TEST_FILES_OUTSIDE" -gt 0 ]; then
    echo -e "${RED}❌ Found $TEST_FILES_OUTSIDE test files outside tests/${NC}"
    find . -name "test_*.py" -not -path "./tests/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" 2>/dev/null
    ISSUES_FOUND=1
else
    echo -e "${GREEN}✅ All test files properly organized${NC}"
fi
echo ""

# Check example files organization
echo "📁 Checking example files organization..."
EXAMPLE_FILES_OUTSIDE=$(find . -name "example_*.py" -not -path "./examples/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" 2>/dev/null | wc -l)
if [ "$EXAMPLE_FILES_OUTSIDE" -gt 0 ]; then
    echo -e "${RED}❌ Found $EXAMPLE_FILES_OUTSIDE example files outside examples/${NC}"
    find . -name "example_*.py" -not -path "./examples/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./build/*" 2>/dev/null
    ISSUES_FOUND=1
else
    echo -e "${GREEN}✅ All example files properly organized${NC}"
fi
echo ""

# Check for status docs in root
echo "📄 Checking for status docs in root..."
STATUS_DOCS=$(ls -1 *.md 2>/dev/null | grep -E "(SUMMARY|STATUS|COMPLETE|REPORT)\.md$" | wc -l)
if [ "$STATUS_DOCS" -gt 0 ]; then
    echo -e "${RED}❌ Found $STATUS_DOCS status/summary docs in root${NC}"
    ls -1 *.md 2>/dev/null | grep -E "(SUMMARY|STATUS|COMPLETE|REPORT)\.md$"
    echo "   Consider moving to docs/archive/"
    ISSUES_FOUND=1
else
    echo -e "${GREEN}✅ No status docs in root${NC}"
fi
echo ""

# Check for data scripts organization
echo "🔧 Checking data scripts organization..."
DATA_SCRIPTS_OUTSIDE=$(find . -name "pull_*_data.py" -not -path "./tools/*" -not -path "./venv/*" -not -path "./.venv/*" 2>/dev/null | wc -l)
if [ "$DATA_SCRIPTS_OUTSIDE" -gt 0 ]; then
    echo -e "${RED}❌ Found $DATA_SCRIPTS_OUTSIDE data scripts outside tools/${NC}"
    find . -name "pull_*_data.py" -not -path "./tools/*" -not -path "./venv/*" -not -path "./.venv/*" 2>/dev/null
    ISSUES_FOUND=1
else
    echo -e "${GREEN}✅ Data scripts properly organized${NC}"
fi
echo ""

# Check for empty directories
echo "📂 Checking for empty directories..."
EMPTY_DIRS=$(find ./src -type d -empty 2>/dev/null | wc -l)
if [ "$EMPTY_DIRS" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $EMPTY_DIRS empty directories${NC}"
    find ./src -type d -empty 2>/dev/null
    echo "   Consider removing if not needed"
fi
echo ""

# Check for hardcoded values
echo "🔍 Checking for hardcoded values..."
echo ""

# Risk limits
RISK_LIMITS=$(grep -r 'max_.*=' src/ 2>/dev/null | grep -E '0\.[0-9]+|[1-9][0-9]*' | grep -v config | grep -v test | wc -l)
echo "  Risk limits: $RISK_LIMITS hardcoded values found"
if [ "$RISK_LIMITS" -gt 0 ]; then
    echo -e "${YELLOW}  Examples:${NC}"
    grep -r 'max_.*=' src/ 2>/dev/null | grep -E '0\.[0-9]+|[1-9][0-9]*' | grep -v config | grep -v test | head -3
fi

# Thresholds
THRESHOLDS=$(grep -r 'threshold.*=' src/ 2>/dev/null | grep -E '0\.[0-9]+|[1-9][0-9]*' | grep -v config | grep -v test | wc -l)
echo "  Thresholds: $THRESHOLDS hardcoded values found"
if [ "$THRESHOLDS" -gt 0 ]; then
    echo -e "${YELLOW}  Examples:${NC}"
    grep -r 'threshold.*=' src/ 2>/dev/null | grep -E '0\.[0-9]+|[1-9][0-9]*' | grep -v config | grep -v test | head -3
fi

# Default constants
DEFAULTS=$(grep -r 'DEFAULT_' src/ 2>/dev/null | grep -v config | grep -v test | wc -l)
echo "  Default constants: $DEFAULTS found"
if [ "$DEFAULTS" -gt 0 ]; then
    echo -e "${YELLOW}  Examples:${NC}"
    grep -r 'DEFAULT_' src/ 2>/dev/null | grep -v config | grep -v test | head -3
fi

# Timeouts and retries
TIMEOUTS=$(grep -r -E 'timeout.*=.*[0-9]+|retry.*=.*[0-9]+' src/ 2>/dev/null | grep -v config | grep -v test | wc -l)
echo "  Timeouts/retries: $TIMEOUTS hardcoded values found"
if [ "$TIMEOUTS" -gt 0 ]; then
    echo -e "${YELLOW}  Examples:${NC}"
    grep -r -E 'timeout.*=.*[0-9]+|retry.*=.*[0-9]+' src/ 2>/dev/null | grep -v config | grep -v test | head -3
fi

TOTAL_HARDCODED=$((RISK_LIMITS + THRESHOLDS + DEFAULTS + TIMEOUTS))
if [ "$TOTAL_HARDCODED" -gt 20 ]; then
    echo -e "${YELLOW}⚠️  Consider moving these to config.yaml or making them adaptive${NC}"
    echo "   See HOUSEKEEPING_GUIDE.md for smart value implementation examples"
fi
echo ""

# Summary statistics
echo "📊 Project Statistics:"
echo "  Root files: $(ls -1 | grep -v -E '^(src|tests|examples|tools|scripts|deployment|docs|data|exports|venv|htmlcov|\.github)$' | wc -l)"
echo "  Test files: $(find tests -name "test_*.py" -type f 2>/dev/null | wc -l)"
echo "  Example files: $(find examples -name "*.py" -type f 2>/dev/null | wc -l)"
echo "  Tool scripts: $(find tools -name "*.py" -type f 2>/dev/null | wc -l)"
echo "  Source modules: $(find src -name "*.py" -type f 2>/dev/null | wc -l)"
echo ""

# Final status
if [ "$ISSUES_FOUND" -eq 0 ]; then
    echo -e "${GREEN}✨ Housekeeping check passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  - Review hardcoded values for potential improvements"
    echo "  - Consider implementing smart/adaptive parameters"
    echo "  - See docs/archive/SMART_VALUES_MIGRATION.md for guidance"
else
    echo -e "${RED}❌ Housekeeping issues found!${NC}"
    echo ""
    echo "Please run: 'Please run project housekeeping according to HOUSEKEEPING_GUIDE.md'"
    exit 1
fi
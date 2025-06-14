#!/bin/bash
# Master script to clean up wheel trading test suite

set -e  # Exit on error

echo "======================================================================"
echo "Wheel Trading Test Suite Cleanup - Master Script"
echo "======================================================================"
echo ""
echo "This script will:"
echo "1. Set up test infrastructure"
echo "2. Fix import issues"
echo "3. Remove mocks and modernize tests"
echo "4. Fix diagnose mode"
echo "5. Configure optimized test runner"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

print_step() {
    echo -e "\n${YELLOW}$1${NC}"
    echo "----------------------------------------------------------------------"
}

# Change to project root
cd "$(dirname "$0")/../.."

# Step 1: Set up test infrastructure
print_step "Step 1: Setting up test infrastructure"
python scripts/test_cleanup/setup_test_infrastructure.py
print_status $? "Test infrastructure setup complete"

# Step 2: Fix imports (dry run first)
print_step "Step 2: Analyzing import issues"
python scripts/test_cleanup/fix_imports.py
echo ""
echo "Review the import issues above. Apply fixes? (y/n)"
read -n 1 apply_imports
echo ""
if [ "$apply_imports" = "y" ]; then
    python scripts/test_cleanup/fix_imports.py --fix
    print_status $? "Import fixes applied"
else
    echo "Skipping import fixes"
fi

# Step 3: Remove mocks (dry run first)
print_step "Step 3: Analyzing mock usage"
python scripts/test_cleanup/remove_mocks.py | head -50
echo ""
echo "... (output truncated)"
echo ""
echo "Remove mocks and modernize tests? (y/n)"
read -n 1 remove_mocks
echo ""
if [ "$remove_mocks" = "y" ]; then
    python scripts/test_cleanup/remove_mocks.py --apply
    print_status $? "Mocks removed and tests modernized"
    
    # Create example patterns
    python scripts/test_cleanup/remove_mocks.py --examples
    print_status $? "Example test patterns created"
else
    echo "Skipping mock removal"
fi

# Step 4: Fix diagnose mode
print_step "Step 4: Fixing diagnose mode"
python scripts/test_cleanup/fix_diagnose_mode.py
print_status $? "Diagnose mode fixed"

# Test diagnostics
echo "Testing fixed diagnostics..."
python scripts/test_cleanup/test_diagnostics.py
if [ $? -eq 0 ]; then
    print_status 0 "Diagnostics test passed"
else
    echo -e "${YELLOW}⚠ Diagnostics test failed - may need manual review${NC}"
fi

# Step 5: Set up optimized test runner
print_step "Step 5: Configuring optimized test runner"
python scripts/test_cleanup/run_tests_optimized.py --create-scripts
print_status $? "Test runner configured"

# Make scripts executable
chmod +x test-*.sh
chmod +x scripts/test_cleanup/*.py

# Summary
echo ""
echo "======================================================================"
echo -e "${GREEN}Test Suite Cleanup Complete!${NC}"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Run quick tests:     ./test-quick.sh"
echo "2. Run all tests:       ./test-all.sh"
echo "3. Run with coverage:   ./test-coverage.sh"
echo "4. Test diagnose mode:  python -m unity_wheel.cli.run --diagnose"
echo ""
echo "Test database created at: data/wheel_trading_test.duckdb"
echo "Test fixtures in:         tests/fixtures/"
echo "Example patterns in:      tests/examples/"
echo ""
echo "If you encounter issues:"
echo "- Check backup files (*.backup)"
echo "- Review import validator: python scripts/test_cleanup/validate_imports.py"
echo "- Check test database: duckdb data/wheel_trading_test.duckdb"
echo ""
echo "======================================================================"
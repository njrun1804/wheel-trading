#!/bin/bash
# Local CI Runner - Simulates GitHub Actions environment locally
# This prevents the iterative fix-push-wait cycle

set -e  # Exit on error

echo "🚀 Running Local CI Simulation"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}1. Checking Python version...${NC}"
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo -e "${RED}❌ Python 3.12 required (CI uses 3.12), but you have $PYTHON_VERSION${NC}"
    echo "Consider using pyenv or conda to match CI environment"
    exit 1
else
    echo -e "${GREEN}✅ Python $PYTHON_VERSION matches CI${NC}"
fi

# Stage 1: Quick syntax and import validation (5 seconds)
echo -e "\n${YELLOW}2. Stage 1: Syntax & Import Validation (5s)...${NC}"
python -m py_compile src/unity_wheel/**/*.py 2>/dev/null || {
    echo -e "${RED}❌ Syntax errors found${NC}"
    exit 1
}

# Test imports
python -c "import unity_wheel" || {
    echo -e "${RED}❌ Import errors found${NC}"
    exit 1
}

pytest --collect-only -q || {
    echo -e "${RED}❌ Test collection failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Syntax and imports OK${NC}"

# Stage 2: Linting (30 seconds)
echo -e "\n${YELLOW}3. Stage 2: Code Quality Checks (30s)...${NC}"
ruff check src/unity_wheel --quiet || {
    echo -e "${RED}❌ Linting failed${NC}"
    echo "Run 'ruff check src/unity_wheel' to see details"
    exit 1
}
echo -e "${GREEN}✅ Linting passed${NC}"

# Stage 3: Fast unit tests (1 minute)
echo -e "\n${YELLOW}4. Stage 3: Fast Unit Tests (1m)...${NC}"
pytest -v -m "not slow" --tb=short -x || {
    echo -e "${RED}❌ Unit tests failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Unit tests passed${NC}"

# Optional: Full test suite
if [[ "$1" == "--full" ]]; then
    echo -e "\n${YELLOW}5. Running full test suite...${NC}"
    pytest -v --cov=src/unity_wheel || {
        echo -e "${RED}❌ Full test suite failed${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ Full test suite passed${NC}"
fi

echo -e "\n${GREEN}🎉 Local CI simulation passed! Safe to push.${NC}"
echo -e "${YELLOW}Tip: Use 'git push' to trigger real CI${NC}"
#!/bin/bash
# Test runner script with common scenarios

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PROFILE="default"
TIMEOUT="60"
WORKERS="auto"

# Show help
show_help() {
    echo "Unity Wheel Bot - Test Runner"
    echo ""
    echo "Usage: ./scripts/test.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  fast      Run fast tests only (no slow/integration)"
    echo "  unit      Run unit tests only"
    echo "  slow      Run slow tests only"
    echo "  all       Run all tests"
    echo "  failed    Run previously failed tests"
    echo "  math      Run math module tests"
    echo "  coverage  Run with coverage report"
    echo "  profile   Profile test performance"
    echo ""
    echo "Options:"
    echo "  --timeout N    Set timeout in seconds (default: 60)"
    echo "  --workers N    Number of parallel workers (default: auto)"
    echo "  --verbose      Extra verbose output"
    echo "  --pdb          Drop into debugger on failure"
    echo ""
    echo "Examples:"
    echo "  ./scripts/test.sh fast"
    echo "  ./scripts/test.sh unit --timeout 30"
    echo "  ./scripts/test.sh coverage --workers 4"
}

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found. Run: python -m venv venv${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo -e "${YELLOW}Warning: Expected Python 3.11, got $PYTHON_VERSION${NC}"
fi

# Parse command
COMMAND=${1:-help}
shift || true

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="-vv"
            shift
            ;;
        --pdb)
            PDB="--pdb"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Base pytest command
BASE_CMD="pytest --timeout=$TIMEOUT"

# Add parallel execution if not debugging
if [ -z "$PDB" ]; then
    BASE_CMD="$BASE_CMD -n $WORKERS --dist loadscope"
fi

# Add verbose flag if set
if [ -n "$VERBOSE" ]; then
    BASE_CMD="$BASE_CMD $VERBOSE"
fi

# Add PDB flag if set
if [ -n "$PDB" ]; then
    BASE_CMD="$BASE_CMD $PDB"
fi

# Execute command
case $COMMAND in
    fast)
        echo -e "${GREEN}Running fast tests (no slow/integration)...${NC}"
        $BASE_CMD -m "not slow and not integration" --durations=10
        ;;
    unit)
        echo -e "${GREEN}Running unit tests only...${NC}"
        $BASE_CMD -m unit --durations=10
        ;;
    slow)
        echo -e "${GREEN}Running slow tests only...${NC}"
        $BASE_CMD -m slow --timeout=300
        ;;
    all)
        echo -e "${GREEN}Running all tests...${NC}"
        $BASE_CMD --durations=20
        ;;
    failed)
        echo -e "${GREEN}Running previously failed tests...${NC}"
        $BASE_CMD --lf
        ;;
    math)
        echo -e "${GREEN}Running math module tests...${NC}"
        $BASE_CMD tests/test_math*.py -v --durations=10
        ;;
    coverage)
        echo -e "${GREEN}Running tests with coverage...${NC}"
        $BASE_CMD --cov=src --cov-report=html --cov-report=term-missing
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    profile)
        echo -e "${GREEN}Profiling test performance...${NC}"
        $BASE_CMD --durations=30 -v
        ;;
    help|*)
        show_help
        ;;
esac

# Show summary
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Tests completed successfully!${NC}"
else
    echo -e "\n${RED}✗ Tests failed!${NC}"
    exit 1
fi

#!/bin/bash
# Optimized pre-commit installation script

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing pre-commit hooks...${NC}"

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo -e "${YELLOW}pre-commit not found. Installing...${NC}"
    pip install pre-commit || {
        echo -e "${RED}Failed to install pre-commit${NC}"
        exit 1
    }
fi

# Install pre-commit hooks
echo -e "${GREEN}Setting up git hooks...${NC}"
pre-commit install --install-hooks

# Run against all files to check current state
echo -e "${GREEN}Running pre-commit on all files (first run may take a moment)...${NC}"
pre-commit run --all-files || {
    echo -e "${YELLOW}Some checks failed. This is normal for first run.${NC}"
    echo -e "${YELLOW}The hooks will auto-fix many issues on commit.${NC}"
}

# Update hooks to latest versions
echo -e "${GREEN}Updating hooks to latest versions...${NC}"
pre-commit autoupdate || {
    echo -e "${YELLOW}Could not auto-update all hooks. Check .pre-commit-config.yaml${NC}"
}

echo -e "${GREEN}✅ Pre-commit hooks installed successfully!${NC}"
echo ""
echo "Hooks will run automatically on:"
echo "  - git commit (on staged files)"
echo "  - pre-commit run --all-files (manual run)"
echo ""
echo "To skip hooks temporarily: git commit --no-verify"

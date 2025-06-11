#!/usr/bin/env bash
set -euo pipefail
# Install git hooks for data validation enforcement


# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📦 Installing git hooks...${NC}"

# Get the git directory
GIT_DIR=$(git rev-parse --git-dir 2>/dev/null || echo ".git")

if [ ! -d "$GIT_DIR" ]; then
    echo -e "${RED}❌ Not a git repository!${NC}"
    exit 1
fi

# Create hooks directory if it doesn't exist
mkdir -p "$GIT_DIR/hooks"

# Copy pre-commit hook
if [ -f ".githooks/pre-commit" ]; then
    cp .githooks/pre-commit "$GIT_DIR/hooks/pre-commit"
    chmod +x "$GIT_DIR/hooks/pre-commit"
    echo -e "${GREEN}✅ Installed pre-commit hook${NC}"
else
    echo -e "${RED}❌ Pre-commit hook not found in .githooks/${NC}"
    exit 1
fi

# Optionally set git to use our hooks directory
echo -e "${YELLOW}Setting git hooks path...${NC}"
git config core.hooksPath .githooks

echo -e "${GREEN}✅ Git hooks installed successfully!${NC}"
echo -e "${YELLOW}The following checks will run before each commit:${NC}"
echo "  • Housekeeping validation"
echo "  • Data validation checks"
echo "  • Print statement detection"
echo "  • TODO/FIXME detection"
echo ""
echo -e "${YELLOW}To bypass hooks (not recommended): git commit --no-verify${NC}"

#!/usr/bin/env bash
# Install and configure pre-commit hooks for Unity Wheel Trading Bot
# This script sets up the optimized pre-commit framework
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Unity Wheel Pre-commit Installer${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Not in a git repository${NC}"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: Python 3 is required${NC}"
    exit 1
fi

# Function to install pre-commit
install_precommit() {
    echo -e "${YELLOW}📦 Installing pre-commit framework...${NC}"
    
    # Try different installation methods
    if command -v pip3 &> /dev/null; then
        pip3 install --user pre-commit
    elif command -v pip &> /dev/null; then
        pip install --user pre-commit
    elif command -v poetry &> /dev/null; then
        poetry add --group dev pre-commit
    else
        echo -e "${RED}❌ No package manager found (pip/poetry)${NC}"
        echo "Please install pre-commit manually: https://pre-commit.com/#install"
        exit 1
    fi
}

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo -e "${YELLOW}⚠️  pre-commit not found${NC}"
    read -p "Install pre-commit now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_precommit
    else
        echo -e "${RED}❌ pre-commit is required for git hooks${NC}"
        exit 1
    fi
fi

# Verify pre-commit config exists
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo -e "${RED}❌ Error: .pre-commit-config.yaml not found${NC}"
    exit 1
fi

# Update pre-commit hooks to latest versions
echo -e "${YELLOW}🔄 Updating hook versions...${NC}"
pre-commit autoupdate || true

# Install the git hook scripts
echo -e "${YELLOW}🔧 Installing git hooks...${NC}"
pre-commit install --install-hooks

# Install additional hook types
echo -e "${YELLOW}🔧 Installing additional hook types...${NC}"
pre-commit install --hook-type pre-push || true
pre-commit install --hook-type commit-msg || true

# Run hooks on all files to check everything is working
echo -e "${YELLOW}🧪 Testing hooks on existing files...${NC}"
if pre-commit run --all-files; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
else
    echo -e "${YELLOW}⚠️  Some checks failed - this is normal for existing code${NC}"
    echo -e "${YELLOW}   The hooks will help maintain code quality going forward${NC}"
fi

# Show installed hooks
echo ""
echo -e "${GREEN}✅ Git hooks installed successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Installed Hooks:${NC}"
echo "  ${GREEN}Pre-commit:${NC}"
echo "    • Trailing whitespace removal"
echo "    • End-of-file fixing"
echo "    • YAML/TOML/JSON validation"
echo "    • Python formatting (Black + isort)"
echo "    • Security scanning (Bandit)"
echo "    • Unity-specific checks"
echo "    • Config validation"
echo ""
echo -e "${BLUE}📖 Quick Reference:${NC}"
echo "  • Run manually: ${YELLOW}pre-commit run --all-files${NC}"
echo "  • Skip hooks: ${YELLOW}git commit --no-verify${NC}"
echo "  • Update hooks: ${YELLOW}pre-commit autoupdate${NC}"
echo "  • Uninstall: ${YELLOW}pre-commit uninstall${NC}"
echo ""
echo -e "${GREEN}✨ Setup complete! Happy coding!${NC}"
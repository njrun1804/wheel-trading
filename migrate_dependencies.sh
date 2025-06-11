#!/bin/bash
# Dependency Migration Script for Unity Wheel Bot
# This script helps migrate from Python 3.13 to Python 3.11 with cleaned dependencies

set -e  # Exit on error

echo "Unity Wheel Bot - Dependency Migration Script"
echo "============================================="
echo ""

# Check current Python version
echo "Current Python version:"
python --version

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv is not installed. Please install pyenv first:"
    echo "   brew install pyenv"
    exit 1
fi

echo ""
echo "This script will:"
echo "1. Backup current environment"
echo "2. Install Python 3.11.10"
echo "3. Create new virtual environment"
echo "4. Install cleaned dependencies"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Migration cancelled."
    exit 0
fi

# Step 1: Backup current environment
echo ""
echo "Step 1: Backing up current environment..."
if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements-backup-$(date +%Y%m%d-%H%M%S).txt
fi
if [ -d "venv" ]; then
    echo "  Creating pip freeze backup..."
    source venv/bin/activate 2>/dev/null || true
    pip freeze > requirements-freeze-backup-$(date +%Y%m%d-%H%M%S).txt
    deactivate 2>/dev/null || true
fi

# Step 2: Install Python 3.11.10
echo ""
echo "Step 2: Installing Python 3.11.10..."
if pyenv versions | grep -q "3.11.10"; then
    echo "  Python 3.11.10 already installed"
else
    echo "  Installing Python 3.11.10 (this may take a few minutes)..."
    pyenv install 3.11.10
fi

# Step 3: Set local Python version
echo ""
echo "Step 3: Setting local Python version to 3.11.10..."
pyenv local 3.11.10

# Verify version
NEW_PYTHON_VERSION=$(python --version)
echo "  New Python version: $NEW_PYTHON_VERSION"

if [[ ! $NEW_PYTHON_VERSION =~ "3.11.10" ]]; then
    echo "❌ Failed to switch to Python 3.11.10"
    echo "   You may need to restart your shell or run: eval \"\$(pyenv init -)\""
    exit 1
fi

# Step 4: Remove old virtual environment
echo ""
echo "Step 4: Removing old virtual environment..."
if [ -d "venv" ]; then
    rm -rf venv
    echo "  Old venv removed"
fi

# Step 5: Create new virtual environment
echo ""
echo "Step 5: Creating new virtual environment..."
python -m venv venv
source venv/bin/activate

# Step 6: Upgrade pip
echo ""
echo "Step 6: Upgrading pip..."
pip install --upgrade pip

# Step 7: Install new dependencies
echo ""
echo "Step 7: Installing cleaned dependencies..."
if [ -f "requirements-recommended.txt" ]; then
    echo "  Installing from requirements-recommended.txt..."
    pip install -r requirements-recommended.txt
else
    echo "❌ requirements-recommended.txt not found!"
    echo "   Please ensure the file exists before running this script."
    exit 1
fi

if [ -f "requirements-dev-recommended.txt" ]; then
    echo "  Installing dev dependencies..."
    pip install -r requirements-dev-recommended.txt
fi

# Step 8: Verify key packages
echo ""
echo "Step 8: Verifying key package versions..."
echo "  Python: $(python --version)"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "  SciPy: $(python -c 'import scipy; print(scipy.__version__)')"
echo "  Pandas: $(python -c 'import pandas; print(pandas.__version__)')"
echo "  Pytest: $(python -c 'import pytest; print(pytest.__version__)')"

# Step 9: Run quick test
echo ""
echo "Step 9: Running quick test..."
if pytest tests/test_math_simple.py -v --tb=short; then
    echo "✅ Quick test passed!"
else
    echo "⚠️  Quick test failed - you may need to fix some code compatibility issues"
fi

# Step 10: Update requirements files
echo ""
echo "Step 10: Updating official requirements files..."
read -p "Replace requirements.txt with requirements-recommended.txt? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cp requirements-recommended.txt requirements.txt
    cp requirements-dev-recommended.txt requirements-dev.txt
    echo "✅ Requirements files updated"
fi

# Final summary
echo ""
echo "============================================="
echo "Migration Complete!"
echo "============================================="
echo ""
echo "✅ Python 3.11.10 installed and activated"
echo "✅ New virtual environment created"
echo "✅ Dependencies installed"
echo ""
echo "Next steps:"
echo "1. Run full test suite: pytest -m 'not slow' --timeout=60"
echo "2. Update pyproject.toml with recommended changes"
echo "3. Run: pip install -e . (if using editable install)"
echo "4. Commit the updated dependency files"
echo ""
echo "If you encounter issues:"
echo "- Check DEPENDENCY_ASSESSMENT.md for detailed recommendations"
echo "- Review backup files created in this directory"
echo "- Run: source venv/bin/activate (to activate virtual environment)"
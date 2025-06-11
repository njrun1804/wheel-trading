#!/usr/bin/env bash
set -euo pipefail
# Fix Import Conflicts - Handles the math module shadowing issue
# This is a workaround for the unity_trading.math shadowing stdlib math

echo "🔧 FIXING IMPORT CONFLICTS"
echo "========================="
echo ""
echo "The project has a 'math' module that conflicts with Python's stdlib."
echo "This script installs packages using alternative methods."
echo ""

# Method 1: Use system pip if available
echo "Method 1: Trying system pip..."
if command -v pip3 >/dev/null 2>&1; then
    pip3 install --user numpy pandas scipy pydantic python-dateutil 2>/dev/null && {
        echo "✅ Installed via system pip"
        exit 0
    }
fi

# Method 2: Download and use get-pip
echo ""
echo "Method 2: Using get-pip.py..."
cd /tmp
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user 2>/dev/null && {
    python3 -m pip install --user numpy pandas scipy pydantic python-dateutil
    echo "✅ Installed via get-pip"
    cd - >/dev/null
    exit 0
}

# Method 3: Use conda/mamba if available
echo ""
echo "Method 3: Checking for conda..."
if command -v conda >/dev/null 2>&1; then
    conda install -y numpy pandas scipy pydantic python-dateutil 2>/dev/null && {
        echo "✅ Installed via conda"
        exit 0
    }
fi

# Method 4: Manual installation reminder
echo ""
echo "⚠️  Automatic installation failed."
echo ""
echo "📋 Manual installation steps:"
echo "1. Open a new terminal (outside this project directory)"
echo "2. Run: pip3 install numpy pandas scipy pydantic python-dateutil"
echo "3. Return here and run: source .codex/.env"
echo ""
echo "Alternative: The project will work in PURE_PYTHON mode without these packages."
echo ""

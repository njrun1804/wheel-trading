#!/bin/bash
# Quick Container Setup - Minimal approach
# Handles the math module naming conflict properly

echo "🚀 QUICK CONTAINER SETUP"
echo "======================="
echo ""

# 1. Install Python dependencies BEFORE setting PYTHONPATH
echo "📦 Installing Python packages..."
echo "   (Installing from /tmp to avoid import conflicts)"

# Save current directory
current_dir=$(pwd)

# Move to /tmp to avoid math module conflict
cd /tmp

# Install packages
if command -v pip3 >/dev/null 2>&1; then
    pip3 install numpy pandas pydantic python-dateutil scipy
else
    # Get pip if needed
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    python3 -m pip install numpy pandas pydantic python-dateutil scipy
fi || {
    echo "   ⚠️  Some packages failed to install"
    echo "   This is OK - will use pure Python fallbacks"
}

# Return to project directory
cd "$current_dir"

# 2. NOW set environment after packages are installed
echo ""
echo "🔧 Setting environment..."
cat > .codex/.env <<EOF
export USE_MOCK_DATA=true
export OFFLINE_MODE=true
export DATABENTO_SKIP_VALIDATION=true
export PYTHONPATH="$(pwd):$(pwd)/unity_trading:\$PYTHONPATH"
export LOG_LEVEL=INFO
EOF

source .codex/.env

# 3. Quick test
echo ""
echo "🧪 Testing setup..."
python3 -c "
import numpy as np
import pandas as pd
print('✅ NumPy version:', np.__version__)
print('✅ Pandas version:', pd.__version__)
print('✅ Basic imports work!')
"

echo ""
echo "✅ Setup complete! To activate in new shells:"
echo "   source .codex/.env"
echo ""

#!/usr/bin/env bash
set -euo pipefail
# Offline setup script for Codex environment


echo "🚀 CODEX OFFLINE SETUP"
echo "======================"

# Set environment variables for offline mode
export USE_PURE_PYTHON=true
export USE_MOCK_DATA=true
export DATABENTO_SKIP_VALIDATION=true
export SKIP_VALIDATION=true
export OFFLINE_MODE=true
export LOG_LEVEL=INFO

echo "📝 Setting environment variables..."
echo "   USE_PURE_PYTHON=true"
echo "   USE_MOCK_DATA=true"
echo "   DATABENTO_SKIP_VALIDATION=true"
echo "   SKIP_VALIDATION=true"
echo "   OFFLINE_MODE=true"

# Create environment file for persistence
cat > .codex/.env <<EOF
# Codex Environment Configuration
USE_PURE_PYTHON=true
USE_MOCK_DATA=true
DATABENTO_SKIP_VALIDATION=true
SKIP_VALIDATION=true
OFFLINE_MODE=true
LOG_LEVEL=INFO
PYTHONPATH=\${PYTHONPATH}:\$(pwd)
EOF

echo "✅ Environment file created: .codex/.env"

# Test Python availability
echo "🐍 Testing Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please ensure Python 3.9+ is available."
    exit 1
fi

PYTHON_VERSION=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "   Python version: $PYTHON_VERSION"

# Test critical imports
echo "📦 Testing critical imports..."
python -c "
import json, datetime, typing, dataclasses, abc, logging, os, sys
print('✅ Standard library imports successful')
" || {
    echo "❌ Critical imports failed!"
    exit 1
}

# Test optional imports with fallbacks
echo "📦 Testing optional imports..."
python -c "
import sys

# Test numpy
try:
    import numpy as np
    print('✅ numpy available')
    NUMPY_AVAILABLE = True
except ImportError:
    print('⚠️  numpy not available - using pure Python fallbacks')
    NUMPY_AVAILABLE = False

# Test pandas
try:
    import pandas as pd
    print('✅ pandas available')
except ImportError:
    print('⚠️  pandas not available - using pure Python fallbacks')

# Test scipy
try:
    import scipy.stats
    print('✅ scipy available')
except ImportError:
    print('⚠️  scipy not available - using pure Python fallbacks')

# Test pydantic
try:
    import pydantic
    print('✅ pydantic available')
except ImportError:
    print('⚠️  pydantic not available - using simple validation')

print(f'\\n📊 Environment status: {\"Full\" if NUMPY_AVAILABLE else \"Limited\"} capabilities')
"

# Test Unity Wheel imports
echo "🎯 Testing Unity Wheel imports..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -c "
try:
    from src.unity_wheel.math.options import black_scholes_price_validated
    print('✅ Math module imported successfully')
except ImportError as e:
    print(f'❌ Math module import failed: {e}')
    exit(1)
"

python -c "
try:
    from src.unity_wheel.strategy.wheel import WheelStrategy
    print('✅ Strategy module imported successfully')
except ImportError as e:
    print(f'❌ Strategy module import failed: {e}')
    exit(1)
"

python -c "
try:
    from src.unity_wheel.utils.position_sizing import calculate_position_size
    print('✅ Position sizing module imported successfully')
except ImportError as e:
    print(f'❌ Position sizing module import failed: {e}')
    exit(1)
"

# Test core functionality
echo "🧪 Testing core functionality..."
python -c "
from src.unity_wheel.math.options import black_scholes_price_validated as bs
result = bs(100, 100, 1, 0.05, 0.2, 'call')
if result.confidence > 0.9:
    print(f'✅ Options pricing works: ${result.value:.2f} (confidence: {result.confidence:.1%})')
else:
    print(f'⚠️  Options pricing has low confidence: {result.confidence:.1%}')
"

# Create activation script
cat > .codex/activate.sh <<'EOF'
#!/bin/bash
# Activate Codex environment

# Load environment variables
if [ -f .codex/.env ]; then
    export $(cat .codex/.env | grep -v '^#' | xargs)
    echo "🔄 Loaded Codex environment variables"
else
    echo "⚠️  No .codex/.env file found"
fi

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "📁 Added $(pwd) to PYTHONPATH"

# Verify setup
python -c "
from src.unity_wheel import __version__
print(f'✅ Unity Wheel Trading Bot v{__version__} ready!')
" 2>/dev/null || echo "⚠️  Unity Wheel import issues - check setup"

echo "🎯 Codex environment activated!"
EOF

chmod +x .codex/activate.sh

echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
echo "✅ Offline environment configured"
echo "✅ Environment variables set"
echo "✅ Core imports tested"
echo "✅ Activation script created"
echo ""
echo "🔧 To activate in new sessions:"
echo "   source .codex/activate.sh"
echo ""
echo "🚀 Quick test commands:"
echo "   python -c \"from src.unity_wheel.math import black_scholes_price_validated as bs; print(bs(100, 100, 1, 0.05, 0.2, 'call'))\""
echo "   python -c \"from src.unity_wheel.strategy.wheel import WheelStrategy; print('Strategy ready')\""
echo ""
echo "📖 For full documentation: .codex/ENVIRONMENT_SETUP.md"

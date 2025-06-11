#!/bin/bash
# Container Diagnostic Script
# Helps debug container environment issues

echo "🔍 CONTAINER DIAGNOSTICS"
echo "======================="
echo ""

# Python info
echo "🐍 Python Environment:"
echo "   Version: $(python3 --version 2>&1)"
echo "   Location: $(which python3)"
echo "   Pip: $(pip3 --version 2>&1 | head -1)"
echo ""

# Check installed packages
echo "📦 Installed Packages:"
python3 -c "
import subprocess
import sys

# Packages we care about
packages = ['numpy', 'pandas', 'scipy', 'pydantic', 'python-dateutil']

# Check what's installed
for pkg in packages:
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', pkg],
                               capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split()[1]
                    print(f'   ✅ {pkg} {version}')
                    break
        else:
            print(f'   ❌ {pkg} not installed')
    except:
        print(f'   ❌ {pkg} check failed')
"
echo ""

# Environment variables
echo "🔧 Environment Variables:"
echo "   USE_MOCK_DATA: ${USE_MOCK_DATA:-not set}"
echo "   OFFLINE_MODE: ${OFFLINE_MODE:-not set}"
echo "   USE_PURE_PYTHON: ${USE_PURE_PYTHON:-not set}"
echo "   PYTHONPATH: ${PYTHONPATH:-not set}"
echo ""

# File system
echo "📁 Project Structure:"
echo "   Current dir: $(pwd)"
echo "   Unity trading exists: $([ -d unity_trading ] && echo 'Yes' || echo 'No')"
echo "   Config exists: $([ -f config.yaml ] && echo 'Yes' || echo 'No')"
echo "   .codex dir exists: $([ -d .codex ] && echo 'Yes' || echo 'No')"
echo ""

# Try basic import
echo "🧪 Import Test:"
python3 -c "
import sys
print(f'   Python path entries: {len(sys.path)}')

# Try standard library
try:
    import json, math, datetime
    print('   ✅ Standard library imports work')
except:
    print('   ❌ Standard library broken!')

# Try numpy
try:
    import numpy as np
    print(f'   ✅ NumPy {np.__version__} works')
except ImportError:
    print('   ⚠️  NumPy not available')

# Try project import
sys.path.insert(0, '.')
try:
    from src.unity_wheel.math import options
    print('   ✅ Unity Wheel imports work')
except ImportError as e:
    print(f'   ⚠️  Unity Wheel import failed: {str(e).split(\":\")[0]}')
"
echo ""

# Memory and disk
echo "💾 System Resources:"
df -h . | grep -E "Filesystem|/" | head -2
echo ""

echo "✅ Diagnostics complete"
echo ""
echo "💡 Quick fixes:"
echo "   - Missing packages: Run .codex/quick_setup.sh"
echo "   - Import errors: Check PYTHONPATH and file structure"
echo "   - Environment not set: source .codex/.env"
echo ""

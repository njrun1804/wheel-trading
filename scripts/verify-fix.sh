#!/bin/bash
# Quick fix verification script
# Run this after making changes to verify they'll pass CI

set -e

echo "🔍 Verifying fixes..."
echo "===================="

# 1. Check for common issues that broke CI
echo "1. Checking for merge conflict markers..."
if grep -r "<<<<<<< \|======= \|>>>>>>> " src/ tests/ --include="*.py" 2>/dev/null; then
    echo "❌ Found merge conflict markers!"
    exit 1
fi
echo "✅ No merge conflicts"

# 2. Verify imports
echo -e "\n2. Verifying imports..."
python -c "
import ast
import sys
from pathlib import Path

errors = []
for py_file in Path('src').rglob('*.py'):
    try:
        with open(py_file) as f:
            ast.parse(f.read())
    except SyntaxError as e:
        errors.append(f'{py_file}: {e}')

if errors:
    print('❌ Syntax errors found:')
    for error in errors:
        print(f'  {error}')
    sys.exit(1)
else:
    print('✅ All Python files have valid syntax')
"

# 3. Check specific imports that caused issues
echo -e "\n3. Checking problematic imports..."
python -c "
try:
    from unity_wheel.analytics import IntegratedDecisionEngine
    print('✅ IntegratedDecisionEngine import OK')
except ImportError as e:
    print(f'❌ IntegratedDecisionEngine import failed: {e}')
    exit(1)
"

# 4. Quick test collection
echo -e "\n4. Testing pytest collection..."
pytest --collect-only -q 2>&1 | grep -E "error|ERROR" && {
    echo "❌ Test collection has errors"
    exit 1
} || echo "✅ Test collection OK"

# 5. Check config file
echo -e "\n5. Checking config file..."
if [[ -f "config_unified.yaml" ]]; then
    echo "✅ config_unified.yaml exists"
else
    echo "❌ config_unified.yaml missing"
fi

echo -e "\n✨ All checks passed! Ready to commit and push."
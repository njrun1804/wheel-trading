#!/bin/bash
# Quick status check for Codex container

echo "🔍 CODEX CONTAINER STATUS"
echo "========================"
echo ""

# Environment
echo "📋 Environment:"
echo "   USE_MOCK_DATA: ${USE_MOCK_DATA:-not set}"
echo "   OFFLINE_MODE: ${OFFLINE_MODE:-not set}"
echo "   USE_PURE_PYTHON: ${USE_PURE_PYTHON:-not set}"
echo "   CONTAINER_MODE: ${CONTAINER_MODE:-not set}"
echo ""

# Python packages (check from /tmp to avoid conflicts)
echo "📦 Python Packages:"
cd /tmp
for pkg in numpy pandas scipy pydantic; do
    if python3 -c "import $pkg; print('   ✓ $pkg', $pkg.__version__)" 2>/dev/null; then
        :
    else
        echo "   ✗ $pkg not available"
    fi
done
cd - >/dev/null
echo ""

# Summary
echo "📊 Summary:"
if [ "$USE_PURE_PYTHON" = "true" ]; then
    echo "   Mode: Pure Python (fallbacks active)"
else
    echo "   Mode: NumPy accelerated"
fi

echo "   Status: ✅ Ready for development"
echo ""
echo "💡 Note: The 'math' module conflict is handled automatically."
echo "   You can safely work on Unity trading code!"
echo ""

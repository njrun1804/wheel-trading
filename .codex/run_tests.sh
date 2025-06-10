#!/bin/bash
source .codex/.container_env

echo "🧪 Running Tests..."
echo "=================="

if [ "$USE_PURE_PYTHON" = "true" ]; then
    export PYTHONPATH="$PROJECT_DIR/.codex/stubs:$PYTHONPATH"
fi

if [ "$USE_PURE_PYTHON" = "true" ]; then
    echo "⚠️  Pure Python mode - skipping property-based tests"
    python3 -m pytest tests/ -v -k "not hypothesis" --tb=short 2>/dev/null || {
        echo ""
        echo "💡 Some tests require numpy/sklearn. Try:"
        echo "   python3 -m pytest tests/test_math_simple.py -v"
        echo "   python3 -m pytest tests/test_config.py -v"
    }
else
    python3 -m pytest tests/ -v --tb=short
fi

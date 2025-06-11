#!/bin/bash
if [ -f .codex/.container_env ]; then
    source .codex/.container_env
    echo "✅ Container environment activated"
    echo "   Mode: $([ "$USE_PURE_PYTHON" = "true" ] && echo "Pure Python (stubs)" || echo "Full (numpy+sklearn)")"
    echo "   Tests: $([ "$HYPOTHESIS_AVAILABLE" = "true" ] && echo "Property-based enabled" || echo "Basic only")"
else
    echo "❌ No container environment found!"
    echo "   Run: ./.codex/container_setup.sh"
fi

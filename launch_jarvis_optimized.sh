#!/bin/bash
# Optimized Jarvis 2.0 launcher with proper environment setup

# Fix all macOS-specific issues
export KMP_DUPLICATE_LIB_OK=TRUE
export MTL_DEBUG_LAYER=0
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optimize FAISS for CPU
export OMP_NUM_THREADS=8  # Use P-cores

# Suppress warnings
export PYTHONWARNINGS="ignore::UserWarning"

# Clear any cached bytecode
find jarvis2 -name "*.pyc" -delete 2>/dev/null
find jarvis2 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Launch with optimized settings
echo "🚀 Launching Jarvis 2.0 (Optimized for M4 Pro)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python jarvis2_optimized.py "$@"
#!/bin/bash

echo "🎯 Starting Optimized Trading System"
echo "===================================="

# Set optimal environment variables for M4 Pro
export OMP_NUM_THREADS=8  # Use performance cores
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=8

# Set memory limits
export MALLOC_ARENA_MAX=4

# MLX optimization
export MLX_GPU_MEMORY_POOL_SIZE=2048

echo "🔧 Environment optimized for M4 Pro (8 P-cores + 4 E-cores)"
echo "💾 Memory management configured"
echo "🎮 GPU acceleration enabled"
echo ""

# Run the trading system
python3 trading_system_with_optimization.py "$@"
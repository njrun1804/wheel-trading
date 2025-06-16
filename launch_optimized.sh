#!/bin/bash
# Launch optimized production environment for M4 Pro

echo "🚀 Launching M4 Pro Optimized Environment"
echo "=========================================="

# Set file descriptor limits
ulimit -n 8192

# M4 Pro optimization environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=12
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=4

# Metal GPU optimization
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_FORCE_INTEL=0

# Memory optimization
export PYTHON_GIL_DISABLED=0
export PYTHONMALLOC=malloc

# Performance monitoring
export PYTHONPROFILEIMPORTTIME=1

echo "✅ Environment variables set"
echo "   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "   MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "   File descriptors: $(ulimit -n)"

# Run optimization activation
echo ""
echo "🔧 Activating optimizations..."
python3 quick_optimize.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Production environment ready!"
    echo "   All M4 Pro optimizations active"
    echo "   Maximum performance enabled"
    echo ""
    echo "💡 You can now run:"
    echo "   python run.py --help"
    echo "   python jarvis2.py"
    echo "   python start_complete_meta_system.py"
else
    echo ""
    echo "⚠️  Some optimizations may not be active"
fi
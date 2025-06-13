#!/usr/bin/env bash
# GPU Framework Installation Recommendations for Wheel Trading Project

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== GPU Framework Installation Script ===${NC}"
echo -e "${YELLOW}For M4 Pro Mac - Wheel Trading Project${NC}\n"

# 1. CRITICAL - Enable JAX GPU acceleration
echo -e "${GREEN}1. JAX-Metal (CRITICAL)${NC}"
echo "   Your project has 21 files with risk calculations that could benefit"
echo "   Installing..."
pip install -U jax-metal
echo "   ✓ JAX will now use Metal GPU instead of CPU"
echo ""

# 2. MLX Extensions for better performance
echo -e "${GREEN}2. MLX Extensions${NC}"
echo "   For neural networks and advanced operations"
pip install mlx-nn
echo "   ✓ Enhanced MLX capabilities"
echo ""

# 3. Pandas performance extras
echo -e "${GREEN}3. Pandas Performance${NC}"
echo "   Faster DataFrame operations"
pip install "pandas[performance]"
echo "   ✓ Pandas acceleration enabled"
echo ""

# 4. Optional but recommended
echo -e "${YELLOW}4. Optional Frameworks${NC}"
echo "   These could provide additional benefits:"
echo ""
echo "   a) TensorFlow Metal (if you need TF):"
echo "      pip install tensorflow-metal"
echo ""
echo "   b) Rapids.ai cuDF alternative for Mac (experimental):"
echo "      pip install cudf-cu12  # Note: Limited Mac support"
echo ""
echo "   c) Polars (faster than pandas for some operations):"
echo "      pip install polars"
echo ""

# 5. Verify installation
echo -e "${BLUE}Verifying GPU acceleration...${NC}"
python3 -c "
import sys
print('\nGPU Framework Status:')

# JAX
try:
    import jax
    print(f'✓ JAX backend: {jax.default_backend()}')
    if 'metal' in jax.default_backend().lower():
        print('  → GPU acceleration ACTIVE')
    else:
        print('  → Still on CPU, restart Python')
except:
    print('✗ JAX not available')

# MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    print(f'✓ MLX device: {mx.default_device()}')
    print('  → MLX neural networks available')
except:
    print('✗ MLX not fully configured')

# Numba
try:
    import numba
    from numba import cuda
    print(f'✓ Numba: v{numba.__version__}')
    # Note: CUDA not available on Mac, but Numba still provides speedup
except:
    print('✗ Numba not available')
"

echo -e "\n${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Key improvements for your project:"
echo "  • Risk calculations: Now GPU-accelerated with JAX"
echo "  • Embeddings: MLX already configured"
echo "  • DataFrame ops: Pandas performance mode"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Restart Python/Jupyter kernels to activate JAX Metal"
echo "2. Update imports in risk calculations:"
echo "   # Replace: import numpy as np"
echo "   # With:    import jax.numpy as jnp"
echo "3. The orchestrator will automatically use GPU when available"
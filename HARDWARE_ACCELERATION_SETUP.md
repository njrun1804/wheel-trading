# Hardware Acceleration Setup for Unity Wheel + Jarvis2

## Complete M4 Pro Optimization Guide

### 1. Unified Startup Script

The main startup script `startup_unified.sh` now:
- ✅ Detects M4 Pro hardware (12 cores, 16 GPU cores, 24GB RAM)
- ✅ Sets optimal environment variables for Metal/GPU acceleration
- ✅ Configures Jarvis2 for MLX (preferred) and PyTorch MPS
- ✅ Allocates P-cores for search, E-cores for learning
- ✅ Enables WezTerm GPU acceleration if detected
- ✅ Creates quick launchers for both systems

Run with:
```bash
./startup_unified.sh
```

### 2. Environment Variables Set

#### Metal/GPU Optimization
- `PYTORCH_ENABLE_MPS_FALLBACK=1` - Enable Metal Performance Shaders
- `PYTORCH_METAL_WORKSPACE_LIMIT_BYTES=19327352832` - 18GB limit for M4 Pro
- `MTL_DEBUG_LAYER=0` - Disable debug for performance
- `METAL_DEVICE_WRAPPER_TYPE=1` - Enable Metal wrapper

#### CPU Optimization
- `OMP_NUM_THREADS=12` - Use all CPU cores
- `MKL_NUM_THREADS=12` - Intel Math Kernel Library
- `NUMEXPR_NUM_THREADS=12` - NumExpr parallelization
- `VECLIB_MAXIMUM_THREADS=12` - Apple Accelerate framework
- `OPENBLAS_NUM_THREADS=12` - OpenBLAS linear algebra

#### Jarvis2 Specific
- `JARVIS2_BACKEND_PREFERENCE=mlx,mps,cpu` - Prefer MLX over PyTorch
- `JARVIS2_MEMORY_LIMIT_GB=18` - Stay within Metal limits
- `JARVIS2_SEARCH_WORKERS=8` - Use P-cores for search
- `JARVIS2_NEURAL_WORKERS=2` - Neural evaluation workers
- `JARVIS2_LEARNING_WORKERS=4` - Use E-cores for learning

### 3. WezTerm GPU Configuration

If using WezTerm, apply GPU acceleration settings:

1. Copy settings from `wezterm_gpu_config.lua` to `~/.wezterm.lua`
2. Key features enabled:
   - Metal GPU rendering (`webgpu_preferred_adapter = "Metal"`)
   - 120 FPS capability
   - Hardware-accelerated blur and transparency
   - Optimized font rendering

3. Keyboard shortcuts added:
   - `Cmd+Shift+J` - Launch Jarvis2 interactive mode
   - `Cmd+Shift+T` - Launch trading system

### 4. Quick Commands

After running `startup_unified.sh`, you can use:

#### Jarvis2 (Meta-Coding AI)
```bash
# Interactive mode
python -m jarvis2

# Generate code with query
python -m jarvis2 "Create a binary search tree implementation"

# Quick script (created by startup)
./jarvis2_quick.sh "optimize this sorting algorithm"

# CLI with options
python jarvis2/cli.py --simulations 5000 --verbose
```

#### Unity Wheel (Trading)
```bash
# Get recommendation
python run.py

# Turbo mode orchestrator (all cores)
./orchestrate_turbo.py "analyze SPY options chain"

# Standard orchestrator
./orchestrate "find optimal strike prices"
```

### 5. Hardware Utilization

The setup maximizes your M4 Pro as follows:

| Component | Allocation | Used For |
|-----------|------------|----------|
| P-cores (8) | JARVIS2_SEARCH_WORKERS=8 | Parallel MCTS search |
| E-cores (4) | JARVIS2_LEARNING_WORKERS=4 | Background learning |
| GPU cores (16) | MLX/Metal | Neural networks |
| Neural Engine | MLX framework | Matrix operations |
| Memory (24GB) | 18GB Metal limit | Unified memory pool |

### 6. Verification

To verify everything is working:

```bash
# Check Jarvis2 initialization
python -c "
import asyncio
from jarvis2 import Jarvis2Orchestrator
async def test():
    j = Jarvis2Orchestrator()
    await j.initialize()
    print('✅ Jarvis2 initialized successfully')
    print(f'Stats: {j.get_stats()}')
    await j.shutdown()
asyncio.run(test())
"

# Check hardware detection
python -c "
from jarvis2.core.device_router import get_router
r = get_router()
print(f'Available backends: {r._backends}')
"

# Monitor performance
python scripts/monitor-m4-performance.py
```

### 7. Performance Tips

1. **First Run**: Initial startup takes 1-2s per worker due to spawn method
2. **Warmup**: First code generation may be slower (model loading)
3. **Memory**: Monitor with `echo $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))GB`
4. **GPU**: Check Metal usage with `sudo powermetrics --samplers gpu_power`

### 8. Troubleshooting

If you see:
- `MLX not installed` - Run: `pip install mlx`
- `PyTorch MPS errors` - The fallback is already enabled
- `Memory warnings` - 18GB limit is enforced automatically
- `Slow startup` - Normal due to spawn method on macOS

The system is now fully optimized for your M4 Pro hardware!
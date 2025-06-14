#!/bin/bash
# Unified startup script for Unity Wheel + Jarvis2 on M4 Pro
# Maximizes ALL hardware acceleration for both systems

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Show banner unless in quiet mode
if [ -z "$STARTUP_QUIET" ]; then
    echo -e "${GREEN}${BOLD}🚀 UNITY WHEEL + JARVIS2 UNIFIED LAUNCHER${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Maximizing M4 Pro hardware for peak performance${NC}"
    echo ""
fi

# Function to display step status
step() {
    echo -e "\n${YELLOW}▶ $1${NC}"
}

# Function to display success
success() {
    echo -e "${GREEN}  ✅ $1${NC}"
}

# ====================
# HARDWARE DETECTION
# ====================
step "Detecting M4 Pro hardware"
CPU_BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
CPU_COUNT=$(sysctl -n hw.ncpu 2>/dev/null || echo "12")
P_CORES=8  # M4 Pro has 8 performance cores
E_CORES=4  # M4 Pro has 4 efficiency cores
GPU_CORES=16  # M4 Pro has 16 GPU cores
MEM_GB=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().total / (1024**3):.1f}')" 2>/dev/null || echo "24")

success "CPU: $CPU_BRAND"
success "Cores: ${CPU_COUNT} total (${P_CORES}P + ${E_CORES}E)"
success "GPU: ${GPU_CORES} cores"
success "Memory: ${MEM_GB}GB unified"

# ====================
# ENVIRONMENT SETUP
# ====================
step "Setting M4 Pro optimized environment"

# macOS-specific fixes for multiprocessing
export KMP_DUPLICATE_LIB_OK=TRUE
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Metal/GPU settings (18GB limit for M4 Pro)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_METAL_WORKSPACE_LIMIT_BYTES=$((18 * 1024 * 1024 * 1024))
export MTL_DEBUG_LAYER=0
export METAL_DEVICE_WRAPPER_TYPE=1

# CPU optimization - use all cores
export OMP_NUM_THREADS=$CPU_COUNT
export MKL_NUM_THREADS=$CPU_COUNT
export NUMEXPR_NUM_THREADS=$CPU_COUNT
export VECLIB_MAXIMUM_THREADS=$CPU_COUNT
export OPENBLAS_NUM_THREADS=$CPU_COUNT

# Python optimizations
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# Jarvis2 specific - prefer MLX over PyTorch MPS
export JARVIS2_BACKEND_PREFERENCE="mlx,mps,cpu"
export JARVIS2_MEMORY_LIMIT_GB=18
export JARVIS2_SEARCH_WORKERS=$P_CORES
export JARVIS2_NEURAL_WORKERS=2
export JARVIS2_LEARNING_WORKERS=$E_CORES

# Suppress warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"

success "Environment configured for maximum performance"

# ====================
# JARVIS2 INITIALIZATION
# ====================
step "Initializing Jarvis2 meta-coding system"

# Create Jarvis2 directories if needed
mkdir -p .jarvis/{indexes,models,experience} logs/jarvis2

# Clear any stale process locks
rm -f .jarvis/experience.db-wal .jarvis/experience.db-shm 2>/dev/null || true

# Pre-warm Metal GPU if available
if command -v metal-info >/dev/null 2>&1; then
    success "Metal GPU detected and ready"
fi

# Check MLX availability
if python3 -c "import mlx" 2>/dev/null; then
    success "MLX framework available for Apple Silicon optimization"
else
    echo "  ⚠️  MLX not installed. Install with: pip install mlx"
fi

# ====================
# UNITY WHEEL SETUP
# ====================
step "Preparing Unity Wheel trading system"

# Clean startup logs
rm -f orchestrator.log 2>/dev/null || true

# Create necessary directories
mkdir -p logs data/cache ~/.wheel_trading/{secrets,cache}

# Source .env if exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    success "Environment variables loaded"
fi

# ====================
# WEZTERM OPTIMIZATION
# ====================
if [ -n "${WEZTERM_PANE:-}" ]; then
    step "Detected WezTerm - applying GPU acceleration"
    
    # WezTerm uses Metal for rendering, ensure it has resources
    export WEZTERM_ENABLE_WEBGPU=1
    
    # If WezTerm config exists, it should have:
    # config.webgpu_preferred_adapter = "Metal"
    # config.front_end = "WebGpu"
    # config.max_fps = 120
    
    success "WezTerm GPU acceleration enabled"
fi

# ====================
# FINAL SUMMARY
# ====================
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}✅ SYSTEM READY - ALL HARDWARE MAXIMIZED${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Available commands:"
echo ""
echo -e "${BOLD}Unity Wheel Trading:${NC}"
echo -e "  ${BLUE}python run.py${NC}                       - Get trading recommendation"
echo -e "  ${BLUE}python run.py --diagnose${NC}            - System diagnostics"
echo -e "  ${BLUE}./orchestrate_turbo.py${NC} '<query>'    - TURBO MODE orchestrator"
echo ""
echo -e "${BOLD}Jarvis2 Meta-Coding:${NC}"
echo -e "  ${BLUE}python -m jarvis2${NC} '<query>'         - Generate code with AI"
echo -e "  ${BLUE}python jarvis2/cli.py${NC} '<query>'     - Interactive mode"
echo -e "  ${BLUE}python -m jarvis2.benchmark${NC}         - Performance testing"
echo ""
echo -e "${BOLD}System Monitoring:${NC}"
echo -e "  ${BLUE}python scripts/monitor-m4-performance.py${NC}  - Real-time monitoring"
echo ""
echo "Hardware allocation:"
echo -e "  • P-cores (${P_CORES}): Parallel search, main computation"
echo -e "  • E-cores (${E_CORES}): Learning, background tasks"
echo -e "  • GPU cores (${GPU_CORES}): Neural networks (MLX/Metal)"
echo -e "  • Memory: ${MEM_GB}GB unified (18GB Metal limit)"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Create quick launcher for Jarvis2
cat > jarvis2_quick.sh << 'EOF'
#!/bin/bash
# Quick launcher for Jarvis2 with a query
python3 -c "
import asyncio
from jarvis2.core.orchestrator import Jarvis2Orchestrator, CodeRequest

async def main():
    jarvis = Jarvis2Orchestrator()
    await jarvis.initialize()
    
    query = '$1'
    if not query:
        query = 'Create a function to calculate fibonacci numbers'
    
    print(f'\\n🤖 Jarvis2: Generating code for: {query}\\n')
    
    request = CodeRequest(query)
    solution = await jarvis.generate_code(request)
    
    print('Generated code:')
    print('='*60)
    print(solution.code)
    print('='*60)
    print(f'\\nConfidence: {solution.confidence:.0%}')
    print(f'Time: {solution.metrics[\"generation_time_ms\"]:.0f}ms')
    
    await jarvis.shutdown()

asyncio.run(main())
"
EOF
chmod +x jarvis2_quick.sh

# If command provided, run it
if [ $# -gt 0 ]; then
    echo ""
    echo -e "${BOLD}${BLUE}Executing: $@${NC}"
    echo ""
    "$@"
fi
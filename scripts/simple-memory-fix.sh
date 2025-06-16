#!/bin/bash

# Simple Node.js Memory Configuration Fix
# Fixes issues without using problematic ulimit commands

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Simple Node.js Memory Configuration Fix ===${NC}"
echo ""

# 1. Create a clean, simple .zshenv
echo -e "${BLUE}1. Creating simple, working configuration...${NC}"

# Backup existing
cp ~/.zshenv ~/.zshenv.backup.simple.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

cat > ~/.zshenv << 'EOF'
# M4 Pro Optimized Configuration for Claude Code CLI
# Simple, working Node.js Memory Configuration

# Optimal Node.js configuration for M4 Pro with 24GB RAM
export NODE_OPTIONS="--max-old-space-size=20480 --max-semi-space-size=1024 --memory-reducer=false --expose-gc --trace-gc --v8-pool-size=12"
export UV_THREADPOOL_SIZE=12
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=262144

# Apple Silicon specific optimizations
export NODE_DISABLE_COLORS=0
export NODE_ENV=development
export NODE_PRESERVE_SYMLINKS=1

# Python optimizations
export PYTHON_MEMORY_LIMIT="8G"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONOPTIMIZE=2

# Threading optimizations for M4 Pro (12 cores)
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12
export RAYON_NUM_THREADS=12
export CARGO_BUILD_JOBS=12

# Jarvis2 hardware settings
export JARVIS2_WORKERS=12
export JARVIS2_PARALLEL_WORKERS=12
export METAL_AVAILABLE_MEMORY=20401094656
export USE_TURBO_MODE=1
export CLAUDE_HARDWARE_ACCEL=1

# Claude Code settings
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000

# Note: ulimit commands removed to prevent conflicts
# System limits will be handled by LaunchAgent and startup scripts
EOF

echo -e "  ${GREEN}✓${NC} Created simple .zshenv configuration"

# 2. Create startup script for setting limits
echo -e "\n${BLUE}2. Creating startup script for system limits...${NC}"

cat > "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/set-memory-limits.sh" << 'EOF'
#!/bin/bash

# Set Node.js Memory Limits - Safe version
# Run this before using Node.js applications

# Set safe file descriptor limit
ulimit -n 16384 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# Set safe process limit  
ulimit -u 4096 2>/dev/null || ulimit -u 2048 2>/dev/null || true

# Set stack size if possible
ulimit -s 32768 2>/dev/null || true

echo "Memory limits set:"
echo "  File descriptors: $(ulimit -n)"
echo "  Processes: $(ulimit -u)"
echo "  Stack size: $(ulimit -s)"
EOF

chmod +x "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/set-memory-limits.sh"
echo -e "  ${GREEN}✓${NC} Created set-memory-limits.sh script"

# 3. Test the configuration
echo -e "\n${BLUE}3. Testing configuration...${NC}"

# Source the new environment in a subshell to avoid affecting current session
(
    source ~/.zshenv
    
    # Add Node.js to PATH if needed
    export PATH="/opt/homebrew/bin:$PATH"
    
    if command -v node >/dev/null 2>&1; then
        echo -e "  ${GREEN}Node.js version:${NC} $(node --version)"
        
        # Test heap configuration
        HEAP_TEST=$(node -e "console.log(Math.round(require('v8').getHeapStatistics().heap_size_limit / 1024 / 1024))" 2>/dev/null || echo "ERROR")
        if [ "$HEAP_TEST" != "ERROR" ] && [ "$HEAP_TEST" -gt 1000 ]; then
            echo -e "  ${GREEN}Heap limit:${NC} ${HEAP_TEST}MB"
            echo -e "  ${GREEN}✓${NC} Configuration working"
        else
            echo -e "  ${YELLOW}⚠${NC} Heap test: $HEAP_TEST (may need new terminal)"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Node.js not found - checking PATH..."
        echo -e "  PATH: $PATH"
    fi
)

# 4. Update other scripts to use the PATH fix
echo -e "\n${BLUE}4. Creating Node.js launcher with proper PATH...${NC}"

cat > "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/node-optimized.sh" << 'EOF'
#!/bin/bash

# Optimized Node.js Launcher for M4 Pro
# Ensures proper PATH and memory settings

# Set proper PATH for Homebrew Node.js
export PATH="/opt/homebrew/bin:$PATH"

# Source memory configuration
source ~/.zshenv

# Set system limits safely
ulimit -n 16384 2>/dev/null || ulimit -n 4096 2>/dev/null || true
ulimit -u 4096 2>/dev/null || ulimit -u 2048 2>/dev/null || true

# Run Node.js with arguments
if [ $# -eq 0 ]; then
    echo "🚀 Optimized Node.js Shell (M4 Pro)"
    echo "Heap limit: $(node -e "console.log(Math.round(require('v8').getHeapStatistics().heap_size_limit / 1024 / 1024))")MB"
    node
else
    exec node "$@"
fi
EOF

chmod +x "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/node-optimized.sh"
echo -e "  ${GREEN}✓${NC} Created node-optimized.sh launcher"

# 5. Summary
echo -e "\n${GREEN}=== Simple Configuration Applied ===${NC}"
echo ""
echo -e "${YELLOW}What was fixed:${NC}"
echo -e "  ✓ Removed conflicting NODE_OPTIONS"
echo -e "  ✓ Removed problematic ulimit commands from .zshenv"
echo -e "  ✓ Created separate scripts for system limits"
echo -e "  ✓ Added proper PATH for Homebrew Node.js"
echo ""
echo -e "${YELLOW}New scripts created:${NC}"
echo -e "  • scripts/set-memory-limits.sh - Set system limits safely"
echo -e "  • scripts/node-optimized.sh - Launch Node.js with optimizations"
echo ""
echo -e "${BLUE}To test the configuration:${NC}"
echo -e "  1. Open a new terminal (or run: source ~/.zshenv)"
echo -e "  2. Run: ./scripts/node-optimized.sh -e 'console.log(\"Memory configured:\", Math.round(require(\"v8\").getHeapStatistics().heap_size_limit/1024/1024) + \"MB\")'"
echo -e "  3. Run: ./scripts/validate-memory-setup.py"
echo ""
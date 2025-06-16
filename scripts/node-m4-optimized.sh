#!/bin/bash

# M4 Pro Optimized Node.js Launcher
# Prevents RangeError: Invalid string length

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== M4 Pro Node.js Memory Optimization ===${NC}"

# 1. Set optimal Node.js memory configuration
export NODE_OPTIONS="--max-old-space-size=18432 --max-semi-space-size=512 --optimize-for-size=false --memory-reducer=false --expose-gc"

# 2. System resource limits
ulimit -n 16384      # File descriptors
ulimit -u 4096       # Processes
ulimit -m unlimited  # Memory
ulimit -v unlimited  # Virtual memory

# 3. M4 Pro threading optimizations
export UV_THREADPOOL_SIZE=12  # Use all 12 cores for I/O
export OMP_NUM_THREADS=12     # OpenMP parallelization

# 4. Memory management
export MALLOC_ARENA_MAX=4     # Reduce memory fragmentation
export MALLOC_MMAP_THRESHOLD_=131072  # Use mmap for large allocations

# 5. Process priority (run with higher priority)
renice -10 $$ 2>/dev/null || true

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  • Node.js heap size: 18GB (max-old-space-size)"
echo -e "  • Semi-space size: 512MB (young generation)"
echo -e "  • File descriptors: 16,384"
echo -e "  • Thread pool: 12 cores"
echo -e "  • Memory optimization: Enabled"
echo -e "  • Process priority: High"
echo ""

# 6. Memory pressure monitoring
check_memory() {
    local available_gb=$(node -p "Math.round(require('os').freemem() / 1024 / 1024 / 1024)")
    if [ "$available_gb" -lt 4 ]; then
        echo -e "${YELLOW}Warning: Low memory ($available_gb GB available)${NC}"
        echo -e "Consider closing other applications for optimal performance"
    fi
}

check_memory

# 7. Start Node.js with optimized settings
if [ $# -eq 0 ]; then
    echo -e "${BLUE}Starting optimized Node.js shell...${NC}"
    node
else
    echo -e "${BLUE}Running: node $@${NC}"
    exec node "$@"
fi
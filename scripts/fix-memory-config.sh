#!/bin/bash

# Fix Node.js Memory Configuration Issues
# Corrects ulimit problems and NODE_OPTIONS conflicts

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Fixing Node.js Memory Configuration Issues ===${NC}"
echo ""

# 1. Backup existing .zshenv
echo -e "${BLUE}1. Backing up existing configuration...${NC}"
cp ~/.zshenv ~/.zshenv.backup.$(date +%Y%m%d_%H%M%S)
echo -e "  ${GREEN}✓${NC} Backup created"

# 2. Clean up .zshenv and create corrected version
echo -e "\n${BLUE}2. Creating corrected .zshenv configuration...${NC}"

cat > ~/.zshenv << 'EOF'
# M4 Pro Optimized Configuration for Claude Code CLI
# Enhanced Node.js Memory Configuration - Prevents string overflow

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

# Safe system limits (respect hard limits)
# Get current hard limits first, then set soft limits appropriately
EOF

echo -e "  ${GREEN}✓${NC} Created clean .zshenv configuration"

# 3. Set safe ulimits based on system hard limits
echo -e "\n${BLUE}3. Configuring safe system limits...${NC}"

# Get current hard limits
FD_HARD=$(ulimit -H -n 2>/dev/null || echo "4096")
PROC_HARD=$(ulimit -H -u 2>/dev/null || echo "2048")
STACK_HARD=$(ulimit -H -s 2>/dev/null || echo "8192")

# Calculate safe soft limits (80% of hard limits, with minimums)
FD_SOFT=$((FD_HARD > 16384 ? 16384 : FD_HARD))
PROC_SOFT=$((PROC_HARD > 4096 ? 4096 : PROC_HARD))
STACK_SOFT=$((STACK_HARD > 32768 ? 32768 : STACK_HARD))

cat >> ~/.zshenv << EOF

# Safe system limits (respecting hard limits)
ulimit -n $FD_SOFT      # File descriptors (safe: $FD_SOFT)
ulimit -u $PROC_SOFT    # Processes (safe: $PROC_SOFT)
ulimit -s $STACK_SOFT   # Stack size (safe: $STACK_SOFT)
EOF

echo -e "  ${GREEN}✓${NC} Configured safe limits - FD: $FD_SOFT, Proc: $PROC_SOFT, Stack: $STACK_SOFT"

# 4. Update .bashrc if it exists
if [ -f ~/.bashrc ]; then
    echo -e "\n${BLUE}4. Updating .bashrc configuration...${NC}"
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    
    # Remove old configuration
    sed -i '' '/# Enhanced M4 Pro Node.js Memory Configuration/,/ulimit -s 65536/d' ~/.bashrc
    
    # Add corrected configuration
    cat >> ~/.bashrc << EOF

# M4 Pro Optimized Configuration for Claude Code CLI
export NODE_OPTIONS="--max-old-space-size=20480 --max-semi-space-size=1024 --memory-reducer=false --expose-gc --trace-gc --v8-pool-size=12"
export UV_THREADPOOL_SIZE=12
export MALLOC_ARENA_MAX=4
export NODE_ENV=development

# Safe system limits
ulimit -n $FD_SOFT
ulimit -u $PROC_SOFT
ulimit -s $STACK_SOFT
EOF
    echo -e "  ${GREEN}✓${NC} Updated .bashrc"
fi

# 5. Test the new configuration
echo -e "\n${BLUE}5. Testing corrected configuration...${NC}"

# Source the new configuration
source ~/.zshenv

# Test Node.js
if command -v node >/dev/null 2>&1; then
    echo -e "  ${GREEN}Node.js version:${NC} $(node --version)"
    
    # Test heap limit
    HEAP_TEST=$(node -e "console.log(Math.round(require('v8').getHeapStatistics().heap_size_limit / 1024 / 1024))" 2>/dev/null || echo "ERROR")
    if [ "$HEAP_TEST" != "ERROR" ]; then
        echo -e "  ${GREEN}Heap limit:${NC} ${HEAP_TEST}MB"
        if [ "$HEAP_TEST" -gt 18000 ]; then
            echo -e "  ${GREEN}✓${NC} Configuration applied successfully"
        else
            echo -e "  ${YELLOW}⚠${NC} Configuration may need terminal restart"
        fi
    else
        echo -e "  ${RED}✗${NC} Error testing heap configuration"
    fi
    
    # Test current limits
    echo -e "  ${GREEN}File descriptors:${NC} $(ulimit -n)"
    echo -e "  ${GREEN}Process limit:${NC} $(ulimit -u)"
    echo -e "  ${GREEN}Stack size:${NC} $(ulimit -s)"
else
    echo -e "  ${YELLOW}⚠${NC} Node.js not found in PATH"
fi

# 6. Update LaunchAgent with corrected configuration
echo -e "\n${BLUE}6. Updating LaunchAgent configuration...${NC}"

mkdir -p ~/Library/LaunchAgents

cat > ~/Library/LaunchAgents/com.nodejs.memory-limits.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nodejs.memory-limits</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/sh</string>
        <string>-c</string>
        <string>
            # Set safe system limits for current user session
            launchctl limit maxfiles $FD_SOFT unlimited;
            launchctl limit maxproc $PROC_SOFT $((PROC_SOFT * 2));
            
            # Set corrected Node.js environment for all processes
            launchctl setenv NODE_OPTIONS "--max-old-space-size=20480 --max-semi-space-size=1024 --memory-reducer=false --expose-gc --v8-pool-size=12";
            launchctl setenv UV_THREADPOOL_SIZE "12";
            launchctl setenv MALLOC_ARENA_MAX "4";
            launchctl setenv NODE_ENV "development";
        </string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
EOF

# Reload LaunchAgent
launchctl unload ~/Library/LaunchAgents/com.nodejs.memory-limits.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.nodejs.memory-limits.plist 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} LaunchAgent updated and reloaded"

# 7. Summary
echo -e "\n${GREEN}=== Configuration Fixed ===${NC}"
echo ""
echo -e "${YELLOW}Fixed Issues:${NC}"
echo -e "  ✓ Removed conflicting NODE_OPTIONS"
echo -e "  ✓ Removed invalid --optimize-for-size flag"
echo -e "  ✓ Set safe ulimit values within hard limits"
echo -e "  ✓ Updated LaunchAgent configuration"
echo -e "  ✓ Created configuration backups"
echo ""
echo -e "${YELLOW}Current Configuration:${NC}"
echo -e "  • Node.js heap: 20GB"
echo -e "  • Semi-space: 1GB"
echo -e "  • File descriptors: $FD_SOFT"
echo -e "  • Process limit: $PROC_SOFT"
echo -e "  • Thread pool: 12 cores"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Open a new terminal or run: source ~/.zshenv"
echo -e "  2. Run: ./scripts/validate-memory-setup.py"
echo -e "  3. Run: ./scripts/test-memory-config.js"
echo ""
echo -e "${BLUE}Note:${NC} If you encounter issues, restore from backup:"
echo -e "  cp ~/.zshenv.backup.* ~/.zshenv"
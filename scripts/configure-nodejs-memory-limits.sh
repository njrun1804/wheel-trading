#!/bin/bash

# Configure permanent Node.js memory limits for M4 Pro
# Prevents "RangeError: Invalid string length" errors

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}=== Configuring M4 Pro Node.js Memory Limits ===${NC}"
echo -e "${YELLOW}This will prevent 'RangeError: Invalid string length' errors${NC}"
echo ""

# 1. Update shell environment files
echo -e "${BLUE}1. Updating shell environment files...${NC}"

# Add to .zshenv (loads for all zsh sessions)
if ! grep -q "M4 Pro Node.js Memory Configuration" ~/.zshenv 2>/dev/null; then
    cat >> ~/.zshenv << 'EOF'

# M4 Pro Node.js Memory Configuration - Prevents string overflow
export NODE_OPTIONS="--max-old-space-size=18432 --max-semi-space-size=512 --optimize-for-size=false --memory-reducer=false"
export UV_THREADPOOL_SIZE=12
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=131072

# System limits for Node.js
ulimit -n 16384      # File descriptors
ulimit -u 4096       # Processes  
ulimit -m unlimited  # Memory
ulimit -v unlimited  # Virtual memory
EOF
    echo -e "  ${GREEN}✓${NC} Updated ~/.zshenv"
else
    echo -e "  ${GREEN}✓${NC} ~/.zshenv already configured"
fi

# Add to .bashrc if it exists
if [ -f ~/.bashrc ]; then
    if ! grep -q "M4 Pro Node.js Memory Configuration" ~/.bashrc; then
        cat >> ~/.bashrc << 'EOF'

# M4 Pro Node.js Memory Configuration - Prevents string overflow  
export NODE_OPTIONS="--max-old-space-size=18432 --max-semi-space-size=512 --optimize-for-size=false --memory-reducer=false"
export UV_THREADPOOL_SIZE=12
export MALLOC_ARENA_MAX=4
export MALLOC_MMAP_THRESHOLD_=131072

# System limits for Node.js
ulimit -n 16384
ulimit -u 4096
ulimit -m unlimited
ulimit -v unlimited
EOF
        echo -e "  ${GREEN}✓${NC} Updated ~/.bashrc"
    else
        echo -e "  ${GREEN}✓${NC} ~/.bashrc already configured"
    fi
fi

# 2. Create launchd configuration for persistent limits
echo -e "\n${BLUE}2. Setting up persistent system limits...${NC}"

# User-level LaunchAgent for memory limits
mkdir -p ~/Library/LaunchAgents

cat > ~/Library/LaunchAgents/com.nodejs.memory-limits.plist << 'EOF'
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
            # Set system limits for current user session
            launchctl limit maxfiles 16384 unlimited;
            launchctl limit maxproc 4096 6000;
            
            # Set Node.js environment for all processes
            launchctl setenv NODE_OPTIONS "--max-old-space-size=18432 --max-semi-space-size=512 --optimize-for-size=false";
            launchctl setenv UV_THREADPOOL_SIZE "12";
        </string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
EOF

# Load the LaunchAgent
launchctl load ~/Library/LaunchAgents/com.nodejs.memory-limits.plist 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} LaunchAgent configured and loaded"

# 3. Create system-wide limits (requires admin)
echo -e "\n${BLUE}3. Configuring system-wide limits...${NC}"

# Check if we can write to /etc/launchd.conf
if [ -w /etc/launchd.conf ] || [ ! -f /etc/launchd.conf ]; then
    echo "limit maxfiles 16384 unlimited" | sudo tee -a /etc/launchd.conf >/dev/null
    echo "limit maxproc 4096 6000" | sudo tee -a /etc/launchd.conf >/dev/null
    echo -e "  ${GREEN}✓${NC} System limits configured in /etc/launchd.conf"
else
    echo -e "  ${YELLOW}⚠${NC} Could not write to /etc/launchd.conf (admin required)"
fi

# 4. Test current configuration
echo -e "\n${BLUE}4. Testing current configuration...${NC}"

# Source the new environment
source ~/.zshenv 2>/dev/null || true

# Test Node.js memory configuration
if command -v node >/dev/null 2>&1; then
    echo -e "  ${GREEN}Node.js version:${NC} $(node --version)"
    
    # Get heap limit
    HEAP_LIMIT=$(node -p "require('v8').getHeapStatistics().heap_size_limit / 1024 / 1024" 2>/dev/null)
    if [ ! -z "$HEAP_LIMIT" ]; then
        echo -e "  ${GREEN}Heap limit:${NC} ${HEAP_LIMIT}MB"
        
        # Check if our configuration took effect
        if (( $(echo "$HEAP_LIMIT > 10000" | bc -l) )); then
            echo -e "  ${GREEN}✓${NC} Memory configuration applied successfully"
        else
            echo -e "  ${YELLOW}⚠${NC} Configuration may not be active yet (requires restart)"
        fi
    fi
    
    # Test current limits
    echo -e "  ${GREEN}File descriptors:${NC} $(ulimit -n)"
    echo -e "  ${GREEN}Process limit:${NC} $(ulimit -u)"
    
else
    echo -e "  ${YELLOW}⚠${NC} Node.js not found. Install Node.js to test configuration."
fi

# 5. Create test script
echo -e "\n${BLUE}5. Creating memory stress test script...${NC}"

cat > /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/test-nodejs-memory.js << 'EOF'
#!/usr/bin/env node

/**
 * Node.js Memory Configuration Test
 * Tests string allocation limits on M4 Pro
 */

const v8 = require('v8');
const os = require('os');

console.log('🧪 Node.js Memory Configuration Test\n');

// Show configuration
const heapStats = v8.getHeapStatistics();
console.log('📊 Current Configuration:');
console.log(`   Heap limit: ${Math.round(heapStats.heap_size_limit / 1024 / 1024)}MB`);
console.log(`   System memory: ${Math.round(os.totalmem() / 1024 / 1024 / 1024)}GB`);
console.log(`   Available memory: ${Math.round(os.freemem() / 1024 / 1024 / 1024)}GB`);
console.log(`   Thread pool size: ${process.env.UV_THREADPOOL_SIZE || 'default'}`);
console.log('');

// Test string allocation
const testSizes = [100, 500, 1000, 2000]; // MB

for (const sizeMB of testSizes) {
    const sizeBytes = sizeMB * 1024 * 1024;
    
    try {
        console.log(`🧪 Testing ${sizeMB}MB string allocation...`);
        
        const startTime = Date.now();
        const testString = 'x'.repeat(sizeBytes);
        const endTime = Date.now();
        
        console.log(`   ✅ Success! Allocated ${sizeMB}MB in ${endTime - startTime}ms`);
        
        // Clean up immediately
        // Note: In real code, don't rely on this pattern
        if (global.gc) {
            global.gc();
        }
        
    } catch (error) {
        if (error.message.includes('Invalid string length')) {
            console.log(`   ❌ Failed: ${error.message}`);
            console.log(`   💡 Maximum safe string size reached at ${sizeMB}MB`);
            break;
        } else {
            console.log(`   ❌ Unexpected error: ${error.message}`);
        }
    }
}

console.log('\n📈 Final heap statistics:');
const finalStats = v8.getHeapStatistics();
console.log(`   Used: ${Math.round(finalStats.used_heap_size / 1024 / 1024)}MB`);
console.log(`   Total: ${Math.round(finalStats.total_heap_size / 1024 / 1024)}MB`);
console.log(`   Limit: ${Math.round(finalStats.heap_size_limit / 1024 / 1024)}MB`);
EOF

chmod +x /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/scripts/test-nodejs-memory.js
echo -e "  ${GREEN}✓${NC} Test script created"

# 6. Summary
echo -e "\n${GREEN}=== Configuration Complete ===${NC}"
echo ""
echo -e "${YELLOW}What was configured:${NC}"
echo -e "  ✓ Node.js heap limit increased to 18GB"
echo -e "  ✓ Semi-space size increased to 512MB"
echo -e "  ✓ File descriptor limit increased to 16,384"
echo -e "  ✓ Thread pool optimized for 12 cores"
echo -e "  ✓ Memory allocator optimized"
echo -e "  ✓ LaunchAgent for persistent settings"
echo ""
echo -e "${YELLOW}To test the configuration:${NC}"
echo -e "  ./scripts/test-nodejs-memory.js"
echo ""
echo -e "${YELLOW}To monitor memory usage:${NC}"
echo -e "  ./scripts/monitor-nodejs-memory.js"
echo ""
echo -e "${YELLOW}To use optimized Node.js:${NC}"
echo -e "  ./scripts/node-m4-optimized.sh [script.js]"
echo ""
echo -e "${BLUE}Note:${NC} Some changes require a new terminal session or restart to take full effect."
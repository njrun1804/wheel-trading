#!/bin/bash

# Make all Node.js memory optimizations permanent across reboots
# This script ensures all changes survive computer restarts

set -e

echo "🔧 Making Node.js memory optimizations permanent..."

# 1. Add to shell profile (.zshrc for macOS)
ZSHRC="$HOME/.zshrc"

echo "📝 Adding environment variables to $ZSHRC..."

# Remove any existing Claude optimizations
sed -i '' '/# Claude Code Memory Optimizations/,/# End Claude Code Optimizations/d' "$ZSHRC" 2>/dev/null || true

# Add new optimizations
cat >> "$ZSHRC" << 'EOF'

# Claude Code Memory Optimizations (Auto-added)
export NODE_OPTIONS="--max-old-space-size=20480 --max-semi-space-size=1024 --max-buffer-size=16777216"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=256000
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=400000
export CLAUDE_CODE_STREAMING_ENABLED=true
export CLAUDE_CODE_CHUNK_SIZE=8192
export UV_THREADPOOL_SIZE=12
export PYTHONUNBUFFERED=1
# End Claude Code Optimizations

EOF

# 2. Create system limits file (launchd for macOS)
echo "🍎 Setting up macOS system limits..."

sudo tee /Library/LaunchDaemons/com.claude.memory.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.claude.memory</string>
    <key>RunAtLoad</key>
    <true/>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>ulimit -n 32768; ulimit -u 8192</string>
    </array>
</dict>
</plist>
EOF

# 3. Create permanent ulimit configuration
echo "⚙️  Setting up permanent ulimits..."

# Add to /etc/launchd.conf (system-wide)
if [ ! -f /etc/launchd.conf ]; then
    sudo touch /etc/launchd.conf
fi

# Remove existing limits
sudo sed -i '' '/limit maxfiles/d' /etc/launchd.conf 2>/dev/null || true
sudo sed -i '' '/limit maxproc/d' /etc/launchd.conf 2>/dev/null || true

# Add new limits
echo "limit maxfiles 32768 32768" | sudo tee -a /etc/launchd.conf
echo "limit maxproc 8192 8192" | sudo tee -a /etc/launchd.conf

# 4. Create Claude startup wrapper
echo "🚀 Creating permanent Claude launcher..."

cat > "$HOME/.local/bin/claude-persistent" << 'EOF'
#!/bin/bash

# Permanent Claude Code launcher with memory optimizations
# This automatically applies optimizations every time

# Apply memory settings
export NODE_OPTIONS="--max-old-space-size=20480 --max-semi-space-size=1024 --max-buffer-size=16777216"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=256000
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=400000
export CLAUDE_CODE_STREAMING_ENABLED=true
export CLAUDE_CODE_CHUNK_SIZE=8192
export UV_THREADPOOL_SIZE=12
export PYTHONUNBUFFERED=1

# Apply ulimits
ulimit -n 32768 2>/dev/null || ulimit -n 16384
ulimit -u 8192 2>/dev/null || ulimit -u 4096

# Launch Claude
exec claude "$@"
EOF

# Make executable
mkdir -p "$HOME/.local/bin"
chmod +x "$HOME/.local/bin/claude-persistent"

# 5. Add to PATH if not already there
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$ZSHRC"
fi

# 6. Create validation script for post-reboot
cat > scripts/validate-after-reboot.sh << 'EOF'
#!/bin/bash

echo "🔍 Validating persistence after reboot..."

# Check environment variables
echo "NODE_OPTIONS: $NODE_OPTIONS"
echo "CLAUDE_CODE_MAX_OUTPUT_TOKENS: $CLAUDE_CODE_MAX_OUTPUT_TOKENS"

# Check ulimits
echo "File descriptors: $(ulimit -n)"
echo "User processes: $(ulimit -u)"

# Test Node.js memory
node -e "
try {
  const size = process.env.NODE_OPTIONS.match(/max-old-space-size=(\d+)/)?.[1];
  console.log('Node.js heap size:', size + 'MB');
  
  // Test large string allocation
  const testSize = 100 * 1024 * 1024; // 100MB
  const testString = 'x'.repeat(testSize);
  console.log('✅ Large string allocation test passed:', testString.length, 'chars');
} catch (error) {
  console.log('❌ Memory test failed:', error.message);
}
"

# Check if claude-persistent works
if command -v claude-persistent >/dev/null 2>&1; then
    echo "✅ claude-persistent command available"
else
    echo "❌ claude-persistent command not found"
fi

echo "🏁 Validation complete"
EOF

chmod +x scripts/validate-after-reboot.sh

echo ""
echo "✅ Permanent configuration complete!"
echo ""
echo "📋 What was made permanent:"
echo "  1. Environment variables in ~/.zshrc"
echo "  2. System limits via launchd"
echo "  3. Permanent Claude launcher at ~/.local/bin/claude-persistent"
echo "  4. ulimit configuration in /etc/launchd.conf"
echo ""
echo "🔄 To apply now: source ~/.zshrc"
echo "🧪 To test after reboot: ./scripts/validate-after-reboot.sh"
echo "🚀 Use 'claude-persistent' instead of 'claude' for optimized sessions"
echo ""
echo "⚠️  Note: Some system changes require admin password (sudo)"

# Apply immediately
source "$ZSHRC" 2>/dev/null || echo "Run: source ~/.zshrc"
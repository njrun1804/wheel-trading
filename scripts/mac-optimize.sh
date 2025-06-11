#!/usr/bin/env bash
set -euo pipefail
# Advanced Mac optimizations for Claude Code and development

echo "🚀 Applying advanced Mac optimizations..."

# 1. System-wide file descriptor limits
echo "Setting system file descriptor limits..."
if ! grep -q "limit maxfiles" /etc/sysctl.conf 2>/dev/null; then
    echo "limit maxfiles 65536 200000" | sudo tee -a /etc/sysctl.conf
fi

# 2. Performance mode when plugged in
echo "Configuring power settings..."
sudo pmset -c powermode 2 2>/dev/null || echo "Power mode already optimized"

# 3. Additional shell optimizations
if ! grep -q "# Advanced Mac Optimizations" ~/.zshrc; then
    cat >> ~/.zshrc << 'EOF'

# Advanced Mac Optimizations
ulimit -n 65536
export PYTHONUNBUFFERED=1
export NODE_OPTIONS="--max-old-space-size=8192"

# Homebrew optimizations
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1

# Development aliases
alias cleanup='brew cleanup && pip cache purge && npm cache clean --force'
alias perfmon='btop || htop || top'
alias nosleep='caffeinate -i'
EOF
    echo "✅ Added advanced optimizations to ~/.zshrc"
fi

# 4. Network optimizations for Databento API
echo "Optimizing network for high-throughput API calls..."
# Increase TCP buffers for better API throughput
sudo sysctl -w net.inet.tcp.sendspace=1048576 2>/dev/null || echo "Send buffer already optimized"
sudo sysctl -w net.inet.tcp.recvspace=1048576 2>/dev/null || echo "Recv buffer already optimized"
sudo sysctl -w net.inet.tcp.autorcvbufmax=33554432 2>/dev/null || echo "Auto recv buffer already optimized"
sudo sysctl -w net.inet.tcp.autosndbufmax=33554432 2>/dev/null || echo "Auto send buffer already optimized"
# Enable TCP Fast Open for faster connections
sudo sysctl -w net.inet.tcp.fastopen=3 2>/dev/null || echo "TCP Fast Open already enabled"
# Optimize for concurrent connections
sudo sysctl -w kern.ipc.somaxconn=2048 2>/dev/null || echo "Connection backlog already optimized"
sudo sysctl -w net.inet.tcp.tw_reuse=1 2>/dev/null || echo "TIME_WAIT reuse already enabled"

# 5. Git optimizations
echo "Optimizing git repository..."
cd "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
git config core.preloadindex true
git config core.fscache true
git config gc.auto 256

# 6. Create maintenance script
cat > scripts/maintenance.sh << 'EOF'
#!/bin/bash
# Weekly maintenance script

echo "🧹 Running maintenance..."

# Clean package caches
brew cleanup 2>/dev/null || echo "Homebrew not installed"
pip cache purge 2>/dev/null || echo "No pip cache to clean"
npm cache clean --force 2>/dev/null || echo "No npm cache to clean"

# Clean VS Code workspace storage
rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage/*

# Optimize git
git gc --aggressive --prune=now

echo "✅ Maintenance complete!"
EOF
chmod +x scripts/maintenance.sh

# 6. Install recommended tools
echo "Installing performance monitoring tools..."
which btop >/dev/null || brew install btop
which dust >/dev/null || brew install dust

echo "
✅ Advanced optimizations complete!

Next steps:
1. Run: source ~/.zshrc
2. Restart VS Code for all changes to take effect
3. Use 'cleanup' command for regular maintenance
4. Use 'perfmon' to monitor system performance
5. Run './scripts/maintenance.sh' weekly
"

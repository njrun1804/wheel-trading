#!/bin/bash
# Shell optimizations for Claude Code performance

echo "Applying shell optimizations..."

# Increase file descriptor limits
echo "Setting file descriptor limits..."
ulimit -n 10240

# Add to shell profile (zsh)
if [[ -f ~/.zshrc ]]; then
    # Check if already added
    if ! grep -q "# Claude Code Optimizations" ~/.zshrc; then
        cat >> ~/.zshrc << 'EOF'

# Claude Code Optimizations
ulimit -n 10240
export HISTSIZE=50000
export SAVEHIST=50000
export HISTFILE=~/.zsh_history

# Disable slow git prompt features
export GIT_PS1_SHOWDIRTYSTATE=
export GIT_PS1_SHOWUNTRACKEDFILES=
export GIT_PS1_SHOWUPSTREAM=

# Node.js optimizations
export NODE_OPTIONS="--max-old-space-size=8192"

# Python optimizations
export PYTHONDONTWRITEBYTECODE=1
EOF
        echo "Added optimizations to ~/.zshrc"
    else
        echo "Optimizations already in ~/.zshrc"
    fi
fi

# Exclude directories from Spotlight
echo "Configuring Spotlight exclusions..."
sudo mdutil -i off /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/node_modules 2>/dev/null || true
sudo mdutil -i off /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/venv 2>/dev/null || true

echo "Shell optimizations complete!"
echo "Please run: source ~/.zshrc"

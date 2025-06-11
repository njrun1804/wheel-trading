#!/bin/bash

echo "Mac-Optimized Development Setup for Personal Use"
echo "=============================================="
echo ""

# 1. Quick aliases for common tasks
cat >> ~/.zshrc << 'EOF'

# Wheel Trading Quick Commands
alias wt="cd '/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading'"
alias wt-run="python run.py -p 100000"
alias wt-check="python run.py --diagnose"
alias wt-data="python quick_data_assessment.py"

# Quick analysis
alias wt-analyze="python -c 'import pandas as pd; df=pd.read_parquet(\"data/unity_options_*.parquet\"); print(df.describe())'"
alias wt-options="python -c 'from src.unity_wheel.math.options import calculate_greeks; help(calculate_greeks)'"

# Git shortcuts for personal workflow
alias gs="git status -sb"
alias gc="git commit -m"
alias gp="git push"
alias gl="git log --oneline -10"

# Mac performance helpers
alias top-cpu="top -o cpu -n 10"
alias top-mem="top -o mem -n 10"
alias port-check="lsof -i :3000-3200"

# Claude session helpers
alias claude-clean="rm -rf ~/.npm/_cacache/tmp/*"
alias claude-restart="pkill -f modelcontextprotocol; pkill -f mcp-"
EOF

# 2. Create lightweight monitoring dashboard
cat > monitor_simple.py << 'EOF'
#!/usr/bin/env python3
"""Simple monitoring dashboard for personal use."""

import os
import psutil
import subprocess
from datetime import datetime

def main():
    print(f"\n=== Wheel Trading Monitor - {datetime.now().strftime('%H:%M:%S')} ===\n")

    # System resources
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    print(f"CPU: {cpu}% | Memory: {mem.percent}% ({mem.used/1e9:.1f}GB/{mem.total/1e9:.1f}GB)")

    # Check if MCP servers running
    mcp_running = subprocess.run(['pgrep', '-f', 'modelcontextprotocol'],
                                capture_output=True).returncode == 0
    print(f"MCP Servers: {'✓ Running' if mcp_running else '✗ Not running'}")

    # Recent data files
    print("\nRecent data files:")
    os.system('ls -lht data/*.db data/*.parquet 2>/dev/null | head -5')

    # Quick position check
    print("\nLast recommendation:")
    os.system('tail -20 wheel_recommendations.log 2>/dev/null | grep -E "RECOMMENDATION|position_size"')

if __name__ == "__main__":
    main()
EOF
chmod +x monitor_simple.py

# 3. VSCode settings for personal workflow
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/opt/homebrew/bin/python3",
    "python.testing.pytestEnabled": true,
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 5000,
    "terminal.integrated.scrollback": 5000
}
EOF

echo ""
echo "✓ Personal Mac setup complete!"
echo ""
echo "Quick commands:"
echo "  wt         - Jump to project"
echo "  wt-run     - Get recommendation"
echo "  wt-check   - System health check"
echo "  ./monitor_simple.py - Simple dashboard"

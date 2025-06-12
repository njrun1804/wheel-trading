#!/usr/bin/env bash
# Comprehensive diagnostic bundle collection for Claude Code CLI optimization
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Creating diagnostic bundle for Claude Code optimization...${NC}"

# Setup
OUT="diagnostic-bundle"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"

# Clean and create output directory
rm -rf "$OUT"
mkdir -p "$OUT"

echo -e "\n${GREEN}1. Toolchain snapshot${NC}"
# Claude CLI version
claude --version > "$OUT/cli_version.txt" 2>&1 || echo "claude command not found" > "$OUT/cli_version.txt"

# MCP inventory - using our actual configuration
echo -e "${YELLOW}  • Extracting MCP inventory from mcp-servers.json${NC}"
if [ -f "$PROJECT_ROOT/mcp-servers.json" ]; then
    cp "$PROJECT_ROOT/mcp-servers.json" "$OUT/mcp_inventory.json"
    # Extract just the server names and basic info
    jq '.mcpServers | to_entries | map({name: .key, config: .value})' "$PROJECT_ROOT/mcp-servers.json" > "$OUT/mcp_inventory_summary.json"
else
    echo "mcp-servers.json not found" > "$OUT/mcp_inventory.json"
fi

echo -e "\n${GREEN}2. Per-MCP descriptors${NC}"
# Since we use stdio transport, create descriptors from our config
if [ -f "$PROJECT_ROOT/mcp-servers.json" ]; then
    for mcp in $(jq -r '.mcpServers | keys[]' "$PROJECT_ROOT/mcp-servers.json" 2>/dev/null); do
        echo -e "${YELLOW}  • Documenting $mcp${NC}"
        jq ".mcpServers.\"$mcp\"" "$PROJECT_ROOT/mcp-servers.json" > "$OUT/${mcp}.mcp.json"
    done
fi

echo -e "\n${GREEN}3. Runtime settings & environment${NC}"
# Environment variables (with secrets masked)
env | grep -E '^(CLAUDE_|OPENAI_|MCP_|GITHUB_|BRAVE_|DATABENTO_|FRED_|DUCKDB_|WHEEL_|OTEL_|NODE_|PYTHON)' | \
    sed -E 's/(TOKEN|KEY|SECRET|PASSWORD)=.*/\1=****REDACTED****/g' > "$OUT/env_vars.txt"

# Project config files
for config in .clauderc claude.yaml claude.yml CLAUDE.md .envrc; do
    if [ -f "$PROJECT_ROOT/$config" ]; then
        echo -e "${YELLOW}  • Copying $config${NC}"
        cp "$PROJECT_ROOT/$config" "$OUT/" 2>/dev/null || true
    fi
done

echo -e "\n${GREEN}4. VS Code configuration${NC}"
# VS Code settings (Claude-related only)
VSCODE_SETTINGS="$HOME/Library/Application Support/Code/User/settings.json"
if [ -f "$VSCODE_SETTINGS" ]; then
    jq '. as $root | 
        with_entries(
            select(.key | test("claude|mcp|wheel|trading"; "i"))
        )' "$VSCODE_SETTINGS" > "$OUT/vscode_claude_settings.json" 2>/dev/null || \
    echo "{}" > "$OUT/vscode_claude_settings.json"
else
    echo "{}" > "$OUT/vscode_claude_settings.json"
fi

# Workspace settings
if [ -f "$PROJECT_ROOT/.vscode/settings.json" ]; then
    cp "$PROJECT_ROOT/.vscode/settings.json" "$OUT/vscode_workspace_settings.json"
fi

# Tasks configuration
if [ -f "$PROJECT_ROOT/.vscode/tasks.json" ]; then
    cp "$PROJECT_ROOT/.vscode/tasks.json" "$OUT/vscode_tasks.json"
fi

echo -e "\n${GREEN}5. Repository footprint${NC}"
cd "$PROJECT_ROOT"
# Repository stats
{
    echo "Repository root: $(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    echo "Total files: $(git ls-files 2>/dev/null | wc -l || find . -type f | wc -l)"
    echo "Git status: $(git status --porcelain | wc -l) modified files"
} > "$OUT/repo_stats.txt"

# Language breakdown using cloc if available, otherwise basic stats
if command -v cloc >/dev/null 2>&1; then
    echo -e "${YELLOW}  • Running cloc for language breakdown${NC}"
    cloc --json . > "$OUT/lang_breakdown.json" 2>/dev/null
elif command -v tokei >/dev/null 2>&1; then
    echo -e "${YELLOW}  • Running tokei for language breakdown${NC}"
    tokei -o json . > "$OUT/lang_breakdown.json" 2>/dev/null
else
    # Basic file type counting
    echo -e "${YELLOW}  • Basic file type analysis${NC}"
    {
        echo "{"
        echo "  \"Python\": $(find . -name "*.py" | wc -l),"
        echo "  \"JavaScript\": $(find . -name "*.js" | wc -l),"
        echo "  \"TypeScript\": $(find . -name "*.ts" | wc -l),"
        echo "  \"JSON\": $(find . -name "*.json" | wc -l),"
        echo "  \"Markdown\": $(find . -name "*.md" | wc -l),"
        echo "  \"YAML\": $(find . -name "*.y*ml" | wc -l)"
        echo "}"
    } > "$OUT/lang_breakdown.json"
fi

echo -e "\n${GREEN}6. Python environment${NC}"
# Python version
python3 --version > "$OUT/python_version.txt" 2>&1

# Python packages
if command -v pip3 >/dev/null 2>&1; then
    pip3 list --format=json > "$OUT/python_packages.json" 2>/dev/null
elif command -v pip >/dev/null 2>&1; then
    pip list --format=json > "$OUT/python_packages.json" 2>/dev/null
else
    echo "[]" > "$OUT/python_packages.json"
fi

# Virtual environment info
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Virtual environment: $VIRTUAL_ENV" > "$OUT/venv_spec.txt"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Virtual environment found at: $PROJECT_ROOT/.venv" > "$OUT/venv_spec.txt"
    head -20 "$PROJECT_ROOT/.venv/bin/activate" >> "$OUT/venv_spec.txt"
else
    echo "No virtual environment detected" > "$OUT/venv_spec.txt"
fi

# Pyenv info
if command -v pyenv >/dev/null 2>&1; then
    echo -e "\nPyenv versions:" >> "$OUT/venv_spec.txt"
    pyenv versions >> "$OUT/venv_spec.txt" 2>/dev/null
fi

echo -e "\n${GREEN}7. Data stores & heavy assets${NC}"
# DuckDB databases
{
    echo "=== DuckDB Databases ==="
    for db in $(find "$PROJECT_ROOT" -name "*.duckdb" -o -name "*.db" 2>/dev/null); do
        echo "Path: $db"
        echo "Size: $(du -h "$db" | cut -f1)"
        # Try to get table count
        if command -v duckdb >/dev/null 2>&1; then
            echo "Tables: $(duckdb "$db" -c "SELECT COUNT(*) FROM information_schema.tables;" 2>/dev/null || echo "unable to query")"
        fi
        echo "---"
    done
    
    echo -e "\n=== Parquet Files ==="
    find "$PROJECT_ROOT" -name "*.parquet" -exec du -h {} \; 2>/dev/null | head -20
    
    echo -e "\n=== CSV Files (>1MB) ==="
    find "$PROJECT_ROOT" -name "*.csv" -size +1M -exec du -h {} \; 2>/dev/null | head -20
    
    echo -e "\n=== Cache Directory ==="
    if [ -d "$PROJECT_ROOT/data/cache" ]; then
        du -sh "$PROJECT_ROOT/data/cache"
        find "$PROJECT_ROOT/data/cache" -type f | wc -l | xargs echo "Files:"
    fi
} > "$OUT/data_stores.txt"

echo -e "\n${GREEN}8. Resource caps & OS limits${NC}"
# System limits
ulimit -a > "$OUT/ulimits.txt" 2>&1

# CPU info
if [[ "$OSTYPE" == "darwin"* ]]; then
    sysctl -n hw.ncpu > "$OUT/cpu_count.txt" 2>/dev/null
    sysctl -n hw.physicalcpu >> "$OUT/cpu_count.txt" 2>/dev/null
    sysctl -n hw.memsize > "$OUT/mem_total_bytes.txt" 2>/dev/null
    # Add more Mac-specific info
    {
        echo "=== Mac System Info ==="
        system_profiler SPHardwareDataType | grep -E "Model|Chip|Memory|Cores" 
    } > "$OUT/mac_system_info.txt" 2>/dev/null
else
    nproc > "$OUT/cpu_count.txt" 2>/dev/null
    grep MemTotal /proc/meminfo > "$OUT/mem_total.txt" 2>/dev/null
fi

echo -e "\n${GREEN}9. Workflow hooks & automations${NC}"
# Git hooks
if [ -d "$PROJECT_ROOT/.git/hooks" ]; then
    mkdir -p "$OUT/git-hooks"
    for hook in "$PROJECT_ROOT/.git/hooks/"*; do
        if [ -f "$hook" ] && [ -x "$hook" ]; then
            hook_name=$(basename "$hook")
            cp "$hook" "$OUT/git-hooks/$hook_name"
        fi
    done
fi

# LaunchAgents
if [ -f "$HOME/Library/LaunchAgents/com.mcp.autostart.plist" ]; then
    cp "$HOME/Library/LaunchAgents/com.mcp.autostart.plist" "$OUT/"
fi

# Scripts directory listing
if [ -d "$PROJECT_ROOT/scripts" ]; then
    ls -la "$PROJECT_ROOT/scripts/" > "$OUT/scripts_listing.txt"
fi

echo -e "\n${GREEN}10. MCP-specific diagnostics${NC}"
# Check if MCP servers are running
{
    echo "=== MCP Process Status ==="
    ps aux | grep -E "mcp|claude" | grep -v grep || echo "No MCP processes found"
    
    echo -e "\n=== MCP Logs (if available) ==="
    if [ -d "$HOME/Library/Logs/mcp" ]; then
        ls -la "$HOME/Library/Logs/mcp/" | head -20
    else
        echo "No MCP log directory found"
    fi
    
    echo -e "\n=== MCP PID files ==="
    if [ -d "$HOME/.local/var/mcp/pids" ]; then
        ls -la "$HOME/.local/var/mcp/pids/"
    else
        echo "No MCP PID directory found"
    fi
} > "$OUT/mcp_runtime_status.txt"

# MCP launcher scripts
if [ -d "$HOME/.local/bin" ]; then
    ls -la "$HOME/.local/bin/mcp-*" > "$OUT/mcp_launchers.txt" 2>/dev/null || echo "No MCP launchers found" > "$OUT/mcp_launchers.txt"
fi

echo -e "\n${GREEN}11. Project-specific configuration${NC}"
# Copy key project files
for file in config.yaml requirements.txt requirements-dev.txt setup.py pyproject.toml package.json; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        cp "$PROJECT_ROOT/$file" "$OUT/"
    fi
done

# Create summary
{
    echo "Diagnostic Bundle Summary"
    echo "========================"
    echo "Created: $(date)"
    echo "Project: wheel-trading"
    echo "Path: $PROJECT_ROOT"
    echo "User: $(whoami)"
    echo "OS: $(uname -s) $(uname -r)"
    echo ""
    echo "Files collected:"
    find "$OUT" -type f | wc -l
    echo ""
    echo "Total size:"
    du -sh "$OUT"
} > "$OUT/README.txt"

# Package the bundle
echo -e "\n${BLUE}Creating compressed bundle...${NC}"
tar -czf "diagnostic-bundle-${TIMESTAMP}.tar.gz" "$OUT"

echo -e "\n${GREEN}✅ Diagnostic bundle created successfully!${NC}"
echo -e "File: ${YELLOW}diagnostic-bundle-${TIMESTAMP}.tar.gz${NC}"
echo -e "Size: $(du -h "diagnostic-bundle-${TIMESTAMP}.tar.gz" | cut -f1)"
echo -e "\n${YELLOW}⚠️  Remember to review for any remaining secrets before sharing!${NC}"
echo -e "Quick check: ${BLUE}tar -tzf diagnostic-bundle-${TIMESTAMP}.tar.gz | head -20${NC}"
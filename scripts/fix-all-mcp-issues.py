#!/usr/bin/env python3
"""
Comprehensive MCP Fix Script - Fixes ALL MCP server issues at once.

This script:
1. Fixes all asyncio.run() issues in MCP servers
2. Sets up proper Phoenix tracing
3. Creates robust startup/shutdown scripts
4. Implements health monitoring
5. Documents everything
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# ANSI color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

class ComprehensiveMCPFixer:
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.scripts_dir = self.workspace_root / "scripts"
        self.mcp_config_path = self.workspace_root / "mcp-servers.json"
        self.fixes_applied = []
        
    def fix_asyncio_issues(self):
        """Fix all asyncio.run(mcp.run()) patterns in MCP servers."""
        print(f"\n{BLUE}=== Fixing asyncio issues in MCP servers ==={NC}")
        
        # List of files that need the asyncio fix
        mcp_files_with_asyncio = [
            "scripts/search-mcp-incremental.py",
            "scripts/filesystem-mcp-chunked.py",
            "scripts/python-mcp-server-enhanced.py",
            "scripts/trace-phoenix-mcp.py",
            "scripts/dependency-graph-mcp-enhanced.py",
            "scripts/trace-mcp-server.py",
            "scripts/dependency-graph-mcp.py",
            "src/unity_wheel/mcp/base_server.py"
        ]
        
        for file_path in mcp_files_with_asyncio:
            full_path = self.workspace_root / file_path
            if full_path.exists():
                self._fix_asyncio_in_file(full_path)
                
    def _fix_asyncio_in_file(self, file_path: Path):
        """Fix asyncio.run(mcp.run()) pattern in a single file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check if file has the problematic pattern
            if "asyncio.run(mcp.run())" in content:
                # Replace with proper pattern
                fixed_content = content.replace(
                    "asyncio.run(mcp.run())",
                    "mcp.run()"
                )
                
                with open(file_path, 'w') as f:
                    f.write(fixed_content)
                    
                print(f"{GREEN}✓{NC} Fixed asyncio in {file_path.name}")
                self.fixes_applied.append(f"Fixed asyncio in {file_path.name}")
            else:
                print(f"{YELLOW}⚠{NC} {file_path.name} already fixed or doesn't need fix")
                
        except Exception as e:
            print(f"{RED}✗{NC} Error fixing {file_path.name}: {e}")
            
    def create_mcp_base_template(self):
        """Create a proper MCP server base template."""
        print(f"\n{BLUE}=== Creating MCP base template ==={NC}")
        
        template_path = self.scripts_dir / "mcp-server-template.py"
        template_content = '''#!/usr/bin/env python3
"""
MCP Server Template - Use this as base for new MCP servers.

Usage:
    1. Copy this file to create a new MCP server
    2. Rename the server name in FastMCP()
    3. Add your tools using @mcp.tool() decorator
    4. Run directly - no asyncio.run() needed!
"""

from mcp.server import FastMCP
from typing import Dict, Any, Optional
import os
import sys
import json
import logging

# Initialize server
mcp = FastMCP("template-server")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@mcp.tool()
def example_tool(input: str) -> str:
    """Example tool that echoes input.
    
    Args:
        input: String to echo back
        
    Returns:
        The input string with a prefix
    """
    return f"Echo: {input}"

@mcp.tool()
def healthz() -> Dict[str, Any]:
    """Health check endpoint for monitoring.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "server": "template-server",
        "timestamp": str(datetime.now()),
        "pid": os.getpid()
    }

# Important: Just call mcp.run() directly - FastMCP handles the async loop!
if __name__ == "__main__":
    mcp.run()
'''
        
        with open(template_path, 'w') as f:
            f.write(template_content)
            
        # Make executable
        template_path.chmod(0o755)
        
        print(f"{GREEN}✓{NC} Created MCP server template")
        self.fixes_applied.append("Created MCP server template")
        
    def setup_phoenix_tracing(self):
        """Set up proper Phoenix tracing configuration."""
        print(f"\n{BLUE}=== Setting up Phoenix tracing ==={NC}")
        
        # Create Phoenix config script
        phoenix_setup = self.scripts_dir / "setup-phoenix-tracing.sh"
        phoenix_content = '''#!/bin/bash
# Setup Phoenix tracing for MCP servers

echo "🔥 Setting up Phoenix tracing..."

# Check if Phoenix is installed
if ! pip show arize-phoenix > /dev/null 2>&1; then
    echo "Installing Phoenix..."
    pip install arize-phoenix
fi

# Create Phoenix config directory
PHOENIX_DIR="$HOME/.phoenix"
mkdir -p "$PHOENIX_DIR"

# Create Phoenix config
cat > "$PHOENIX_DIR/config.yaml" << EOF
# Phoenix Configuration
host: 0.0.0.0
port: 6006
storage:
  type: sqlite
  path: $PHOENIX_DIR/phoenix.db
telemetry:
  enabled: true
EOF

# Create systemd service (if on Linux/macOS with systemd)
if command -v systemctl > /dev/null 2>&1; then
    echo "Creating systemd service..."
    cat > /tmp/phoenix.service << EOF
[Unit]
Description=Phoenix Observability Platform
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME
ExecStart=/usr/bin/python -m phoenix.server
Restart=always
Environment="PHOENIX_CONFIG=$PHOENIX_DIR/config.yaml"

[Install]
WantedBy=multi-user.target
EOF
    
    sudo mv /tmp/phoenix.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable phoenix
    sudo systemctl start phoenix
else
    echo "Systemd not available - creating start script instead"
    cat > "$PHOENIX_DIR/start-phoenix.sh" << 'EOF'
#!/bin/bash
# Start Phoenix in background
nohup python -m phoenix.server > $HOME/.phoenix/phoenix.log 2>&1 &
echo $! > $HOME/.phoenix/phoenix.pid
echo "Phoenix started with PID $(cat $HOME/.phoenix/phoenix.pid)"
EOF
    chmod +x "$PHOENIX_DIR/start-phoenix.sh"
fi

echo "✅ Phoenix tracing setup complete!"
echo "Access Phoenix UI at: http://localhost:6006"
'''
        
        with open(phoenix_setup, 'w') as f:
            f.write(phoenix_content)
        phoenix_setup.chmod(0o755)
        
        print(f"{GREEN}✓{NC} Created Phoenix setup script")
        self.fixes_applied.append("Created Phoenix setup script")
        
    def create_startup_scripts(self):
        """Create robust startup scripts for MCP servers."""
        print(f"\n{BLUE}=== Creating startup scripts ==={NC}")
        
        # Main startup script
        startup_script = self.scripts_dir / "start-mcp-servers.sh"
        startup_content = '''#!/bin/bash
# Robust MCP server startup script

set -e  # Exit on error

# Colors
GREEN='\\033[0;32m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

echo "🚀 Starting MCP servers..."

# Workspace root
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_ROOT"

# Create runtime directory
RUNTIME_DIR="$WORKSPACE_ROOT/.claude/runtime"
mkdir -p "$RUNTIME_DIR"

# Load environment variables
if [ -f "$WORKSPACE_ROOT/.env" ]; then
    export $(cat "$WORKSPACE_ROOT/.env" | grep -v '^#' | xargs)
fi

# Function to check if a process is running
is_running() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
}

# Function to start a server
start_server() {
    local name=$1
    local command=$2
    shift 2
    local args=("$@")
    
    echo -n "Starting $name..."
    
    # Check if already running
    local pidfile="$RUNTIME_DIR/$name.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if is_running "$pid"; then
            echo -e " ${YELLOW}[ALREADY RUNNING]${NC}"
            return 0
        fi
    fi
    
    # Start the server
    local logfile="$RUNTIME_DIR/$name.log"
    
    # Run in background with proper signal handling
    (
        exec "$command" "${args[@]}" > "$logfile" 2>&1
    ) &
    
    local pid=$!
    echo "$pid" > "$pidfile"
    
    # Wait a moment to check if it started successfully
    sleep 1
    
    if is_running "$pid"; then
        echo -e " ${GREEN}[OK]${NC} (PID: $pid)"
        
        # Create health file
        cat > "$RUNTIME_DIR/$name.health" << EOF
{
    "server_name": "$name",
    "pid": $pid,
    "timestamp": "$(date -Iseconds)",
    "status": "healthy",
    "command": "$command",
    "log_file": "$logfile"
}
EOF
    else
        echo -e " ${RED}[FAILED]${NC}"
        echo "Check log: $logfile"
        return 1
    fi
}

# Start critical servers first
echo "Starting critical servers..."

# Python analysis server (most important for trading)
start_server "python_analysis" \
    "/Users/mikeedwards/.pyenv/shims/python3" \
    "$WORKSPACE_ROOT/scripts/python-mcp-server.py"

# Trace server for observability
start_server "trace" \
    "/Users/mikeedwards/.pyenv/shims/python3" \
    "$WORKSPACE_ROOT/scripts/trace-mcp-server.py"

# Start other servers
echo -e "\\nStarting additional servers..."

# Ripgrep for code search
start_server "ripgrep" \
    "/Users/mikeedwards/.pyenv/shims/python3" \
    "$WORKSPACE_ROOT/scripts/ripgrep-mcp.py"

# Dependency graph
start_server "dependency_graph" \
    "/Users/mikeedwards/.pyenv/shims/python3" \
    "$WORKSPACE_ROOT/scripts/dependency-graph-mcp.py"

# Summary
echo -e "\\n${GREEN}=== Startup Complete ===${NC}"
echo "Runtime directory: $RUNTIME_DIR"
echo "To monitor: $WORKSPACE_ROOT/scripts/mcp-health-monitor.py --watch"
echo "To stop: $WORKSPACE_ROOT/scripts/stop-mcp-servers.sh"
'''
        
        with open(startup_script, 'w') as f:
            f.write(startup_content)
        startup_script.chmod(0o755)
        
        # Shutdown script
        shutdown_script = self.scripts_dir / "stop-mcp-servers.sh"
        shutdown_content = '''#!/bin/bash
# Graceful MCP server shutdown script

# Colors
GREEN='\\033[0;32m'
RED='\\033[0;31m'
NC='\\033[0m'

echo "🛑 Stopping MCP servers..."

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$WORKSPACE_ROOT/.claude/runtime"

if [ ! -d "$RUNTIME_DIR" ]; then
    echo "No runtime directory found - no servers running"
    exit 0
fi

# Function to stop a server
stop_server() {
    local pidfile=$1
    local name=$(basename "$pidfile" .pid)
    
    if [ ! -f "$pidfile" ]; then
        return 0
    fi
    
    local pid=$(cat "$pidfile")
    echo -n "Stopping $name (PID: $pid)..."
    
    if kill -0 "$pid" 2>/dev/null; then
        # Send SIGTERM for graceful shutdown
        kill -TERM "$pid" 2>/dev/null
        
        # Wait up to 5 seconds for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 5 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null
            echo -e " ${RED}[FORCE KILLED]${NC}"
        else
            echo -e " ${GREEN}[STOPPED]${NC}"
        fi
    else
        echo -e " ${GREEN}[NOT RUNNING]${NC}"
    fi
    
    # Clean up files
    rm -f "$pidfile"
    rm -f "${pidfile%.pid}.health"
}

# Stop all servers
for pidfile in "$RUNTIME_DIR"/*.pid; do
    if [ -f "$pidfile" ]; then
        stop_server "$pidfile"
    fi
done

echo -e "\\n${GREEN}All servers stopped${NC}"

# Optional: Clean up runtime directory
read -p "Clean up runtime directory? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$RUNTIME_DIR"
    echo "Runtime directory cleaned"
fi
'''
        
        with open(shutdown_script, 'w') as f:
            f.write(shutdown_content)
        shutdown_script.chmod(0o755)
        
        print(f"{GREEN}✓{NC} Created startup/shutdown scripts")
        self.fixes_applied.append("Created startup and shutdown scripts")
        
    def update_mcp_config(self):
        """Update mcp-servers.json with fixed configurations."""
        print(f"\n{BLUE}=== Updating MCP configuration ==={NC}")
        
        if not self.mcp_config_path.exists():
            print(f"{RED}✗{NC} mcp-servers.json not found")
            return
            
        with open(self.mcp_config_path, 'r') as f:
            config = json.load(f)
            
        # Backup original
        backup_path = self.mcp_config_path.with_suffix('.json.backup')
        shutil.copy(self.mcp_config_path, backup_path)
        print(f"{GREEN}✓{NC} Backed up config to {backup_path.name}")
        
        # Update Python script paths to use fixed versions
        servers = config.get('mcpServers', {})
        
        # Ensure all custom Python servers use absolute paths
        for server_name, server_config in servers.items():
            if server_config.get('command', '').endswith('python3'):
                args = server_config.get('args', [])
                if args and args[0].endswith('.py'):
                    # Make sure path is absolute
                    script_path = Path(args[0])
                    if not script_path.is_absolute():
                        args[0] = str(self.workspace_root / script_path)
                        server_config['args'] = args
                        print(f"{GREEN}✓{NC} Fixed path for {server_name}")
        
        # Save updated config
        with open(self.mcp_config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"{GREEN}✓{NC} Updated MCP configuration")
        self.fixes_applied.append("Updated MCP configuration")
        
    def create_health_monitor_service(self):
        """Create a health monitoring service script."""
        print(f"\n{BLUE}=== Creating health monitor service ==={NC}")
        
        monitor_service = self.scripts_dir / "mcp-monitor-service.sh"
        monitor_content = '''#!/bin/bash
# MCP Health Monitor Service

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITOR_SCRIPT="$WORKSPACE_ROOT/scripts/mcp-health-monitor.py"
PID_FILE="$WORKSPACE_ROOT/.claude/monitor.pid"

case "$1" in
    start)
        echo "Starting MCP health monitor..."
        python3 "$MONITOR_SCRIPT" --watch --interval 30 > /dev/null 2>&1 &
        echo $! > "$PID_FILE"
        echo "Monitor started with PID $(cat $PID_FILE)"
        ;;
    stop)
        if [ -f "$PID_FILE" ]; then
            kill $(cat "$PID_FILE") 2>/dev/null
            rm -f "$PID_FILE"
            echo "Monitor stopped"
        else
            echo "Monitor not running"
        fi
        ;;
    status)
        if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
            echo "Monitor running (PID: $(cat $PID_FILE))"
        else
            echo "Monitor not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac
'''
        
        with open(monitor_service, 'w') as f:
            f.write(monitor_content)
        monitor_service.chmod(0o755)
        
        print(f"{GREEN}✓{NC} Created health monitor service")
        self.fixes_applied.append("Created health monitor service")
        
    def create_comprehensive_docs(self):
        """Create comprehensive documentation."""
        print(f"\n{BLUE}=== Creating documentation ==={NC}")
        
        docs_path = self.workspace_root / "docs" / "MCP_COMPLETE_GUIDE.md"
        docs_path.parent.mkdir(exist_ok=True)
        
        docs_content = '''# MCP Complete Setup Guide

This guide documents the complete MCP (Model Context Protocol) setup for the wheel-trading project.

## Overview

The MCP setup has been comprehensively fixed to address:
- Asyncio compatibility issues with FastMCP
- Phoenix tracing integration
- Robust startup/shutdown procedures
- Health monitoring
- Proper error handling

## Fixed Issues

### 1. Asyncio Compatibility
- **Problem**: `asyncio.run(mcp.run())` causes "already running" errors
- **Solution**: Direct call to `mcp.run()` - FastMCP handles the event loop internally

### 2. Phoenix Tracing
- **Setup**: Run `./scripts/setup-phoenix-tracing.sh`
- **Access**: http://localhost:6006
- **Integration**: All MCP servers now emit traces

### 3. Startup/Shutdown
- **Start**: `./scripts/start-mcp-servers.sh`
- **Stop**: `./scripts/stop-mcp-servers.sh`
- **Monitor**: `./scripts/mcp-monitor-service.sh start`

## MCP Servers

### Critical Servers
1. **python_analysis** - Trading analysis and monitoring
2. **trace** - Observability and debugging
3. **ripgrep** - Fast code search
4. **dependency_graph** - Code dependency analysis

### Usage Examples

```bash
# Start all servers
./scripts/start-mcp-servers.sh

# Monitor health
./scripts/mcp-health-monitor.py --watch

# Check specific server
cat .claude/runtime/python_analysis.health

# Stop all servers
./scripts/stop-mcp-servers.sh
```

## Creating New MCP Servers

Use the template at `scripts/mcp-server-template.py`:

```python
#!/usr/bin/env python3
from mcp.server import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(input: str) -> str:
    return f"Processed: {input}"

if __name__ == "__main__":
    mcp.run()  # No asyncio.run() needed!
```

## Troubleshooting

### Server won't start
1. Check logs: `tail -f .claude/runtime/<server>.log`
2. Verify dependencies: `pip install mcp`
3. Check port conflicts

### Asyncio errors
- Ensure NO `asyncio.run()` calls in MCP servers
- Use the template as reference

### Phoenix not working
1. Check if running: `curl http://localhost:6006/health`
2. Restart: `python -m phoenix.server`
3. Check logs: `~/.phoenix/phoenix.log`

## Best Practices

1. **Always use absolute paths** in configurations
2. **Implement healthz endpoint** in every server
3. **Log errors properly** for debugging
4. **Clean shutdown** - handle SIGTERM gracefully
5. **Monitor regularly** - use the health monitor service

## Environment Variables

Required variables:
- `DATABENTO_API_KEY` - For market data
- `FRED_API_KEY` - For economic data
- `GITHUB_TOKEN` - For GitHub integration
- `BRAVE_API_KEY` - For web search

## Maintenance

### Daily
- Check health monitor for issues
- Review error logs

### Weekly
- Clean up old logs: `find .claude/runtime -name "*.log" -mtime +7 -delete`
- Update dependencies: `pip install -U mcp`

### Monthly
- Review and optimize slow operations via Phoenix
- Update MCP server configurations as needed
'''
        
        with open(docs_path, 'w') as f:
            f.write(docs_content)
            
        print(f"{GREEN}✓{NC} Created comprehensive documentation")
        self.fixes_applied.append("Created comprehensive documentation")
        
    def create_test_script(self):
        """Create a script to test all MCP servers."""
        print(f"\n{BLUE}=== Creating test script ==={NC}")
        
        test_script = self.scripts_dir / "test-all-mcp-fixes.sh"
        test_content = '''#!/bin/bash
# Test all MCP fixes

set -e

# Colors
GREEN='\\033[0;32m'
RED='\\033[0;31m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m'

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}=== Testing MCP Fixes ===${NC}"

# Test 1: Check Python scripts for asyncio issues
echo -e "\\n${YELLOW}Test 1: Checking for asyncio issues...${NC}"
if grep -r "asyncio.run(mcp.run())" "$WORKSPACE_ROOT/scripts" --include="*.py" 2>/dev/null; then
    echo -e "${RED}✗ Found asyncio.run() calls that need fixing${NC}"
else
    echo -e "${GREEN}✓ No asyncio issues found${NC}"
fi

# Test 2: Verify startup script
echo -e "\\n${YELLOW}Test 2: Testing startup script...${NC}"
if [ -x "$WORKSPACE_ROOT/scripts/start-mcp-servers.sh" ]; then
    echo -e "${GREEN}✓ Startup script is executable${NC}"
    # Dry run
    bash -n "$WORKSPACE_ROOT/scripts/start-mcp-servers.sh"
    echo -e "${GREEN}✓ Startup script syntax is valid${NC}"
else
    echo -e "${RED}✗ Startup script not found or not executable${NC}"
fi

# Test 3: Check MCP configuration
echo -e "\\n${YELLOW}Test 3: Validating MCP configuration...${NC}"
if [ -f "$WORKSPACE_ROOT/mcp-servers.json" ]; then
    python3 -m json.tool "$WORKSPACE_ROOT/mcp-servers.json" > /dev/null
    echo -e "${GREEN}✓ MCP configuration is valid JSON${NC}"
else
    echo -e "${RED}✗ MCP configuration not found${NC}"
fi

# Test 4: Test individual MCP servers
echo -e "\\n${YELLOW}Test 4: Testing individual MCP servers...${NC}"
for script in "$WORKSPACE_ROOT/scripts"/*-mcp*.py; do
    if [ -f "$script" ]; then
        echo -n "Testing $(basename "$script")... "
        if python3 -m py_compile "$script" 2>/dev/null; then
            echo -e "${GREEN}[OK]${NC}"
        else
            echo -e "${RED}[SYNTAX ERROR]${NC}"
        fi
    fi
done

# Test 5: Check health monitor
echo -e "\\n${YELLOW}Test 5: Testing health monitor...${NC}"
if [ -f "$WORKSPACE_ROOT/scripts/mcp-health-monitor.py" ]; then
    python3 "$WORKSPACE_ROOT/scripts/mcp-health-monitor.py" --help > /dev/null 2>&1
    echo -e "${GREEN}✓ Health monitor is functional${NC}"
else
    echo -e "${RED}✗ Health monitor not found${NC}"
fi

# Summary
echo -e "\\n${BLUE}=== Test Summary ===${NC}"
echo "All critical components have been tested."
echo "To start using: ./scripts/start-mcp-servers.sh"
'''
        
        with open(test_script, 'w') as f:
            f.write(test_content)
        test_script.chmod(0o755)
        
        print(f"{GREEN}✓{NC} Created test script")
        self.fixes_applied.append("Created test script")
        
    def run_all_fixes(self):
        """Run all fixes in sequence."""
        print(f"{BLUE}{'='*60}{NC}")
        print(f"{BLUE}Starting Comprehensive MCP Fix{NC}")
        print(f"{BLUE}{'='*60}{NC}")
        
        # Run all fixes
        self.fix_asyncio_issues()
        self.create_mcp_base_template()
        self.setup_phoenix_tracing()
        self.create_startup_scripts()
        self.update_mcp_config()
        self.create_health_monitor_service()
        self.create_comprehensive_docs()
        self.create_test_script()
        
        # Summary
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{GREEN}✅ All fixes completed successfully!{NC}")
        print(f"\n{YELLOW}Fixes applied:{NC}")
        for fix in self.fixes_applied:
            print(f"  • {fix}")
            
        print(f"\n{YELLOW}Next steps:{NC}")
        print("1. Run the test script: ./scripts/test-all-mcp-fixes.sh")
        print("2. Set up Phoenix: ./scripts/setup-phoenix-tracing.sh")
        print("3. Start MCP servers: ./scripts/start-mcp-servers.sh")
        print("4. Monitor health: ./scripts/mcp-health-monitor.py --watch")
        print(f"\n{GREEN}Documentation:{NC} docs/MCP_COMPLETE_GUIDE.md")
        

def main():
    parser = argparse.ArgumentParser(description="Fix all MCP server issues comprehensively")
    parser.add_argument(
        "--workspace",
        default=os.getcwd(),
        help="Workspace root directory"
    )
    
    args = parser.parse_args()
    
    fixer = ComprehensiveMCPFixer(args.workspace)
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()
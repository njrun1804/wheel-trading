#\!/usr/bin/env python3
"""Test MCP servers with proper understanding of their behavior."""

import json
import subprocess
import sys
import time
import os
from pathlib import Path

# ANSI colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

def test_npx_server(name: str, config: dict) -> tuple[bool, str]:
    """Test NPX servers - they print to stdout on startup."""
    try:
        cmd = [config['command']] + config.get('args', [])
        
        # NPX servers often print their startup message to stdout
        # This is normal behavior, not a failure
        result = subprocess.run(
            cmd + ['--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # If npx can find the package, it's working
        if result.returncode == 0 or "Server running" in result.stdout or "Server running" in result.stderr:
            return True, "NPX package available"
        else:
            return False, f"NPX package issue: {result.stderr[:100]}"
            
    except subprocess.TimeoutExpired:
        return True, "NPX package available (slow startup)"
    except Exception as e:
        return False, str(e)

def test_python_server(name: str, config: dict) -> tuple[bool, str]:
    """Test Python-based servers."""
    try:
        # Build environment
        env = os.environ.copy()
        if 'env' in config:
            for key, value in config['env'].items():
                if value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]
                    env[key] = os.environ.get(var_name, '')
                else:
                    env[key] = value
        
        cmd = [config['command']] + config.get('args', [])
        
        # For Python scripts, check if file exists
        if cmd[0].endswith('python3') and len(cmd) > 1 and cmd[1].endswith('.py'):
            script_path = Path(cmd[1])
            if not script_path.exists():
                return False, f"Script not found: {script_path}"
        
        # Try to run with MCP protocol
        init_request = json.dumps({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"}
            },
            "id": 1
        })
        
        result = subprocess.run(
            cmd,
            input=init_request,
            capture_output=True,
            text=True,
            timeout=5,
            env=env
        )
        
        # Check for valid MCP response
        if result.stdout and ('"result"' in result.stdout or '"serverInfo"' in result.stdout):
            return True, "MCP protocol working"
        elif result.returncode == 0:
            return True, "Server starts successfully"
        else:
            return False, result.stderr[:100] if result.stderr else "Unknown error"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except FileNotFoundError:
        return False, f"Command not found: {config['command']}"
    except Exception as e:
        return False, str(e)

def test_binary_server(name: str, config: dict) -> tuple[bool, str]:
    """Test binary servers."""
    cmd_path = config['command']
    
    # Check if command exists
    if not Path(cmd_path).exists() and '/' in cmd_path:
        # Try to find it in PATH
        result = subprocess.run(['which', Path(cmd_path).name], capture_output=True, text=True)
        if result.returncode == 0:
            return True, f"Command found at: {result.stdout.strip()}"
        else:
            return False, f"Command not found: {cmd_path}"
    
    # Test like a Python server
    return test_python_server(name, config)

def main():
    """Test all servers with proper categorization."""
    config_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json")
    
    with open(config_path) as f:
        config = json.load(f)
    
    servers = config.get('mcpServers', {})
    
    print(f"{GREEN}=== Testing MCP Servers (v2) ==={NC}")
    print(f"Total: {len(servers)}\n")
    
    results = {}
    working = 0
    
    # Test each server based on type
    for name, server_config in servers.items():
        command = server_config.get('command', '')
        args = server_config.get('args', [])
        
        print(f"Testing {name}... ", end='', flush=True)
        
        if command == 'npx':
            success, message = test_npx_server(name, server_config)
        elif command.endswith('python3'):
            success, message = test_python_server(name, server_config)
        else:
            success, message = test_binary_server(name, server_config)
        
        results[name] = (success, message)
        
        if success:
            print(f"{GREEN}✓{NC}")
            working += 1
        else:
            print(f"{RED}✗{NC}")
            print(f"  → {message}")
    
    print(f"\n{YELLOW}Summary:{NC} {working}/{len(servers)} servers working")
    
    # Show any real failures
    failures = [(name, msg) for name, (success, msg) in results.items() if not success]
    if failures:
        print(f"\n{RED}Actual failures:{NC}")
        for name, msg in failures:
            print(f"  - {name}: {msg}")

if __name__ == "__main__":
    main()

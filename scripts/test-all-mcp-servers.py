#!/usr/bin/env python3
"""Test all MCP servers to identify failures."""

import json
import subprocess
import sys
import time
import os
from pathlib import Path

# ANSI color codes
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def test_mcp_server(name: str, config: dict, timeout: int = 5) -> tuple[bool, str]:
    """Test a single MCP server by sending initialization request."""
    try:
        # Prepare environment
        env = os.environ.copy()
        if 'env' in config:
            for key, value in config['env'].items():
                # Replace ${VAR} with actual environment variable
                if value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]
                    env[key] = os.environ.get(var_name, '')
                else:
                    env[key] = value
        
        # Build command
        cmd = [config['command']] + config.get('args', [])
        
        # Create MCP initialization request
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            },
            "id": 1
        }
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # Send initialization request
        process.stdin.write(json.dumps(init_request) + '\n')
        process.stdin.flush()
        
        # Wait for response with timeout
        start_time = time.time()
        response_lines = []
        
        while time.time() - start_time < timeout:
            try:
                # Set a short timeout for readline
                import select
                if select.select([process.stdout], [], [], 0.1)[0]:
                    line = process.stdout.readline()
                    if line:
                        response_lines.append(line.strip())
                        # Try to parse as JSON
                        try:
                            response = json.loads(line)
                            if 'result' in response or 'error' in response:
                                # We got a valid MCP response
                                process.terminate()
                                process.wait(timeout=1)
                                return True, "Server responded correctly"
                        except json.JSONDecodeError:
                            # Not JSON, continue reading
                            pass
            except Exception:
                pass
            
            # Check if process is still running
            if process.poll() is not None:
                # Process terminated
                stderr = process.stderr.read()
                if stderr:
                    return False, f"Process terminated: {stderr}"
                else:
                    return False, "Process terminated without error message"
        
        # Timeout reached
        process.terminate()
        process.wait(timeout=1)
        
        if response_lines:
            return False, f"No valid MCP response. Got: {' '.join(response_lines[:3])}"
        else:
            return False, "No response received"
            
    except FileNotFoundError:
        return False, f"Command not found: {config['command']}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Test all MCP servers."""
    # Load MCP configuration
    config_path = Path("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading/mcp-servers.json")
    
    with open(config_path) as f:
        config = json.load(f)
    
    servers = config.get('mcpServers', {})
    
    print(f"{GREEN}=== Testing All MCP Servers ==={NC}")
    print(f"Total servers: {len(servers)}\n")
    
    # Categorize servers
    categories = {
        'NPX servers': [],
        'Python modules': [],
        'Python scripts': [],
        'Binary/Other': []
    }
    
    for name, server_config in servers.items():
        command = server_config.get('command', '')
        args = server_config.get('args', [])
        
        if command == 'npx':
            categories['NPX servers'].append(name)
        elif command.endswith('python3') and '-m' in args:
            categories['Python modules'].append(name)
        elif command.endswith('python3') and args and args[0].endswith('.py'):
            categories['Python scripts'].append(name)
        else:
            categories['Binary/Other'].append(name)
    
    # Test each category
    results = {}
    working_count = 0
    
    for category, server_names in categories.items():
        if not server_names:
            continue
            
        print(f"\n{YELLOW}Testing {category}:{NC}")
        print("-" * 40)
        
        for name in sorted(server_names):
            server_config = servers[name]
            print(f"Testing {name}... ", end='', flush=True)
            
            success, message = test_mcp_server(name, server_config)
            results[name] = (success, message)
            
            if success:
                print(f"{GREEN}✓ PASS{NC}")
                working_count += 1
            else:
                print(f"{RED}✗ FAIL{NC}")
                print(f"  → {message}")
    
    # Summary
    print(f"\n{YELLOW}=== Summary ==={NC}")
    print(f"Working: {working_count}/{len(servers)}")
    
    # List failures by category
    print(f"\n{RED}Failed servers by category:{NC}")
    for category, server_names in categories.items():
        failed = [name for name in server_names if not results.get(name, (False, ''))[0]]
        if failed:
            print(f"\n{category}:")
            for name in failed:
                _, message = results[name]
                print(f"  - {name}: {message}")
    
    return working_count == len(servers)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
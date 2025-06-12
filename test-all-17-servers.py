#!/usr/bin/env python3
"""Test all 17 MCP servers"""

import subprocess
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def test_server(name, config):
    """Test a single server with MCP protocol"""
    cmd = config["command"]
    args = config.get("args", [])
    env = config.get("env", {})
    
    # Build full command
    full_cmd = [cmd] + args
    
    # MCP initialization message
    init_msg = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0.0"}
        }
    })
    
    try:
        # Merge environment variables
        import os
        full_env = os.environ.copy()
        full_env.update(env)
        
        # Run with timeout
        start = time.time()
        result = subprocess.run(
            full_cmd,
            input=init_msg,
            capture_output=True,
            text=True,
            timeout=10,
            env=full_env
        )
        elapsed = time.time() - start
        
        # Check response
        if result.stdout and '"result"' in result.stdout:
            try:
                response = json.loads(result.stdout.strip().split('\n')[0])
                server_name = response.get("result", {}).get("serverInfo", {}).get("name", "unknown")
                return True, f"✓ Running ({server_name}) - {elapsed:.1f}s"
            except:
                return True, f"✓ Running - {elapsed:.1f}s"
        else:
            stderr = result.stderr.strip()[:100] if result.stderr else "No error output"
            return False, f"✗ Failed: {stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "✗ Timeout (10s)"
    except FileNotFoundError:
        return False, f"✗ Command not found: {cmd}"
    except Exception as e:
        return False, f"✗ Error: {str(e)[:50]}"

def main():
    # Load configuration
    with open("mcp-servers.json") as f:
        config = json.load(f)
    
    servers = config["mcpServers"]
    total = len(servers)
    
    print(f"Testing {total} MCP servers...\n")
    
    # Test servers in parallel for speed
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(test_server, name, server_config): name 
            for name, server_config in servers.items()
        }
        
        for future in as_completed(futures):
            name = futures[future]
            success, message = future.result()
            results[name] = (success, message)
    
    # Display results by category
    categories = {
        "File & Code": ["filesystem", "github", "ripgrep", "dependency-graph"],
        "Web & Browser": ["brave", "puppeteer"],
        "Development": ["memory", "sequential-thinking", "pyrepl"],
        "Data & Analytics": ["statsource", "duckdb", "sklearn", "optionsflow"],
        "Trading": ["python_analysis"],
        "Observability": ["mlflow", "logfire", "trace"]
    }
    
    working = 0
    for category, server_list in categories.items():
        print(f"\n{category}:")
        for server in server_list:
            if server in results:
                success, message = results[server]
                print(f"  {server:20} {message}")
                if success:
                    working += 1
    
    print(f"\n{'='*50}")
    print(f"Summary: {working}/{total} servers working")
    
    if working < total:
        print("\nTroubleshooting failed servers:")
        print("1. Check if required commands are installed")
        print("2. Verify environment variables are set")
        print("3. Ensure file paths exist")
        print("4. Run with individual server for detailed errors")

if __name__ == "__main__":
    main()

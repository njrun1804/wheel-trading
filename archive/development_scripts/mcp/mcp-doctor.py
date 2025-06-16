#!/usr/bin/env python3
"""Simple MCP server diagnostic tool"""

import json
import os
import shutil
import subprocess


def check_server(name, config):
    """Test if a server can start and respond to init"""
    cmd = config["command"]
    args = config.get("args", [])

    # Check if command exists
    if cmd == "npx":
        # NPX packages download on demand, assume OK
        return True, "NPX package (downloads on first use)"

    if not shutil.which(cmd):
        return False, f"Command not found: {cmd}"

    # For Python scripts, check if file exists
    if cmd.endswith("python3") and args:
        script = args[0]
        if not os.path.exists(script):
            return False, f"Script not found: {script}"

    # Try to run with init message
    full_cmd = [cmd] + args
    init_msg = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"doctor","version":"1.0.0"}}}'

    try:
        result = subprocess.run(
            full_cmd, input=init_msg, capture_output=True, text=True, timeout=5
        )

        if '"result"' in result.stdout:
            return True, "Server responds correctly"
        else:
            return False, f"Invalid response: {result.stderr[:100]}"

    except subprocess.TimeoutExpired:
        return False, "Server timeout (5s)"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


def main():
    """Check all configured MCP servers"""
    print("MCP Server Health Check\n")

    # Load configuration
    with open("mcp-servers-minimal.json") as f:
        config = json.load(f)

    servers = config["mcpServers"]
    working = 0
    total = len(servers)

    for name, server_config in servers.items():
        ok, message = check_server(name, server_config)
        status = "✓" if ok else "✗"
        color = "\033[0;32m" if ok else "\033[0;31m"
        nc = "\033[0m"

        print(f"{color}{status}{nc} {name}: {message}")
        if ok:
            working += 1

    print(f"\nSummary: {working}/{total} servers ready")

    if working < total:
        print("\nTroubleshooting:")
        print("1. Install missing commands (npm install -g ...)")
        print("2. Check file paths in configuration")
        print("3. Set required environment variables")
        print("4. Run with MCP_DEBUG=1 for more info")


if __name__ == "__main__":
    main()

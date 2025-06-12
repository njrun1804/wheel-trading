#!/usr/bin/env python3
"""Manage MCP server configuration"""

import json
import sys
import argparse

def load_config(file="mcp-servers-enhanced.json"):
    with open(file) as f:
        return json.load(f)

def save_config(config, file="mcp-servers-enhanced.json"):
    with open(file, 'w') as f:
        json.dump(config, f, indent=2)
        f.write('\n')

def list_servers(config):
    print("Active servers:")
    for name, server in config.get("mcpServers", {}).items():
        if name != "comment":
            purpose = server.get("_purpose", "No description")
            print(f"  ✓ {name}: {purpose}")
    
    print("\nOptional servers:")
    for name, server in config.get("optionalServers", {}).items():
        if name != "comment":
            purpose = server.get("_purpose", "No description")
            print(f"  - {name}: {purpose}")

def enable_server(config, server_name):
    optional = config.get("optionalServers", {})
    if server_name in optional:
        server = optional[server_name].copy()
        # Remove metadata
        server.pop("_enable", None)
        
        # Move to active
        config["mcpServers"][server_name] = server
        del optional[server_name]
        
        print(f"✓ Enabled {server_name}")
        return True
    else:
        print(f"✗ Server '{server_name}' not found in optional servers")
        return False

def disable_server(config, server_name):
    active = config.get("mcpServers", {})
    if server_name in active and server_name not in ["filesystem", "github", "python_analysis"]:
        server = active[server_name].copy()
        
        # Move to optional
        config.setdefault("optionalServers", {})[server_name] = server
        del active[server_name]
        
        print(f"✓ Disabled {server_name}")
        return True
    else:
        print(f"✗ Cannot disable core server or server not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Manage MCP servers")
    parser.add_argument("action", choices=["list", "enable", "disable"], help="Action to perform")
    parser.add_argument("server", nargs="?", help="Server name for enable/disable")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.action == "list":
        list_servers(config)
    elif args.action == "enable" and args.server:
        if enable_server(config, args.server):
            save_config(config)
    elif args.action == "disable" and args.server:
        if disable_server(config, args.server):
            save_config(config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

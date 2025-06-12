#!/usr/bin/env python3
"""Monitor health of all MCP servers and provide auto-cleanup."""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import subprocess
from typing import Dict, List, Optional

def check_server_health(health_file: Path) -> Dict:
    """Check health status of a single server."""
    try:
        with open(health_file) as f:
            data = json.load(f)
            
        # Check if process is still alive
        pid = data.get("pid")
        if pid:
            try:
                process = psutil.Process(pid)
                data["process_alive"] = True
                data["process_status"] = process.status()
            except psutil.NoSuchProcess:
                data["process_alive"] = False
                data["process_status"] = "dead"
        else:
            data["process_alive"] = False
            data["process_status"] = "no_pid"
            
        # Check age
        timestamp = datetime.fromisoformat(data.get("timestamp", ""))
        age_seconds = (datetime.now() - timestamp).total_seconds()
        data["age_seconds"] = age_seconds
        data["stale"] = age_seconds > 300  # 5 minutes
        
        return data
        
    except Exception as e:
        return {
            "error": str(e),
            "file": str(health_file),
            "process_alive": False,
            "stale": True
        }

def send_health_check(server_name: str, mcp_config: Dict) -> Optional[Dict]:
    """Send healthz request to an MCP server."""
    try:
        # Prepare healthz tool call
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "healthz",
                "arguments": {}
            }
        }
        
        # Get server command from config
        server_config = mcp_config.get("mcpServers", {}).get(server_name)
        if not server_config:
            return None
            
        cmd = [server_config["command"]] + server_config.get("args", [])
        
        # Send request
        result = subprocess.run(
            cmd,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0 and result.stdout:
            response = json.loads(result.stdout)
            if "result" in response:
                return response["result"]
                
    except Exception:
        pass
        
    return None

def cleanup_stale_servers(runtime_dir: Path, dry_run: bool = True):
    """Clean up health files from dead servers."""
    cleaned = []
    
    for health_file in runtime_dir.glob("*.health"):
        health = check_server_health(health_file)
        
        should_clean = False
        reason = ""
        
        if not health.get("process_alive", False):
            should_clean = True
            reason = "process dead"
        elif health.get("stale", False):
            should_clean = True
            reason = "stale (>5 min)"
        elif health.get("error"):
            should_clean = True
            reason = f"error: {health['error']}"
            
        if should_clean:
            if dry_run:
                print(f"Would clean {health_file.name}: {reason}")
            else:
                health_file.unlink()
                print(f"Cleaned {health_file.name}: {reason}")
            cleaned.append(health_file.name)
            
    return cleaned

def monitor_servers(workspace_root: str, watch: bool = False, interval: int = 5):
    """Monitor all MCP servers."""
    runtime_dir = Path(workspace_root) / ".claude" / "runtime"
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"MCP Server Health Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        if not runtime_dir.exists():
            print("No runtime directory found - no servers running")
            if not watch:
                break
            time.sleep(interval)
            continue
            
        servers = []
        total_memory = 0
        total_requests = 0
        total_errors = 0
        
        for health_file in sorted(runtime_dir.glob("*.health")):
            health = check_server_health(health_file)
            servers.append(health)
            
            if health.get("process_alive"):
                total_memory += health.get("memory_mb", 0)
                total_requests += health.get("request_count", 0)
                total_errors += health.get("error_count", 0)
                
        # Summary
        alive_count = sum(1 for s in servers if s.get("process_alive"))
        print(f"\nServers: {alive_count}/{len(servers)} alive")
        print(f"Total Memory: {total_memory:.1f} MB")
        print(f"Total Requests: {total_requests:,} (Errors: {total_errors})")
        print()
        
        # Server details
        print("Server Details:")
        print("-" * 80)
        print(f"{'Server':<20} {'Status':<10} {'Uptime':<12} {'Requests':<10} {'Errors':<8} {'Memory':<10}")
        print("-" * 80)
        
        for server in servers:
            if "error" in server:
                print(f"{server.get('file', 'unknown'):<20} {'ERROR':<10} {'-':<12} {'-':<10} {'-':<8} {'-':<10}")
                continue
                
            name = server.get("server_name", "unknown")[:20]
            status = "ALIVE" if server.get("process_alive") else "DEAD"
            if server.get("stale"):
                status = "STALE"
            elif server.get("status") == "degraded":
                status = "DEGRADED"
                
            uptime = server.get("uptime_human", "-")[:12]
            requests = str(server.get("request_count", 0))
            errors = str(server.get("error_count", 0))
            memory = f"{server.get('memory_mb', 0):.1f} MB"
            
            # Color coding (basic ANSI)
            if status == "ALIVE" and server.get("status") == "healthy":
                status_str = f"\033[92m{status}\033[0m"  # Green
            elif status == "DEGRADED":
                status_str = f"\033[93m{status}\033[0m"  # Yellow
            else:
                status_str = f"\033[91m{status}\033[0m"  # Red
                
            print(f"{name:<20} {status_str:<19} {uptime:<12} {requests:<10} {errors:<8} {memory:<10}")
            
        # Show recent errors
        print("\nRecent Errors:")
        print("-" * 80)
        error_found = False
        for server in servers:
            if server.get("last_error") and server.get("process_alive"):
                print(f"{server.get('server_name', 'unknown')}: {server['last_error'][:60]}...")
                error_found = True
                
        if not error_found:
            print("No recent errors")
            
        if not watch:
            break
            
        print(f"\nRefreshing in {interval} seconds... (Ctrl+C to quit)")
        time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Monitor MCP server health")
    parser.add_argument(
        "--workspace",
        default=os.environ.get("WORKSPACE_ROOT", os.getcwd()),
        help="Workspace root directory"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor servers"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up stale health files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without doing it"
    )
    
    args = parser.parse_args()
    
    if args.cleanup:
        runtime_dir = Path(args.workspace) / ".claude" / "runtime"
        if runtime_dir.exists():
            cleaned = cleanup_stale_servers(runtime_dir, args.dry_run)
            if cleaned:
                print(f"Cleaned {len(cleaned)} stale health files")
            else:
                print("No stale health files found")
        else:
            print("No runtime directory found")
    else:
        monitor_servers(args.workspace, args.watch, args.interval)

if __name__ == "__main__":
    main()
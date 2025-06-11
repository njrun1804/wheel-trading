#!/usr/bin/env python3
"""
Real-time monitoring dashboard for wheel trading system
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime


def clear_screen():
    os.system("clear")


def get_system_stats():
    """Get system performance stats."""
    try:
        import psutil

        return {
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory().percent,
            "disk_free": psutil.disk_usage("/").free / 1e9,
        }
    except ImportError:
        return {"error": "psutil not installed"}


def check_mcp_servers():
    """Check which MCP servers are running."""
    servers = {
        "filesystem": False,
        "github": False,
        "web": False,
        "wiki": False,
        "python_analysis": False,
    }

    try:
        result = subprocess.run(
            ["pgrep", "-af", "modelcontextprotocol"], capture_output=True, text=True
        )
        if result.stdout:
            for line in result.stdout.split("\n"):
                if "filesystem" in line:
                    servers["filesystem"] = True
                elif "github" in line:
                    servers["github"] = True
                elif "web-search" in line:
                    servers["web"] = True
                elif "wikipedia" in line:
                    servers["wiki"] = True

        # Check python analysis server
        result = subprocess.run(
            ["pgrep", "-af", "python-mcp-server"], capture_output=True, text=True
        )
        if result.stdout:
            servers["python_analysis"] = True

    except Exception:
        pass

    return servers


def check_data_freshness():
    """Check how fresh our data is."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return {"error": "No data directory"}

    files = []
    for file in os.listdir(data_dir):
        if file.endswith(".db") or file.endswith(".parquet"):
            path = os.path.join(data_dir, file)
            mtime = os.path.getmtime(path)
            age_hours = (time.time() - mtime) / 3600
            files.append(
                {"name": file, "age_hours": age_hours, "size_mb": os.path.getsize(path) / 1e6}
            )

    return sorted(files, key=lambda x: x["age_hours"])


def get_recent_logs():
    """Get recent log entries."""
    try:
        result = subprocess.run(
            ["tail", "-5", "wheel_recommendations.log"], capture_output=True, text=True
        )
        return result.stdout.split("\n")[-3:] if result.stdout else []
    except:
        return ["No logs found"]


def main():
    """Main monitoring loop."""
    print("Starting real-time monitor... Press Ctrl+C to exit")

    try:
        while True:
            clear_screen()

            print("🎯 WHEEL TRADING SYSTEM MONITOR")
            print("=" * 50)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

            # System stats
            stats = get_system_stats()
            if "error" not in stats:
                print(
                    f"💻 System: CPU {stats['cpu']:5.1f}% | RAM {stats['memory']:5.1f}% | Disk {stats['disk_free']:5.1f}GB"
                )

            # MCP Servers
            servers = check_mcp_servers()
            print("\n🔧 MCP Servers:")
            for name, running in servers.items():
                status = "✅ Running" if running else "❌ Stopped"
                print(f"   {name:15} {status}")

            # Data freshness
            data = check_data_freshness()
            print("\n📊 Data Status:")
            if "error" not in data:
                for file in data[:3]:  # Show top 3 most recent
                    age = f"{file['age_hours']:.1f}h ago"
                    size = f"{file['size_mb']:.1f}MB"
                    print(f"   {file['name']:20} {age:>10} ({size})")

            # Recent activity
            logs = get_recent_logs()
            print("\n📝 Recent Activity:")
            for log in logs:
                if log.strip():
                    print(f"   {log[:60]}...")

            print("\n" + "=" * 50)
            print("Commands: 'make analyze' | 'wt-run' | 'wt-check'")

            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    main()

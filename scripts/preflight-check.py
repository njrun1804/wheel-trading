#!/usr/bin/env python3
"""Pre-flight checks for Wheel Trading startup."""

import json
import sys
from pathlib import Path


def check_environment():
    """Run pre-flight checks and return status."""
    checks_passed = True

    # Check Python version

    # Check critical imports
    try:
        import duckdb
        import pandas

        import unity_wheel
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        checks_passed = False

    # Check database
    db_found = False
    for db_path in [
        "data/wheel_trading_optimized.duckdb",
        "data/wheel_trading_master.duckdb",
    ]:
        if Path(db_path).exists():
            db_found = True
            break

    if not db_found:
        print("⚠️  No database found")

    # Check config
    if not Path("config.yaml").exists():
        print("⚠️  config.yaml not found")

    # Check .env
    if not Path(".env").exists():
        print("⚠️  .env file not found")

    # Check MCP servers
    if Path("mcp-servers.json").exists():
        try:
            with open("mcp-servers.json") as f:
                servers = json.load(f)
                if "mcpServers" in servers:
                    active = len(
                        [
                            s
                            for s in servers["mcpServers"].values()
                            if s.get("autostart")
                        ]
                    )
                    print(f"✅ {active} MCP servers configured for autostart")
        except:
            pass

    return checks_passed


if __name__ == "__main__":
    if check_environment():
        sys.exit(0)
    else:
        sys.exit(1)

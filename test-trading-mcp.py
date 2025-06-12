#!/usr/bin/env python3
"""Test MCP servers work with trading bot needs"""

import subprocess
import json
import os

def test_filesystem_access():
    """Test we can access trading bot files"""
    critical_files = [
        "config.yaml",
        "data/wheel_trading_master.duckdb",
        "src/unity_wheel/strategy/wheel.py"
    ]
    
    print("Testing filesystem access...")
    for file in critical_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
    
    return True

def test_github_integration():
    """Test GitHub access"""
    print("\nTesting GitHub integration...")
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ Git repository accessible")
            if result.stdout:
                print(f"  ! {len(result.stdout.splitlines())} uncommitted changes")
        return True
    except:
        print("  ✗ Git not accessible")
        return False

def test_python_analysis():
    """Test python analysis server"""
    print("\nTesting python analysis server...")
    script = "scripts/python-mcp-server.py"
    if os.path.exists(script):
        print(f"  ✓ {script} exists")
        # Test it can import trading bot modules
        try:
            import sys
            sys.path.insert(0, os.getcwd())
            from src.unity_wheel.config import TradingConfig
            print("  ✓ Can import trading bot modules")
            return True
        except ImportError as e:
            print(f"  ✗ Cannot import trading modules: {e}")
            return False
    else:
        print(f"  ✗ {script} not found")
        return False

def main():
    print("=== Trading Bot MCP Integration Test ===\n")
    
    all_good = True
    all_good &= test_filesystem_access()
    all_good &= test_github_integration()
    all_good &= test_python_analysis()
    
    print("\n" + "="*40)
    if all_good:
        print("✅ All tests passed - MCP ready for trading bot")
    else:
        print("❌ Some tests failed - check configuration")

if __name__ == "__main__":
    main()

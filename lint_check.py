#!/usr/bin/env python3
"""
Quick linting check script to avoid shell file handle issues.
"""
import subprocess
import sys
import os

def run_ruff_check():
    """Run ruff check and report statistics."""
    try:
        # Change to project directory
        os.chdir("/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading")
        
        # Run ruff check with statistics
        result = subprocess.run(
            ["ruff", "check", "--statistics"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print("RUFF CHECK RESULTS:")
        print("=" * 50)
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Ruff check timed out")
        return False
    except Exception as e:
        print(f"Error running ruff check: {e}")
        return False

if __name__ == "__main__":
    run_ruff_check()
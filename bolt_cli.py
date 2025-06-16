#!/usr/bin/env python3
"""
Bolt CLI Compatibility Wrapper
==============================
This is a compatibility wrapper for the legacy bolt_cli.py interface.
All functionality has been moved to the unified './bob solve' command.

Usage: python bolt_cli.py <query>
Recommended: ./bob solve <query>
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("⚠️  DEPRECATION WARNING: bolt_cli.py is deprecated")
    print("   Use './bob solve' instead of 'python bolt_cli.py'")
    print("   Migration guide: ./bob help")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bob_executable = script_dir / "bob"
    
    if not bob_executable.exists():
        print("❌ Error: Unified BOB executable not found")
        print("   Run: ./setup_bob_symlinks.sh")
        sys.exit(1)
    
    # Forward all arguments to unified BOB solve command
    args = ["solve"] + sys.argv[1:]  # Add 'solve' prefix
    
    try:
        # Execute unified BOB with solve command
        result = subprocess.run([str(bob_executable)] + args, 
                              capture_output=False,
                              text=True)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Error executing unified BOB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

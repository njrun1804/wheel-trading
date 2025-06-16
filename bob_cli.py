#!/usr/bin/env python3
"""
BOB CLI Compatibility Wrapper
============================
This is a compatibility wrapper for the legacy bob_cli.py interface.
All functionality has been moved to the unified './bob' command.

Usage: python bob_cli.py <command>
Recommended: ./bob <command>
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("⚠️  DEPRECATION WARNING: bob_cli.py is deprecated")
    print("   Use './bob' instead of 'python bob_cli.py'")
    print("   Migration guide: ./bob help")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bob_executable = script_dir / "bob"
    
    if not bob_executable.exists():
        print("❌ Error: Unified BOB executable not found")
        print("   Run: ./setup_bob_symlinks.sh")
        sys.exit(1)
    
    # Forward all arguments to unified BOB
    args = sys.argv[1:]  # Skip script name
    
    try:
        # Execute unified BOB with the same arguments
        result = subprocess.run([str(bob_executable)] + args, 
                              capture_output=False,
                              text=True)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Error executing unified BOB: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

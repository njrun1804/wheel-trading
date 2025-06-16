#!/usr/bin/env python3
"""
Automated linting fix that uses subprocess to apply ruff fixes.
This approach should work around the file handle issue.
"""
import subprocess
import sys
import os
import time

def run_command_safe(cmd, description, cwd, timeout=60):
    """Run a command safely with timeout."""
    print(f"\n{description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.stdout.strip():
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr.strip():
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error: {e}")
        return False, "", str(e)

def main():
    """Run automated linting fixes."""
    cwd = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
    
    print("Starting automated linting fixes...")
    print("=" * 60)
    
    # Step 1: Apply safe automated fixes
    print("\nSTEP 1: Applying safe automated fixes...")
    success, stdout, stderr = run_command_safe(
        ["ruff", "check", "--fix", "src/"],
        "Apply safe automated fixes to src/",
        cwd,
        timeout=120
    )
    
    if success:
        print("✓ Safe fixes applied successfully")
    else:
        print("⚠ Some safe fixes may have failed")
    
    # Step 2: Apply format fixes
    print("\nSTEP 2: Applying format fixes...")
    success, stdout, stderr = run_command_safe(
        ["ruff", "format", "src/"],
        "Format code in src/",
        cwd,
        timeout=60
    )
    
    if success:
        print("✓ Code formatted successfully")
    else:
        print("⚠ Code formatting may have failed")
    
    # Step 3: Check current status
    print("\nSTEP 3: Checking current status...")
    success, stdout, stderr = run_command_safe(
        ["ruff", "check", "--statistics", "src/"],
        "Check linting status",
        cwd,
        timeout=60
    )
    
    # Step 4: Try unsafe fixes if needed
    print("\nSTEP 4: Applying unsafe fixes...")
    success, stdout, stderr = run_command_safe(
        ["ruff", "check", "--fix", "--unsafe-fixes", "src/"],
        "Apply unsafe fixes to src/",
        cwd,
        timeout=120
    )
    
    if success:
        print("✓ Unsafe fixes applied successfully")
    else:
        print("⚠ Some unsafe fixes may have failed")
    
    # Final status check
    print("\nFINAL STATUS CHECK:")
    success, stdout, stderr = run_command_safe(
        ["ruff", "check", "--statistics", "src/"],
        "Final linting status",
        cwd,
        timeout=60
    )
    
    print("\n" + "=" * 60)
    print("AUTOMATED LINTING COMPLETE")
    print("=" * 60)
    
    if stdout and "violations" in stdout.lower():
        print("Check the statistics above for remaining issues.")
    
    print("\nNext steps:")
    print("1. Review any remaining F401 (unused imports)")
    print("2. Review any remaining F841 (unused variables)")
    print("3. Add # noqa comments where needed")
    print("4. Update pyproject.toml per-file-ignores if needed")

if __name__ == "__main__":
    main()
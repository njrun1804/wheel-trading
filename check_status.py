#!/usr/bin/env python3
"""Quick status check for linting issues."""
import subprocess
import sys

def run_ruff_check():
    """Run ruff check with limited output."""
    try:
        # Check just a few key files first
        files_to_check = [
            "src/unity_wheel/api/advisor.py",
            "src/unity_wheel/cli/run.py",
            "src/unity_wheel/strategy/wheel.py",
            "src/unity_wheel/utils/__init__.py",
            "src/unity_wheel/utils/position_sizing.py",
        ]
        
        result = subprocess.run(
            ["ruff", "check"] + files_to_check,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print("RUFF CHECK RESULTS (sample files):")
        print("=" * 50)
        if result.stdout:
            lines = result.stdout.split('\n')
            error_count = 0
            for line in lines:
                if line.strip():
                    print(line)
                    error_count += 1
            print(f"\nTotal errors in sample: {error_count}")
        else:
            print("No errors found in sample files!")
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        
        return error_count
        
    except subprocess.TimeoutExpired:
        print("Ruff check timed out")
        return -1
    except Exception as e:
        print(f"Error running ruff check: {e}")
        return -1

if __name__ == "__main__":
    run_ruff_check()
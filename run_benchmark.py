#!/usr/bin/env python3
"""
Simple script to run the performance benchmark without shell dependencies
"""

import subprocess
import sys
import os

def run_benchmark():
    """Run the benchmark script directly"""
    try:
        # Change to the correct directory
        script_dir = "/Users/mikeedwards/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading"
        os.chdir(script_dir)
        
        # Run the benchmark
        result = subprocess.run([
            sys.executable, 
            "emergency_performance_benchmark.py"
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
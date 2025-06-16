#!/usr/bin/env python3
"""Emergency cleanup for meta processes and PID files."""

import os
import signal
import subprocess
import time
import sys

def main():
    print("=== Emergency Meta Process Cleanup ===")
    
    # Target PIDs from our investigation
    pids = [39124, 84182, 77238, 77250]
    
    # Check and kill PIDs
    for pid in pids:
        try:
            # Check if process exists
            os.kill(pid, 0)
            print(f"PID {pid} is running - attempting to terminate")
            
            # Try graceful shutdown
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid}")
            except ProcessLookupError:
                print(f"PID {pid} already terminated")
                continue
            except PermissionError:
                print(f"Permission denied for PID {pid}")
                continue
                
        except ProcessLookupError:
            print(f"PID {pid} not running")
            continue
        except PermissionError:
            print(f"Permission denied checking PID {pid}")
            continue
    
    # Wait a moment for graceful shutdown
    time.sleep(2)
    
    # Force kill any remaining
    for pid in pids:
        try:
            os.kill(pid, 0)  # Check if still alive
            print(f"Force killing PID {pid}")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Already dead
        except PermissionError:
            print(f"Permission denied force killing PID {pid}")
    
    # Kill by pattern using subprocess
    patterns = ["meta.*py", "meta_daemon", "meta_system", "memory_monitor"]
    for pattern in patterns:
        try:
            result = subprocess.run(["pkill", "-f", pattern], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Killed processes matching pattern: {pattern}")
        except Exception as e:
            print(f"Error killing pattern {pattern}: {e}")
    
    # Remove PID files
    pid_files = [
        "meta_system.pid",
        "meta_daemon.pid", 
        "pids/memory_monitor.pid",
        "pids/process_manager.pid"
    ]
    
    for pid_file in pid_files:
        try:
            if os.path.exists(pid_file):
                os.remove(pid_file)
                print(f"Removed PID file: {pid_file}")
        except Exception as e:
            print(f"Error removing {pid_file}: {e}")
    
    # Final verification - check if any meta processes remain
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        if result.returncode == 0:
            meta_processes = [line for line in result.stdout.split('\n') 
                            if 'meta' in line.lower() and 'grep' not in line]
            if meta_processes:
                print("\n⚠️  WARNING: Some meta processes may still be running:")
                for proc in meta_processes:
                    print(f"  {proc}")
            else:
                print("\n✅ No meta processes found - cleanup successful")
    except Exception as e:
        print(f"Error checking remaining processes: {e}")
    
    print("=== Cleanup Complete ===")

if __name__ == "__main__":
    main()
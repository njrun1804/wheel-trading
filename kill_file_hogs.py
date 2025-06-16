#!/usr/bin/env python3
import os
import signal
import sys

# PIDs from PID files that are likely consuming file handles
pids = [39124, 84182, 77238, 77250]

print("=== EMERGENCY FILE HANDLE CLEANUP ===")
print("Killing processes that may be consuming excessive file handles...")

for pid in pids:
    try:
        # Check if process exists
        os.kill(pid, 0)
        print(f"Killing PID {pid}...")
        # Try graceful termination first
        os.kill(pid, signal.SIGTERM)
    except OSError:
        print(f"PID {pid} not running")
    except Exception as e:
        print(f"Error with PID {pid}: {e}")

print("\nWaiting 2 seconds for graceful shutdown...")
import time
time.sleep(2)

# Force kill any remaining
for pid in pids:
    try:
        os.kill(pid, 0)  # Check if still running
        print(f"Force killing PID {pid}...")
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass  # Already dead
    except Exception as e:
        print(f"Force kill error PID {pid}: {e}")

# Clean up PID files
pid_files = [
    "meta_daemon.pid",
    "meta_system.pid", 
    "pids/memory_monitor.pid",
    "pids/process_manager.pid"
]

print("\nCleaning up PID files...")
for pid_file in pid_files:
    try:
        os.remove(pid_file)
        print(f"Removed {pid_file}")
    except:
        pass

print("\n=== CLEANUP COMPLETE ===")
print("Try running a simple command now like: echo 'test'")
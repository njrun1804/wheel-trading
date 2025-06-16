#!/usr/bin/env python3
"""
Process Investigation Tool
Simple tool to investigate what processes might be consuming file descriptors
"""

import os
import subprocess
import sys
from pathlib import Path


def check_pid_files():
    """Check all PID files and their associated processes"""
    print("🔍 Checking PID files...")
    
    pid_files = list(Path(".").glob("*.pid"))
    
    if not pid_files:
        print("  No PID files found")
        return
    
    for pid_file in pid_files:
        try:
            pid_content = pid_file.read_text().strip()
            print(f"\n📄 {pid_file.name}:")
            print(f"  Content: {pid_content}")
            
            try:
                pid = int(pid_content)
                # Check if process exists
                try:
                    os.kill(pid, 0)  # Just check, don't actually kill
                    print(f"  Status: ✅ Process {pid} is running")
                    
                    # Try to get process info
                    try:
                        result = subprocess.run(
                            ["ps", "-p", str(pid), "-o", "pid,ppid,command"],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            print(f"  Details:\n{result.stdout}")
                    except:
                        pass
                        
                except OSError:
                    print(f"  Status: ❌ Process {pid} is not running (stale PID file)")
                    
            except ValueError:
                print(f"  Status: ❌ Invalid PID content")
                
        except Exception as e:
            print(f"  Status: ❌ Error reading file: {e}")


def check_database_locks():
    """Check for database lock files"""
    print("\n🔒 Checking database files and locks...")
    
    db_files = list(Path(".").glob("*.db"))
    wal_files = list(Path(".").glob("*.db-wal"))
    shm_files = list(Path(".").glob("*.db-shm"))
    
    print(f"Database files: {len(db_files)}")
    print(f"WAL files: {len(wal_files)}")  
    print(f"SHM files: {len(shm_files)}")
    
    if wal_files:
        print("\n📝 WAL files (may indicate open connections):")
        for wal in wal_files:
            print(f"  {wal.name}")
            
    if shm_files:
        print("\n💾 SHM files (shared memory, may indicate open connections):")
        for shm in shm_files:
            print(f"  {shm.name}")


def check_python_processes():
    """Check for Python processes that might be from this project"""
    print("\n🐍 Checking Python processes...")
    
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            python_lines = [line for line in lines if 'python' in line.lower()]
            
            relevant_processes = []
            for line in python_lines:
                if any(keyword in line for keyword in [
                    'meta_', 'einstein', 'bolt', 'jarvis', 'daemon', 
                    'monitor', 'watcher', 'trading'
                ]):
                    relevant_processes.append(line)
            
            if relevant_processes:
                print("Potentially relevant Python processes:")
                for proc in relevant_processes:
                    print(f"  {proc}")
            else:
                print("No obviously relevant Python processes found")
                
    except Exception as e:
        print(f"Error checking processes: {e}")


def check_file_descriptors():
    """Try to check file descriptor usage"""
    print("\n📊 File descriptor information...")
    
    try:
        # Try to get ulimit info
        result = subprocess.run(
            ["sh", "-c", "ulimit -n"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"File descriptor limit: {result.stdout.strip()}")
    except:
        print("Could not get file descriptor limit")
    
    try:
        # Try to count open files for current process
        result = subprocess.run(
            ["sh", "-c", "lsof -p $$ | wc -l"], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Current process file descriptors: {result.stdout.strip()}")
    except:
        print("Could not count current file descriptors")


def main():
    """Run investigation"""
    print("🔬 PROCESS INVESTIGATION")
    print("=" * 40)
    
    check_pid_files()
    check_database_locks()
    check_python_processes()
    check_file_descriptors()
    
    print("\n💡 If you see many WAL/SHM files or running processes,")
    print("   run: python cleanup_file_descriptor_leak.py")


if __name__ == "__main__":
    main()
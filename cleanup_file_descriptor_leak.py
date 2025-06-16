#!/usr/bin/env python3
"""
Emergency File Descriptor Leak Cleanup
Addresses the critical file descriptor exhaustion by:
1. Stopping all background processes
2. Closing database connections
3. Cleaning up database WAL/SHM files
4. Monitoring file descriptor usage
"""

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Set


class FileDescriptorCleanup:
    """Emergency cleanup for file descriptor leaks"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.stopped_processes: Set[int] = set()
        self.closed_databases: Set[str] = set()
        
    def get_current_fd_usage(self) -> int:
        """Get current file descriptor usage"""
        try:
            result = subprocess.run(
                ["sh", "-c", "lsof -p $$ | wc -l"], 
                capture_output=True, 
                text=True
            )
            return int(result.stdout.strip()) if result.returncode == 0 else -1
        except:
            return -1
    
    def stop_background_processes(self):
        """Stop all background processes that might be holding file descriptors"""
        print("🛑 Stopping background processes...")
        
        # Stop processes from PID files
        pid_files = list(self.base_path.glob("*.pid"))
        for pid_file in pid_files:
            try:
                pid = int(pid_file.read_text().strip())
                if self._is_process_running(pid):
                    print(f"  Stopping process {pid} from {pid_file.name}")
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(1)
                    
                    # Force kill if still running
                    if self._is_process_running(pid):
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(0.5)
                    
                    self.stopped_processes.add(pid)
                
                pid_file.unlink()
                print(f"  ✅ Cleaned up {pid_file.name}")
                
            except (ValueError, ProcessLookupError, FileNotFoundError):
                pid_file.unlink()  # Clean up stale PID file
        
        # Kill processes by name
        process_patterns = [
            "meta_daemon",
            "meta_monitoring", 
            "meta_watcher",
            "einstein",
            "bolt",
            "jarvis2",
            "memory_monitor",
            "system_monitor",
            "gpu_monitor",
            "thermal_monitor"
        ]
        
        for pattern in process_patterns:
            try:
                result = subprocess.run(
                    ["pgrep", "-f", pattern], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    pids = [int(p) for p in result.stdout.strip().split('\n') if p]
                    for pid in pids:
                        if pid not in self.stopped_processes:
                            print(f"  Stopping {pattern} process {pid}")
                            try:
                                os.kill(pid, signal.SIGTERM)
                                time.sleep(0.5)
                                if self._is_process_running(pid):
                                    os.kill(pid, signal.SIGKILL)
                                self.stopped_processes.add(pid)
                            except ProcessLookupError:
                                pass
            except:
                pass
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if process is still running"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def close_database_connections(self):
        """Close all database connections and clean up WAL/SHM files"""
        print("🔒 Cleaning up database connections...")
        
        # Find all database files
        db_files = list(self.base_path.glob("*.db"))
        
        for db_file in db_files:
            print(f"  Processing {db_file.name}")
            
            # Try to connect and close cleanly
            try:
                conn = sqlite3.connect(str(db_file), timeout=1.0)
                # Force WAL checkpoint
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
                self.closed_databases.add(db_file.name)
                print(f"    ✅ Cleanly closed {db_file.name}")
            except Exception as e:
                print(f"    ⚠️ Could not cleanly close {db_file.name}: {e}")
            
            # Clean up WAL and SHM files
            wal_files = list(self.base_path.glob(f"{db_file.stem}*.db-wal"))
            shm_files = list(self.base_path.glob(f"{db_file.stem}*.db-shm"))
            
            for wal_file in wal_files:
                try:
                    wal_file.unlink()
                    print(f"    ✅ Removed {wal_file.name}")
                except:
                    pass
                    
            for shm_file in shm_files:
                try:
                    shm_file.unlink() 
                    print(f"    ✅ Removed {shm_file.name}")
                except:
                    pass
    
    def cleanup_temp_files(self):
        """Clean up temporary files that might hold file descriptors"""
        print("🧹 Cleaning up temporary files...")
        
        temp_patterns = [
            "*.tmp",
            "*.temp", 
            "*~",
            ".#*",
            "*.lock",
            "*.socket"
        ]
        
        for pattern in temp_patterns:
            temp_files = list(self.base_path.glob(pattern))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    print(f"  ✅ Removed {temp_file.name}")
                except:
                    pass
    
    def increase_fd_limit(self):
        """Attempt to increase file descriptor limit"""
        print("📈 Attempting to increase file descriptor limit...")
        
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"  Current limits: soft={soft}, hard={hard}")
            
            # Try to increase soft limit to hard limit
            if soft < hard:
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
                new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                print(f"  ✅ Increased to: soft={new_soft}, hard={new_hard}")
            else:
                print("  ℹ️ Already at maximum")
                
        except Exception as e:
            print(f"  ❌ Could not increase limits: {e}")
    
    def monitor_fd_usage(self, duration: int = 5):
        """Monitor file descriptor usage for a short period"""
        print(f"📊 Monitoring file descriptor usage for {duration}s...")
        
        for i in range(duration):
            fd_count = self.get_current_fd_usage()
            if fd_count > 0:
                print(f"  File descriptors in use: {fd_count}")
            time.sleep(1)
    
    def emergency_cleanup(self):
        """Perform emergency cleanup of file descriptors"""
        print("🚨 EMERGENCY FILE DESCRIPTOR CLEANUP")
        print("=" * 50)
        
        # Check initial state
        initial_fd = self.get_current_fd_usage()
        print(f"Initial file descriptor usage: {initial_fd}")
        
        # Step 1: Increase limits if possible
        self.increase_fd_limit()
        
        # Step 2: Stop background processes
        self.stop_background_processes()
        
        # Step 3: Close database connections
        self.close_database_connections()
        
        # Step 4: Clean temp files
        self.cleanup_temp_files()
        
        # Step 5: Force garbage collection
        import gc
        gc.collect()
        
        # Check final state
        time.sleep(2)
        final_fd = self.get_current_fd_usage()
        print(f"\nFinal file descriptor usage: {final_fd}")
        
        if final_fd > 0 and initial_fd > 0:
            reduction = initial_fd - final_fd
            print(f"Reduction: {reduction} file descriptors")
        
        print("\n📋 Cleanup Summary:")
        print(f"  Stopped processes: {len(self.stopped_processes)}")
        print(f"  Closed databases: {len(self.closed_databases)}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        print("  1. Check for processes that restart automatically")
        print("  2. Review database connection pooling")
        print("  3. Add proper cleanup in process shutdown")
        print("  4. Monitor file descriptor usage regularly")
        
        return final_fd < 8000  # Reasonable threshold


def main():
    """Run emergency cleanup"""
    cleanup = FileDescriptorCleanup()
    
    try:
        success = cleanup.emergency_cleanup()
        
        if success:
            print("\n✅ Emergency cleanup completed successfully")
            print("You should now be able to run basic commands")
        else:
            print("\n⚠️ Emergency cleanup completed but file descriptor usage still high")
            print("May need system restart or manual intervention")
            
    except Exception as e:
        print(f"\n❌ Emergency cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
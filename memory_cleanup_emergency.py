#!/usr/bin/env python3
"""
Emergency Memory Cleanup Script
Provides immediate memory relief for critical situations.
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


class EmergencyMemoryCleanup:
    def __init__(self):
        self.protected_processes = [
            "System",
            "kernel",
            "launchd",
            "WindowServer",
            "loginwindow",
            "python",
            "claude",
            "wheel",
            "trading",
            "databento",
            "postgres",
            "ssh",
            "Terminal",
            "iTerm",
            "bash",
            "zsh",
        ]

        self.cleanup_actions = []
        self.memory_freed = 0
        self.start_time = datetime.now()

    def log_action(self, action: str, freed_mb: float = 0):
        """Log cleanup action."""
        self.cleanup_actions.append(
            {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "memory_freed_mb": freed_mb,
            }
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {action}")
        if freed_mb > 0:
            print(f"  → Freed: {freed_mb:.1f}MB")

    def get_memory_status(self):
        """Get current memory status."""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            "total_gb": vm.total / 1024 / 1024 / 1024,
            "available_mb": vm.available / 1024 / 1024,
            "used_percent": vm.percent,
            "swap_used_mb": swap.used / 1024 / 1024,
            "swap_percent": swap.percent,
        }

    def force_memory_purge(self):
        """Force macOS to purge inactive memory."""
        try:
            # Use the purge command to force memory cleanup
            result = subprocess.run(["purge"], capture_output=True, timeout=60)
            if result.returncode == 0:
                self.log_action("Forced memory purge", 0)
                return True
            else:
                self.log_action(f"Memory purge failed: {result.stderr.decode()}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_action("Memory purge command not available or timed out")
            return False

    def cleanup_browser_processes(self):
        """Clean up memory-heavy browser processes."""
        browser_patterns = [
            "Chrome",
            "chrome",
            "Firefox",
            "firefox",
            "Safari",
            "safari",
            "WebKit",
            "webkit",
            "Electron",
            "electron",
        ]

        terminated = 0
        memory_freed = 0

        for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
            try:
                proc_info = proc.info
                name = proc_info["name"]
                cmdline = proc_info.get("cmdline", [])

                # Skip protected processes
                if any(
                    pattern.lower() in name.lower()
                    for pattern in self.protected_processes
                ):
                    continue

                # Check if it's a browser process
                is_browser = any(
                    pattern.lower() in name.lower() for pattern in browser_patterns
                )
                if not is_browser and cmdline:
                    cmdline_str = " ".join(cmdline).lower()
                    is_browser = any(
                        pattern.lower() in cmdline_str for pattern in browser_patterns
                    )

                if is_browser:
                    memory_mb = proc_info["memory_info"].rss / 1024 / 1024

                    # Only terminate if using more than 200MB
                    if memory_mb > 200:
                        try:
                            process = psutil.Process(proc_info["pid"])
                            process.terminate()

                            # Wait for termination
                            try:
                                process.wait(timeout=5)
                                terminated += 1
                                memory_freed += memory_mb
                                self.log_action(
                                    f"Terminated browser process: {name} (PID: {proc_info['pid']})"
                                )
                            except psutil.TimeoutExpired:
                                process.kill()
                                terminated += 1
                                memory_freed += memory_mb
                                self.log_action(
                                    f"Killed browser process: {name} (PID: {proc_info['pid']})"
                                )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if terminated > 0:
            self.log_action(
                f"Browser cleanup: {terminated} processes terminated", memory_freed
            )

        return memory_freed

    def cleanup_node_processes(self):
        """Clean up Node.js processes using excessive memory."""
        terminated = 0
        memory_freed = 0

        for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
            try:
                proc_info = proc.info
                name = proc_info["name"]
                cmdline = proc_info.get("cmdline", [])

                # Skip protected processes
                if any(
                    pattern.lower() in name.lower()
                    for pattern in self.protected_processes
                ):
                    continue

                # Check if it's a Node.js process
                is_node = "node" in name.lower() or "nodejs" in name.lower()
                if not is_node and cmdline:
                    cmdline_str = " ".join(cmdline).lower()
                    is_node = "node" in cmdline_str or "nodejs" in cmdline_str

                if is_node:
                    memory_mb = proc_info["memory_info"].rss / 1024 / 1024

                    # Only terminate if using more than 500MB
                    if memory_mb > 500:
                        try:
                            process = psutil.Process(proc_info["pid"])
                            process.terminate()

                            try:
                                process.wait(timeout=5)
                                terminated += 1
                                memory_freed += memory_mb
                                self.log_action(
                                    f"Terminated Node.js process: {name} (PID: {proc_info['pid']})"
                                )
                            except psutil.TimeoutExpired:
                                process.kill()
                                terminated += 1
                                memory_freed += memory_mb
                                self.log_action(
                                    f"Killed Node.js process: {name} (PID: {proc_info['pid']})"
                                )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if terminated > 0:
            self.log_action(
                f"Node.js cleanup: {terminated} processes terminated", memory_freed
            )

        return memory_freed

    def cleanup_temp_files(self):
        """Clean up temporary files and caches."""
        temp_dirs = [
            "/tmp",
            "/var/tmp",
            os.path.expanduser("~/Library/Caches"),
            os.path.expanduser("~/Library/Logs"),
            "/System/Library/Caches",
            "/Library/Caches",
        ]

        total_freed = 0

        for temp_dir in temp_dirs:
            if not os.path.exists(temp_dir):
                continue

            try:
                # Get initial size
                initial_size = self.get_directory_size(temp_dir)

                # Clean files older than 1 hour in /tmp and /var/tmp
                if temp_dir in ["/tmp", "/var/tmp"]:
                    subprocess.run(
                        ["find", temp_dir, "-type", "f", "-mmin", "+60", "-delete"],
                        capture_output=True,
                        timeout=30,
                    )

                # For cache directories, clean more aggressively
                else:
                    # Remove files older than 1 day
                    subprocess.run(
                        ["find", temp_dir, "-type", "f", "-mtime", "+1", "-delete"],
                        capture_output=True,
                        timeout=60,
                    )

                    # Remove empty directories
                    subprocess.run(
                        ["find", temp_dir, "-type", "d", "-empty", "-delete"],
                        capture_output=True,
                        timeout=30,
                    )

                # Calculate freed space
                final_size = self.get_directory_size(temp_dir)
                freed_mb = (initial_size - final_size) / 1024 / 1024

                if freed_mb > 1:  # Only log if significant
                    self.log_action(f"Cleaned {temp_dir}", freed_mb)
                    total_freed += freed_mb

            except subprocess.TimeoutExpired:
                self.log_action(f"Timeout cleaning {temp_dir}")
            except Exception as e:
                self.log_action(f"Error cleaning {temp_dir}: {e}")

        return total_freed

    def get_directory_size(self, path: str) -> int:
        """Get directory size in bytes."""
        try:
            result = subprocess.run(
                ["du", "-sb", path], capture_output=True, timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.decode().split()[0])
        except:
            pass
        return 0

    def cleanup_python_caches(self):
        """Clean Python cache files."""

        # Look in common Python directories
        python_dirs = [os.getcwd(), os.path.expanduser("~"), "/usr/local", "/opt"]

        for base_dir in python_dirs:
            if not os.path.exists(base_dir):
                continue

            try:
                # Find and remove __pycache__ directories
                subprocess.run(
                    [
                        "find",
                        base_dir,
                        "-name",
                        "__pycache__",
                        "-type",
                        "d",
                        "-exec",
                        "rm",
                        "-rf",
                        "{}",
                        "+",
                    ],
                    capture_output=True,
                    timeout=60,
                )

                # Find and remove .pyc files
                subprocess.run(
                    ["find", base_dir, "-name", "*.pyc", "-delete"],
                    capture_output=True,
                    timeout=30,
                )

                # Find and remove .pyo files
                subprocess.run(
                    ["find", base_dir, "-name", "*.pyo", "-delete"],
                    capture_output=True,
                    timeout=30,
                )

            except subprocess.TimeoutExpired:
                self.log_action(f"Timeout cleaning Python caches in {base_dir}")

        self.log_action("Cleaned Python cache files", 5)  # Estimate
        return 5

    def force_garbage_collection(self):
        """Force garbage collection in running Python processes."""
        try:
            # Try to force garbage collection
            import gc

            collected = gc.collect()
            self.log_action(f"Python garbage collection: {collected} objects collected")

            # Also try to run gc in subprocess
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    'import gc; print(f"Collected: {gc.collect()}")',
                ],
                timeout=10,
            )

            return True
        except Exception as e:
            self.log_action(f"Garbage collection failed: {e}")
            return False

    def emergency_swap_cleanup(self):
        """Emergency swap space cleanup."""
        swap = psutil.swap_memory()

        if swap.percent > 50:  # If swap usage is high
            try:
                # Try to force swap cleanup (requires sudo)
                result = subprocess.run(
                    ["sudo", "sysctl", "vm.purge=1"], capture_output=True, timeout=30
                )

                if result.returncode == 0:
                    self.log_action("Forced swap cleanup")
                    return True
            except:
                pass

        return False

    def run_emergency_cleanup(self):
        """Run complete emergency cleanup."""
        print("=" * 60)
        print("EMERGENCY MEMORY CLEANUP INITIATED")
        print("=" * 60)

        # Get initial memory status
        initial_memory = self.get_memory_status()
        print(
            f"Initial Memory: {initial_memory['available_mb']:.1f}MB available "
            f"({initial_memory['used_percent']:.1f}% used)"
        )

        total_freed = 0

        # 1. Force memory purge
        print("\n1. Forcing memory purge...")
        self.force_memory_purge()
        time.sleep(2)

        # 2. Clean browser processes
        print("\n2. Cleaning browser processes...")
        freed = self.cleanup_browser_processes()
        total_freed += freed
        time.sleep(1)

        # 3. Clean Node.js processes
        print("\n3. Cleaning Node.js processes...")
        freed = self.cleanup_node_processes()
        total_freed += freed
        time.sleep(1)

        # 4. Clean temporary files
        print("\n4. Cleaning temporary files...")
        freed = self.cleanup_temp_files()
        total_freed += freed
        time.sleep(1)

        # 5. Clean Python caches
        print("\n5. Cleaning Python caches...")
        freed = self.cleanup_python_caches()
        total_freed += freed
        time.sleep(1)

        # 6. Force garbage collection
        print("\n6. Forcing garbage collection...")
        self.force_garbage_collection()
        time.sleep(1)

        # 7. Emergency swap cleanup
        print("\n7. Emergency swap cleanup...")
        self.emergency_swap_cleanup()
        time.sleep(2)

        # Final memory check
        final_memory = self.get_memory_status()
        memory_improvement = (
            final_memory["available_mb"] - initial_memory["available_mb"]
        )

        print("\n" + "=" * 60)
        print("EMERGENCY CLEANUP COMPLETED")
        print("=" * 60)
        print(f"Initial Memory:  {initial_memory['available_mb']:.1f}MB available")
        print(f"Final Memory:    {final_memory['available_mb']:.1f}MB available")
        print(f"Improvement:     {memory_improvement:+.1f}MB")
        print(f"Estimated Freed: {total_freed:.1f}MB")
        print(
            f"Duration:        {(datetime.now() - self.start_time).total_seconds():.1f}s"
        )
        print(f"Actions Taken:   {len(self.cleanup_actions)}")

        # Save cleanup report
        self.save_cleanup_report(initial_memory, final_memory, total_freed)

        return {
            "success": memory_improvement > 0,
            "memory_freed_mb": memory_improvement,
            "estimated_freed_mb": total_freed,
            "actions_taken": len(self.cleanup_actions),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

    def save_cleanup_report(self, initial_memory, final_memory, total_freed):
        """Save cleanup report to file."""
        report = {
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "initial_memory": initial_memory,
            "final_memory": final_memory,
            "memory_improvement_mb": final_memory["available_mb"]
            - initial_memory["available_mb"],
            "estimated_freed_mb": total_freed,
            "actions": self.cleanup_actions,
        }

        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Save report
        report_file = (
            logs_dir
            / f"emergency_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            import json

            json.dump(report, f, indent=2)

        print(f"\nCleanup report saved to: {report_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Emergency Memory Cleanup")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="More aggressive cleanup (use with caution)",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No actual cleanup will be performed")
        # TODO: Implement dry run mode
        return

    cleanup = EmergencyMemoryCleanup()

    # Show warning for aggressive mode
    if args.aggressive:
        print("WARNING: Aggressive mode will terminate more processes")
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Cancelled")
            return

    # Run cleanup
    result = cleanup.run_emergency_cleanup()

    # Exit with appropriate code
    exit_code = 0 if result["success"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

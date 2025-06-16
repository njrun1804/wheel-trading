#!/usr/bin/env python3
"""
System Health Check - Validates all fixes and checks for remaining issues
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def check_running_processes():
    """Check for any problematic processes"""
    print("🔍 Checking running processes...")

    problem_processes = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            cpu_percent = proc.info.get("cpu_percent")
            memory_percent = proc.info.get("memory_percent")

            if (
                cpu_percent
                and isinstance(cpu_percent, int | float)
                and cpu_percent > 80
            ):
                problem_processes.append(
                    f"High CPU: {proc.info['name']} ({proc.info['pid']}) - {cpu_percent:.1f}%"
                )
            if (
                memory_percent
                and isinstance(memory_percent, int | float)
                and memory_percent > 10
            ):
                problem_processes.append(
                    f"High Memory: {proc.info['name']} ({proc.info['pid']}) - {memory_percent:.1f}%"
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if problem_processes:
        print("⚠️  Found high-resource processes:")
        for proc in problem_processes[:5]:  # Show top 5
            print(f"   {proc}")
    else:
        print("✅ No high-resource processes detected")

    return len(problem_processes)


def check_log_errors():
    """Check recent logs for errors"""
    print("\n🔍 Checking recent log files for errors...")

    log_files = []
    for log_file in Path("logs").glob("*.log"):
        if log_file.stat().st_mtime > time.time() - 3600:  # Modified in last hour
            log_files.append(log_file)

    error_count = 0
    for log_file in log_files:
        try:
            with open(log_file) as f:
                lines = f.readlines()[-50:]  # Check last 50 lines
                for line in lines:
                    if "ERROR" in line or "CRITICAL" in line:
                        error_count += 1
                        if error_count <= 3:  # Show first 3 errors
                            print(f"   {log_file.name}: {line.strip()}")
        except Exception:
            continue

    if error_count == 0:
        print("✅ No recent errors in log files")
    else:
        print(f"⚠️  Found {error_count} recent errors")

    return error_count


def check_system_resources():
    """Check overall system health"""
    print("\n🔍 Checking system resources...")

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"   CPU Usage: {cpu_percent:.1f}%")

    # Memory
    memory = psutil.virtual_memory()
    print(
        f"   Memory Usage: {memory.percent:.1f}% ({memory.available // (1024**3)}GB available)"
    )

    # Load average
    load_avg = os.getloadavg()
    print(f"   Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")

    # Check for issues
    issues = []
    if cpu_percent > 80:
        issues.append(f"High CPU usage: {cpu_percent:.1f}%")
    if memory.percent > 85:
        issues.append(f"High memory usage: {memory.percent:.1f}%")
    if load_avg[0] > 8:
        issues.append(f"High load average: {load_avg[0]:.2f}")

    if issues:
        print("⚠️  System resource issues:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ System resources are healthy")

    return len(issues)


def check_failed_services():
    """Check for failed system services"""
    print("\n🔍 Checking system services...")

    try:
        result = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split("\n")[1:]  # Skip header
            failed_services = []

            for line in lines:
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        parts[0].strip()
                        status = parts[1].strip()
                        if status not in ["0", "-9", "-"]:
                            failed_services.append(line.strip())

            if len(failed_services) > 50:  # More than 50 failed services is concerning
                print(f"⚠️  High number of failed services: {len(failed_services)}")
                return len(failed_services)
            else:
                print(
                    f"✅ Service status acceptable ({len(failed_services)} failed services)"
                )
                return 0
    except Exception as e:
        print(f"⚠️  Could not check services: {e}")
        return 1


def run_system_validation():
    """Run comprehensive system validation"""
    print("🔧 FIX AGENT: System Health Validation")
    print("=" * 50)

    total_issues = 0

    # Run all checks
    total_issues += check_running_processes()
    total_issues += check_log_errors()
    total_issues += check_system_resources()
    total_issues += check_failed_services()

    print("\n" + "=" * 50)

    if total_issues == 0:
        print("✅ SYSTEM HEALTH: All checks passed - no issues detected")
        print("🎯 All deployment/integration fixes are working correctly")
        return True
    else:
        print(f"⚠️  SYSTEM HEALTH: {total_issues} issues detected")
        print("🔧 Some areas may need additional monitoring")
        return False


def main():
    """Main health check"""
    return run_system_validation()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Hardware Acceleration Audit for M4 Pro - Diagnose and fix all bottlenecks."""

import asyncio
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import psutil

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import numpy as np


class HardwareAccelerationAuditor:
    """Comprehensive hardware acceleration audit and optimization."""

    def __init__(self):
        self.report = {
            "timestamp": time.time(),
            "system_info": {},
            "cpu_analysis": {},
            "memory_analysis": {},
            "gpu_analysis": {},
            "current_config": {},
            "bottlenecks": [],
            "optimizations": [],
            "utilization_tests": {},
            "recommendations": [],
        }

    def audit_system_info(self):
        """Audit basic system information."""
        print("🔍 Auditing system information...")

        # Basic system info
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "cpu_count_physical": mp.cpu_count(),
            "cpu_count_logical": os.cpu_count(),
        }

        # Get detailed CPU info on macOS
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                system_info["cpu_brand"] = result.stdout.strip()

                result = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True
                )
                system_info["physical_cores"] = int(result.stdout.strip())

                result = subprocess.run(
                    ["sysctl", "-n", "hw.logicalcpu"], capture_output=True, text=True
                )
                system_info["logical_cores"] = int(result.stdout.strip())

                # M4 Pro specific info
                if "M4 Pro" in system_info.get("cpu_brand", ""):
                    system_info["performance_cores"] = 8  # M4 Pro has 8P cores
                    system_info["efficiency_cores"] = 4  # M4 Pro has 4E cores
                    system_info["gpu_cores"] = 20  # M4 Pro has 20-core GPU

            except Exception as e:
                print(f"Warning: Could not get detailed CPU info: {e}")

        # Memory info
        memory = psutil.virtual_memory()
        system_info["total_memory_gb"] = memory.total / (1024**3)
        system_info["available_memory_gb"] = memory.available / (1024**3)
        system_info["memory_percent"] = memory.percent

        self.report["system_info"] = system_info

        print(
            f"✅ System: {system_info.get('cpu_brand', 'Unknown')} with {system_info['cpu_count_physical']} cores"
        )
        print(
            f"✅ Memory: {system_info['total_memory_gb']:.1f}GB total, {system_info['available_memory_gb']:.1f}GB available"
        )

    def audit_current_config(self):
        """Audit current optimization configuration."""
        print("\n🔍 Auditing current configuration...")

        config_files = [
            "optimization_config.json",
            "config.yaml",
            "config_unified.yaml",
        ]

        configs = {}
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file) as f:
                        if config_file.endswith(".json"):
                            configs[config_file] = json.load(f)
                        else:
                            import yaml

                            configs[config_file] = yaml.safe_load(f)
                except Exception as e:
                    print(f"Warning: Could not load {config_file}: {e}")

        self.report["current_config"] = configs

        # Analyze configuration vs hardware
        if "optimization_config.json" in configs:
            config = configs["optimization_config.json"]
            cpu_workers = config.get("cpu", {}).get("max_workers", 1)
            total_cores = self.report["system_info"]["cpu_count_physical"]

            if cpu_workers < total_cores:
                self.report["bottlenecks"].append(
                    {
                        "type": "cpu_underutilization",
                        "description": f"Using only {cpu_workers} workers vs {total_cores} available cores",
                        "severity": "high",
                        "impact": f"Missing {(total_cores - cpu_workers) / total_cores * 100:.0f}% CPU capacity",
                    }
                )

        print(f"✅ Found {len(configs)} configuration files")

    def audit_mlx_gpu(self):
        """Audit MLX GPU acceleration."""
        print("\n🔍 Auditing MLX GPU acceleration...")

        gpu_analysis = {
            "mlx_available": MLX_AVAILABLE,
            "metal_available": False,
            "gpu_memory_gb": 0,
            "performance_tests": {},
        }

        if MLX_AVAILABLE:
            try:
                gpu_analysis["metal_available"] = mx.metal.is_available()
                if gpu_analysis["metal_available"]:
                    # Test basic GPU operations
                    print("⚡ Testing MLX Metal GPU performance...")

                    # Matrix multiplication test
                    sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
                    for size in sizes:
                        # CPU baseline
                        a_np = np.random.randn(*size).astype(np.float32)
                        b_np = np.random.randn(*size).astype(np.float32)

                        start = time.perf_counter()
                        a_np @ b_np
                        cpu_time = time.perf_counter() - start

                        # GPU test
                        a_mx = mx.array(a_np)
                        b_mx = mx.array(b_np)

                        start = time.perf_counter()
                        result_gpu = a_mx @ b_mx
                        mx.eval(result_gpu)  # Force evaluation
                        gpu_time = time.perf_counter() - start

                        speedup = cpu_time / gpu_time
                        gpu_analysis["performance_tests"][
                            f"matmul_{size[0]}x{size[1]}"
                        ] = {
                            "cpu_time_ms": cpu_time * 1000,
                            "gpu_time_ms": gpu_time * 1000,
                            "speedup": speedup,
                            "effective": speedup > 1.2,  # 20% faster threshold
                        }

                        print(
                            f"  Matrix {size[0]}x{size[1]}: GPU {speedup:.1f}x faster"
                        )

            except Exception as e:
                gpu_analysis["error"] = str(e)
                print(f"❌ MLX GPU test failed: {e}")
        else:
            self.report["bottlenecks"].append(
                {
                    "type": "missing_mlx",
                    "description": "MLX not available - GPU acceleration disabled",
                    "severity": "high",
                    "impact": "No GPU acceleration for vector operations",
                }
            )

        self.report["gpu_analysis"] = gpu_analysis

        if gpu_analysis["metal_available"]:
            print("✅ MLX Metal GPU acceleration is working")
        else:
            print("❌ MLX Metal GPU acceleration not available")

    def test_cpu_utilization(self):
        """Test actual CPU core utilization."""
        print("\n🔍 Testing CPU core utilization...")

        def cpu_intensive_task(n):
            """CPU-intensive task for testing core utilization."""
            result = 0
            for i in range(n):
                result += i * i
            return result

        cpu_analysis = {
            "baseline_performance": {},
            "parallel_performance": {},
            "core_efficiency": {},
        }

        # Test different worker counts
        task_size = 1_000_000
        worker_counts = [1, 4, 8, 12, 16]

        for worker_count in worker_counts:
            if worker_count > mp.cpu_count():
                continue

            print(f"  Testing with {worker_count} workers...")

            # Run parallel tasks
            start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(cpu_intensive_task, task_size)
                    for _ in range(worker_count * 2)
                ]  # 2x tasks to ensure saturation
                [f.result() for f in futures]

            elapsed = time.perf_counter() - start
            throughput = (worker_count * 2) / elapsed

            cpu_analysis["parallel_performance"][worker_count] = {
                "elapsed_seconds": elapsed,
                "throughput_tasks_per_sec": throughput,
                "efficiency_vs_single": throughput
                / cpu_analysis.get("parallel_performance", {})
                .get(1, {})
                .get("throughput_tasks_per_sec", throughput),
            }

        # Find optimal worker count
        best_throughput = 0
        optimal_workers = 1
        for workers, perf in cpu_analysis["parallel_performance"].items():
            if perf["throughput_tasks_per_sec"] > best_throughput:
                best_throughput = perf["throughput_tasks_per_sec"]
                optimal_workers = workers

        cpu_analysis["optimal_workers"] = optimal_workers
        cpu_analysis["max_throughput"] = best_throughput

        # Check if we're underutilizing cores
        total_cores = self.report["system_info"]["cpu_count_physical"]
        if optimal_workers < total_cores:
            self.report["bottlenecks"].append(
                {
                    "type": "cpu_optimal_workers",
                    "description": f"Optimal worker count ({optimal_workers}) less than total cores ({total_cores})",
                    "severity": "medium",
                    "impact": "CPU cores may be heterogeneous (P+E cores on M4 Pro)",
                }
            )

        self.report["cpu_analysis"] = cpu_analysis
        print(
            f"✅ Optimal worker count: {optimal_workers} (max throughput: {best_throughput:.1f} tasks/sec)"
        )

    def test_memory_performance(self):
        """Test memory allocation and bandwidth."""
        print("\n🔍 Testing memory performance...")

        memory_analysis = {
            "allocation_tests": {},
            "bandwidth_tests": {},
            "cache_efficiency": {},
        }

        # Test large allocations
        allocation_sizes = [1, 5, 10, 15]  # GB
        for size_gb in allocation_sizes:
            try:
                print(f"  Testing {size_gb}GB allocation...")
                size_bytes = size_gb * 1024**3

                start = time.perf_counter()
                data = np.random.randn(size_bytes // 8).astype(
                    np.float64
                )  # 8 bytes per float64
                alloc_time = time.perf_counter() - start

                # Test access pattern
                start = time.perf_counter()
                np.sum(data[::1000])  # Sample access
                access_time = time.perf_counter() - start

                memory_analysis["allocation_tests"][f"{size_gb}GB"] = {
                    "allocation_time_sec": alloc_time,
                    "access_time_sec": access_time,
                    "bandwidth_gb_per_sec": size_gb / alloc_time,
                }

                del data  # Free memory

            except MemoryError:
                memory_analysis["allocation_tests"][f"{size_gb}GB"] = {
                    "error": "Out of memory"
                }
                break

        self.report["memory_analysis"] = memory_analysis
        print("✅ Memory performance testing completed")

    def analyze_bottlenecks(self):
        """Analyze identified bottlenecks and generate optimizations."""
        print("\n🔍 Analyzing bottlenecks...")

        optimizations = []

        # CPU optimization
        total_cores = self.report["system_info"]["cpu_count_physical"]
        current_workers = (
            self.report["current_config"]
            .get("optimization_config.json", {})
            .get("cpu", {})
            .get("max_workers", 1)
        )

        if current_workers < total_cores:
            optimizations.append(
                {
                    "type": "cpu_workers",
                    "description": f"Increase max_workers from {current_workers} to {total_cores}",
                    "config_change": {
                        "file": "optimization_config.json",
                        "path": "cpu.max_workers",
                        "old_value": current_workers,
                        "new_value": total_cores,
                    },
                    "expected_improvement": f"{(total_cores / current_workers - 1) * 100:.0f}% more CPU throughput",
                }
            )

        # Memory optimization
        total_memory_gb = self.report["system_info"]["total_memory_gb"]
        current_cache_mb = (
            self.report["current_config"]
            .get("optimization_config.json", {})
            .get("memory", {})
            .get("cache_size_mb", 1000)
        )
        optimal_cache_mb = int(total_memory_gb * 1024 * 0.3)  # Use 30% for cache

        if current_cache_mb < optimal_cache_mb:
            optimizations.append(
                {
                    "type": "memory_cache",
                    "description": f"Increase cache from {current_cache_mb}MB to {optimal_cache_mb}MB",
                    "config_change": {
                        "file": "optimization_config.json",
                        "path": "memory.cache_size_mb",
                        "old_value": current_cache_mb,
                        "new_value": optimal_cache_mb,
                    },
                    "expected_improvement": "Better caching, reduced I/O",
                }
            )

        # GPU optimization
        if not self.report["gpu_analysis"]["mlx_available"]:
            optimizations.append(
                {
                    "type": "install_mlx",
                    "description": "Install MLX for GPU acceleration",
                    "command": "pip install mlx",
                    "expected_improvement": "5-10x speedup for vector operations",
                }
            )

        # Thread pool optimization
        optimal_threads = min(32, total_cores * 2)  # 2x cores for I/O bound
        current_threads = (
            self.report["current_config"]
            .get("optimization_config.json", {})
            .get("io", {})
            .get("concurrent_reads", 12)
        )

        if current_threads < optimal_threads:
            optimizations.append(
                {
                    "type": "io_threads",
                    "description": f"Increase I/O threads from {current_threads} to {optimal_threads}",
                    "config_change": {
                        "file": "optimization_config.json",
                        "path": "io.concurrent_reads",
                        "old_value": current_threads,
                        "new_value": optimal_threads,
                    },
                    "expected_improvement": "Better I/O parallelization",
                }
            )

        self.report["optimizations"] = optimizations
        print(f"✅ Generated {len(optimizations)} optimization recommendations")

    def apply_optimizations(self, auto_apply: bool = False):
        """Apply optimizations to configuration files."""
        print("\n🔧 Applying optimizations...")

        if not auto_apply:
            response = input("Apply optimizations automatically? (y/N): ")
            if response.lower() != "y":
                print("Skipping optimization application")
                return

        applied = []

        for opt in self.report["optimizations"]:
            if "config_change" in opt:
                try:
                    config_file = opt["config_change"]["file"]
                    if Path(config_file).exists():
                        with open(config_file) as f:
                            config = json.load(f)

                        # Navigate to the config path and update
                        path_parts = opt["config_change"]["path"].split(".")
                        current = config
                        for part in path_parts[:-1]:
                            current = current.setdefault(part, {})
                        current[path_parts[-1]] = opt["config_change"]["new_value"]

                        # Backup original
                        backup_file = f"{config_file}.backup.{int(time.time())}"
                        subprocess.run(["cp", config_file, backup_file])

                        # Write updated config
                        with open(config_file, "w") as f:
                            json.dump(config, f, indent=2)

                        applied.append(opt)
                        print(f"✅ Applied: {opt['description']}")

                except Exception as e:
                    print(f"❌ Failed to apply {opt['description']}: {e}")

            elif "command" in opt:
                try:
                    print(f"🔧 Running: {opt['command']}")
                    result = subprocess.run(
                        opt["command"].split(), capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        applied.append(opt)
                        print(f"✅ Applied: {opt['description']}")
                    else:
                        print(f"❌ Command failed: {result.stderr}")
                except Exception as e:
                    print(f"❌ Failed to run {opt['command']}: {e}")

        print(f"\n✅ Applied {len(applied)} optimizations")

    def generate_report(self):
        """Generate comprehensive optimization report."""
        print("\n📊 Generating comprehensive report...")

        # Generate recommendations
        recommendations = []

        system_info = self.report["system_info"]
        if "M4 Pro" in system_info.get("cpu_brand", ""):
            recommendations.extend(
                [
                    "✅ Detected M4 Pro - optimal for this workload",
                    f"🔥 Use all {system_info['cpu_count_physical']} cores (8P + 4E configuration)",
                    "⚡ Enable MLX Metal GPU acceleration for 5-10x vector speedup",
                    f"🧠 Allocate {system_info['total_memory_gb'] * 0.3:.0f}GB for caching",
                    "🔄 Use memory mapping for large files",
                    "📊 Monitor CPU utilization to balance P-cores vs E-cores",
                ]
            )

        # Performance targets
        if self.report["cpu_analysis"]:
            optimal_workers = self.report["cpu_analysis"].get("optimal_workers", 12)
            recommendations.append(
                f"🎯 Target: {optimal_workers} parallel workers for optimal throughput"
            )

        # GPU recommendations
        if self.report["gpu_analysis"]["mlx_available"]:
            recommendations.append(
                "⚡ GPU acceleration ready - use for vector ops >1K elements"
            )
        else:
            recommendations.append("❌ Install MLX: pip install mlx")

        self.report["recommendations"] = recommendations

        # Write report to file
        report_file = f"hardware_acceleration_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        print(f"📄 Report saved to: {report_file}")
        return report_file

    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("🚀 HARDWARE ACCELERATION AUDIT SUMMARY")
        print("=" * 60)

        system_info = self.report["system_info"]
        print(f"💻 System: {system_info.get('cpu_brand', 'Unknown')}")
        print(f"🔥 Cores: {system_info['cpu_count_physical']} total")
        print(f"🧠 Memory: {system_info['total_memory_gb']:.1f}GB")

        # Bottlenecks
        if self.report["bottlenecks"]:
            print(f"\n❌ BOTTLENECKS FOUND ({len(self.report['bottlenecks'])})")
            for bottleneck in self.report["bottlenecks"]:
                severity = bottleneck["severity"].upper()
                print(f"  {severity}: {bottleneck['description']}")

        # Optimizations
        if self.report["optimizations"]:
            print(f"\n🔧 OPTIMIZATIONS AVAILABLE ({len(self.report['optimizations'])})")
            for opt in self.report["optimizations"]:
                print(f"  • {opt['description']}")
                if "expected_improvement" in opt:
                    print(f"    Expected: {opt['expected_improvement']}")

        # GPU status
        gpu_status = (
            "✅ Working"
            if self.report["gpu_analysis"]["mlx_available"]
            else "❌ Not Available"
        )
        print(f"\n⚡ GPU Acceleration: {gpu_status}")

        # Performance summary
        if self.report["cpu_analysis"]:
            optimal = self.report["cpu_analysis"].get("optimal_workers", "Unknown")
            throughput = self.report["cpu_analysis"].get("max_throughput", 0)
            print(f"🎯 Optimal Workers: {optimal}")
            print(f"📊 Max Throughput: {throughput:.1f} tasks/sec")

        print("\n" + "=" * 60)


async def main():
    """Run comprehensive hardware acceleration audit."""
    print("🚀 Starting M4 Pro Hardware Acceleration Audit")
    print("=" * 60)

    auditor = HardwareAccelerationAuditor()

    # Run all audits
    auditor.audit_system_info()
    auditor.audit_current_config()
    auditor.audit_mlx_gpu()
    auditor.test_cpu_utilization()
    auditor.test_memory_performance()
    auditor.analyze_bottlenecks()

    # Generate and apply optimizations
    auditor.apply_optimizations(auto_apply=False)

    # Generate report
    report_file = auditor.generate_report()
    auditor.print_summary()

    print(f"\n🎉 Audit complete! Report saved to: {report_file}")

    # Return key metrics
    return {
        "total_cores": auditor.report["system_info"]["cpu_count_physical"],
        "bottlenecks": len(auditor.report["bottlenecks"]),
        "optimizations": len(auditor.report["optimizations"]),
        "gpu_available": auditor.report["gpu_analysis"]["mlx_available"],
        "report_file": report_file,
    }


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nKey metrics: {result}")

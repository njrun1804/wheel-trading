#!/usr/bin/env python3
"""Comprehensive hardware assessment for M4 Pro Mac to optimize implementations."""

import subprocess
import platform
import psutil
import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_command(cmd: str) -> str:
    """Run shell command and return output."""
    try:
        result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        return result.strip()
    except:
        return "N/A"


def assess_hardware():
    """Comprehensive hardware assessment."""
    print("🔍 M4 Pro Mac Hardware Assessment")
    print("=" * 60)
    
    assessment = {}
    
    # Basic system info
    print("\n📱 System Information:")
    assessment["system"] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0]
    }
    
    chip_info = run_command("sysctl -n machdep.cpu.brand_string")
    print(f"  Chip: {chip_info}")
    assessment["chip"] = chip_info
    
    # CPU details
    print("\n⚡ CPU Details:")
    cpu_info = {
        "physical_cores": int(run_command("sysctl -n hw.physicalcpu")),
        "logical_cores": int(run_command("sysctl -n hw.logicalcpu")),
        "performance_cores": int(run_command("sysctl -n hw.perflevel0.physicalcpu") or "0"),
        "efficiency_cores": int(run_command("sysctl -n hw.perflevel1.physicalcpu") or "0"),
        "cpu_frequency_max": int(run_command("sysctl -n hw.cpufrequency_max") or "0") / 1e9,
        "cache_sizes": {
            "l1_icache": run_command("sysctl -n hw.l1icachesize"),
            "l1_dcache": run_command("sysctl -n hw.l1dcachesize"),
            "l2_cache": run_command("sysctl -n hw.l2cachesize"),
            "l3_cache": run_command("sysctl -n hw.l3cachesize") or "N/A"
        }
    }
    
    for key, value in cpu_info.items():
        if key != "cache_sizes":
            print(f"  {key}: {value}")
    
    print(f"  Cache sizes: L1i={cpu_info['cache_sizes']['l1_icache']}, "
          f"L1d={cpu_info['cache_sizes']['l1_dcache']}, "
          f"L2={cpu_info['cache_sizes']['l2_cache']}")
    
    assessment["cpu"] = cpu_info
    
    # Memory details
    print("\n💾 Memory Details:")
    memory_bytes = int(run_command("sysctl -n hw.memsize"))
    memory_gb = memory_bytes / (1024**3)
    
    vm = psutil.virtual_memory()
    memory_info = {
        "total_gb": memory_gb,
        "available_gb": vm.available / (1024**3),
        "used_gb": vm.used / (1024**3),
        "percent_used": vm.percent,
        "memory_bandwidth": run_command("sysctl -n hw.memorytype") or "LPDDR5",
        "unified_memory": True  # M4 Pro has unified memory
    }
    
    print(f"  Total: {memory_info['total_gb']:.1f}GB")
    print(f"  Available: {memory_info['available_gb']:.1f}GB ({100-vm.percent:.1f}%)")
    print(f"  Type: {memory_info['memory_bandwidth']} (Unified)")
    
    assessment["memory"] = memory_info
    
    # GPU details
    print("\n🎮 GPU Details:")
    gpu_info = {
        "gpu_cores": "20" if "M4 Pro" in chip_info else "Unknown",
        "metal_version": run_command("system_profiler SPDisplaysDataType | grep 'Metal' | head -1"),
        "gpu_family": run_command("system_profiler SPDisplaysDataType | grep 'Chipset Model' | head -1")
    }
    
    # Check for GPU libraries
    gpu_libs = {}
    
    # MLX
    try:
        import mlx.core as mx
        gpu_libs["mlx"] = {
            "available": True,
            "version": getattr(mx, "__version__", "Unknown"),
            "device": "gpu"
        }
        print("  MLX: ✅ Available")
    except ImportError:
        gpu_libs["mlx"] = {"available": False}
        print("  MLX: ❌ Not installed")
    
    # PyTorch MPS
    try:
        import torch
        mps_available = torch.backends.mps.is_available()
        gpu_libs["pytorch_mps"] = {
            "available": mps_available,
            "version": torch.__version__,
            "built_with_mps": torch.backends.mps.is_built()
        }
        print(f"  PyTorch MPS: {'✅' if mps_available else '❌'} {torch.__version__}")
    except ImportError:
        gpu_libs["pytorch_mps"] = {"available": False}
        print("  PyTorch MPS: ❌ Not installed")
    
    # TensorFlow Metal
    try:
        import tensorflow as tf
        gpu_libs["tensorflow_metal"] = {
            "available": len(tf.config.list_physical_devices('GPU')) > 0,
            "version": tf.__version__
        }
        print(f"  TensorFlow Metal: {'✅' if gpu_libs['tensorflow_metal']['available'] else '❌'}")
    except ImportError:
        gpu_libs["tensorflow_metal"] = {"available": False}
        print("  TensorFlow Metal: ❌ Not installed")
    
    gpu_info["libraries"] = gpu_libs
    assessment["gpu"] = gpu_info
    
    # Storage details
    print("\n💽 Storage Details:")
    disk = psutil.disk_usage('/')
    storage_info = {
        "total_gb": disk.total / (1024**3),
        "available_gb": disk.free / (1024**3),
        "used_percent": disk.percent,
        "ssd_type": "NVMe"  # M4 Pro has NVMe SSD
    }
    
    print(f"  Total: {storage_info['total_gb']:.1f}GB")
    print(f"  Available: {storage_info['available_gb']:.1f}GB ({100-disk.percent:.1f}%)")
    print(f"  Type: {storage_info['ssd_type']}")
    
    assessment["storage"] = storage_info
    
    # Performance features
    print("\n🚀 Performance Features:")
    features = {
        "simd_width": run_command("sysctl -n hw.optional.neon") == "1",
        "arm_features": run_command("sysctl -n hw.optional.arm64") == "1",
        "accelerate_framework": os.path.exists("/System/Library/Frameworks/Accelerate.framework"),
        "metal_performance_shaders": os.path.exists("/System/Library/Frameworks/MetalPerformanceShaders.framework"),
        "neural_engine": "M4" in chip_info,  # M4 has 38 TOPS Neural Engine
        "ray_tracing": "M4" in chip_info,  # M4 Pro has hardware ray tracing
        "av1_decode": "M4" in chip_info  # M4 has AV1 decode
    }
    
    for feature, available in features.items():
        print(f"  {feature}: {'✅' if available else '❌'}")
    
    assessment["features"] = features
    
    # Optimization recommendations
    print("\n📊 Optimization Recommendations:")
    recommendations = []
    
    if cpu_info["performance_cores"] >= 8:
        recommendations.append("Use ProcessPoolExecutor with 8-10 workers for CPU-bound tasks")
    
    if memory_info["total_gb"] >= 24:
        recommendations.append("Can safely use 16-20GB for memory-intensive operations")
    
    if gpu_libs["mlx"]["available"]:
        recommendations.append("Use MLX for matrix operations and neural networks")
    
    if features["neural_engine"]:
        recommendations.append("Leverage Neural Engine for ML inference via CoreML")
    
    if features["metal_performance_shaders"]:
        recommendations.append("Use MPS for image processing and parallel compute")
    
    for rec in recommendations:
        print(f"  • {rec}")
    
    assessment["recommendations"] = recommendations
    
    # Benchmark quick test
    print("\n⏱️  Quick Performance Test:")
    
    # CPU benchmark
    start = time.perf_counter()
    _ = sum(i*i for i in range(10_000_000))
    cpu_time = (time.perf_counter() - start) * 1000
    print(f"  CPU (10M operations): {cpu_time:.1f}ms")
    
    # Memory benchmark
    start = time.perf_counter()
    data = bytearray(100 * 1024 * 1024)  # 100MB
    _ = data[::-1]  # Reverse
    mem_time = (time.perf_counter() - start) * 1000
    print(f"  Memory (100MB reverse): {mem_time:.1f}ms")
    
    assessment["benchmarks"] = {
        "cpu_10m_ops_ms": cpu_time,
        "memory_100mb_reverse_ms": mem_time
    }
    
    # Save assessment
    with open("hardware_assessment.json", "w") as f:
        json.dump(assessment, f, indent=2)
    
    print(f"\n💾 Full assessment saved to hardware_assessment.json")
    
    return assessment


def generate_optimization_config(assessment: dict) -> dict:
    """Generate optimization configuration based on assessment."""
    
    config = {
        "cpu": {
            "max_workers": assessment["cpu"]["performance_cores"],
            "batch_size": 2048 if assessment["memory"]["total_gb"] >= 24 else 1024,
            "chunk_size": 65536 if assessment["memory"]["total_gb"] >= 24 else 32768
        },
        "memory": {
            "cache_size_mb": int(assessment["memory"]["total_gb"] * 1024 * 0.2),  # 20% for cache
            "buffer_size_mb": int(assessment["memory"]["total_gb"] * 1024 * 0.3),  # 30% for buffers
            "max_allocation_gb": int(assessment["memory"]["total_gb"] * 0.8)  # 80% max
        },
        "gpu": {
            "preferred_backend": "mlx" if assessment["gpu"]["libraries"]["mlx"]["available"] else "cpu",
            "batch_size": 4096 if assessment["gpu"]["libraries"]["mlx"]["available"] else 1024,
            "use_unified_memory": True
        },
        "io": {
            "concurrent_reads": assessment["cpu"]["logical_cores"] * 2,
            "prefetch_size_mb": 64 if assessment["storage"]["ssd_type"] == "NVMe" else 32,
            "use_mmap": True
        }
    }
    
    return config


if __name__ == "__main__":
    assessment = assess_hardware()
    
    print("\n⚙️  Generating Optimization Config...")
    config = generate_optimization_config(assessment)
    
    with open("optimization_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Optimization config saved to optimization_config.json")
    
    print("\n🎯 Key Optimizations for Your M4 Pro:")
    print(f"  • CPU Workers: {config['cpu']['max_workers']} performance cores")
    print(f"  • Memory Cache: {config['memory']['cache_size_mb']/1024:.1f}GB")
    print(f"  • GPU Backend: {config['gpu']['preferred_backend']}")
    print(f"  • I/O Streams: {config['io']['concurrent_reads']} concurrent")
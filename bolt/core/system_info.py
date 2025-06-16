"""
System information and status utilities for Bolt.

Provides comprehensive system information including hardware capabilities,
GPU availability, and accelerated tools status.
"""

import platform
from typing import Any

import psutil

# GPU backends
try:
    import mlx.core as mx

    HAS_MLX = mx.metal.is_available()
except ImportError:
    HAS_MLX = False

try:
    import torch

    HAS_TORCH_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH_MPS = False

try:
    import pynvml

    HAS_NVIDIA = True
except ImportError:
    HAS_NVIDIA = False


def get_system_status() -> dict[str, Any]:
    """Get comprehensive system status information."""

    # Basic system info
    system_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Hardware detection
    hardware = "Unknown"
    if system_info["platform"] == "Darwin" and system_info["machine"] == "arm64":
        # Try to detect specific Apple Silicon chip
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                brand = result.stdout.strip()
                if "M4" in brand:
                    hardware = "Apple M4"
                elif "M3" in brand:
                    hardware = "Apple M3"
                elif "M2" in brand:
                    hardware = "Apple M2"
                elif "M1" in brand:
                    hardware = "Apple M1"
                else:
                    hardware = f"Apple Silicon ({brand})"
            else:
                hardware = "Apple Silicon"
        except Exception:
            hardware = "Apple Silicon"
    elif system_info["platform"] == "Linux":
        hardware = "Linux x86_64"
    elif system_info["platform"] == "Windows":
        hardware = "Windows x86_64"

    # Memory info
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)

    # CPU info
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)

    # GPU backend detection
    gpu_backend = "None"
    gpu_memory_gb = None

    if HAS_MLX:
        gpu_backend = "MLX (Metal)"
        try:
            # Try to get Metal memory info (if API exists)
            if hasattr(mx.metal, "get_memory_info"):
                gpu_memory_gb = mx.metal.get_memory_info().get("peak", 0) / (1024**3)
            else:
                # Fallback for older MLX versions
                gpu_memory_gb = None
        except Exception:
            pass
    elif HAS_TORCH_MPS:
        gpu_backend = "PyTorch MPS"
        try:
            # Try to get MPS memory info
            if hasattr(torch.backends.mps, "driver_allocated_memory"):
                gpu_memory_gb = torch.backends.mps.driver_allocated_memory() / (
                    1024**3
                )
        except Exception:
            pass
    elif HAS_NVIDIA:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_backend = f"NVIDIA ({name})"
            gpu_memory_gb = memory_info.total / (1024**3)
        except Exception:
            gpu_backend = "NVIDIA (Error)"

    # Accelerated tools status
    accelerated_tools = check_accelerated_tools_status()

    return {
        "hardware": hardware,
        "platform": system_info["platform"],
        "machine": system_info["machine"],
        "python_version": system_info["python_version"],
        "memory_gb": memory_gb,
        "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads,
        "gpu_backend": gpu_backend,
        "gpu_memory_gb": gpu_memory_gb,
        "accelerated_tools": accelerated_tools,
    }


def check_accelerated_tools_status() -> dict[str, bool]:
    """Check availability of accelerated tools."""

    tools_status = {}

    # Check tool availability using importlib
    import importlib

    tools_to_check = {
        "ripgrep_turbo": "src.unity_wheel.accelerated_tools.ripgrep_turbo",
        "dependency_graph": "src.unity_wheel.accelerated_tools.dependency_graph_turbo",
        "python_analysis": "src.unity_wheel.accelerated_tools.python_analysis_turbo",
        "duckdb_turbo": "src.unity_wheel.accelerated_tools.duckdb_turbo",
        "trace_turbo": "src.unity_wheel.accelerated_tools.trace_simple",
        "python_helpers": "src.unity_wheel.accelerated_tools.python_helpers_turbo",
    }

    for tool_name, module_name in tools_to_check.items():
        try:
            importlib.import_module(module_name)
            tools_status[tool_name] = True
        except ImportError:
            tools_status[tool_name] = False

    return tools_status


def get_hardware_capabilities() -> dict[str, Any]:
    """Get detailed hardware capabilities for optimization."""

    capabilities: dict[str, Any] = {
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "has_gpu": False,
        "gpu_type": None,
        "parallel_capable": True,
        "metal_available": False,
        "mlx_available": HAS_MLX,
        "torch_mps_available": HAS_TORCH_MPS,
    }

    # Check for Metal/GPU capabilities
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        capabilities["metal_available"] = True
        capabilities["has_gpu"] = True
        capabilities["gpu_type"] = "Metal"

    # Check NVIDIA
    if HAS_NVIDIA:
        try:
            pynvml.nvmlInit()
            capabilities["has_gpu"] = True
            capabilities["gpu_type"] = "NVIDIA"
        except Exception:
            pass

    return capabilities

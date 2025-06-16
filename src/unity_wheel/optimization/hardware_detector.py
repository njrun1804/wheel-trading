"""Hardware capabilities detection for Mac optimization."""

import multiprocessing
import platform
from typing import Any

import psutil


class HardwareCapabilities:
    """Detect and report hardware capabilities."""

    def __init__(self):
        self.cpu_cores = multiprocessing.cpu_count()
        self.platform_info = platform.platform()

        # M4 Pro specific
        if "arm64" in platform.machine().lower():
            # M4 Pro has 12 cores (8 P-cores + 4 E-cores)
            self.performance_cores = 8
            self.efficiency_cores = 4
            self.gpu_cores = 20  # M4 Pro GPU cores
        else:
            self.performance_cores = self.cpu_cores
            self.efficiency_cores = 0
            self.gpu_cores = 0

        # Memory info
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)

    def get_optimal_workers(self, task_type: str = "compute") -> int:
        """Get optimal number of workers for task type."""
        if task_type == "compute":
            return self.performance_cores
        elif task_type == "io":
            return self.cpu_cores * 2
        else:
            return self.cpu_cores

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_cores": self.cpu_cores,
            "performance_cores": self.performance_cores,
            "efficiency_cores": self.efficiency_cores,
            "gpu_cores": self.gpu_cores,
            "total_memory_gb": self.total_memory_gb,
            "platform": self.platform_info,
        }

"""Unified Hardware State Detection Service for M4 Pro.

Provides a singleton interface for consistent hardware detection and resource management
across the entire system. Consolidates hardware detection from multiple sources into
a single source of truth with <5ms access times.
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import psutil

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CPUInfo:
    """CPU hardware information."""

    brand: str
    physical_cores: int
    logical_cores: int
    p_cores: int  # Performance cores
    e_cores: int  # Efficiency cores
    frequency_mhz: float = 0.0
    frequency_max_mhz: float = 0.0


@dataclass
class GPUInfo:
    """GPU hardware information."""

    name: str
    cores: int
    metal_supported: bool
    unified_memory: bool
    vram_mb: int = 0  # 0 for unified memory systems


@dataclass
class MemoryInfo:
    """Memory hardware information."""

    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    page_size: int
    unified: bool
    metal_limit_gb: float  # Max memory available to Metal GPU


@dataclass
class ResourceBudget:
    """Dynamic resource allocation budget."""

    cpu_workers: int
    p_core_workers: int
    e_core_workers: int
    gpu_workers: int
    memory_pool_mb: int
    batch_size: int
    timestamp: float = field(default_factory=time.time)


class HardwareState:
    """Unified hardware state detection and management.

    Singleton pattern ensures consistent hardware detection across the system.
    Provides fast (<5ms) access to hardware capabilities and dynamic resource allocation.
    """

    _instance = None
    _lock = Lock()
    _cache_duration = 60.0  # Refresh hardware info every 60 seconds

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize hardware state (only runs once due to singleton)."""
        if self._initialized:
            return

        self._cpu_info: CPUInfo | None = None
        self._gpu_info: GPUInfo | None = None
        self._memory_info: MemoryInfo | None = None
        self._last_update: float = 0
        self._detection_lock = Lock()
        self._resource_budgets: dict[str, ResourceBudget] = {}

        # Cache file for persistence across restarts
        self._cache_file = Path.home() / ".hardware_state_cache.json"

        # Detect hardware on initialization
        self._detect_hardware()
        self._initialized = True

        logger.info(f"HardwareState initialized: {self.get_summary()}")

    def _detect_hardware(self) -> None:
        """Detect all hardware capabilities."""
        with self._detection_lock:
            # Try to load from cache first
            if self._load_from_cache():
                return

            # Detect CPU
            self._cpu_info = self._detect_cpu()

            # Detect GPU
            self._gpu_info = self._detect_gpu()

            # Detect memory
            self._memory_info = self._detect_memory()

            # Save to cache
            self._save_to_cache()

            self._last_update = time.time()

    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU configuration."""
        try:
            # Get CPU brand
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()

            # Get core counts
            physical = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"], text=True
                ).strip()
            )

            logical = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.logicalcpu"], text=True
                ).strip()
            )

            # Detect M4 Pro core configuration
            if "Apple M4" in brand:
                if physical == 12:
                    p_cores = 8
                    e_cores = 4
                elif physical == 14:
                    p_cores = 10
                    e_cores = 4
                else:
                    # Default M4 configuration
                    p_cores = max(8, physical - 4)
                    e_cores = 4
            else:
                # Non-M4 systems
                p_cores = max(4, physical // 2)
                e_cores = physical - p_cores

            # Get CPU frequency
            cpu_freq = psutil.cpu_freq()
            frequency_mhz = cpu_freq.current if cpu_freq else 0
            frequency_max_mhz = cpu_freq.max if cpu_freq else 0

            return CPUInfo(
                brand=brand,
                physical_cores=physical,
                logical_cores=logical,
                p_cores=p_cores,
                e_cores=e_cores,
                frequency_mhz=frequency_mhz,
                frequency_max_mhz=frequency_max_mhz,
            )

        except Exception as e:
            logger.warning(f"CPU detection failed: {e}, using defaults")
            return CPUInfo(
                brand="Unknown CPU",
                physical_cores=psutil.cpu_count(logical=False) or 8,
                logical_cores=psutil.cpu_count(logical=True) or 12,
                p_cores=8,
                e_cores=4,
            )

    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU configuration."""
        try:
            # Use system_profiler for GPU detection
            sp_output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"], text=True
            )
            sp_data = json.loads(sp_output)

            displays = sp_data.get("SPDisplaysDataType", [])
            if displays:
                gpu_data = displays[0]
                gpu_name = gpu_data.get("sppci_model", "Unknown GPU")

                # Detect M4 Pro GPU cores
                if "M4" in gpu_name or "Apple M4" in self._cpu_info.brand:
                    # M4 Pro has 16 or 20 GPU cores
                    if "20-core" in gpu_name or "Max" in gpu_name:
                        cores = 20
                    else:
                        cores = 16  # Default M4 Pro
                else:
                    cores = 8  # Default for other systems

                return GPUInfo(
                    name=gpu_name,
                    cores=cores,
                    metal_supported=True,
                    unified_memory=True,
                    vram_mb=0,  # Unified memory
                )
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}, using defaults")

        # Default M4 Pro GPU
        return GPUInfo(
            name="Apple M4 Pro GPU",
            cores=20,  # Assume higher-end model
            metal_supported=True,
            unified_memory=True,
            vram_mb=0,
        )

    def _detect_memory(self) -> MemoryInfo:
        """Detect memory configuration."""
        vm = psutil.virtual_memory()

        # Get page size
        try:
            vm_stat = subprocess.check_output(["vm_stat"], text=True)
            page_size = 16384  # Default
            for line in vm_stat.split("\n"):
                if "page size" in line:
                    import re

                    match = re.search(r"(\d+)", line)
                    if match:
                        page_size = int(match.group(1))
                    break
        except (subprocess.CalledProcessError, ValueError, AttributeError) as e:
            logger.debug(f"Could not get page size: {e}")
            page_size = 16384

        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
        used_gb = vm.used / (1024**3)

        # For Apple Silicon, Metal can use up to 75% of system memory
        metal_limit_gb = total_gb * 0.75 if self._is_apple_silicon() else 0

        return MemoryInfo(
            total_gb=round(total_gb, 1),
            available_gb=round(available_gb, 1),
            used_gb=round(used_gb, 1),
            percent_used=vm.percent,
            page_size=page_size,
            unified=self._is_apple_silicon(),
            metal_limit_gb=round(metal_limit_gb, 1),
        )

    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        try:
            arch = subprocess.check_output(["uname", "-m"], text=True).strip()
            return arch == "arm64"
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"Could not detect ARM64 architecture: {e}")
            return False

    def _load_from_cache(self) -> bool:
        """Load hardware info from cache if recent."""
        try:
            if not self._cache_file.exists():
                return False

            # Check cache age
            cache_age = time.time() - self._cache_file.stat().st_mtime
            if cache_age > self._cache_duration:
                return False

            with open(self._cache_file) as f:
                data = json.load(f)

            # Reconstruct dataclasses
            self._cpu_info = CPUInfo(**data["cpu"])
            self._gpu_info = GPUInfo(**data["gpu"])
            # Memory info is always refreshed
            self._memory_info = self._detect_memory()
            self._last_update = data["timestamp"]

            return True

        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save hardware info to cache."""
        try:
            data = {
                "cpu": self._cpu_info.__dict__,
                "gpu": self._gpu_info.__dict__,
                "timestamp": self._last_update,
            }

            with open(self._cache_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    def refresh(self) -> None:
        """Force refresh of hardware information."""
        self._detect_hardware()

    @property
    def cpu(self) -> CPUInfo:
        """Get CPU information."""
        if time.time() - self._last_update > self._cache_duration:
            self.refresh()
        return self._cpu_info

    @property
    def gpu(self) -> GPUInfo:
        """Get GPU information."""
        if time.time() - self._last_update > self._cache_duration:
            self.refresh()
        return self._gpu_info

    @property
    def gpu_backend(self) -> str:
        """Get the GPU backend type as a string."""
        if self._gpu_info is None:
            return "none"

        if self._gpu_info.metal_supported:
            # Check for MLX availability
            try:
                import importlib

                importlib.import_module("mlx.core")
                return "mlx"
            except ImportError:
                pass

            # Check for PyTorch MPS
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass

            # Generic Metal support
            return "metal"
        else:
            return "cpu"

    @property
    def memory(self) -> MemoryInfo:
        """Get current memory information (always fresh)."""
        self._memory_info = self._detect_memory()
        return self._memory_info

    def get_resource_budget(self, task_type: str = "general") -> ResourceBudget:
        """Get recommended resource allocation for task type.

        Args:
            task_type: Type of task ("cpu", "gpu", "io", "memory", "general")

        Returns:
            ResourceBudget with recommended allocations
        """
        mem = self.memory

        # Base calculations
        available_memory_mb = int(mem.available_gb * 1024)
        safe_memory_mb = int(available_memory_mb * 0.8)  # Leave 20% buffer

        if task_type == "cpu":
            # CPU-intensive: Use all P-cores
            return ResourceBudget(
                cpu_workers=self.cpu.p_cores,
                p_core_workers=self.cpu.p_cores,
                e_core_workers=0,
                gpu_workers=0,
                memory_pool_mb=safe_memory_mb,
                batch_size=self.cpu.p_cores * 4,
            )

        elif task_type == "gpu":
            # GPU-intensive: Minimal CPU, max GPU
            return ResourceBudget(
                cpu_workers=2,
                p_core_workers=2,
                e_core_workers=0,
                gpu_workers=min(4, self.gpu.cores // 4),
                memory_pool_mb=int(mem.metal_limit_gb * 1024 * 0.8),
                batch_size=256 if self.gpu.cores >= 16 else 128,
            )

        elif task_type == "io":
            # I/O bound: Use E-cores
            return ResourceBudget(
                cpu_workers=self.cpu.e_cores,
                p_core_workers=0,
                e_core_workers=self.cpu.e_cores,
                gpu_workers=0,
                memory_pool_mb=min(4096, safe_memory_mb),
                batch_size=self.cpu.e_cores * 8,
            )

        elif task_type == "memory":
            # Memory-intensive: Balance CPU and memory
            return ResourceBudget(
                cpu_workers=self.cpu.p_cores // 2,
                p_core_workers=self.cpu.p_cores // 2,
                e_core_workers=0,
                gpu_workers=0,
                memory_pool_mb=safe_memory_mb,
                batch_size=max(1, safe_memory_mb // 1024),
            )

        else:  # general
            # Balanced allocation
            return ResourceBudget(
                cpu_workers=self.cpu.p_cores - 2,
                p_core_workers=self.cpu.p_cores - 2,
                e_core_workers=self.cpu.e_cores,
                gpu_workers=1,
                memory_pool_mb=safe_memory_mb // 2,
                batch_size=self.cpu.physical_cores * 2,
            )

    def allocate_resources(
        self, agent_id: str, requested_memory_mb: int, task_type: str = "general"
    ) -> ResourceBudget | None:
        """Allocate resources for an agent with memory safety.

        Args:
            agent_id: Unique agent identifier
            requested_memory_mb: Memory requested by agent
            task_type: Type of task

        Returns:
            ResourceBudget if allocation successful, None if insufficient resources
        """
        mem = self.memory
        int(mem.available_gb * 1024)

        # Check total allocated memory
        total_allocated = sum(
            budget.memory_pool_mb for budget in self._resource_budgets.values()
        )

        # Prevent overcommit (max 85% of total memory)
        max_allowed = int(mem.total_gb * 1024 * 0.85)
        if total_allocated + requested_memory_mb > max_allowed:
            logger.warning(
                f"Memory allocation denied for {agent_id}: "
                f"would exceed 85% limit ({total_allocated + requested_memory_mb}MB > {max_allowed}MB)"
            )
            return None

        # Get base budget
        budget = self.get_resource_budget(task_type)
        budget.memory_pool_mb = min(requested_memory_mb, budget.memory_pool_mb)

        # Store allocation
        self._resource_budgets[agent_id] = budget

        return budget

    def release_resources(self, agent_id: str) -> None:
        """Release resources allocated to an agent."""
        if agent_id in self._resource_budgets:
            del self._resource_budgets[agent_id]
            logger.debug(f"Released resources for agent {agent_id}")

    def get_utilization(self) -> dict[str, float]:
        """Get current system utilization metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = self.memory

        # Estimate GPU utilization (would need metal_monitor integration for real values)
        gpu_percent = 0.0
        try:
            # Simple heuristic based on GPU-likely processes
            for proc in psutil.process_iter(["name", "memory_percent"]):
                name = proc.info["name"].lower()
                if any(
                    gpu_key in name
                    for gpu_key in ["python", "mlx", "tensorflow", "torch"]
                ):
                    if proc.info["memory_percent"] > 5:
                        gpu_percent += proc.info["memory_percent"] * 0.5
        except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Could not estimate GPU usage from processes: {e}")

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": mem.percent_used,
            "gpu_percent": min(100.0, gpu_percent),
            "allocated_agents": len(self._resource_budgets),
        }

    def get_summary(self) -> str:
        """Get hardware summary string."""
        return (
            f"M4 Pro: {self.cpu.p_cores}P+{self.cpu.e_cores}E cores, "
            f"{self.gpu.cores} GPU cores, {self.memory.total_gb}GB RAM"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert hardware state to dictionary."""
        return {
            "cpu": self.cpu.__dict__,
            "gpu": self.gpu.__dict__,
            "memory": self.memory.__dict__,
            "utilization": self.get_utilization(),
            "allocated_budgets": {
                agent_id: {
                    "memory_mb": budget.memory_pool_mb,
                    "cpu_workers": budget.cpu_workers,
                    "timestamp": budget.timestamp,
                }
                for agent_id, budget in self._resource_budgets.items()
            },
        }


# Convenience function for singleton access
def get_hardware_state() -> HardwareState:
    """Get the hardware state singleton instance."""
    return HardwareState()


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate hardware state usage."""
        # Get singleton instance
        hw = get_hardware_state()

        print("=== Hardware Detection ===")
        print(f"Summary: {hw.get_summary()}")
        print(f"\nCPU: {hw.cpu}")
        print(f"\nGPU: {hw.gpu}")
        print(f"\nMemory: {hw.memory}")

        print("\n=== Resource Budgets ===")
        for task_type in ["cpu", "gpu", "io", "memory", "general"]:
            budget = hw.get_resource_budget(task_type)
            print(f"\n{task_type.upper()} task budget:")
            print(
                f"  Workers: {budget.cpu_workers} ({budget.p_core_workers}P + {budget.e_core_workers}E)"
            )
            print(f"  GPU workers: {budget.gpu_workers}")
            print(f"  Memory: {budget.memory_pool_mb}MB")
            print(f"  Batch size: {budget.batch_size}")

        print("\n=== Agent Resource Allocation ===")
        # Simulate agent allocation
        agent1 = hw.allocate_resources("agent1", 4096, "cpu")
        print(f"Agent1 allocated: {agent1}")

        agent2 = hw.allocate_resources("agent2", 8192, "gpu")
        print(f"Agent2 allocated: {agent2}")

        # Show utilization
        print(f"\nCurrent utilization: {hw.get_utilization()}")

        # Release resources
        hw.release_resources("agent1")
        print(f"After releasing agent1: {hw.get_utilization()}")

        print("\n=== Performance Test ===")
        # Test access speed
        start = time.time()
        for _ in range(1000):
            _ = hw.cpu
            _ = hw.gpu
            _ = hw.memory
        elapsed = (time.time() - start) * 1000
        print(
            f"1000 hardware queries: {elapsed:.2f}ms ({elapsed/1000:.2f}ms per query)"
        )

    # Run demo
    asyncio.run(demo())

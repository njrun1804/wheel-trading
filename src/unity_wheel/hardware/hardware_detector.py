"""Unified Hardware Detection System.

Single source of truth for hardware capabilities detection across the system.
Provides <5ms access to hardware information with intelligent caching.
"""

import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Complete hardware information."""

    # CPU
    cpu_brand: str
    cpu_cores: int
    p_cores: int
    e_cores: int
    cpu_frequency_mhz: float

    # GPU
    gpu_name: str
    gpu_cores: int
    metal_supported: bool
    ane_cores: int

    # Memory
    memory_gb: float
    unified_memory: bool

    # System
    platform: str
    architecture: str
    macos_version: str

    def get_summary(self) -> str:
        """Get concise hardware summary."""
        return (
            f"{self.cpu_brand}: {self.p_cores}P+{self.e_cores}E cores, "
            f"{self.gpu_cores} GPU cores, {self.ane_cores} ANE cores, "
            f"{self.memory_gb}GB RAM"
        )


class HardwareDetector:
    """Singleton hardware detector with fast cached access.

    Consolidates hardware detection from multiple implementations into a single,
    efficient system with <5ms access times.
    """

    _instance = None
    _lock = Lock()
    _cache_file = Path.home() / ".unity_hardware_cache.json"
    _cache_duration = 3600  # 1 hour cache

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize hardware detection."""
        if self._initialized:
            return

        self._hardware_info: HardwareInfo | None = None
        self._detection_time: float = 0
        self._detect_lock = Lock()

        # Detect hardware
        self._detect_hardware()
        self._initialized = True

    def _detect_hardware(self) -> None:
        """Detect all hardware capabilities."""
        start_time = time.time()

        with self._detect_lock:
            # Try cache first
            if self._load_from_cache():
                detection_time = (time.time() - start_time) * 1000
                logger.debug(f"Hardware loaded from cache in {detection_time:.1f}ms")
                return

            # Detect components
            cpu_info = self._detect_cpu()
            gpu_info = self._detect_gpu()
            memory_info = self._detect_memory()
            system_info = self._detect_system()

            # Create hardware info
            self._hardware_info = HardwareInfo(
                # CPU
                cpu_brand=cpu_info[0],
                cpu_cores=cpu_info[1],
                p_cores=cpu_info[2],
                e_cores=cpu_info[3],
                cpu_frequency_mhz=cpu_info[4],
                # GPU
                gpu_name=gpu_info[0],
                gpu_cores=gpu_info[1],
                metal_supported=gpu_info[2],
                ane_cores=gpu_info[3],
                # Memory
                memory_gb=memory_info[0],
                unified_memory=memory_info[1],
                # System
                platform=system_info[0],
                architecture=system_info[1],
                macos_version=system_info[2],
            )

            # Save to cache
            self._save_to_cache()

            self._detection_time = time.time()
            detection_time = (time.time() - start_time) * 1000
            logger.info(
                f"Hardware detected in {detection_time:.1f}ms: {self._hardware_info.get_summary()}"
            )

    def _detect_cpu(self) -> tuple[str, int, int, int, float]:
        """Detect CPU configuration."""
        try:
            # CPU brand
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()

            # Core count
            physical = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"], text=True
                ).strip()
            )

            # Detect M-series configuration
            if "Apple M" in brand:
                if "M4 Pro" in brand:
                    if physical == 14:
                        p_cores, e_cores = 10, 4
                    else:  # 12 core model
                        p_cores, e_cores = 8, 4
                elif "M4 Max" in brand:
                    p_cores = physical - 4
                    e_cores = 4
                elif "M4" in brand:
                    p_cores, e_cores = 4, 4
                elif "M3" in brand or "M2" in brand or "M1" in brand:
                    # Simplified detection for older chips
                    if physical >= 10:
                        p_cores = physical - 2
                        e_cores = 2
                    else:
                        p_cores = 4
                        e_cores = physical - 4
                else:
                    # Default Apple Silicon
                    p_cores = max(4, physical // 2)
                    e_cores = physical - p_cores
            else:
                # Intel or unknown
                p_cores = physical
                e_cores = 0

            # CPU frequency
            try:
                freq = psutil.cpu_freq()
                frequency = freq.current if freq else 0.0
            except:
                frequency = 0.0

            return brand, physical, p_cores, e_cores, frequency

        except Exception as e:
            logger.warning(f"CPU detection error: {e}")
            return "Unknown CPU", 8, 8, 0, 0.0

    def _detect_gpu(self) -> tuple[str, int, bool, int]:
        """Detect GPU configuration."""
        try:
            # Check for Apple Silicon GPU
            cpu_brand = self._hardware_info.cpu_brand if self._hardware_info else ""

            if "M4 Pro" in cpu_brand:
                # M4 Pro variants
                if "20-core" in cpu_brand or "Max" in cpu_brand:
                    return "Apple M4 Pro GPU", 20, True, 16
                else:
                    return "Apple M4 Pro GPU", 16, True, 16
            elif "M4 Max" in cpu_brand:
                return "Apple M4 Max GPU", 40, True, 16
            elif "M4" in cpu_brand:
                return "Apple M4 GPU", 10, True, 16
            elif "Apple M" in cpu_brand:
                # Generic Apple Silicon
                return "Apple Silicon GPU", 8, True, 16
            else:
                # Try system_profiler for more info
                try:
                    sp_output = subprocess.check_output(
                        ["system_profiler", "SPDisplaysDataType", "-json"],
                        text=True,
                        timeout=2,
                    )
                    sp_data = json.loads(sp_output)
                    displays = sp_data.get("SPDisplaysDataType", [])
                    if displays:
                        gpu_name = displays[0].get("sppci_model", "Unknown GPU")
                        return gpu_name, 0, False, 0
                except:
                    pass

                return "Unknown GPU", 0, False, 0

        except Exception as e:
            logger.warning(f"GPU detection error: {e}")
            return "Unknown GPU", 0, False, 0

    def _detect_memory(self) -> tuple[float, bool]:
        """Detect memory configuration."""
        try:
            vm = psutil.virtual_memory()
            total_gb = round(vm.total / (1024**3), 1)

            # Check if unified memory (Apple Silicon)
            unified = "arm64" in platform.machine().lower()

            return total_gb, unified

        except Exception as e:
            logger.warning(f"Memory detection error: {e}")
            return 16.0, False

    def _detect_system(self) -> tuple[str, str, str]:
        """Detect system information."""
        try:
            # Platform
            system_platform = platform.system()

            # Architecture
            architecture = platform.machine()

            # macOS version
            if system_platform == "Darwin":
                version = platform.mac_ver()[0]
            else:
                version = platform.version()

            return system_platform, architecture, version

        except Exception as e:
            logger.warning(f"System detection error: {e}")
            return "Unknown", "Unknown", "Unknown"

    def _load_from_cache(self) -> bool:
        """Load hardware info from cache."""
        try:
            if not self._cache_file.exists():
                return False

            # Check cache age
            cache_age = time.time() - self._cache_file.stat().st_mtime
            if cache_age > self._cache_duration:
                return False

            with open(self._cache_file) as f:
                data = json.load(f)

            # Reconstruct HardwareInfo
            self._hardware_info = HardwareInfo(**data["hardware"])
            self._detection_time = data["timestamp"]

            return True

        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save hardware info to cache."""
        try:
            data = {
                "hardware": self._hardware_info.__dict__,
                "timestamp": self._detection_time,
            }

            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    @property
    def cpu_cores(self) -> int:
        """Get total CPU cores."""
        return self._hardware_info.cpu_cores

    @property
    def p_cores(self) -> int:
        """Get performance cores."""
        return self._hardware_info.p_cores

    @property
    def e_cores(self) -> int:
        """Get efficiency cores."""
        return self._hardware_info.e_cores

    @property
    def gpu_cores(self) -> int:
        """Get GPU cores."""
        return self._hardware_info.gpu_cores

    @property
    def ane_cores(self) -> int:
        """Get ANE cores."""
        return self._hardware_info.ane_cores

    @property
    def memory_gb(self) -> float:
        """Get total memory in GB."""
        return self._hardware_info.memory_gb

    @property
    def metal_supported(self) -> bool:
        """Check if Metal is supported."""
        return self._hardware_info.metal_supported

    @property
    def unified_memory(self) -> bool:
        """Check if system has unified memory."""
        return self._hardware_info.unified_memory

    def get_optimal_workers(self, task_type: str = "general") -> dict[str, int]:
        """Get optimal worker configuration for task type."""
        if task_type == "cpu":
            return {
                "total": self.p_cores,
                "p_cores": self.p_cores,
                "e_cores": 0,
                "gpu": 0,
            }
        elif task_type == "io":
            return {
                "total": self.cpu_cores,
                "p_cores": 2,
                "e_cores": self.e_cores,
                "gpu": 0,
            }
        elif task_type == "gpu":
            return {
                "total": 4,
                "p_cores": 2,
                "e_cores": 0,
                "gpu": min(4, self.gpu_cores // 4),
            }
        else:  # general
            return {
                "total": self.cpu_cores - 2,
                "p_cores": self.p_cores - 2,
                "e_cores": self.e_cores,
                "gpu": 1,
            }

    def get_memory_budget(self, component: str) -> float:
        """Get recommended memory budget for component in GB."""
        total = self.memory_gb

        budgets = {
            "duckdb": 0.4,  # 40% for DuckDB
            "jarvis": 0.15,  # 15% for Jarvis
            "einstein": 0.1,  # 10% for Einstein
            "metal": 0.2,  # 20% for Metal GPU
            "cache": 0.1,  # 10% for caches
            "other": 0.05,  # 5% buffer
        }

        percentage = budgets.get(component, 0.05)
        return round(total * percentage, 1)

    def get_info(self) -> HardwareInfo:
        """Get complete hardware information."""
        return self._hardware_info

    def get_summary(self) -> str:
        """Get hardware summary string."""
        return self._hardware_info.get_summary()

    def to_dict(self) -> dict[str, any]:
        """Convert to dictionary."""
        return {
            "hardware": self._hardware_info.__dict__,
            "detection_time": self._detection_time,
            "cache_file": str(self._cache_file),
        }

    def refresh(self) -> None:
        """Force refresh hardware detection."""
        self._detect_hardware()


# Performance test
if __name__ == "__main__":
    import asyncio

    async def test_performance():
        """Test hardware detection performance."""
        print("=== Hardware Detection Performance Test ===")

        # First detection (cold)
        start = time.time()
        hw1 = HardwareDetector()
        cold_time = (time.time() - start) * 1000
        print(f"\nCold detection: {cold_time:.1f}ms")
        print(f"Hardware: {hw1.get_summary()}")

        # Second detection (cached in memory)
        start = time.time()
        hw2 = HardwareDetector()
        warm_time = (time.time() - start) * 1000
        print(f"\nWarm detection: {warm_time:.1f}ms")

        # Access test
        access_times = []
        for _ in range(1000):
            start = time.time()
            _ = hw1.cpu_cores
            _ = hw1.gpu_cores
            _ = hw1.memory_gb
            access_times.append((time.time() - start) * 1000)

        avg_access = sum(access_times) / len(access_times)
        print(f"\nAverage access time (1000 queries): {avg_access:.3f}ms")

        # Show detailed info
        print("\n=== Detailed Hardware Info ===")
        info = hw1.get_info()
        for key, value in info.__dict__.items():
            print(f"{key}: {value}")

        # Test optimal workers
        print("\n=== Optimal Worker Configurations ===")
        for task in ["cpu", "gpu", "io", "general"]:
            workers = hw1.get_optimal_workers(task)
            print(f"{task}: {workers}")

        # Test memory budgets
        print("\n=== Memory Budgets ===")
        for component in ["duckdb", "jarvis", "einstein", "metal", "cache"]:
            budget = hw1.get_memory_budget(component)
            print(f"{component}: {budget}GB")

    asyncio.run(test_performance())

"""
Unified Hardware Detection - Single source of truth for hardware capabilities.

Consolidates hardware detection from:
- einstein/einstein_config.py HardwareDetector
- bolt/core/system_info.py 
- jarvis2/hardware/m4_detector.py
- src/unity_wheel/optimization/hardware_detector.py
"""

import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Unified hardware configuration."""

    # CPU
    cpu_cores: int
    cpu_performance_cores: int
    cpu_efficiency_cores: int
    cpu_threads: int

    # Memory
    memory_total_gb: float
    memory_available_gb: float

    # GPU
    has_gpu: bool
    gpu_cores: int

    # Platform
    platform_type: str  # 'apple_silicon', 'intel', 'amd', 'unknown'
    architecture: str  # 'arm64', 'x86_64', etc.
    os_type: str  # 'Darwin', 'Linux', 'Windows'

    # Optional fields with defaults
    unified_memory: bool = False
    gpu_name: str = "Unknown"
    gpu_memory_gb: float | None = None
    metal_supported: bool = False
    metal_limit_gb: float | None = None

    # Apple Neural Engine
    has_ane: bool = False
    ane_cores: int = 0

    # Platform - optional with defaults
    model_name: str = "Unknown"
    chip_name: str = "Unknown"

    # Capabilities
    mlx_available: bool = False
    torch_mps_available: bool = False
    cuda_available: bool = False

    def get_optimal_workers(self, task_type: str = "compute") -> int:
        """Get optimal number of workers for task type."""
        if task_type == "compute":
            return self.cpu_performance_cores
        elif task_type == "io":
            return self.cpu_cores * 2
        elif task_type == "mixed":
            return self.cpu_cores
        elif task_type == "gpu":
            return min(2, self.gpu_cores // 8) if self.has_gpu else 0
        else:
            return self.cpu_cores

    def get_memory_limit(self, fraction: float = 0.8) -> float:
        """Get recommended memory limit in GB."""
        return min(self.memory_available_gb * fraction, self.memory_total_gb * 0.9)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            # CPU
            "cpu_cores": self.cpu_cores,
            "cpu_performance_cores": self.cpu_performance_cores,
            "cpu_efficiency_cores": self.cpu_efficiency_cores,
            "cpu_threads": self.cpu_threads,
            # Memory
            "memory_total_gb": self.memory_total_gb,
            "memory_available_gb": self.memory_available_gb,
            "unified_memory": self.unified_memory,
            # GPU
            "has_gpu": self.has_gpu,
            "gpu_cores": self.gpu_cores,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "metal_supported": self.metal_supported,
            "metal_limit_gb": self.metal_limit_gb,
            # ANE
            "has_ane": self.has_ane,
            "ane_cores": self.ane_cores,
            # Platform
            "platform_type": self.platform_type,
            "architecture": self.architecture,
            "os_type": self.os_type,
            "model_name": self.model_name,
            "chip_name": self.chip_name,
            # Capabilities
            "mlx_available": self.mlx_available,
            "torch_mps_available": self.torch_mps_available,
            "cuda_available": self.cuda_available,
        }


class HardwareDetector:
    """Unified hardware detection implementation."""

    _cache: HardwareConfig | None = None

    @classmethod
    def detect_hardware(cls) -> HardwareConfig:
        """Detect current hardware configuration."""
        if cls._cache is not None:
            return cls._cache

        try:
            # Basic system info
            cpu_cores = psutil.cpu_count(logical=False) or 8
            cpu_threads = psutil.cpu_count(logical=True) or cpu_cores
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)

            # Platform detection
            os_type = platform.system()
            architecture = platform.machine().lower()
            platform_type = cls._detect_platform_type()

            # Apple Silicon specific detection
            if platform_type == "apple_silicon":
                cpu_info = cls._detect_apple_silicon_details()
                cpu_perf_cores = cpu_info["p_cores"]
                cpu_eff_cores = cpu_info["e_cores"]
                model_name = cpu_info["model_name"]
                chip_name = cpu_info["chip_name"]
            else:
                cpu_perf_cores = cpu_cores
                cpu_eff_cores = 0
                model_name = platform.node()
                chip_name = cls._detect_cpu_brand()

            # GPU detection
            gpu_info = cls._detect_gpu(platform_type)

            # ANE detection
            has_ane, ane_cores = cls._detect_ane(platform_type)

            # Check ML framework availability
            mlx_available = cls._check_mlx()
            torch_mps_available = cls._check_torch_mps()
            cuda_available = cls._check_cuda()

            config = HardwareConfig(
                # CPU
                cpu_cores=cpu_cores,
                cpu_performance_cores=cpu_perf_cores,
                cpu_efficiency_cores=cpu_eff_cores,
                cpu_threads=cpu_threads,
                # Memory
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb,
                unified_memory=(platform_type == "apple_silicon"),
                # GPU
                has_gpu=gpu_info["has_gpu"],
                gpu_cores=gpu_info["gpu_cores"],
                gpu_name=gpu_info["gpu_name"],
                gpu_memory_gb=gpu_info["gpu_memory_gb"],
                metal_supported=gpu_info["metal_supported"],
                metal_limit_gb=gpu_info.get("metal_limit_gb"),
                # ANE
                has_ane=has_ane,
                ane_cores=ane_cores,
                # Platform
                platform_type=platform_type,
                architecture=architecture,
                os_type=os_type,
                model_name=model_name,
                chip_name=chip_name,
                # Capabilities
                mlx_available=mlx_available,
                torch_mps_available=torch_mps_available,
                cuda_available=cuda_available,
            )

            cls._cache = config
            return config

        except Exception as e:
            logger.error(f"Hardware detection failed: {e}", exc_info=True)
            return cls._get_default_hardware()

    @staticmethod
    def _detect_platform_type() -> str:
        """Detect platform type."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "darwin" and "arm" in machine:
            return "apple_silicon"
        elif system == "darwin" and "x86" in machine:
            return "intel_mac"
        else:
            # Try to detect CPU brand
            try:
                cpu_info = subprocess.run(
                    ["cat", "/proc/cpuinfo"], capture_output=True, text=True
                )
                if cpu_info.returncode == 0:
                    if "intel" in cpu_info.stdout.lower():
                        return "intel"
                    elif "amd" in cpu_info.stdout.lower():
                        return "amd"
            except:
                pass

            return "unknown"

    @staticmethod
    def _detect_apple_silicon_details() -> dict[str, Any]:
        """Detect detailed Apple Silicon configuration."""
        info = {
            "p_cores": 8,
            "e_cores": 4,
            "model_name": "Mac",
            "chip_name": "Apple Silicon",
        }

        try:
            # Get CPU brand string
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                brand = result.stdout.strip()
                info["chip_name"] = brand

                # Extract chip model
                if "M4" in brand:
                    chip_family = "M4"
                elif "M3" in brand:
                    chip_family = "M3"
                elif "M2" in brand:
                    chip_family = "M2"
                elif "M1" in brand:
                    chip_family = "M1"
                else:
                    chip_family = "Apple Silicon"

                info["chip_name"] = chip_family

            # Get physical CPU count for core distribution
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                physical_cores = int(result.stdout.strip())

                # Try to get P-cores directly
                result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    info["p_cores"] = int(result.stdout.strip())
                    info["e_cores"] = physical_cores - info["p_cores"]
                else:
                    # Estimate based on known configurations
                    if physical_cores == 8:  # M1
                        info["p_cores"] = 4
                        info["e_cores"] = 4
                    elif physical_cores == 10:  # M1/M2 Pro
                        info["p_cores"] = 6
                        info["e_cores"] = 4
                    elif physical_cores == 12:  # M4 Pro, M2 Pro
                        info["p_cores"] = 8
                        info["e_cores"] = 4
                    elif physical_cores == 14:  # M4 Pro Max variant
                        info["p_cores"] = 10
                        info["e_cores"] = 4
                    elif physical_cores >= 16:  # Max variants
                        info["p_cores"] = max(10, physical_cores - 6)
                        info["e_cores"] = physical_cores - info["p_cores"]

            # Get model name
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Model Name:" in line:
                        info["model_name"] = line.split(":", 1)[1].strip()
                        break

        except Exception as e:
            logger.debug(f"Failed to detect Apple Silicon details: {e}")

        return info

    @staticmethod
    def _detect_cpu_brand() -> str:
        """Detect CPU brand on non-Apple systems."""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        return lines[1].strip()
        except:
            pass

        return platform.processor() or "Unknown CPU"

    @staticmethod
    def _detect_gpu(platform_type: str) -> dict[str, Any]:
        """Detect GPU capabilities."""
        gpu_info = {
            "has_gpu": False,
            "gpu_cores": 0,
            "gpu_name": "None",
            "gpu_memory_gb": None,
            "metal_supported": False,
            "metal_limit_gb": None,
        }

        if platform_type == "apple_silicon":
            try:
                # Check Metal GPU via system_profiler
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    sp_data = json.loads(result.stdout)
                    displays = sp_data.get("SPDisplaysDataType", [])
                    if displays:
                        gpu_data = displays[0]
                        gpu_name = gpu_data.get("sppci_model", "Apple GPU")
                        gpu_info["gpu_name"] = gpu_name
                        gpu_info["has_gpu"] = True
                        gpu_info["metal_supported"] = True

                        # Estimate GPU cores based on chip
                        if "M4" in gpu_name:
                            if "20-core" in gpu_name or "Max" in gpu_name:
                                gpu_info["gpu_cores"] = 20
                            else:
                                gpu_info["gpu_cores"] = 16
                        elif "M3" in gpu_name:
                            gpu_info["gpu_cores"] = 18
                        elif "M2" in gpu_name:
                            gpu_info["gpu_cores"] = 16
                        elif "M1" in gpu_name:
                            gpu_info["gpu_cores"] = 14
                        else:
                            gpu_info["gpu_cores"] = 16  # Conservative estimate

                        # Unified memory - Metal can use up to 75%
                        memory_gb = psutil.virtual_memory().total / (1024**3)
                        gpu_info["metal_limit_gb"] = round(memory_gb * 0.75, 1)
                        gpu_info["gpu_memory_gb"] = gpu_info["metal_limit_gb"]

            except Exception as e:
                logger.debug(f"Failed to detect Metal GPU: {e}")
                # Fallback for Apple Silicon
                gpu_info["has_gpu"] = True
                gpu_info["gpu_cores"] = 16
                gpu_info["gpu_name"] = "Apple GPU"
                gpu_info["metal_supported"] = True

        else:
            # Check for NVIDIA GPUs
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(handle).decode()
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    gpu_info["has_gpu"] = True
                    gpu_info["gpu_name"] = name
                    gpu_info["gpu_memory_gb"] = memory_info.total / (1024**3)
                    # Estimate cores (rough approximation)
                    gpu_info["gpu_cores"] = 1000  # Placeholder

            except ImportError:
                logger.debug("pynvml not available for NVIDIA detection")
            except Exception as e:
                logger.debug(f"Failed to detect NVIDIA GPU: {e}")

        return gpu_info

    @staticmethod
    def _detect_ane(platform_type: str) -> tuple[bool, int]:
        """Detect Apple Neural Engine availability."""
        if platform_type != "apple_silicon":
            return False, 0

        # All modern Apple Silicon has 16 ANE cores
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                cpu_brand = result.stdout.strip().lower()
                if any(chip in cpu_brand for chip in ["m1", "m2", "m3", "m4"]):
                    return True, 16
        except:
            pass

        # Check if MLX is available as a proxy for ANE
        try:
            import mlx.core as mx

            if hasattr(mx, "metal") and mx.metal.is_available():
                return True, 16
        except ImportError:
            pass

        return False, 0

    @staticmethod
    def _check_mlx() -> bool:
        """Check if MLX is available."""
        try:
            import mlx.core as mx

            return hasattr(mx, "metal") and mx.metal.is_available()
        except ImportError:
            return False

    @staticmethod
    def _check_torch_mps() -> bool:
        """Check if PyTorch MPS is available."""
        try:
            import torch

            return torch.backends.mps.is_available()
        except ImportError:
            return False

    @staticmethod
    def _check_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _get_default_hardware() -> HardwareConfig:
        """Get default hardware configuration as fallback."""
        return HardwareConfig(
            cpu_cores=8,
            cpu_performance_cores=6,
            cpu_efficiency_cores=2,
            cpu_threads=8,
            memory_total_gb=16.0,
            memory_available_gb=12.0,
            unified_memory=False,
            has_gpu=False,
            gpu_cores=0,
            gpu_name="None",
            gpu_memory_gb=None,
            metal_supported=False,
            metal_limit_gb=None,
            has_ane=False,
            ane_cores=0,
            platform_type="unknown",
            architecture="unknown",
            os_type=platform.system(),
            model_name="Unknown",
            chip_name="Unknown",
            mlx_available=False,
            torch_mps_available=False,
            cuda_available=False,
        )


def get_hardware_config() -> HardwareConfig:
    """Get the current hardware configuration."""
    return HardwareDetector.detect_hardware()


def reset_hardware_cache():
    """Reset the hardware detection cache."""
    HardwareDetector._cache = None


if __name__ == "__main__":
    # Test hardware detection
    print("🔍 Unified Hardware Detection")
    print("=" * 50)

    config = get_hardware_config()

    print(f"\n💻 Platform: {config.platform_type} ({config.os_type})")
    print(f"   Model: {config.model_name}")
    print(f"   Chip: {config.chip_name}")
    print(f"   Architecture: {config.architecture}")

    print("\n🧠 CPU:")
    print(f"   Total Cores: {config.cpu_cores} ({config.cpu_threads} threads)")
    print(f"   Performance Cores: {config.cpu_performance_cores}")
    print(f"   Efficiency Cores: {config.cpu_efficiency_cores}")

    print("\n💾 Memory:")
    print(f"   Total: {config.memory_total_gb:.1f} GB")
    print(f"   Available: {config.memory_available_gb:.1f} GB")
    print(f"   Unified Memory: {'✅' if config.unified_memory else '❌'}")

    print("\n🎮 GPU:")
    print(f"   Available: {'✅' if config.has_gpu else '❌'}")
    if config.has_gpu:
        print(f"   Name: {config.gpu_name}")
        print(f"   Cores: {config.gpu_cores}")
        if config.gpu_memory_gb:
            print(f"   Memory: {config.gpu_memory_gb:.1f} GB")
        print(f"   Metal Support: {'✅' if config.metal_supported else '❌'}")
        if config.metal_limit_gb:
            print(f"   Metal Limit: {config.metal_limit_gb:.1f} GB")

    print("\n🤖 ANE (Apple Neural Engine):")
    print(f"   Available: {'✅' if config.has_ane else '❌'}")
    if config.has_ane:
        print(f"   Cores: {config.ane_cores}")

    print("\n🚀 ML Frameworks:")
    print(f"   MLX: {'✅' if config.mlx_available else '❌'}")
    print(f"   PyTorch MPS: {'✅' if config.torch_mps_available else '❌'}")
    print(f"   CUDA: {'✅' if config.cuda_available else '❌'}")

    print("\n⚙️  Optimal Workers:")
    print(f"   Compute Tasks: {config.get_optimal_workers('compute')}")
    print(f"   I/O Tasks: {config.get_optimal_workers('io')}")
    print(f"   Mixed Tasks: {config.get_optimal_workers('mixed')}")
    print(f"   GPU Tasks: {config.get_optimal_workers('gpu')}")
    print(f"   Memory Limit: {config.get_memory_limit():.1f} GB")

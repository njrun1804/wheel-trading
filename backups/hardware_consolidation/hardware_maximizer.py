"""Hardware maximizer that ensures all available resources are utilized."""

import os
import platform
import subprocess
import psutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HardwareMaximizer:
    """Automatically maximizes hardware utilization for any task."""
    
    def __init__(self):
        self.hardware_info = self._detect_hardware()
        self._apply_maximum_performance()
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect all available hardware resources."""
        info = {
            "platform": platform.system(),
            "cpu_count": mp.cpu_count(),
            "cpu_freq_mhz": 0,
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "gpu_type": None,
            "gpu_cores": 0,
            "is_m4_pro": False
        }
        
        if platform.system() == "Darwin":
            try:
                # Get chip info
                chip = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
                info["chip"] = chip
                info["is_m4_pro"] = "M4 Pro" in chip or "M4" in chip
                
                # Get performance cores
                perf_cores = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True
                ).strip())
                eff_cores = int(subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel1.physicalcpu"], text=True
                ).strip())
                
                info["performance_cores"] = perf_cores
                info["efficiency_cores"] = eff_cores
                
                # GPU detection
                if info["is_m4_pro"]:
                    info["gpu_type"] = "Metal"
                    info["gpu_cores"] = 20  # M4 Pro has 20 GPU cores
                    
            except Exception as e:
                logger.warning(f"Hardware detection partial: {e}")
        
        return info
    
    def _apply_maximum_performance(self):
        """Apply all performance optimizations."""
        hw = self.hardware_info
        
        # CPU: Use ALL cores
        total_cores = str(hw["cpu_count"])
        os.environ["NUMBA_NUM_THREADS"] = total_cores
        os.environ["OMP_NUM_THREADS"] = total_cores
        os.environ["MKL_NUM_THREADS"] = total_cores
        os.environ["OPENBLAS_NUM_THREADS"] = total_cores
        os.environ["VECLIB_MAXIMUM_THREADS"] = total_cores
        os.environ["BLIS_NUM_THREADS"] = total_cores
        os.environ["TBB_NUM_THREADS"] = total_cores
        
        # Memory: Use as much as safely possible (80%)
        memory_limit_gb = int(hw["memory_gb"] * 0.8)
        os.environ["DUCKDB_MEMORY_LIMIT"] = f"{memory_limit_gb}GB"
        os.environ["ARROW_MEMORY_POOL"] = str(memory_limit_gb * 1024)  # MB
        
        # Python: Maximum optimization
        os.environ["PYTHONOPTIMIZE"] = "2"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["PYTHONHASHSEED"] = "0"
        
        # GPU: Enable all acceleration
        if hw["gpu_type"] == "Metal":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"
            os.environ["METAL_DEBUG_ERROR_MODE"] = "0"
            os.environ["MPS_PREFETCH_ENABLE"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["METAL_GPU_FORCE_COMPILE"] = "1"
            os.environ["MLX_GPU_MEMORY_LIMIT"] = str(int(hw["memory_gb"] * 0.5 * 1024))
            os.environ["MLX_DEFAULT_STREAM"] = "gpu"
            os.environ["USE_GPU_ACCELERATION"] = "true"
        
        # Parallel execution
        os.environ["WHEEL_PARALLEL_WORKERS"] = total_cores
        os.environ["WHEEL_BATCH_SIZE"] = "2048"  # Larger batches
        os.environ["WHEEL_PERFORMANCE_MODE"] = "maximum"
        
        # Memory allocation
        os.environ["MALLOC_MMAP_MAX_"] = "40960"
        os.environ["MALLOC_TRIM_THRESHOLD_"] = "131072"
        os.environ["MALLOC_ARENA_MAX"] = total_cores
        
        logger.info(f"⚡ Hardware maximized: {hw['cpu_count']} CPUs, "
                   f"{hw['memory_gb']:.1f}GB RAM, {hw['gpu_type'] or 'No GPU'}")
    
    def get_executors(self) -> Dict[str, Any]:
        """Get optimized executors for parallel work."""
        return {
            "thread_pool": ThreadPoolExecutor(max_workers=self.hardware_info["cpu_count"] * 2),
            "process_pool": ProcessPoolExecutor(max_workers=self.hardware_info["cpu_count"]),
            "cpu_count": self.hardware_info["cpu_count"],
            "optimal_batch_size": 2048 if self.hardware_info["is_m4_pro"] else 1024
        }
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """Get optimization parameters for algorithms."""
        hw = self.hardware_info
        
        if hw["is_m4_pro"]:
            return {
                "parallel_searches": hw["cpu_count"],
                "mcts_simulations": 10000,  # High for M4 Pro
                "batch_size": 2048,
                "use_gpu": True,
                "gpu_backend": "mlx",
                "memory_limit_gb": hw["memory_gb"] * 0.8,
                "cache_size_mb": 4096,
                "prefetch_size": 64,
                "vector_size": 512  # Larger vectors for M4 Pro
            }
        else:
            return {
                "parallel_searches": max(4, hw["cpu_count"] // 2),
                "mcts_simulations": 1000,
                "batch_size": 512,
                "use_gpu": False,
                "gpu_backend": None,
                "memory_limit_gb": hw["memory_gb"] * 0.5,
                "cache_size_mb": 1024,
                "prefetch_size": 16,
                "vector_size": 128
            }


# Global instance
_maximizer: Optional[HardwareMaximizer] = None


def get_hardware_maximizer() -> HardwareMaximizer:
    """Get or create the global hardware maximizer."""
    global _maximizer
    if _maximizer is None:
        _maximizer = HardwareMaximizer()
    return _maximizer


def maximize_hardware():
    """Convenience function to ensure hardware is maximized."""
    return get_hardware_maximizer()
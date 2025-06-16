#!/usr/bin/env python3
"""
Unified Hardware Abstraction Layer

Provides a consistent hardware interface for both Einstein and Bolt systems,
ensuring optimal resource utilization and preventing conflicts.

Key Features:
- M4 Pro hardware detection and optimization
- CPU core allocation (P-cores vs E-cores)
- Metal GPU acceleration management
- Memory pressure monitoring
- Unified configuration interface
- Cross-system resource sharing
"""

import asyncio
import logging
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfiguration:
    """Unified hardware configuration."""
    
    # CPU Configuration
    cpu_brand: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    p_cores: int = 0  # Performance cores
    e_cores: int = 0  # Efficiency cores
    cpu_frequency_ghz: float = 0.0
    
    # GPU Configuration
    gpu_name: str = ""
    gpu_cores: int = 0
    metal_supported: bool = False
    unified_memory: bool = False
    
    # Memory Configuration
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    memory_pressure_threshold: float = 0.8
    
    # Platform Information
    platform_type: str = ""
    os_version: str = ""
    is_m4_pro: bool = False
    
    # Resource Allocation
    max_parallel_threads: int = 0
    max_concurrent_tasks: int = 0
    memory_pool_size_mb: int = 0


@dataclass
class ResourceAllocation:
    """Current resource allocation."""
    
    component_name: str
    cpu_cores_allocated: int = 0
    gpu_cores_allocated: int = 0
    memory_allocated_mb: int = 0
    priority: int = 0  # 0=highest, 5=lowest
    active: bool = False
    last_updated: float = field(default_factory=time.time)


class UnifiedHardwareAbstraction:
    """Unified hardware abstraction layer for Einstein and Bolt."""
    
    _instance: Optional['UnifiedHardwareAbstraction'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.config = HardwareConfiguration()
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Initialize hardware detection
        self._detect_hardware()
        
        logger.info(f"🔧 Unified Hardware Abstraction initialized")
        logger.info(f"   Platform: {self.config.platform_type}")
        logger.info(f"   CPU: {self.config.cpu_brand} ({self.config.p_cores}P+{self.config.e_cores}E cores)")
        logger.info(f"   GPU: {self.config.gpu_name} ({self.config.gpu_cores} cores)")
        logger.info(f"   Memory: {self.config.total_memory_gb:.1f}GB")
    
    def _detect_hardware(self):
        """Detect hardware configuration."""
        try:
            # Platform detection
            self.config.platform_type = platform.system()
            self.config.os_version = platform.release()
            
            # CPU detection
            self.config.physical_cores = psutil.cpu_count(logical=False)
            self.config.logical_cores = psutil.cpu_count(logical=True)
            
            # Try to get CPU brand
            try:
                if self.config.platform_type == "Darwin":
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True, timeout=5
                    )
                    self.config.cpu_brand = result.stdout.strip()
                    
                    # Detect M4 Pro
                    if "M4 Pro" in self.config.cpu_brand:
                        self.config.is_m4_pro = True
                        self.config.p_cores = 12  # M4 Pro has 12 P-cores
                        self.config.e_cores = 0   # No E-cores in M4 Pro
                    elif "M4" in self.config.cpu_brand:
                        self.config.p_cores = 8   # Regular M4
                        self.config.e_cores = 4
                    else:
                        # Default assumption for Apple Silicon
                        self.config.p_cores = max(1, self.config.physical_cores // 2)
                        self.config.e_cores = self.config.physical_cores - self.config.p_cores
                        
            except Exception as e:
                logger.warning(f"Could not detect CPU brand: {e}")
                self.config.cpu_brand = "Unknown"
                self.config.p_cores = self.config.physical_cores
                self.config.e_cores = 0
            
            # GPU detection
            self._detect_gpu()
            
            # Memory detection
            memory = psutil.virtual_memory()
            self.config.total_memory_gb = memory.total / (1024**3)
            self.config.available_memory_gb = memory.available / (1024**3)
            
            # Calculate optimal resource allocation
            self._calculate_optimal_allocation()
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            self._set_fallback_config()
    
    def _detect_gpu(self):
        """Detect GPU configuration."""
        try:
            if self.config.platform_type == "Darwin":
                # Try to detect Metal GPU
                try:
                    result = subprocess.run(
                        ["system_profiler", "SPDisplaysDataType"],
                        capture_output=True, text=True, timeout=10
                    )
                    output = result.stdout
                    
                    if "M4 Pro" in output:
                        self.config.gpu_name = "M4 Pro GPU"
                        self.config.gpu_cores = 20  # M4 Pro has 20 GPU cores
                        self.config.metal_supported = True
                        self.config.unified_memory = True
                    elif "M4" in output:
                        self.config.gpu_name = "M4 GPU"
                        self.config.gpu_cores = 10  # Regular M4 has 10 GPU cores
                        self.config.metal_supported = True
                        self.config.unified_memory = True
                    else:
                        # Generic Metal detection
                        self.config.gpu_name = "Metal GPU"
                        self.config.gpu_cores = 8  # Conservative estimate
                        self.config.metal_supported = True
                        self.config.unified_memory = True
                        
                except subprocess.TimeoutExpired:
                    logger.warning("GPU detection timed out")
                    self.config.gpu_name = "Unknown GPU"
                    self.config.metal_supported = False
                    
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self.config.gpu_name = "Unknown GPU"
            self.config.metal_supported = False
    
    def _calculate_optimal_allocation(self):
        """Calculate optimal resource allocation parameters."""
        # Max parallel threads (leave some cores for system)
        available_cores = max(1, self.config.physical_cores - 2)
        self.config.max_parallel_threads = available_cores
        
        # Max concurrent tasks (based on memory and cores)
        memory_factor = int(self.config.total_memory_gb / 4)  # 4GB per task
        core_factor = self.config.physical_cores
        self.config.max_concurrent_tasks = min(memory_factor, core_factor, 12)
        
        # Memory pool size (80% of available memory)
        self.config.memory_pool_size_mb = int(
            self.config.total_memory_gb * 0.8 * 1024
        )
    
    def _set_fallback_config(self):
        """Set fallback configuration."""
        self.config.cpu_brand = "Unknown CPU"
        self.config.physical_cores = 4
        self.config.logical_cores = 8
        self.config.p_cores = 4
        self.config.e_cores = 0
        self.config.gpu_name = "Unknown GPU"
        self.config.gpu_cores = 4
        self.config.total_memory_gb = 8.0
        self.config.max_parallel_threads = 4
        self.config.max_concurrent_tasks = 4
        self.config.memory_pool_size_mb = 6144
    
    async def allocate_resources(
        self, 
        component_name: str, 
        cpu_cores: int = 0, 
        gpu_cores: int = 0,
        memory_mb: int = 0,
        priority: int = 3
    ) -> bool:
        """Allocate hardware resources to a component."""
        try:
            # Check if allocation is possible
            if not self._can_allocate(cpu_cores, gpu_cores, memory_mb):
                return False
            
            # Create allocation
            allocation = ResourceAllocation(
                component_name=component_name,
                cpu_cores_allocated=cpu_cores,
                gpu_cores_allocated=gpu_cores,
                memory_allocated_mb=memory_mb,
                priority=priority,
                active=True
            )
            
            self.allocations[component_name] = allocation
            logger.info(f"✅ Allocated resources to {component_name}: "
                       f"{cpu_cores} CPU cores, {gpu_cores} GPU cores, {memory_mb}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Resource allocation failed for {component_name}: {e}")
            return False
    
    def _can_allocate(self, cpu_cores: int, gpu_cores: int, memory_mb: int) -> bool:
        """Check if resources can be allocated."""
        # Calculate current allocations
        current_cpu = sum(a.cpu_cores_allocated for a in self.allocations.values() if a.active)
        current_gpu = sum(a.gpu_cores_allocated for a in self.allocations.values() if a.active)
        current_memory = sum(a.memory_allocated_mb for a in self.allocations.values() if a.active)
        
        # Check limits
        if current_cpu + cpu_cores > self.config.max_parallel_threads:
            return False
        if current_gpu + gpu_cores > self.config.gpu_cores:
            return False
        if current_memory + memory_mb > self.config.memory_pool_size_mb:
            return False
        
        return True
    
    async def deallocate_resources(self, component_name: str) -> bool:
        """Deallocate resources from a component."""
        try:
            if component_name in self.allocations:
                del self.allocations[component_name]
                logger.info(f"🔄 Deallocated resources from {component_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Resource deallocation failed for {component_name}: {e}")
            return False
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        active_allocations = [a for a in self.allocations.values() if a.active]
        
        total_cpu = sum(a.cpu_cores_allocated for a in active_allocations)
        total_gpu = sum(a.gpu_cores_allocated for a in active_allocations)
        total_memory = sum(a.memory_allocated_mb for a in active_allocations)
        
        return {
            "cpu_usage": {
                "allocated": total_cpu,
                "available": self.config.max_parallel_threads,
                "utilization": total_cpu / self.config.max_parallel_threads if self.config.max_parallel_threads > 0 else 0
            },
            "gpu_usage": {
                "allocated": total_gpu,
                "available": self.config.gpu_cores,
                "utilization": total_gpu / self.config.gpu_cores if self.config.gpu_cores > 0 else 0
            },
            "memory_usage": {
                "allocated_mb": total_memory,
                "available_mb": self.config.memory_pool_size_mb,
                "utilization": total_memory / self.config.memory_pool_size_mb if self.config.memory_pool_size_mb > 0 else 0
            },
            "active_components": len(active_allocations),
            "allocations": {a.component_name: {
                "cpu_cores": a.cpu_cores_allocated,
                "gpu_cores": a.gpu_cores_allocated,
                "memory_mb": a.memory_allocated_mb,
                "priority": a.priority
            } for a in active_allocations}
        }
    
    def get_optimal_einstein_config(self) -> Dict[str, Any]:
        """Get optimal configuration for Einstein system."""
        return {
            "cpu_cores": min(8, self.config.p_cores),  # Use up to 8 P-cores
            "gpu_cores": min(16, self.config.gpu_cores),  # Use up to 16 GPU cores
            "memory_mb": min(8192, self.config.memory_pool_size_mb // 2),  # Use up to half memory
            "max_concurrent_searches": self.config.max_concurrent_tasks,
            "use_metal_acceleration": self.config.metal_supported,
            "embedding_batch_size": 64 if self.config.is_m4_pro else 32,
            "faiss_index_type": "HNSW" if self.config.is_m4_pro else "IVF"
        }
    
    def get_optimal_bolt_config(self) -> Dict[str, Any]:
        """Get optimal configuration for Bolt system."""
        return {
            "cpu_cores": min(12, self.config.p_cores),  # Use up to 12 P-cores
            "gpu_cores": min(20, self.config.gpu_cores),  # Use up to 20 GPU cores
            "memory_mb": min(12288, self.config.memory_pool_size_mb // 2),  # Use up to half memory
            "max_agents": 8 if self.config.is_m4_pro else 4,
            "use_metal_acceleration": self.config.metal_supported,
            "parallel_task_limit": self.config.max_concurrent_tasks,
            "memory_pool_enabled": True,
            "thermal_throttling": True
        }
    
    async def start_monitoring(self):
        """Start hardware monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("📊 Hardware monitoring started")
    
    async def stop_monitoring(self):
        """Stop hardware monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("📊 Hardware monitoring stopped")
    
    async def _monitoring_loop(self):
        """Hardware monitoring loop."""
        try:
            while self.monitoring_active:
                # Update current memory status
                memory = psutil.virtual_memory()
                self.config.available_memory_gb = memory.available / (1024**3)
                
                # Check for memory pressure
                memory_usage = 1.0 - (memory.available / memory.total)
                if memory_usage > self.config.memory_pressure_threshold:
                    logger.warning(f"⚠️  Memory pressure detected: {memory_usage:.1%}")
                    await self._handle_memory_pressure()
                
                # Check CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                if cpu_usage > 90:
                    logger.warning(f"⚠️  High CPU usage detected: {cpu_usage:.1f}%")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Hardware monitoring error: {e}")
    
    async def _handle_memory_pressure(self):
        """Handle memory pressure by optimizing allocations."""
        try:
            # Find lowest priority allocations
            sorted_allocations = sorted(
                self.allocations.values(),
                key=lambda a: a.priority,
                reverse=True
            )
            
            # Reduce memory allocation for lowest priority components
            for allocation in sorted_allocations[:2]:  # Reduce top 2 lowest priority
                if allocation.memory_allocated_mb > 512:
                    reduced_memory = allocation.memory_allocated_mb * 0.8
                    allocation.memory_allocated_mb = int(reduced_memory)
                    logger.info(f"🔄 Reduced memory allocation for {allocation.component_name} "
                               f"to {allocation.memory_allocated_mb}MB due to memory pressure")
                    
        except Exception as e:
            logger.error(f"Memory pressure handling failed: {e}")
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get complete hardware information."""
        return {
            "cpu": {
                "brand": self.config.cpu_brand,
                "physical_cores": self.config.physical_cores,
                "logical_cores": self.config.logical_cores,
                "p_cores": self.config.p_cores,
                "e_cores": self.config.e_cores,
                "frequency_ghz": self.config.cpu_frequency_ghz
            },
            "gpu": {
                "name": self.config.gpu_name,
                "cores": self.config.gpu_cores,
                "metal_supported": self.config.metal_supported,
                "unified_memory": self.config.unified_memory
            },
            "memory": {
                "total_gb": self.config.total_memory_gb,
                "available_gb": self.config.available_memory_gb,
                "pressure_threshold": self.config.memory_pressure_threshold
            },
            "platform": {
                "type": self.config.platform_type,
                "os_version": self.config.os_version,
                "is_m4_pro": self.config.is_m4_pro
            },
            "optimization": {
                "max_parallel_threads": self.config.max_parallel_threads,
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "memory_pool_size_mb": self.config.memory_pool_size_mb
            }
        }


# Global instance
_hardware_abstraction = None

def get_hardware_abstraction() -> UnifiedHardwareAbstraction:
    """Get the global hardware abstraction instance."""
    global _hardware_abstraction
    if _hardware_abstraction is None:
        _hardware_abstraction = UnifiedHardwareAbstraction()
    return _hardware_abstraction


async def initialize_hardware_abstraction() -> UnifiedHardwareAbstraction:
    """Initialize and return the hardware abstraction layer."""
    hardware = get_hardware_abstraction()
    await hardware.start_monitoring()
    return hardware


if __name__ == "__main__":
    async def main():
        hardware = await initialize_hardware_abstraction()
        
        print("Hardware Information:")
        print("=" * 50)
        
        info = hardware.get_hardware_info()
        for category, details in info.items():
            print(f"\n{category.upper()}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        print("\nOptimal Configurations:")
        print("=" * 50)
        
        einstein_config = hardware.get_optimal_einstein_config()
        print(f"\nEinstein: {einstein_config}")
        
        bolt_config = hardware.get_optimal_bolt_config()
        print(f"\nBolt: {bolt_config}")
        
        # Test resource allocation
        print("\nTesting Resource Allocation:")
        print("=" * 50)
        
        await hardware.allocate_resources("Einstein", cpu_cores=4, gpu_cores=8, memory_mb=4096)
        await hardware.allocate_resources("Bolt", cpu_cores=6, gpu_cores=12, memory_mb=6144)
        
        usage = hardware.get_resource_usage()
        print(f"\nResource Usage: {usage}")
        
        await hardware.stop_monitoring()
    
    asyncio.run(main())
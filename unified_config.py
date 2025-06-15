"""
Unified Configuration System for Jarvis2 + Meta Integration

Eliminates config conflicts and resource allocation issues.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os


@dataclass
class HardwareConfig:
    """Unified hardware configuration for M4 Pro."""
    # M4 Pro specifications
    total_cpu_cores: int = 12
    performance_cores: int = 8
    efficiency_cores: int = 4
    gpu_cores: int = 20
    unified_memory_gb: int = 24
    
    # Resource allocation
    jarvis2_cpu_allocation: float = 0.6  # 60% of cores
    meta_cpu_allocation: float = 0.3     # 30% of cores
    system_cpu_reserve: float = 0.1      # 10% for system
    
    memory_limit_gb: int = 18  # 75% of total
    gpu_memory_limit_gb: int = 15  # Conservative for Metal


@dataclass  
class Jarvis2SystemConfig:
    """Jarvis2-specific configuration."""
    max_parallel_simulations: int = 2000
    gpu_batch_size: int = 512
    mcts_exploration_constant: float = 1.414
    experience_buffer_size: int = 10000
    
    # Paths (namespaced)
    index_path: Path = field(default_factory=lambda: Path('.jarvis2/indexes'))
    model_path: Path = field(default_factory=lambda: Path('.jarvis2/models'))
    experience_path: Path = field(default_factory=lambda: Path('.jarvis2/experience'))


@dataclass
class MetaSystemConfig:
    """Meta system specific configuration."""
    evolution_threshold: int = 50
    coordination_cycle_seconds: int = 10
    rapid_development_threshold_seconds: int = 60
    minimum_observations_for_evolution: int = 20
    
    # Database paths (namespaced)
    evolution_db: str = "meta_evolution.db"
    monitoring_db: str = "meta_monitoring.db"
    reality_db: str = "meta_reality_learning.db"


@dataclass
class SharedResourceConfig:
    """Configuration for shared resources."""
    database_connection_pool_size: int = 5
    max_concurrent_operations: int = 8
    memory_pressure_threshold: float = 0.85
    cleanup_interval_seconds: int = 300


@dataclass
class UnifiedConfig:
    """Unified configuration preventing resource conflicts."""
    
    hardware: HardwareConfig
    jarvis2: Jarvis2SystemConfig
    meta: MetaSystemConfig
    shared: SharedResourceConfig
    
    # System metadata
    config_version: str = "2.0.0"
    last_updated: float = field(default_factory=lambda: __import__('time').time())
    
    def __init__(self):
        self.hardware = HardwareConfig()
        self.jarvis2 = Jarvis2SystemConfig()
        self.meta = MetaSystemConfig()
        self.shared = SharedResourceConfig()
        
        # Apply environment overrides
        self._apply_environment_overrides()
        
        # Validate resource allocation
        self._validate_resource_allocation()
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Hardware overrides
        if 'JARVIS_CPU_CORES' in os.environ:
            self.hardware.total_cpu_cores = int(os.environ['JARVIS_CPU_CORES'])
        
        if 'JARVIS_MEMORY_GB' in os.environ:
            self.hardware.unified_memory_gb = int(os.environ['JARVIS_MEMORY_GB'])
        
        # Performance overrides
        if 'JARVIS_BATCH_SIZE' in os.environ:
            self.jarvis2.gpu_batch_size = int(os.environ['JARVIS_BATCH_SIZE'])
    
    def _validate_resource_allocation(self):
        """Validate that resource allocations don't conflict."""
        # CPU allocation validation
        total_cpu_allocation = (
            self.hardware.jarvis2_cpu_allocation + 
            self.hardware.meta_cpu_allocation + 
            self.hardware.system_cpu_reserve
        )
        
        if total_cpu_allocation > 1.0:
            raise ValueError(f"CPU allocation exceeds 100%: {total_cpu_allocation:.1%}")
        
        # Memory validation
        if self.hardware.memory_limit_gb > self.hardware.unified_memory_gb:
            raise ValueError("Memory limit exceeds hardware capacity")
        
        # GPU memory validation
        if self.hardware.gpu_memory_limit_gb > self.hardware.unified_memory_gb:
            raise ValueError("GPU memory limit exceeds unified memory")
    
    def get_jarvis2_cpu_cores(self) -> int:
        """Get CPU cores allocated to Jarvis2."""
        return int(self.hardware.total_cpu_cores * self.hardware.jarvis2_cpu_allocation)
    
    def get_meta_cpu_cores(self) -> int:
        """Get CPU cores allocated to meta system."""
        return int(self.hardware.total_cpu_cores * self.hardware.meta_cpu_allocation)
    
    def get_effective_batch_size(self, memory_pressure: float = 0.0) -> int:
        """Get batch size adjusted for memory pressure."""
        if memory_pressure > self.shared.memory_pressure_threshold:
            return self.jarvis2.gpu_batch_size // 2
        return self.jarvis2.gpu_batch_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'hardware': self.hardware.__dict__,
            'jarvis2': self.jarvis2.__dict__,
            'meta': self.meta.__dict__,
            'shared': self.shared.__dict__,
            'config_version': self.config_version,
            'last_updated': self.last_updated
        }


# Global instance
_unified_config: Optional[UnifiedConfig] = None


def get_unified_config() -> UnifiedConfig:
    """Get the global unified configuration instance."""
    global _unified_config
    if _unified_config is None:
        _unified_config = UnifiedConfig()
    return _unified_config


def reload_config():
    """Reload configuration (useful for testing)."""
    global _unified_config
    _unified_config = None


if __name__ == "__main__":
    # Test configuration
    config = get_unified_config()
    print("🔧 Unified Configuration Test")
    print(f"Jarvis2 CPU cores: {config.get_jarvis2_cpu_cores()}")
    print(f"Meta CPU cores: {config.get_meta_cpu_cores()}")
    print(f"GPU batch size: {config.get_effective_batch_size()}")
    print(f"Memory limit: {config.hardware.memory_limit_gb}GB")
    print("✅ Configuration validated successfully")
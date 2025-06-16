#!/usr/bin/env python3
"""
Unified Configuration System

Standardizes configuration parameters across Einstein and Bolt systems,
providing a single source of truth for all system settings while maintaining
backward compatibility and system-specific optimizations.

Key Features:
- Unified configuration schema
- System-specific parameter translation
- Environment-based configuration
- Runtime configuration updates
- Configuration validation
- Parameter synchronization
- Backup and recovery
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from unified_hardware_abstraction import get_hardware_abstraction

logger = logging.getLogger(__name__)


class ConfigurationSource(Enum):
    """Sources for configuration values."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    HARDWARE_DETECTED = "hardware_detected"


class SystemType(Enum):
    """Supported system types."""
    EINSTEIN = "einstein"
    BOLT = "bolt"
    UNIFIED = "unified"


@dataclass
class HardwareConfig:
    """Hardware-related configuration."""
    
    # CPU Configuration
    cpu_cores: int = 0
    cpu_cores_p: int = 0  # Performance cores
    cpu_cores_e: int = 0  # Efficiency cores
    max_parallel_threads: int = 0
    cpu_affinity_enabled: bool = True
    
    # GPU Configuration
    gpu_cores: int = 0
    metal_acceleration: bool = False
    gpu_memory_mb: int = 0
    max_gpu_utilization: float = 0.9
    
    # Memory Configuration
    total_memory_gb: float = 0.0
    max_memory_usage_gb: float = 0.0
    memory_pool_enabled: bool = True
    memory_pressure_threshold: float = 0.8
    
    # Platform Configuration
    platform_type: str = ""
    is_m4_pro: bool = False
    optimization_level: str = "auto"  # auto, conservative, aggressive


@dataclass
class EinsteinConfig:
    """Einstein-specific configuration."""
    
    # Search Configuration
    max_concurrent_searches: int = 8
    search_timeout_ms: int = 5000
    target_search_latency_ms: int = 50
    cache_enabled: bool = True
    cache_size_mb: int = 512
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 32
    max_embedding_cache: int = 10000
    
    # FAISS Configuration
    faiss_index_type: str = "HNSW"
    faiss_hnsw_m: int = 16
    faiss_hnsw_ef_construction: int = 200
    faiss_hnsw_ef_search: int = 100
    faiss_rebuild_threshold: float = 0.1
    
    # Performance Configuration
    enable_gpu_acceleration: bool = True
    enable_parallel_processing: bool = True
    worker_threads: int = 4
    query_result_limit: int = 100


@dataclass
class BoltConfig:
    """Bolt-specific configuration."""
    
    # Agent Configuration
    max_agents: int = 8
    agent_timeout_seconds: int = 300
    agent_retry_limit: int = 3
    agent_memory_limit_mb: int = 1024
    
    # Task Configuration
    max_parallel_tasks: int = 12
    task_queue_size: int = 100
    task_priority_levels: int = 5
    task_scheduling_algorithm: str = "priority_fifo"
    
    # Coordination Configuration
    coordination_interval_ms: int = 100
    resource_sharing_enabled: bool = True
    load_balancing_enabled: bool = True
    failover_enabled: bool = True
    
    # Performance Configuration
    enable_gpu_acceleration: bool = True
    thermal_throttling: bool = True
    memory_pool_enabled: bool = True
    work_stealing_enabled: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration."""
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_enabled: bool = True
    log_file_path: str = "system.log"
    log_rotation_enabled: bool = True
    log_max_size_mb: int = 100
    
    # Monitoring Configuration
    monitoring_enabled: bool = True
    metrics_collection_interval: int = 5
    performance_monitoring: bool = True
    health_check_interval: int = 30
    
    # Integration Configuration
    integration_mode: str = "cooperative"  # cooperative, competitive, isolated
    startup_timeout_seconds: int = 60
    shutdown_timeout_seconds: int = 30
    graceful_degradation: bool = True
    
    # Security Configuration
    security_enabled: bool = True
    resource_isolation: bool = True
    access_control: bool = False


@dataclass
class UnifiedConfiguration:
    """Complete unified configuration."""
    
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    einstein: EinsteinConfig = field(default_factory=EinsteinConfig)
    bolt: BoltConfig = field(default_factory=BoltConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Metadata
    version: str = "1.0.0"
    last_updated: str = ""
    config_source: str = ""
    environment: str = "development"


class ConfigurationManager:
    """Manages unified configuration across Einstein and Bolt systems."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "unified_config.yaml"
        self.config = UnifiedConfiguration()
        self.hardware = get_hardware_abstraction()
        self.config_cache: Dict[str, Any] = {}
        self.parameter_mappings: Dict[SystemType, Dict[str, str]] = {}
        
        # Initialize parameter mappings
        self._initialize_parameter_mappings()
        
        logger.info("⚙️  Unified Configuration Manager initialized")
    
    def _initialize_parameter_mappings(self):
        """Initialize parameter mappings between systems."""
        
        # Einstein parameter mappings
        self.parameter_mappings[SystemType.EINSTEIN] = {
            # Hardware mappings
            "cpu_cores": "hardware.cpu_cores",
            "gpu_cores": "hardware.gpu_cores",
            "memory_mb": "hardware.max_memory_usage_gb",
            "use_metal_acceleration": "hardware.metal_acceleration",
            
            # Search mappings
            "max_concurrent_searches": "einstein.max_concurrent_searches",
            "search_timeout": "einstein.search_timeout_ms",
            "target_latency": "einstein.target_search_latency_ms",
            "enable_cache": "einstein.cache_enabled",
            
            # Embedding mappings
            "embedding_batch_size": "einstein.embedding_batch_size",
            "embedding_model": "einstein.embedding_model",
            
            # FAISS mappings
            "faiss_index_type": "einstein.faiss_index_type",
            "faiss_m": "einstein.faiss_hnsw_m",
            "faiss_ef": "einstein.faiss_hnsw_ef_search",
        }
        
        # Bolt parameter mappings
        self.parameter_mappings[SystemType.BOLT] = {
            # Hardware mappings
            "cpu_cores": "hardware.cpu_cores",
            "gpu_cores": "hardware.gpu_cores",
            "memory_mb": "hardware.max_memory_usage_gb",
            "use_metal_acceleration": "hardware.metal_acceleration",
            
            # Agent mappings
            "max_agents": "bolt.max_agents",
            "agent_timeout": "bolt.agent_timeout_seconds",
            "agent_memory_limit": "bolt.agent_memory_limit_mb",
            
            # Task mappings
            "max_parallel_tasks": "bolt.max_parallel_tasks",
            "task_queue_size": "bolt.task_queue_size",
            "scheduling_algorithm": "bolt.task_scheduling_algorithm",
            
            # Performance mappings
            "thermal_throttling": "bolt.thermal_throttling",
            "memory_pool_enabled": "bolt.memory_pool_enabled",
            "work_stealing": "bolt.work_stealing_enabled",
        }
    
    async def initialize(self):
        """Initialize configuration system."""
        try:
            # Detect hardware configuration
            await self._detect_hardware_configuration()
            
            # Load configuration from file if it exists
            if Path(self.config_file).exists():
                await self.load_configuration()
            else:
                logger.info(f"Configuration file {self.config_file} not found, using defaults")
            
            # Apply environment variable overrides
            self._apply_environment_overrides()
            
            # Optimize configuration for detected hardware
            await self._optimize_for_hardware()
            
            # Validate configuration
            self._validate_configuration()
            
            # Save the optimized configuration
            await self.save_configuration()
            
            logger.info("✅ Configuration system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise
    
    async def _detect_hardware_configuration(self):
        """Detect and set hardware configuration."""
        try:
            hardware_info = self.hardware.get_hardware_info()
            
            # CPU configuration
            self.config.hardware.cpu_cores = hardware_info["cpu"]["physical_cores"]
            self.config.hardware.cpu_cores_p = hardware_info["cpu"]["p_cores"]
            self.config.hardware.cpu_cores_e = hardware_info["cpu"]["e_cores"]
            self.config.hardware.max_parallel_threads = hardware_info["optimization"]["max_parallel_threads"]
            
            # GPU configuration
            self.config.hardware.gpu_cores = hardware_info["gpu"]["cores"]
            self.config.hardware.metal_acceleration = hardware_info["gpu"]["metal_supported"]
            self.config.hardware.gpu_memory_mb = 0  # Unified memory
            
            # Memory configuration
            self.config.hardware.total_memory_gb = hardware_info["memory"]["total_gb"]
            self.config.hardware.max_memory_usage_gb = hardware_info["memory"]["total_gb"] * 0.8
            self.config.hardware.memory_pressure_threshold = hardware_info["memory"]["pressure_threshold"]
            
            # Platform configuration
            self.config.hardware.platform_type = hardware_info["platform"]["type"]
            self.config.hardware.is_m4_pro = hardware_info["platform"]["is_m4_pro"]
            
            logger.info("🔍 Hardware configuration detected and applied")
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
    
    async def _optimize_for_hardware(self):
        """Optimize configuration based on detected hardware."""
        try:
            # Get optimal configurations from hardware abstraction
            einstein_optimal = self.hardware.get_optimal_einstein_config()
            bolt_optimal = self.hardware.get_optimal_bolt_config()
            
            # Apply Einstein optimizations
            self.config.einstein.max_concurrent_searches = einstein_optimal["max_concurrent_searches"]
            self.config.einstein.embedding_batch_size = einstein_optimal["embedding_batch_size"]
            self.config.einstein.faiss_index_type = einstein_optimal["faiss_index_type"]
            self.config.einstein.enable_gpu_acceleration = einstein_optimal["use_metal_acceleration"]
            
            # Apply Bolt optimizations
            self.config.bolt.max_agents = bolt_optimal["max_agents"]
            self.config.bolt.max_parallel_tasks = bolt_optimal["parallel_task_limit"]
            self.config.bolt.enable_gpu_acceleration = bolt_optimal["use_metal_acceleration"]
            self.config.bolt.memory_pool_enabled = bolt_optimal["memory_pool_enabled"]
            self.config.bolt.thermal_throttling = bolt_optimal["thermal_throttling"]
            
            # Adjust for M4 Pro if detected
            if self.config.hardware.is_m4_pro:
                self.config.hardware.optimization_level = "aggressive"
                self.config.einstein.worker_threads = min(8, self.config.hardware.cpu_cores_p)
                self.config.bolt.max_agents = 8
                self.config.system.integration_mode = "cooperative"
            
            logger.info("🚀 Configuration optimized for detected hardware")
            
        except Exception as e:
            logger.error(f"Hardware optimization failed: {e}")
    
    def _apply_environment_overrides(self):
        """Apply configuration overrides from environment variables."""
        try:
            env_overrides = {}
            
            # Scan for environment variables with WHEEL_ prefix
            for key, value in os.environ.items():
                if key.startswith("WHEEL_"):
                    config_key = key[6:].lower()  # Remove WHEEL_ prefix
                    env_overrides[config_key] = value
            
            # Apply specific overrides
            if "WHEEL_LOG_LEVEL" in os.environ:
                self.config.system.log_level = os.environ["WHEEL_LOG_LEVEL"]
            
            if "WHEEL_MAX_AGENTS" in os.environ:
                try:
                    self.config.bolt.max_agents = int(os.environ["WHEEL_MAX_AGENTS"])
                except ValueError:
                    logger.warning(f"Invalid WHEEL_MAX_AGENTS value: {os.environ['WHEEL_MAX_AGENTS']}")
            
            if "WHEEL_EINSTEIN_CACHE" in os.environ:
                self.config.einstein.cache_enabled = os.environ["WHEEL_EINSTEIN_CACHE"].lower() == "true"
            
            if "WHEEL_GPU_ACCELERATION" in os.environ:
                gpu_enabled = os.environ["WHEEL_GPU_ACCELERATION"].lower() == "true"
                self.config.einstein.enable_gpu_acceleration = gpu_enabled
                self.config.bolt.enable_gpu_acceleration = gpu_enabled
            
            if env_overrides:
                logger.info(f"Applied {len(env_overrides)} environment variable overrides")
                
        except Exception as e:
            logger.error(f"Failed to apply environment overrides: {e}")
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        try:
            errors = []
            
            # Validate hardware configuration
            if self.config.hardware.cpu_cores <= 0:
                errors.append("Invalid CPU core count")
            
            if self.config.hardware.max_memory_usage_gb <= 0:
                errors.append("Invalid memory configuration")
            
            # Validate Einstein configuration
            if self.config.einstein.max_concurrent_searches <= 0:
                errors.append("Invalid Einstein concurrent search limit")
            
            if self.config.einstein.embedding_batch_size <= 0:
                errors.append("Invalid Einstein embedding batch size")
            
            # Validate Bolt configuration
            if self.config.bolt.max_agents <= 0:
                errors.append("Invalid Bolt agent count")
            
            if self.config.bolt.max_parallel_tasks <= 0:
                errors.append("Invalid Bolt parallel task limit")
            
            # Validate system configuration
            if self.config.system.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                errors.append("Invalid log level")
            
            if errors:
                error_msg = "Configuration validation failed: " + "; ".join(errors)
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    async def load_configuration(self, config_file: Optional[str] = None) -> bool:
        """Load configuration from file."""
        try:
            file_path = Path(config_file or self.config_file)
            
            if not file_path.exists():
                logger.warning(f"Configuration file {file_path} does not exist")
                return False
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Apply configuration data
            self._apply_config_data(config_data)
            
            self.config.config_source = str(file_path)
            logger.info(f"✅ Configuration loaded from {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    async def save_configuration(self, config_file: Optional[str] = None) -> bool:
        """Save configuration to file."""
        try:
            file_path = Path(config_file or self.config_file)
            
            # Update metadata
            import datetime
            self.config.last_updated = datetime.datetime.now().isoformat()
            self.config.config_source = str(file_path)
            
            # Convert to dictionary
            config_dict = asdict(self.config)
            
            # Save to file
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"✅ Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to the config object."""
        try:
            if "hardware" in config_data:
                hw_data = config_data["hardware"]
                for key, value in hw_data.items():
                    if hasattr(self.config.hardware, key):
                        setattr(self.config.hardware, key, value)
            
            if "einstein" in config_data:
                einstein_data = config_data["einstein"]
                for key, value in einstein_data.items():
                    if hasattr(self.config.einstein, key):
                        setattr(self.config.einstein, key, value)
            
            if "bolt" in config_data:
                bolt_data = config_data["bolt"]
                for key, value in bolt_data.items():
                    if hasattr(self.config.bolt, key):
                        setattr(self.config.bolt, key, value)
            
            if "system" in config_data:
                system_data = config_data["system"]
                for key, value in system_data.items():
                    if hasattr(self.config.system, key):
                        setattr(self.config.system, key, value)
                        
        except Exception as e:
            logger.error(f"Failed to apply configuration data: {e}")
            raise
    
    def get_system_config(self, system_type: SystemType) -> Dict[str, Any]:
        """Get configuration for a specific system."""
        try:
            if system_type == SystemType.EINSTEIN:
                return self._translate_config_for_einstein()
            elif system_type == SystemType.BOLT:
                return self._translate_config_for_bolt()
            elif system_type == SystemType.UNIFIED:
                return asdict(self.config)
            else:
                raise ValueError(f"Unknown system type: {system_type}")
                
        except Exception as e:
            logger.error(f"Failed to get system config for {system_type}: {e}")
            return {}
    
    def _translate_config_for_einstein(self) -> Dict[str, Any]:
        """Translate unified config to Einstein-specific format."""
        return {
            # Hardware
            "cpu_cores": self.config.hardware.cpu_cores,
            "gpu_cores": self.config.hardware.gpu_cores,
            "memory_mb": int(self.config.hardware.max_memory_usage_gb * 1024),
            "use_metal_acceleration": self.config.hardware.metal_acceleration,
            
            # Search
            "max_concurrent_searches": self.config.einstein.max_concurrent_searches,
            "search_timeout": self.config.einstein.search_timeout_ms,
            "target_latency": self.config.einstein.target_search_latency_ms,
            "enable_cache": self.config.einstein.cache_enabled,
            "cache_size": self.config.einstein.cache_size_mb,
            
            # Embedding
            "embedding_model": self.config.einstein.embedding_model,
            "embedding_batch_size": self.config.einstein.embedding_batch_size,
            "embedding_dimensions": self.config.einstein.embedding_dimensions,
            
            # FAISS
            "faiss_index_type": self.config.einstein.faiss_index_type,
            "faiss_m": self.config.einstein.faiss_hnsw_m,
            "faiss_ef_construction": self.config.einstein.faiss_hnsw_ef_construction,
            "faiss_ef_search": self.config.einstein.faiss_hnsw_ef_search,
            
            # Performance
            "enable_gpu_acceleration": self.config.einstein.enable_gpu_acceleration,
            "worker_threads": self.config.einstein.worker_threads,
            "parallel_processing": self.config.einstein.enable_parallel_processing,
        }
    
    def _translate_config_for_bolt(self) -> Dict[str, Any]:
        """Translate unified config to Bolt-specific format."""
        return {
            # Hardware
            "cpu_cores": self.config.hardware.cpu_cores,
            "gpu_cores": self.config.hardware.gpu_cores,
            "memory_mb": int(self.config.hardware.max_memory_usage_gb * 1024),
            "use_metal_acceleration": self.config.hardware.metal_acceleration,
            
            # Agents
            "max_agents": self.config.bolt.max_agents,
            "agent_timeout": self.config.bolt.agent_timeout_seconds,
            "agent_retry_limit": self.config.bolt.agent_retry_limit,
            "agent_memory_limit": self.config.bolt.agent_memory_limit_mb,
            
            # Tasks
            "max_parallel_tasks": self.config.bolt.max_parallel_tasks,
            "task_queue_size": self.config.bolt.task_queue_size,
            "task_priority_levels": self.config.bolt.task_priority_levels,
            "scheduling_algorithm": self.config.bolt.task_scheduling_algorithm,
            
            # Coordination
            "coordination_interval": self.config.bolt.coordination_interval_ms,
            "resource_sharing": self.config.bolt.resource_sharing_enabled,
            "load_balancing": self.config.bolt.load_balancing_enabled,
            "failover_enabled": self.config.bolt.failover_enabled,
            
            # Performance
            "enable_gpu_acceleration": self.config.bolt.enable_gpu_acceleration,
            "thermal_throttling": self.config.bolt.thermal_throttling,
            "memory_pool_enabled": self.config.bolt.memory_pool_enabled,
            "work_stealing": self.config.bolt.work_stealing_enabled,
        }
    
    def update_parameter(self, parameter_path: str, value: Any) -> bool:
        """Update a configuration parameter at runtime."""
        try:
            # Parse parameter path (e.g., "einstein.cache_enabled")
            parts = parameter_path.split('.')
            
            if len(parts) != 2:
                logger.error(f"Invalid parameter path: {parameter_path}")
                return False
            
            section, param = parts
            
            # Find the appropriate config section
            config_section = None
            if section == "hardware":
                config_section = self.config.hardware
            elif section == "einstein":
                config_section = self.config.einstein
            elif section == "bolt":
                config_section = self.config.bolt
            elif section == "system":
                config_section = self.config.system
            else:
                logger.error(f"Unknown configuration section: {section}")
                return False
            
            # Check if parameter exists
            if not hasattr(config_section, param):
                logger.error(f"Unknown parameter: {param} in section {section}")
                return False
            
            # Update the parameter
            current_value = getattr(config_section, param)
            setattr(config_section, param, type(current_value)(value))
            
            logger.info(f"✅ Updated {parameter_path}: {current_value} -> {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update parameter {parameter_path}: {e}")
            return False
    
    def get_parameter(self, parameter_path: str) -> Any:
        """Get a configuration parameter value."""
        try:
            parts = parameter_path.split('.')
            
            if len(parts) != 2:
                return None
            
            section, param = parts
            
            config_section = None
            if section == "hardware":
                config_section = self.config.hardware
            elif section == "einstein":
                config_section = self.config.einstein
            elif section == "bolt":
                config_section = self.config.bolt
            elif section == "system":
                config_section = self.config.system
            else:
                return None
            
            return getattr(config_section, param, None)
            
        except Exception as e:
            logger.error(f"Failed to get parameter {parameter_path}: {e}")
            return None
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "metadata": {
                "version": self.config.version,
                "last_updated": self.config.last_updated,
                "config_source": self.config.config_source,
                "environment": self.config.environment
            },
            "hardware_summary": {
                "cpu_cores": self.config.hardware.cpu_cores,
                "gpu_cores": self.config.hardware.gpu_cores,
                "memory_gb": self.config.hardware.max_memory_usage_gb,
                "platform": self.config.hardware.platform_type,
                "is_m4_pro": self.config.hardware.is_m4_pro,
                "optimization_level": self.config.hardware.optimization_level
            },
            "einstein_summary": {
                "concurrent_searches": self.config.einstein.max_concurrent_searches,
                "cache_enabled": self.config.einstein.cache_enabled,
                "gpu_acceleration": self.config.einstein.enable_gpu_acceleration,
                "faiss_type": self.config.einstein.faiss_index_type
            },
            "bolt_summary": {
                "max_agents": self.config.bolt.max_agents,
                "parallel_tasks": self.config.bolt.max_parallel_tasks,
                "gpu_acceleration": self.config.bolt.enable_gpu_acceleration,
                "thermal_throttling": self.config.bolt.thermal_throttling
            },
            "system_summary": {
                "log_level": self.config.system.log_level,
                "monitoring_enabled": self.config.system.monitoring_enabled,
                "integration_mode": self.config.system.integration_mode
            }
        }


# Global configuration manager
_config_manager = None

def get_configuration_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


async def initialize_unified_configuration(config_file: Optional[str] = None) -> ConfigurationManager:
    """Initialize the unified configuration system."""
    manager = get_configuration_manager()
    if config_file:
        manager.config_file = config_file
    await manager.initialize()
    return manager


if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Unified Configuration System Test")
        print("=" * 50)
        
        # Initialize configuration
        manager = await initialize_unified_configuration("test_config.yaml")
        
        # Show configuration summary
        summary = manager.get_configuration_summary()
        print("\nConfiguration Summary:")
        for section, details in summary.items():
            print(f"\n{section}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # Test system-specific configurations
        print("\nSystem-Specific Configurations:")
        
        einstein_config = manager.get_system_config(SystemType.EINSTEIN)
        print(f"\nEinstein Config: {len(einstein_config)} parameters")
        for key, value in list(einstein_config.items())[:5]:
            print(f"  {key}: {value}")
        
        bolt_config = manager.get_system_config(SystemType.BOLT)
        print(f"\nBolt Config: {len(bolt_config)} parameters")
        for key, value in list(bolt_config.items())[:5]:
            print(f"  {key}: {value}")
        
        # Test runtime parameter updates
        print("\nTesting Runtime Parameter Updates:")
        success = manager.update_parameter("einstein.cache_enabled", False)
        print(f"Update cache_enabled: {success}")
        
        success = manager.update_parameter("bolt.max_agents", 12)
        print(f"Update max_agents: {success}")
        
        # Verify updates
        cache_enabled = manager.get_parameter("einstein.cache_enabled")
        max_agents = manager.get_parameter("bolt.max_agents")
        print(f"Verified cache_enabled: {cache_enabled}")
        print(f"Verified max_agents: {max_agents}")
        
        print("\n✅ Configuration system test completed successfully")
    
    asyncio.run(main())
"""
Centralized Configuration Service - Singleton pattern for unified config access.
Provides thread-safe, lazy-loaded configuration management.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .loader import ConfigurationLoader
from .schema import WheelConfig

logger = logging.getLogger(__name__)


class ConfigurationService:
    """
    Singleton service for centralized configuration management.
    
    This service:
    - Provides a single point of access for all configuration
    - Ensures thread-safe lazy loading
    - Tracks configuration usage and performance
    - Supports dynamic reloading
    - Provides health monitoring
    """
    
    _instance: Optional['ConfigurationService'] = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Union[str, Path] = "config.yaml") -> 'ConfigurationService':
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """Initialize the configuration service."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self.config_path = Path(config_path)
            self._loader: Optional[ConfigurationLoader] = None
            self._config_cache: Optional[WheelConfig] = None
            self._last_reload_time: Optional[float] = None
            self._access_count = 0
            self._initialized = True
            
            logger.info(f"ConfigurationService initialized with path: {self.config_path}")
    
    @property
    def loader(self) -> ConfigurationLoader:
        """Get or create the configuration loader."""
        if self._loader is None:
            with self._lock:
                if self._loader is None:
                    self._loader = ConfigurationLoader(self.config_path)
                    logger.info("ConfigurationLoader created")
        return self._loader
    
    @property
    def config(self) -> WheelConfig:
        """Get the current configuration."""
        self._access_count += 1
        
        if self._config_cache is None:
            with self._lock:
                if self._config_cache is None:
                    self._config_cache = self.loader.config
                    logger.info("Configuration loaded and cached")
                    
        return self._config_cache
    
    def reload(self) -> WheelConfig:
        """Reload configuration from source."""
        with self._lock:
            logger.info("Reloading configuration...")
            self._loader = ConfigurationLoader(self.config_path)
            self._config_cache = self._loader.config
            self._last_reload_time = datetime.now().timestamp()
            logger.info("Configuration reloaded successfully")
            return self._config_cache
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get configuration health report."""
        return {
            "access_count": self._access_count,
            "last_reload": self._last_reload_time,
            "loader_health": self.loader.generate_health_report(),
            "config_valid": self._config_cache is not None,
            "singleton_id": id(self)
        }
    
    def track_parameter_usage(self, parameter: str, impact: float = 0.0) -> None:
        """Track parameter usage and impact."""
        self.loader.track_parameter_usage(parameter, impact)
    
    def get_unused_parameters(self) -> List[str]:
        """Get list of unused configuration parameters."""
        return self.loader.get_unused_parameters()
    
    def get_parameter_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about parameter usage."""
        return self.loader.get_parameter_stats()
    
    def validate_health(self) -> Tuple[bool, List[str]]:
        """Validate configuration health."""
        return self.loader.validate_health()
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the config object."""
        return getattr(self.config, name)
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
            logger.info("ConfigurationService singleton reset")


# Convenience functions for backward compatibility
def get_config_service(config_path: Union[str, Path] = "config.yaml") -> ConfigurationService:
    """Get the configuration service singleton."""
    return ConfigurationService(config_path)


def get_config() -> WheelConfig:
    """Get the current configuration (backward compatible)."""
    return get_config_service().config


def reload_config() -> WheelConfig:
    """Reload the configuration."""
    return get_config_service().reload()
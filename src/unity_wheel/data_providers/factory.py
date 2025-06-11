"""Data provider factory for creating and configuring providers based on configuration.

Provides a centralized way to create, configure, and manage data providers
based on application configuration and environment settings.
"""

import logging
from typing import Dict, List, Optional, Type

from src.config.loader import get_config
from src.unity_wheel.utils.logging import StructuredLogger

from .unified_provider import (
    BaseDataProvider,
    DatabentoProvider,
    DataSourceType,
    FREDProvider,
    MockProvider,
    UnifiedDataProvider,
)

logger = StructuredLogger(logging.getLogger(__name__))


class DataProviderFactory:
    """Factory for creating and configuring data providers."""

    def __init__(self):
        self._provider_classes: Dict[str, Type[BaseDataProvider]] = {
            "databento": DatabentoProvider,
            "fred": FREDProvider,
            "mock": MockProvider,
        }

        self._provider_configs: Dict[str, Dict] = {}

        logger.info("data_provider_factory_initialized")

    def register_provider_class(self, name: str, provider_class: Type[BaseDataProvider]) -> None:
        """Register a provider class for factory creation."""
        self._provider_classes[name] = provider_class
        logger.info(
            "provider_class_registered", extra={"name": name, "class": provider_class.__name__}
        )

    def set_provider_config(self, name: str, config: Dict) -> None:
        """Set configuration for a specific provider."""
        self._provider_configs[name] = config
        logger.info("provider_config_set", extra={"name": name, "config_keys": list(config.keys())})

    def create_provider(self, name: str, **kwargs) -> BaseDataProvider:
        """Create a provider instance by name."""
        if name not in self._provider_classes:
            raise ValueError(
                f"Unknown provider: {name}. Available: {list(self._provider_classes.keys())}"
            )

        provider_class = self._provider_classes[name]
        provider_config = self._provider_configs.get(name, {})

        # Merge kwargs with stored config
        final_config = {**provider_config, **kwargs}

        try:
            if final_config:
                provider = provider_class(**final_config)
            else:
                provider = provider_class()

            logger.info("provider_created", extra={"name": name, "class": provider_class.__name__})

            return provider

        except Exception as e:
            logger.error("provider_creation_failed", extra={"name": name, "error": str(e)})
            raise

    def create_unified_provider(
        self,
        enabled_providers: Optional[List[str]] = None,
        fallback_chains: Optional[Dict[str, List[str]]] = None,
    ) -> UnifiedDataProvider:
        """Create a fully configured unified provider."""
        config = get_config()

        # Determine which providers to enable
        if enabled_providers is None:
            enabled_providers = self._get_enabled_providers_from_config(config)

        # Create unified provider
        unified = UnifiedDataProvider()

        # Create and register providers
        for provider_name in enabled_providers:
            try:
                provider = self.create_provider(provider_name)
                unified.register_provider(provider)

            except Exception as e:
                logger.warning(
                    "provider_registration_failed",
                    extra={"provider": provider_name, "error": str(e)},
                )
                continue

        # Set up fallback chains
        if fallback_chains is None:
            fallback_chains = self._get_default_fallback_chains(enabled_providers)

        for source_type_str, chain in fallback_chains.items():
            try:
                source_type = DataSourceType(source_type_str)
                unified.set_fallback_chain(source_type, chain)

            except (ValueError, KeyError) as e:
                logger.warning(
                    "fallback_chain_setup_failed",
                    extra={"source_type": source_type_str, "error": str(e)},
                )

        logger.info(
            "unified_provider_created",
            extra={"enabled_providers": enabled_providers, "fallback_chains": fallback_chains},
        )

        return unified

    def _get_enabled_providers_from_config(self, config) -> List[str]:
        """Determine enabled providers from configuration."""
        enabled = []

        # Check for specific provider configurations
        if hasattr(config, "data_providers"):
            data_config = config.data_providers

            # Check individual provider configs
            if hasattr(data_config, "databento") and getattr(
                data_config.databento, "enabled", False
            ):
                enabled.append("databento")

            if hasattr(data_config, "fred") and getattr(data_config.fred, "enabled", False):
                enabled.append("fred")

            if hasattr(data_config, "mock") and getattr(data_config.mock, "enabled", True):
                enabled.append("mock")

        # Fallback to environment-based detection
        if not enabled:
            enabled = self._detect_providers_from_environment()

        # Always include mock as fallback
        if "mock" not in enabled:
            enabled.append("mock")

        return enabled

    def _detect_providers_from_environment(self) -> List[str]:
        """Detect available providers from environment."""
        import os

        providers = []

        # Check for Databento
        if os.getenv("DATABENTO_API_KEY") or os.getenv("DATABENTO_SKIP_VALIDATION"):
            providers.append("databento")

        # Check for FRED
        if os.getenv("FRED_API_KEY"):
            providers.append("fred")

        # Always include mock
        providers.append("mock")

        logger.info("providers_detected_from_environment", extra={"providers": providers})

        return providers

    def _get_default_fallback_chains(self, enabled_providers: List[str]) -> Dict[str, List[str]]:
        """Get default fallback chains based on enabled providers."""
        chains = {}

        # Market data chain
        market_chain = []
        if "databento" in enabled_providers:
            market_chain.append("databento")
        if "mock" in enabled_providers:
            market_chain.append("mock")
        if market_chain:
            chains["market_data"] = market_chain

        # Options data chain
        options_chain = []
        if "databento" in enabled_providers:
            options_chain.append("databento")
        if "mock" in enabled_providers:
            options_chain.append("mock")
        if options_chain:
            chains["options_data"] = options_chain

        # Economic data chain
        econ_chain = []
        if "fred" in enabled_providers:
            econ_chain.append("fred")
        if "mock" in enabled_providers:
            econ_chain.append("mock")
        if econ_chain:
            chains["economic_data"] = econ_chain

        return chains

    def get_available_providers(self) -> List[str]:
        """Get list of available provider types."""
        return list(self._provider_classes.keys())

    def get_provider_info(self, name: str) -> Dict:
        """Get information about a specific provider."""
        if name not in self._provider_classes:
            raise ValueError(f"Unknown provider: {name}")

        provider_class = self._provider_classes[name]

        # Create temporary instance to get supported sources
        try:
            temp_provider = provider_class()
            supported_sources = [s.value for s in temp_provider.get_supported_sources()]
        except Exception:
            supported_sources = ["unknown"]

        return {
            "name": name,
            "class": provider_class.__name__,
            "module": provider_class.__module__,
            "supported_sources": supported_sources,
            "config": self._provider_configs.get(name, {}),
        }


# Environment-specific factory configurations


class DevelopmentFactory(DataProviderFactory):
    """Factory configured for development environment."""

    def __init__(self):
        super().__init__()

        # Development typically uses mock data with some real providers
        self.set_provider_config("mock", {"priority": 100})  # Low priority fallback
        self.set_provider_config("databento", {"priority": 50})
        self.set_provider_config("fred", {"priority": 60})

        logger.info("development_factory_initialized")

    def _get_enabled_providers_from_config(self, config) -> List[str]:
        """Development environment provider selection."""
        # Always enable mock for development
        enabled = ["mock"]

        # Add real providers if credentials are available
        import os

        if os.getenv("DATABENTO_API_KEY"):
            enabled.insert(0, "databento")  # Higher priority

        if os.getenv("FRED_API_KEY"):
            enabled.insert(-1, "fred")  # Before mock

        return enabled


class ProductionFactory(DataProviderFactory):
    """Factory configured for production environment."""

    def __init__(self):
        super().__init__()

        # Production prefers real data sources
        self.set_provider_config("databento", {"priority": 10})  # Highest priority
        self.set_provider_config("fred", {"priority": 20})
        self.set_provider_config("mock", {"priority": 1000})  # Emergency fallback only

        logger.info("production_factory_initialized")

    def _get_enabled_providers_from_config(self, config) -> List[str]:
        """Production environment provider selection."""
        enabled = []

        # Require real data sources in production
        import os

        if os.getenv("DATABENTO_API_KEY"):
            enabled.append("databento")

        if os.getenv("FRED_API_KEY"):
            enabled.append("fred")

        # Mock only as emergency fallback
        if not enabled:
            logger.warning("no_real_providers_in_production")
            enabled.append("mock")
        else:
            enabled.append("mock")  # Emergency fallback

        return enabled


class TestingFactory(DataProviderFactory):
    """Factory configured for testing environment."""

    def __init__(self):
        super().__init__()

        # Testing uses only mock providers by default
        self.set_provider_config("mock", {"priority": 10})

        logger.info("testing_factory_initialized")

    def _get_enabled_providers_from_config(self, config) -> List[str]:
        """Testing environment provider selection."""
        return ["mock"]  # Only mock for testing


# Factory selection based on environment


def get_data_provider_factory() -> DataProviderFactory:
    """Get appropriate factory based on environment."""
    import os

    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionFactory()
    elif env == "testing":
        return TestingFactory()
    else:
        return DevelopmentFactory()


def create_configured_provider() -> UnifiedDataProvider:
    """Create a unified provider configured for the current environment."""
    factory = get_data_provider_factory()
    return factory.create_unified_provider()


# Convenience functions for specific use cases


def create_unity_wheel_provider() -> UnifiedDataProvider:
    """Create a provider optimized for Unity wheel strategy."""
    factory = get_data_provider_factory()

    # Unity wheel strategy needs market data, options data, and economic data
    enabled_providers = factory._get_enabled_providers_from_config(get_config())

    # Custom fallback chains optimized for wheel strategy
    fallback_chains = {
        "market_data": [p for p in ["databento", "mock"] if p in enabled_providers],
        "options_data": [p for p in ["databento", "mock"] if p in enabled_providers],
        "economic_data": [p for p in ["fred", "mock"] if p in enabled_providers],
    }

    unified = factory.create_unified_provider(
        enabled_providers=enabled_providers, fallback_chains=fallback_chains
    )

    logger.info("unity_wheel_provider_created")

    return unified


def create_minimal_provider() -> UnifiedDataProvider:
    """Create a minimal provider with only mock data."""
    factory = DataProviderFactory()
    return factory.create_unified_provider(enabled_providers=["mock"])


def create_development_provider() -> UnifiedDataProvider:
    """Create a provider for development with real data when available."""
    factory = DevelopmentFactory()
    return factory.create_unified_provider()


def create_production_provider() -> UnifiedDataProvider:
    """Create a provider for production with real data sources."""
    factory = ProductionFactory()
    return factory.create_unified_provider()

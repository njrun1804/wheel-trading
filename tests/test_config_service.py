"""Tests for centralized ConfigurationService."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.config.service import ConfigurationService, get_config_service


class TestConfigurationService:
    """Test the ConfigurationService singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        ConfigurationService.reset()

    def test_singleton_pattern(self):
        """Test that only one instance is created."""
        service1 = ConfigurationService()
        service2 = ConfigurationService()
        service3 = get_config_service()

        assert service1 is service2
        assert service2 is service3
        assert id(service1) == id(service2) == id(service3)

    def test_thread_safety(self):
        """Test thread-safe singleton creation."""
        services = []

        def get_service():
            services.append(ConfigurationService())

        # Create multiple threads trying to get the service
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_service) for _ in range(100)]
            for future in futures:
                future.result()

        # All should be the same instance
        first_id = id(services[0])
        assert all(id(service) == first_id for service in services)

    def test_config_access(self):
        """Test configuration access through service."""
        service = get_config_service()
        config = service.config

        # Test that config is loaded
        assert config is not None
        assert hasattr(config, "strategy")
        assert hasattr(config, "risk")
        assert hasattr(config, "operations")

    def test_attribute_delegation(self):
        """Test that service delegates to config object."""
        service = get_config_service()

        # Should be able to access config attributes directly
        assert service.strategy is not None
        assert service.risk is not None
        assert service.operations is not None

    def test_health_report(self):
        """Test health report generation."""
        service = get_config_service()

        # Access config a few times
        _ = service.config
        _ = service.config
        _ = service.config

        health = service.get_health_report()

        assert health["access_count"] >= 3
        assert "singleton_id" in health
        assert health["config_valid"] is True

    def test_parameter_tracking(self):
        """Test parameter usage tracking."""
        service = get_config_service()

        # Track some parameter usage
        service.track_parameter_usage("strategy.greeks.delta_target", 0.95)
        service.track_parameter_usage("risk.position_limits.max_position_size", 0.88)

        stats = service.get_parameter_stats()
        assert "strategy.greeks.delta_target" in stats
        assert "risk.position_limits.max_position_size" in stats

    def test_reset_functionality(self):
        """Test singleton reset (for testing)."""
        service1 = ConfigurationService()
        id1 = id(service1)

        ConfigurationService.reset()

        service2 = ConfigurationService()
        id2 = id(service2)

        assert id1 != id2

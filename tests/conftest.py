"""
Pytest configuration and shared fixtures.
"""

import os
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Verbosity, settings

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure Hypothesis profiles for different testing scenarios
settings.register_profile(
    "fast",
    max_examples=10,
    deadline=200,
    verbosity=Verbosity.normal,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=500,
    verbosity=Verbosity.normal,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "default",
    max_examples=100,
    deadline=1000,
    verbosity=Verbosity.normal,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "debug",
    max_examples=1000,
    deadline=None,
    verbosity=Verbosity.verbose,
)

# Load profile from environment variable
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


# Shared fixtures
@pytest.fixture
def clean_environment(monkeypatch):
    """Provides a clean environment for tests."""
    # Clear any existing env vars that might interfere
    env_vars_to_clear = [
        "WHEEL_STRATEGY__GREEKS__DELTA_TARGET",
        "WHEEL_RISK__MAX_POSITION_SIZE",
        "DATABENTO_API_KEY",
        "SCHWAB_CLIENT_ID",
        "SCHWAB_CLIENT_SECRET",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)
    yield
    # Cleanup happens automatically with monkeypatch


@pytest.fixture
def sample_config():
    """Provides a sample configuration for testing."""
    return {
        "strategy": {
            "greeks": {"delta_target": 0.30, "gamma_threshold": 0.05},
            "expiration": {"days_to_expiry_target": 45},
        },
        "risk": {
            "position_limits": {
                "max_position_size": 0.20,
                "max_contracts_per_trade": 100,
            }
        },
    }


@pytest.fixture
def sample_market_data():
    """Provides sample market data for testing."""
    return {
        "symbol": "U",
        "price": 35.00,
        "volatility": 0.65,
        "volume": 1000000,
        "bid": 34.95,
        "ask": 35.05,
    }


@pytest.fixture
def sample_option_chain():
    """Provides a sample option chain for testing."""
    return [
        {"strike": 30.0, "bid": 5.20, "ask": 5.30, "volume": 100, "open_interest": 500},
        {"strike": 32.5, "bid": 3.10, "ask": 3.20, "volume": 200, "open_interest": 1000},
        {"strike": 35.0, "bid": 1.50, "ask": 1.60, "volume": 300, "open_interest": 1500},
        {"strike": 37.5, "bid": 0.50, "ask": 0.60, "volume": 150, "open_interest": 800},
        {"strike": 40.0, "bid": 0.10, "ask": 0.20, "volume": 50, "open_interest": 300},
    ]


# Performance tracking
@pytest.fixture(autouse=True)
def track_test_duration(request):
    """Automatically track test duration for all tests."""
    import time

    start_time = time.time()
    yield
    duration = time.time() - start_time

    # Log slow tests
    if duration > 1.0:  # Tests taking more than 1 second
        test_name = request.node.name
        print(f"\nSlow test detected: {test_name} took {duration:.2f}s")


# Skip markers for conditional test execution
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "external: marks tests that require external services")


# Test environment setup
def pytest_sessionstart(session):
    """Called after the Session object has been created and before performing collection."""
    # Set test environment
    os.environ["WHEEL_ENV"] = "test"

    # Disable external service validation in tests
    os.environ["DATABENTO_SKIP_VALIDATION"] = "true"

    # Use test configuration if available
    test_config = Path(__file__).parent.parent / "config.test.yaml"
    if test_config.exists():
        os.environ["WHEEL_CONFIG_PATH"] = str(test_config)

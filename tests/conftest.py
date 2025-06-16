"""Global test configuration for BOB unified system testing."""

import pytest
import asyncio
import os
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import duckdb

# Test fixtures for BOB system components
from bob.config import ConfigLoader, EnvironmentDetector
from bob.core import HealthChecker, ServiceManager, IntegrationBridge

# Einstein and Bolt imports
try:
    from einstein.unified_index import UnifiedIndex
    from bolt.solve import BoltSolver
except ImportError:
    # Mock if not available
    UnifiedIndex = None
    BoltSolver = None

# Trading system imports
try:
    from src.unity_wheel.api.advisor import Advisor
    from src.unity_wheel.risk.analytics import RiskAnalytics
except ImportError:
    Advisor = None
    RiskAnalytics = None


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration for BOB system."""
    return {
        'testing': True,
        'log_level': 'DEBUG',
        'einstein': {
            'index_path': ':memory:',
            'embedding_dim': 768,
            'max_results': 100
        },
        'bolt': {
            'num_agents': 4,  # Reduced for testing
            'timeout': 30,
            'memory_limit_mb': 1024
        },
        'trading': {
            'database_path': ':memory:',
            'symbol': 'U',
            'risk_limits': {
                'max_position': 10000,  # Smaller for testing
                'max_delta': 0.30
            }
        },
        'hardware': {
            'cpu_cores': 2,  # Limited for testing
            'memory_limit_gb': 4,
            'gpu_enabled': False  # Disable GPU for CI
        }
    }


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix="bob_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def test_database(temp_dir):
    """Create test database with sample data."""
    db_path = temp_dir / "test_trading.duckdb"
    
    conn = duckdb.connect(str(db_path))
    
    # Create sample tables
    conn.execute("""
        CREATE TABLE positions (
            symbol VARCHAR,
            position_type VARCHAR,
            strike DECIMAL,
            expiration DATE,
            quantity INTEGER,
            entry_price DECIMAL,
            current_price DECIMAL,
            pnl DECIMAL
        )
    """)
    
    conn.execute("""
        CREATE TABLE market_data (
            symbol VARCHAR,
            timestamp TIMESTAMP,
            price DECIMAL,
            volume INTEGER,
            implied_vol DECIMAL
        )
    """)
    
    # Insert sample data
    conn.execute("""
        INSERT INTO positions VALUES 
        ('U', 'cash_secured_put', 50.0, '2024-12-20', -10, 2.50, 1.80, 700),
        ('U', 'covered_call', 55.0, '2024-12-13', 10, 1.20, 0.85, 350)
    """)
    
    conn.execute("""
        INSERT INTO market_data VALUES 
        ('U', '2024-12-01 09:30:00', 52.45, 125000, 0.35),
        ('U', '2024-12-01 10:00:00', 52.60, 98000, 0.34)
    """)
    
    conn.close()
    yield db_path


@pytest.fixture(scope="function")
async def mock_einstein():
    """Mock Einstein search system."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value={
        'results': [
            {
                'file_path': 'src/unity_wheel/strategy/wheel.py',
                'function': 'calculate_position_size',
                'score': 0.95,
                'context': 'Position sizing calculation for wheel strategy'
            },
            {
                'file_path': 'src/unity_wheel/risk/analytics.py', 
                'function': 'assess_portfolio_risk',
                'score': 0.88,
                'context': 'Portfolio risk assessment functions'
            }
        ],
        'total_results': 2,
        'search_time': 0.045
    })
    mock.build_index = AsyncMock()
    mock.get_status = AsyncMock(return_value={'status': 'ready', 'indexed_files': 1322})
    return mock


@pytest.fixture(scope="function")
async def mock_bolt():
    """Mock Bolt orchestration system."""
    mock = AsyncMock()
    mock.solve = AsyncMock(return_value={
        'status': 'completed',
        'agents_used': 4,
        'execution_time': 2.5,
        'results': [
            {'agent_id': 1, 'task': 'code_analysis', 'status': 'success'},
            {'agent_id': 2, 'task': 'risk_calculation', 'status': 'success'},
            {'agent_id': 3, 'task': 'optimization', 'status': 'success'},
            {'agent_id': 4, 'task': 'validation', 'status': 'success'}
        ],
        'recommendations': [
            'Increase delta target to 0.35 for better premium capture',
            'Adjust position sizing based on volatility regime'
        ]
    })
    mock.get_status = AsyncMock(return_value={'agents': 4, 'status': 'healthy'})
    return mock


@pytest.fixture(scope="function")
def mock_trading_advisor(test_database):
    """Mock trading advisor with test data."""
    mock = Mock()
    mock.get_recommendation = Mock(return_value={
        'action': 'sell_put',
        'strike': 50.0,
        'expiration': '2024-12-20',
        'premium_target': 2.50,
        'confidence': 0.85,
        'risk_metrics': {
            'max_loss': 4750,
            'probability_profit': 0.75,
            'expected_return': 0.12
        }
    })
    mock.get_positions = Mock(return_value=[
        {
            'symbol': 'U',
            'type': 'cash_secured_put',
            'strike': 50.0,
            'quantity': -10,
            'pnl': 700
        }
    ])
    return mock


@pytest.fixture(scope="function")
async def bob_test_system(test_config, mock_einstein, mock_bolt, mock_trading_advisor, test_database):
    """Complete BOB test system with all components."""
    from bob.core import BOBSystem
    
    # Create BOB system with mocked dependencies
    bob = BOBSystem(
        config=test_config,
        einstein=mock_einstein,
        bolt=mock_bolt,
        trading_advisor=mock_trading_advisor,
        database_path=test_database
    )
    
    await bob.initialize()
    yield bob
    await bob.shutdown()


@pytest.fixture(scope="function")
def performance_monitor():
    """Performance monitoring fixture."""
    monitor = Mock()
    monitor.start_timing = Mock()
    monitor.end_timing = Mock(return_value=0.123)
    monitor.record_memory = Mock()
    monitor.get_metrics = Mock(return_value={
        'cpu_percent': 45.2,
        'memory_mb': 512,
        'execution_time': 1.234,
        'operations_per_second': 125.5
    })
    return monitor


@pytest.fixture(scope="function")
def sample_market_data():
    """Sample market data for testing."""
    return {
        'symbol': 'U',
        'price': 52.45,
        'volume': 125000,
        'options_chain': {
            '2024-12-13': {
                'calls': {
                    50: {'bid': 2.80, 'ask': 2.90, 'iv': 0.32},
                    55: {'bid': 0.85, 'ask': 0.95, 'iv': 0.35}
                },
                'puts': {
                    50: {'bid': 1.75, 'ask': 1.85, 'iv': 0.34},
                    55: {'bid': 4.10, 'ask': 4.20, 'iv': 0.36}
                }
            }
        },
        'implied_volatility': 0.35,
        'historical_volatility': 0.28,
        'vix': 18.5
    }


@pytest.fixture(scope="function")
def hardware_metrics():
    """Mock hardware metrics for testing."""
    return {
        'cpu': {
            'cores_available': 12,
            'cores_used': 8,
            'usage_percent': 65.5,
            'temperature': 68.2
        },
        'memory': {
            'total_gb': 24.0,
            'used_gb': 12.8,
            'available_gb': 11.2,
            'pressure': 'normal'
        },
        'gpu': {
            'cores': 20,
            'memory_gb': 4.5,
            'utilization_percent': 35.2,
            'metal_supported': True
        },
        'disk': {
            'read_iops': 1250,
            'write_iops': 850,
            'latency_ms': 2.1
        }
    }


@pytest.fixture(scope="function")
def integration_test_commands():
    """Standard test commands for integration testing."""
    return [
        "analyze Unity wheel strategy performance",
        "optimize position sizing for current market conditions",
        "check risk limits and suggest adjustments",
        "evaluate options chain for best premium opportunities",
        "generate trading report for the last week",
        "validate system health and performance metrics"
    ]


@pytest.fixture(scope="function")
def error_test_scenarios():
    """Error scenarios for testing error handling."""
    return [
        {
            'description': 'Network timeout',
            'error': asyncio.TimeoutError("Request timed out"),
            'expected_behavior': 'retry_with_backoff'
        },
        {
            'description': 'Database connection lost',
            'error': Exception("Database connection failed"),
            'expected_behavior': 'reconnect_and_retry'
        },
        {
            'description': 'Memory pressure',
            'error': MemoryError("Insufficient memory"),
            'expected_behavior': 'reduce_batch_size'
        },
        {
            'description': 'Invalid user input',
            'error': ValueError("Invalid parameter"),
            'expected_behavior': 'return_helpful_error'
        }
    ]


@pytest.fixture(scope="function")
def performance_benchmarks():
    """Performance benchmarks for regression testing."""
    return {
        'search_latency_ms': 100,
        'task_throughput_per_second': 1.0,
        'memory_usage_mb': 2048,
        'cpu_utilization_percent': 80,
        'initialization_time_s': 5.0,
        'response_time_95th_percentile_ms': 500
    }


# Utility functions for tests
def assert_performance_within_bounds(metrics: Dict[str, float], benchmarks: Dict[str, float], tolerance: float = 0.2):
    """Assert that performance metrics are within acceptable bounds."""
    for metric, benchmark in benchmarks.items():
        if metric in metrics:
            actual = metrics[metric]
            max_allowed = benchmark * (1 + tolerance)
            assert actual <= max_allowed, f"{metric}: {actual} exceeds benchmark {benchmark} (max: {max_allowed})"


def create_test_position(symbol: str = "U", position_type: str = "cash_secured_put", **kwargs):
    """Create a test trading position."""
    defaults = {
        'symbol': symbol,
        'position_type': position_type,
        'strike': 50.0,
        'expiration': '2024-12-20',
        'quantity': -10,
        'entry_price': 2.50,
        'current_price': 1.80,
        'pnl': 700
    }
    defaults.update(kwargs)
    return defaults


def create_test_command(intent: str, complexity: str = "medium", **kwargs):
    """Create a test command with specified intent and complexity."""
    commands = {
        'analyze': {
            'simple': "check Unity price",
            'medium': "analyze Unity volatility patterns",
            'complex': "perform comprehensive risk analysis of current positions"
        },
        'optimize': {
            'simple': "adjust delta target",
            'medium': "optimize position sizing parameters", 
            'complex': "optimize entire wheel strategy across multiple timeframes"
        },
        'execute': {
            'simple': "place put order",
            'medium': "execute strategy adjustment",
            'complex': "implement multi-leg optimization strategy"
        }
    }
    
    base_command = commands.get(intent, {}).get(complexity, "analyze Unity options")
    
    if kwargs:
        # Add parameters to command
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        base_command += f" with parameters: {params}"
    
    return base_command


async def wait_for_condition(condition_func, timeout: float = 10.0, interval: float = 0.1):
    """Wait for a condition to become true with timeout."""
    start_time = asyncio.get_event_loop().time()
    while True:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise asyncio.TimeoutError(f"Condition not met within {timeout}s")
        
        await asyncio.sleep(interval)


class TestMetricsCollector:
    """Collect and analyze test metrics."""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
        self.end_time = None
    
    def start_collection(self):
        self.start_time = datetime.now()
        self.metrics = []
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        metric = {
            'name': name,
            'value': value,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.metrics.append(metric)
    
    def end_collection(self):
        self.end_time = datetime.now()
        return self.get_summary()
    
    def get_summary(self):
        if not self.metrics:
            return {}
        
        summary = {
            'total_metrics': len(self.metrics),
            'collection_duration': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'metrics_by_type': {},
            'average_values': {}
        }
        
        # Group by metric name
        by_name = {}
        for metric in self.metrics:
            name = metric['name']
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(metric['value'])
        
        # Calculate statistics
        for name, values in by_name.items():
            summary['metrics_by_type'][name] = len(values)
            summary['average_values'][name] = sum(values) / len(values)
        
        return summary


@pytest.fixture(scope="function")
def metrics_collector():
    """Test metrics collector."""
    return TestMetricsCollector()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "regression: mark test as regression test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "trading: mark test as trading system test")
    config.addinivalue_line("markers", "cli: mark test as CLI test")
    config.addinivalue_line("markers", "error_handling: mark test as error handling test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "regression" in str(item.fspath):
            item.add_marker(pytest.mark.regression)
        
        # Mark slow tests
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name or "metal" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark trading tests
        if "trading" in str(item.fspath) or "wheel" in item.name:
            item.add_marker(pytest.mark.trading)

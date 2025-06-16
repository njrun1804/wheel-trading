"""Production test configuration and fixtures."""

import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, List

import pytest
import psutil

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unity_wheel.storage import Storage, StorageConfig
from unity_wheel.api.advisor import WheelAdvisor
from unity_wheel.data_providers.databento import DatabentoClient
from unity_wheel.auth.client_v2 import AuthClient
from unity_wheel.monitoring.performance import PerformanceMonitor
from unity_wheel.utils import setup_structured_logging


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_production.db"
        config = StorageConfig(
            database_url=f"duckdb:///{db_path}",
            connection_pool_size=10,
            enable_wal=True,
            enable_optimizations=True
        )
        storage = Storage(config)
        await storage.initialize()
        yield storage
        await storage.close()


@pytest.fixture(scope="session")
async def auth_client():
    """Initialize auth client for production tests."""
    client = AuthClient()
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture(scope="session")
async def databento_client():
    """Initialize Databento client for production tests."""
    client = DatabentoClient()
    yield client
    await client.close()


@pytest.fixture(scope="session")
async def advisor(temp_storage, auth_client):
    """Initialize wheel advisor for production tests."""
    advisor = WheelAdvisor(
        storage=temp_storage,
        auth_client=auth_client
    )
    await advisor.initialize()
    yield advisor
    await advisor.close()


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def system_resources():
    """System resource monitoring fixture."""
    initial_stats = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'network_io': psutil.net_io_counters(),
        'process_count': len(psutil.pids())
    }
    yield initial_stats


@asynccontextmanager
async def concurrent_operations(operation_count: int, operation_func, *args, **kwargs):
    """Context manager for running concurrent operations."""
    tasks = []
    start_time = time.time()
    
    try:
        # Launch all operations concurrently
        for i in range(operation_count):
            task = asyncio.create_task(operation_func(i, *args, **kwargs))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        errors = [r for r in results if isinstance(r, Exception)]
        
        yield {
            'results': results,
            'successful': successful,
            'failed': failed,
            'errors': errors,
            'duration': duration,
            'operations_per_second': len(results) / duration if duration > 0 else 0
        }
        
    finally:
        # Cancel any remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellations to complete
        await asyncio.gather(*tasks, return_exceptions=True)


class ProductionTestContext:
    """Context manager for production test scenarios."""
    
    def __init__(self, test_name: str, enable_monitoring: bool = True):
        self.test_name = test_name
        self.enable_monitoring = enable_monitoring
        self.start_time = None
        self.metrics = {}
        
    async def __aenter__(self):
        setup_structured_logging()
        self.start_time = time.time()
        
        if self.enable_monitoring:
            self.metrics['initial_memory'] = psutil.virtual_memory().percent
            self.metrics['initial_cpu'] = psutil.cpu_percent()
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics['duration'] = duration
            
        if self.enable_monitoring:
            self.metrics['final_memory'] = psutil.virtual_memory().percent
            self.metrics['final_cpu'] = psutil.cpu_percent()
            self.metrics['memory_delta'] = (
                self.metrics['final_memory'] - self.metrics['initial_memory']
            )
            
        # Log test completion
        print(f"\n{'='*60}")
        print(f"Production Test: {self.test_name}")
        print(f"Duration: {self.metrics.get('duration', 0):.2f}s")
        if self.enable_monitoring:
            print(f"Memory Usage: {self.metrics['initial_memory']:.1f}% → "
                  f"{self.metrics['final_memory']:.1f}% "
                  f"(Δ{self.metrics['memory_delta']:+.1f}%)")
        print(f"{'='*60}")


@pytest.fixture
def production_context():
    """Production test context fixture."""
    return ProductionTestContext


# Stress test configuration
STRESS_TEST_CONFIG = {
    'concurrent_users': [1, 5, 10, 25, 50],
    'operations_per_user': 10,
    'timeout_seconds': 300,
    'acceptable_failure_rate': 0.05,  # 5%
    'performance_thresholds': {
        'response_time_p95': 2.0,  # seconds
        'memory_growth_max': 100,   # MB
        'cpu_usage_max': 80,        # percent
    }
}


# Real-world scenario configurations
REAL_WORLD_SCENARIOS = {
    'morning_market_open': {
        'time_window': '09:30-10:00',
        'expected_operations': 100,
        'data_freshness_max': 300,  # 5 minutes
    },
    'midday_analysis': {
        'time_window': '12:00-13:00',
        'expected_operations': 50,
        'data_freshness_max': 900,  # 15 minutes
    },
    'end_of_day_analysis': {
        'time_window': '15:30-16:00',
        'expected_operations': 200,
        'data_freshness_max': 60,   # 1 minute
    },
    'weekend_preparation': {
        'time_window': 'weekend',
        'expected_operations': 25,
        'data_freshness_max': 3600,  # 1 hour
    }
}


@pytest.fixture(params=REAL_WORLD_SCENARIOS.keys())
def scenario_config(request):
    """Real-world scenario configuration fixture."""
    scenario_name = request.param
    config = REAL_WORLD_SCENARIOS[scenario_name].copy()
    config['name'] = scenario_name
    return config
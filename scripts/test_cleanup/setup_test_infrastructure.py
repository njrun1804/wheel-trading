#!/usr/bin/env python3
"""Setup test infrastructure for wheel trading tests.

This script:
1. Creates an isolated test database from production
2. Loads minimal test data subsets
3. Sets up test fixtures and utilities
"""

import shutil
import sqlite3
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# Configuration
PROD_DB = Path("data/wheel_trading_optimized.duckdb")
TEST_DB = Path("data/wheel_trading_test.duckdb")
TEST_FIXTURES_DIR = Path("tests/fixtures")
MIN_TEST_DAYS = 30  # Minimal dataset for fast tests


class TestDatabaseManager:
    """Manages test database lifecycle."""
    
    def __init__(self, test_db_path: Path = TEST_DB):
        self.test_db_path = test_db_path
        self.prod_db_path = PROD_DB
        
    def create_test_database(self) -> None:
        """Create test database with minimal data subset."""
        print("Creating test database...")
        
        # Ensure test fixtures directory exists
        TEST_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Remove existing test database
        if self.test_db_path.exists():
            self.test_db_path.unlink()
            
        # Connect to production database
        prod_conn = duckdb.connect(str(self.prod_db_path), read_only=True)
        test_conn = duckdb.connect(str(self.test_db_path))
        
        try:
            # Get schema from production
            tables = prod_conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            
            for (table_name,) in tables:
                print(f"  Copying table: {table_name}")
                
                # Get table schema
                schema_query = f"SHOW CREATE TABLE {table_name}"
                schema = prod_conn.execute(schema_query).fetchone()[0]
                
                # Create table in test database
                test_conn.execute(schema)
                
                # Copy limited data based on table type
                if table_name in ['equity_quotes', 'option_quotes']:
                    # For quote data, only copy last N days
                    copy_query = f"""
                    INSERT INTO {table_name}
                    SELECT * FROM prod.{table_name}
                    WHERE ts >= CURRENT_DATE - INTERVAL '{MIN_TEST_DAYS} days'
                    """
                elif table_name in ['positions', 'trades']:
                    # For position/trade data, copy recent records
                    copy_query = f"""
                    INSERT INTO {table_name}
                    SELECT * FROM prod.{table_name}
                    WHERE created_at >= CURRENT_DATE - INTERVAL '{MIN_TEST_DAYS * 2} days'
                    LIMIT 1000
                    """
                else:
                    # For reference data, copy everything
                    copy_query = f"INSERT INTO {table_name} SELECT * FROM prod.{table_name}"
                
                # Attach production database and copy data
                test_conn.execute(f"ATTACH '{self.prod_db_path}' AS prod")
                test_conn.execute(copy_query)
                test_conn.execute("DETACH prod")
                
            # Create indexes for performance
            self._create_test_indexes(test_conn)
            
            # Verify data
            self._verify_test_data(test_conn)
            
            print(f"\nTest database created at: {self.test_db_path}")
            
        finally:
            prod_conn.close()
            test_conn.close()
            
    def _create_test_indexes(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create performance indexes for test database."""
        indexes = [
            "CREATE INDEX idx_equity_quotes_ts ON equity_quotes(ts)",
            "CREATE INDEX idx_option_quotes_ts ON option_quotes(ts)",
            "CREATE INDEX idx_option_quotes_symbol ON option_quotes(symbol)",
        ]
        
        for idx in indexes:
            try:
                conn.execute(idx)
            except Exception:
                pass  # Index might already exist
                
    def _verify_test_data(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Verify test data is properly loaded."""
        print("\nVerifying test data:")
        
        # Check row counts
        tables_to_check = ['equity_quotes', 'option_quotes', 'positions', 'trades']
        for table in tables_to_check:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count:,} rows")
            
        # Verify Unity data exists
        unity_count = conn.execute(
            "SELECT COUNT(*) FROM equity_quotes WHERE symbol = 'U'"
        ).fetchone()[0]
        
        if unity_count == 0:
            raise ValueError("No Unity stock data found in test database!")
            
    def reset_database(self) -> None:
        """Reset test database to clean state."""
        # For now, just recreate it
        self.create_test_database()
        
    def create_memory_database(self) -> duckdb.DuckDBPyConnection:
        """Create in-memory database for ultra-fast tests."""
        mem_conn = duckdb.connect(":memory:")
        
        # Copy minimal schema and data from test database
        test_conn = duckdb.connect(str(self.test_db_path), read_only=True)
        
        # Copy only essential tables
        essential_tables = ['equity_quotes', 'option_quotes']
        
        for table in essential_tables:
            # Get schema
            schema = test_conn.execute(f"SHOW CREATE TABLE {table}").fetchone()[0]
            mem_conn.execute(schema)
            
            # Copy last 7 days of data only
            mem_conn.execute(f"ATTACH '{self.test_db_path}' AS test_db")
            mem_conn.execute(f"""
                INSERT INTO {table}
                SELECT * FROM test_db.{table}
                WHERE ts >= CURRENT_DATE - INTERVAL '7 days'
            """)
            mem_conn.execute("DETACH test_db")
            
        test_conn.close()
        return mem_conn


def create_test_fixtures():
    """Create test fixture files."""
    print("\nCreating test fixtures...")
    
    # Create test configuration
    test_config = """
# Test configuration for wheel trading
unity:
  ticker: "U"
  company_name: "Unity Software Inc."

storage:
  database_path: "data/wheel_trading_test.duckdb"
  
trading:
  target_delta: 0.30
  target_dte: 30
  max_position_size: 0.25
  
risk:
  max_var_95: 0.05
  max_margin_utilization: 0.50
  
# Test mode flags
test_mode: true
use_test_database: true
"""
    
    config_path = TEST_FIXTURES_DIR / "test_config.yaml"
    config_path.write_text(test_config)
    print(f"  Created: {config_path}")
    
    # Create SQL fixture for known test data
    test_data_sql = """
-- Known test data for reproducible tests
-- Insert specific Unity quotes for testing
INSERT INTO equity_quotes (symbol, ts, bid, ask, last, volume) VALUES
    ('U', '2024-01-15 09:30:00', 32.50, 32.55, 32.52, 100000),
    ('U', '2024-01-15 10:00:00', 32.60, 32.65, 32.62, 150000),
    ('U', '2024-01-15 15:30:00', 32.80, 32.85, 32.82, 200000);

-- Insert test options for known calculations
INSERT INTO option_quotes (symbol, ts, bid, ask, last, volume, open_interest, implied_volatility) VALUES
    ('U240215C00030000', '2024-01-15 15:30:00', 3.20, 3.30, 3.25, 500, 1000, 0.35),
    ('U240215P00030000', '2024-01-15 15:30:00', 0.80, 0.90, 0.85, 300, 800, 0.35);
"""
    
    sql_path = TEST_FIXTURES_DIR / "test_data.sql"
    sql_path.write_text(test_data_sql)
    print(f"  Created: {sql_path}")


def setup_pytest_configuration():
    """Create pytest configuration for optimal performance."""
    
    conftest_content = '''"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
from typing import Generator

import duckdb
import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.test_cleanup.setup_test_infrastructure import TestDatabaseManager


@pytest.fixture(scope="session")
def test_db_manager():
    """Provide test database manager for all tests."""
    return TestDatabaseManager()


@pytest.fixture(scope="function")
def test_db(test_db_manager) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Provide clean test database connection for each test."""
    conn = duckdb.connect(str(test_db_manager.test_db_path))
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def memory_db(test_db_manager) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Provide in-memory database for ultra-fast tests."""
    conn = test_db_manager.create_memory_database()
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    from unity_wheel.config.unified_config import UnifiedConfig
    
    # Override with test values
    config = UnifiedConfig()
    config.storage.database_path = "data/wheel_trading_test.duckdb"
    config.storage.use_test_mode = True
    
    return config


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment for all tests."""
    # Set test environment variables
    monkeypatch.setenv("WHEEL_TRADING_TEST_MODE", "1")
    monkeypatch.setenv("DATABASE_PATH", "data/wheel_trading_test.duckdb")
    
    # Disable external API calls by default
    monkeypatch.setenv("DATABENTO_TEST_MODE", "1")
    monkeypatch.setenv("FRED_TEST_MODE", "1")


# Performance monitoring fixture
@pytest.fixture
def benchmark(request):
    """Simple benchmark fixture for performance tests."""
    import time
    
    start_time = time.perf_counter()
    
    def _benchmark():
        return time.perf_counter() - start_time
    
    request.addfinalizer(lambda: print(f"\\nTest duration: {_benchmark():.3f}s"))
    
    return _benchmark
'''
    
    conftest_path = Path("tests/conftest.py")
    conftest_path.parent.mkdir(exist_ok=True)
    conftest_path.write_text(conftest_content)
    print(f"\nCreated pytest configuration: {conftest_path}")


def main():
    """Run test infrastructure setup."""
    print("Setting up wheel trading test infrastructure...")
    print("=" * 60)
    
    # Create test database
    db_manager = TestDatabaseManager()
    db_manager.create_test_database()
    
    # Create test fixtures
    create_test_fixtures()
    
    # Setup pytest configuration
    setup_pytest_configuration()
    
    print("\n" + "=" * 60)
    print("Test infrastructure setup complete!")
    print(f"\nTest database: {TEST_DB}")
    print(f"Test fixtures: {TEST_FIXTURES_DIR}")
    print("\nNext steps:")
    print("1. Run: python scripts/test_cleanup/fix_imports.py")
    print("2. Run: python scripts/test_cleanup/remove_mocks.py")
    print("3. Run: pytest -xvs tests/")


if __name__ == "__main__":
    main()
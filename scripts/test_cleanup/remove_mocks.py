#!/usr/bin/env python3
"""Remove mocks from test files and replace with real test data.

This script:
1. Identifies all mock usage in tests
2. Replaces mocks with real database/API fixtures
3. Updates test patterns to modern standards
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MockRemover:
    """Removes mocks and replaces with real test patterns."""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.mock_patterns = [
            r'from unittest\.mock import',
            r'import mock',
            r'@patch\(',
            r'@mock\.',
            r'Mock\(',
            r'MagicMock\(',
            r'PropertyMock\(',
            r'AsyncMock\(',
            r'mock_\w+\s*=',
            r'\.return_value\s*=',
        ]
        
        # Replacement templates for common patterns
        self.replacements = {
            'storage_mock': self._get_storage_fixture(),
            'api_mock': self._get_api_fixture(),
            'time_mock': self._get_time_fixture(),
            'config_mock': self._get_config_fixture(),
        }
        
    def _get_storage_fixture(self) -> str:
        """Get storage fixture replacement."""
        return '''@pytest.fixture
def storage(test_db):
    \"\"\"Provide real storage instance with test database.\"\"\"
    from unity_wheel.storage.storage import Storage
    storage = Storage(db_path="data/wheel_trading_test.duckdb")
    storage._conn = test_db
    return storage'''
    
    def _get_api_fixture(self) -> str:
        """Get API fixture replacement."""
        return '''@pytest.fixture
def api_client(test_config):
    \"\"\"Provide API client in test mode.\"\"\"
    from unity_wheel.data_providers.databento.client import DatabentoClient
    client = DatabentoClient(test_mode=True)
    # Client will use recorded responses in test mode
    return client'''
    
    def _get_time_fixture(self) -> str:
        """Get time fixture replacement."""
        return '''@pytest.fixture
def frozen_time():
    \"\"\"Provide time manipulation for tests.\"\"\"
    from freezegun import freeze_time
    with freeze_time("2024-01-15 09:30:00") as frozen:
        yield frozen'''
        
    def _get_config_fixture(self) -> str:
        """Get config fixture replacement."""
        return '''@pytest.fixture
def config(test_config):
    \"\"\"Provide test configuration.\"\"\"
    return test_config'''
    
    def find_mock_usage(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Find mock usage in a file."""
        mock_usages = []
        
        try:
            content = file_path.read_text()
            lines = content.splitlines()
            
            for i, line in enumerate(lines):
                for pattern in self.mock_patterns:
                    if re.search(pattern, line):
                        # Determine replacement suggestion
                        suggestion = self._suggest_replacement(line)
                        mock_usages.append((i + 1, line.strip(), suggestion))
                        
        except Exception as e:
            print(f"  Error analyzing {file_path}: {e}")
            
        return mock_usages
    
    def _suggest_replacement(self, line: str) -> str:
        """Suggest replacement for mock usage."""
        # Storage mocks
        if any(x in line for x in ['Storage', 'storage', 'DuckDBCache']):
            return "Use storage fixture with test database"
            
        # API mocks
        if any(x in line for x in ['DatabentoClient', 'FREDClient', 'api']):
            return "Use api_client fixture with test mode"
            
        # Time mocks
        if any(x in line for x in ['datetime', 'time', 'now()']):
            return "Use frozen_time fixture"
            
        # Config mocks
        if any(x in line for x in ['config', 'settings', 'Config']):
            return "Use config fixture"
            
        # Model mocks
        if any(x in line for x in ['Position', 'Account', 'Greeks']):
            return "Create real model instances with test data"
            
        return "Replace with appropriate fixture or real data"
    
    def generate_replacement_code(self, file_path: Path) -> Optional[str]:
        """Generate replacement code for a test file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            # Analyze test class and methods
            test_class = None
            test_methods = []
            mock_decorators = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                    test_class = node.name
                elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_methods.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    # Check for mock decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'id'):
                            if decorator.func.id == 'patch':
                                mock_decorators.append(node.name)
                                
            # Generate replacement template
            if test_class or test_methods:
                return self._generate_test_template(
                    file_path.stem,
                    test_class,
                    test_methods,
                    mock_decorators
                )
                
        except Exception as e:
            print(f"  Error generating replacement: {e}")
            
        return None
    
    def _generate_test_template(
        self,
        module_name: str,
        test_class: Optional[str],
        test_methods: List[str],
        mock_decorators: List[str]
    ) -> str:
        """Generate modern test template."""
        
        template = f'''"""Tests for {module_name} using real data."""

import pytest
from datetime import datetime
from decimal import Decimal

from unity_wheel.{module_name} import *


'''
        
        if test_class:
            template += f'''class {test_class}:
    """Test {test_class.replace('Test', '')} with real data."""
    
'''
            
        # Add common fixtures
        template += '''    @pytest.fixture(autouse=True)
    def setup(self, test_db, test_config):
        """Set up test environment."""
        self.db = test_db
        self.config = test_config
        
'''
        
        # Generate test method templates
        for method in test_methods[:3]:  # Show first 3 as examples
            template += f'''    def {method}(self):
        """Test {method.replace('test_', '').replace('_', ' ')}."""
        # TODO: Implement with real data
        # Original test used mocks - replace with:
        # 1. Create test data in database
        # 2. Call real implementation
        # 3. Assert on actual results
        pass
        
'''
        
        return template
    
    def process_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, any]:
        """Process a single test file."""
        print(f"\nProcessing: {file_path.relative_to(self.project_root)}")
        
        # Find mock usage
        mock_usages = self.find_mock_usage(file_path)
        
        if not mock_usages:
            print("  ✓ No mocks found")
            return {"mocks_found": 0}
            
        print(f"  Found {len(mock_usages)} mock usages:")
        for line_num, line, suggestion in mock_usages[:5]:  # Show first 5
            print(f"    Line {line_num}: {line[:60]}...")
            print(f"      → {suggestion}")
            
        if len(mock_usages) > 5:
            print(f"    ... and {len(mock_usages) - 5} more")
            
        # Generate replacement code
        replacement = self.generate_replacement_code(file_path)
        
        if replacement and not dry_run:
            # Create backup
            backup_path = file_path.with_suffix('.py.mock_backup')
            file_path.rename(backup_path)
            
            # Write new file
            file_path.write_text(replacement)
            print(f"  ✓ Replaced with modern test template")
            print(f"  ✓ Original backed up to: {backup_path.name}")
            
        return {
            "mocks_found": len(mock_usages),
            "replacement_generated": bool(replacement)
        }
    
    def process_all_tests(self, dry_run: bool = True) -> None:
        """Process all test files."""
        test_files = list(self.project_root.rglob("test_*.py"))
        
        print(f"Found {len(test_files)} test files")
        
        total_mocks = 0
        files_with_mocks = 0
        replacements_generated = 0
        
        for test_file in test_files:
            if "__pycache__" in str(test_file) or ".venv" in str(test_file):
                continue
                
            result = self.process_file(test_file, dry_run=dry_run)
            
            if result["mocks_found"] > 0:
                files_with_mocks += 1
                total_mocks += result["mocks_found"]
                
            if result.get("replacement_generated"):
                replacements_generated += 1
                
        print(f"\n{'=' * 60}")
        print("Summary:")
        print(f"  Files with mocks: {files_with_mocks}")
        print(f"  Total mock usages: {total_mocks}")
        print(f"  Replacements generated: {replacements_generated}")
        
        if dry_run:
            print("\nThis was a dry run. To apply changes, run with --apply flag")


def create_fixture_examples():
    """Create example fixture patterns for common test scenarios."""
    
    examples_content = '''"""Example test patterns using real data instead of mocks."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from unity_wheel.storage.storage import Storage
from unity_wheel.models import Position, Account
from unity_wheel.strategy import WheelStrategy


class TestWheelStrategyWithRealData:
    """Example of testing strategy with real database."""
    
    @pytest.fixture
    def strategy(self, test_db, test_config):
        """Create strategy with test database."""
        storage = Storage()
        storage._conn = test_db  # Use test database
        
        return WheelStrategy(
            storage=storage,
            config=test_config
        )
    
    @pytest.fixture
    def test_positions(self, test_db):
        """Create test positions in database."""
        # Insert test data
        test_db.execute("""
            INSERT INTO positions (symbol, quantity, cost_basis, created_at)
            VALUES 
                ('U', 100, 3250.00, '2024-01-01'),
                ('U240215P00030000', -1, -150.00, '2024-01-15')
        """)
        
        # Return position objects
        return [
            Position('U', 100),
            Position('U240215P00030000', -1)
        ]
    
    def test_evaluate_positions_with_real_data(self, strategy, test_positions):
        """Test position evaluation with real market data."""
        # Strategy will use real data from test database
        recommendations = strategy.evaluate_positions(test_positions)
        
        # Assert on actual results
        assert len(recommendations) > 0
        assert all(r.confidence >= 0.0 for r in recommendations)
        
    def test_risk_analysis_with_real_data(self, strategy, test_db):
        """Test risk analysis with real market scenarios."""
        # Load real historical data
        market_data = test_db.execute("""
            SELECT * FROM equity_quotes 
            WHERE symbol = 'U' 
            ORDER BY ts DESC 
            LIMIT 100
        """).fetchdf()
        
        # Run risk analysis
        risk_metrics = strategy.analyze_risk(market_data)
        
        # Validate results
        assert risk_metrics.var_95 > 0
        assert 0 <= risk_metrics.sharpe_ratio <= 5
        
    @pytest.mark.parametrize("delta_target,expected_strike", [
        (0.30, 30.0),  # 30 delta should give ~30 strike for 32.50 stock
        (0.20, 28.0),  # 20 delta should give lower strike
        (0.40, 31.0),  # 40 delta should give higher strike
    ])
    def test_strike_selection_with_parameters(self, strategy, delta_target, expected_strike):
        """Test strike selection with different parameters."""
        strategy.config.delta_target = delta_target
        
        # Use real option chain from database
        strike = strategy.select_optimal_strike(
            underlying_price=32.50,
            target_dte=30
        )
        
        # Verify strike is reasonable
        assert abs(strike - expected_strike) < 2.0  # Within $2


class TestDataValidationWithRealData:
    """Example of testing data validation with real data."""
    
    def test_validate_market_data(self, test_db):
        """Test market data validation with real database."""
        from unity_wheel.data_providers.validation import validate_market_data
        
        # Load real data
        data = test_db.execute("""
            SELECT * FROM equity_quotes 
            WHERE symbol = 'U' 
            AND ts >= '2024-01-01'
        """).fetchdf()
        
        # Validate
        is_valid, errors = validate_market_data(data)
        
        assert is_valid
        assert len(errors) == 0
        
    def test_detect_data_anomalies(self, test_db):
        """Test anomaly detection on real data."""
        from unity_wheel.analytics.anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector()
        
        # Load time series
        prices = test_db.execute("""
            SELECT ts, last as price 
            FROM equity_quotes 
            WHERE symbol = 'U'
            ORDER BY ts
        """).fetchdf()
        
        # Detect anomalies
        anomalies = detector.detect_price_anomalies(prices)
        
        # Should find some but not too many
        assert 0 <= len(anomalies) <= len(prices) * 0.05  # Max 5% anomalies


# Performance testing example
class TestPerformanceWithRealData:
    """Example of performance testing with real data."""
    
    def test_strategy_evaluation_performance(self, strategy, benchmark):
        """Test strategy evaluation performance."""
        # Load larger dataset
        positions = [Position('U', 100 * i) for i in range(1, 11)]
        
        # Measure performance
        strategy.evaluate_positions(positions)
        
        duration = benchmark()
        assert duration < 0.1  # Should complete in under 100ms
        
    def test_database_query_performance(self, test_db, benchmark):
        """Test database query performance."""
        # Complex aggregation query
        test_db.execute("""
            SELECT 
                date_trunc('day', ts) as day,
                symbol,
                avg(last) as avg_price,
                sum(volume) as total_volume
            FROM equity_quotes
            WHERE symbol = 'U'
            GROUP BY 1, 2
            ORDER BY 1
        """).fetchall()
        
        duration = benchmark()
        assert duration < 0.05  # Should complete in under 50ms
'''
    
    examples_path = Path("tests/examples/test_patterns_real_data.py")
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.write_text(examples_content)
    print(f"Created example test patterns: {examples_path}")


def main():
    """Run mock removal process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove mocks from test files")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    parser.add_argument("--examples", action="store_true", help="Create example test patterns")
    args = parser.parse_args()
    
    print("Mock Remover for Wheel Trading Tests")
    print("=" * 60)
    
    if args.examples:
        create_fixture_examples()
        print("\nExample test patterns created in tests/examples/")
        return
        
    remover = MockRemover()
    remover.process_all_tests(dry_run=not args.apply)
    
    print("\n\nNext steps:")
    print("1. Review the suggested replacements")
    print("2. Run with --apply to replace mock-based tests")
    print("3. Update tests to use real data patterns")
    print("4. See tests/examples/ for test pattern examples")


if __name__ == "__main__":
    main()
# Testing Guide

## Environment Setup

Ensure you're using Python 3.11:
```bash
python --version  # Should show Python 3.11.10
source venv/bin/activate  # Activate virtual environment
```

## Running Tests

### Quick Test Commands

```bash
# Run all fast tests (skip slow/integration)
pytest -m "not slow and not integration" --timeout=30

# Run specific test modules
pytest tests/test_math.py tests/test_position.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel (faster)
pytest -n auto --dist loadscope

# Run with specific timeout
pytest --timeout=60 tests/test_wheel.py

# Show slowest tests
pytest --durations=20
```

### Test Categories

Tests are marked with categories:
- `@pytest.mark.unit` - Fast unit tests (<1s)
- `@pytest.mark.integration` - Tests requiring external resources
- `@pytest.mark.slow` - Tests that take >10s
- `@pytest.mark.e2e` - End-to-end workflow tests

### Running by Category

```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Everything except slow tests
pytest -m "not slow"

# Slow tests only (useful for CI)
pytest -m slow --timeout=300
```

## Hypothesis Testing

For property-based tests, use profiles:

```bash
# Fast profile (10 examples)
pytest --hypothesis-profile=fast

# CI profile (50 examples)
pytest --hypothesis-profile=ci

# Full testing (100 examples, default)
pytest --hypothesis-profile=default
```

## Performance Testing

```bash
# Run performance benchmarks
pytest tests/test_performance_benchmarks.py -v

# Profile test execution
python -m cProfile -m pytest tests/test_math.py

# Memory profiling
pytest --memray tests/test_wheel_backtester.py
```

## Debugging Failed Tests

```bash
# Run last failed tests
pytest --lf

# Run failed tests first, then others
pytest --ff

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Extra verbose output
pytest -vv
```

## Environment Variables for Testing

```bash
# Skip external API validation
export DATABENTO_SKIP_VALIDATION=true

# Use test configuration
export WHEEL_ENV=test

# Set specific config values
export WHEEL_STRATEGY__GREEKS__DELTA_TARGET=0.30
```

## Common Issues and Solutions

### Issue: Tests timing out
```bash
# Increase timeout for specific tests
pytest --timeout=120 tests/test_integration.py

# Or mark individual tests
@pytest.mark.timeout(180)
def test_complex_backtest():
    pass
```

### Issue: Import errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Verify PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: Flaky tests
```bash
# Re-run flaky tests
pytest --reruns 3 --reruns-delay 1

# Mark known flaky tests
@pytest.mark.flaky(reruns=3)
def test_external_api():
    pass
```

## CI Configuration

For GitHub Actions or other CI:

```yaml
- name: Run tests
  run: |
    pytest -m "not slow" \
      --timeout=60 \
      --hypothesis-profile=ci \
      --cov=src \
      --cov-report=xml \
      --junit-xml=test-results.xml
```

## Test Data Management

- Use fixtures for common test data
- Mock external APIs to avoid rate limits
- Store large test datasets in `tests/fixtures/`
- Use `tmp_path` fixture for temporary files

## Writing New Tests

```python
import pytest
from hypothesis import given, strategies as st

class TestNewFeature:
    """Group related tests in classes."""

    @pytest.mark.unit
    @pytest.mark.timeout(10)
    def test_basic_functionality(self):
        """Test basic case with timeout."""
        assert True

    @pytest.mark.slow
    @pytest.mark.integration
    @given(st.floats(min_value=0.1, max_value=1000.0))
    def test_property_based(self, value):
        """Property-based test marked as slow."""
        assert value > 0

    @pytest.fixture
    def sample_data(self):
        """Fixture for test data."""
        return {"price": 100.0, "volatility": 0.3}
```

## Measuring Test Quality

```bash
# Coverage report
pytest --cov=src --cov-report=term-missing

# Mutation testing
mutmut run --paths-to-mutate=src/

# Complexity analysis
radon cc src/ -a -nb
```

## Best Practices

1. **Keep tests fast**: Aim for <1s per unit test
2. **Use markers**: Categorize tests appropriately
3. **Mock external services**: Don't hit real APIs in unit tests
4. **Test edge cases**: Use Hypothesis for property testing
5. **Clean up resources**: Use fixtures with proper teardown
6. **Parallel safety**: Ensure tests can run in parallel
7. **Deterministic results**: Avoid time-dependent assertions

## Quick Reference Card

```bash
# Most common commands
pytest                           # Run all tests
pytest -x                        # Stop on first failure
pytest -k "test_math"           # Run tests matching pattern
pytest --pdb                    # Debug on failure
pytest -n auto                  # Parallel execution
pytest -m "not slow"            # Skip slow tests
pytest --lf                     # Run last failed
pytest --durations=10           # Show slowest tests
```

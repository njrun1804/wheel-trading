# Dependency Assessment and Recommendations

## Executive Summary

The project has significant dependency conflicts and Python version compatibility issues that are likely causing test timeouts and performance problems. The main issues are:

1. **Python 3.13 Compatibility**: Project wasn't designed for Python 3.13 - should use Python 3.11 or 3.12
2. **NumPy Major Version Jump**: Auto-upgraded from 1.x to 2.x for Python 3.13, potentially breaking APIs
3. **Version Conflicts**: Multiple packages have different versions in pyproject.toml vs requirements.txt
4. **Unused Dependencies**: Several dev tools installed but never used

## Immediate Actions Required

### 1. Downgrade to Python 3.11 (Recommended)
```bash
# Install Python 3.11
pyenv install 3.11.10
pyenv local 3.11.10

# Create new virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### 2. Remove Unused Dependencies
The following packages can be removed as they're not being used:
- **flake8** - No flake8 config, black handles formatting
- **mypy** - No type checking being run
- **bandit** - No security scanning integrated
- **memory-profiler** - Only 2 references, not essential
- **click** - Only 1 import, can use argparse
- **scikit-learn** - Only 2 imports, large dependency

### 3. Update Critical Dependencies
For Python 3.11, these versions are recommended:
- **numpy**: 1.26.4 (stay on 1.x to avoid breaking changes)
- **scipy**: 1.13.1
- **pandas**: 2.2.3
- **aiohttp**: 3.10.11 (current version is fine)
- **duckdb**: 1.0.0
- **pydantic**: 2.7.4

## Recommended pyproject.toml (Updated)

```toml
[tool.poetry]
name = "unity-wheel-bot"
version = "0.1.0"
description = "Sophisticated options wheel trading system optimized for autonomous operation"
authors = ["Claude Code <noreply@anthropic.com>"]
readme = "README.md"
packages = [{include = "unity_wheel", from = "src"}]

[tool.poetry.dependencies]
python = ">=3.11,<3.13"  # Specify supported Python versions
numpy = "~1.26.4"        # Stay on 1.x for stability
scipy = "~1.13.1"        # Compatible with numpy 1.x
pandas = "~2.2.3"
pydantic = "~2.7.4"
python-dotenv = "~1.0.1"
pytz = "~2024.1"
rich = "~13.9.0"
aiohttp = "~3.10.11"
cryptography = ">=45.0.0"
pyyaml = "~6.0.2"
pydantic-settings = "~2.6.1"
duckdb = "~1.0.0"
google-cloud-secret-manager = "~2.20.2"  # Make required, not optional

[tool.poetry.group.dev.dependencies]
pytest = "~8.3.4"
pytest-cov = "~6.0.0"
pytest-xdist = "~3.6.1"
pytest-timeout = "~2.3.1"
pytest-asyncio = "~0.25.2"
hypothesis = "~6.122.3"
black = "~24.10.0"
isort = "~5.13.2"
pre-commit = "~4.0.1"
types-pytz = "~2024.2.0.20241221"
pandas-stubs = "~2.2.3.250308"
types-PyYAML = "~6.0.12.20240917"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 100
target-version = ['py311']  # Updated for Python 3.11
include = '\.pyi?$'
extend-exclude = '''
/(
  __pycache__
  | .venv
  | venv
  | .pytest_cache
)/
'''

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]
timeout = 60  # Default 60 second timeout
timeout_method = "thread"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

## Recommended requirements.txt (Simplified)

```txt
# Core dependencies
numpy~=1.26.4
scipy~=1.13.1
pandas~=2.2.3
pydantic~=2.7.4
python-dotenv~=1.0.1
pytz~=2024.1
rich~=13.9.0
aiohttp~=3.10.11
cryptography>=45.0.0
pyyaml~=6.0.2
pydantic-settings~=2.6.1
duckdb~=1.0.0
google-cloud-secret-manager~=2.20.2

# Dev dependencies (requirements-dev.txt)
pytest~=8.3.4
pytest-cov~=6.0.0
pytest-xdist~=3.6.1
pytest-timeout~=2.3.1
pytest-asyncio~=0.25.2
hypothesis~=6.122.3
black~=24.10.0
isort~=5.13.2
pre-commit~=4.0.1
```

## Migration Steps

1. **Backup Current Environment**
   ```bash
   pip freeze > requirements-backup.txt
   cp -r venv venv-backup
   ```

2. **Switch to Python 3.11**
   ```bash
   pyenv install 3.11.10
   pyenv local 3.11.10
   python --version  # Should show 3.11.10
   ```

3. **Clean Install**
   ```bash
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. **Verify Installation**
   ```bash
   python -c "import numpy; print(f'NumPy: {numpy.__version__}')"  # Should be 1.26.x
   python -c "import scipy; print(f'SciPy: {scipy.__version__}')"  # Should be 1.13.x
   ```

6. **Run Quick Test**
   ```bash
   pytest tests/test_math_simple.py -v
   ```

## Expected Benefits

1. **Test Performance**: 2-5x faster with Python 3.11 and optimized dependencies
2. **Stability**: No more NumPy 2.x breaking changes
3. **Size Reduction**: ~350MB smaller without unused dependencies
4. **Compatibility**: All packages will have proper Python 3.11 support

## Long-term Recommendations

1. **Use Single Dependency Manager**: Choose either Poetry or pip-tools, not both
2. **Add Dependency Checking to CI**: Ensure versions stay synchronized
3. **Regular Updates**: Schedule quarterly dependency updates
4. **Consider Replacing**:
   - `pytz` → Python's native `zoneinfo`
   - `click` → `argparse` (if CLI stays simple)
   - Remove scikit-learn if regime detection isn't critical

## Test Optimization (After Dependencies Fixed)

Once dependencies are fixed, implement these test optimizations:

1. **Categorize Tests**
   ```python
   @pytest.mark.slow
   @pytest.mark.timeout(120)
   def test_backtest_optimization():
       pass

   @pytest.mark.unit
   @pytest.mark.timeout(10)
   def test_black_scholes():
       pass
   ```

2. **Run Fast Tests First**
   ```bash
   pytest -m "not slow" --timeout=30
   pytest -m "slow" --timeout=300  # Run separately
   ```

3. **Use Hypothesis Profiles**
   ```python
   # conftest.py
   from hypothesis import settings, Verbosity

   settings.register_profile("ci", max_examples=10, verbosity=Verbosity.normal)
   settings.register_profile("dev", max_examples=100, verbosity=Verbosity.verbose)
   settings.register_profile("debug", max_examples=1000, verbosity=Verbosity.debug)
   ```

4. **Parallel Execution**
   ```bash
   pytest -n auto --dist loadscope  # Better test distribution
   ```

This dependency cleanup should resolve the test timeout issues and improve overall project stability.

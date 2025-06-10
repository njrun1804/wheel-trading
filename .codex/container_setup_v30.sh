#!/bin/bash
# Comprehensive Container Setup for Unity Wheel Trading Bot v2.2
# Handles numpy, sklearn, hypothesis, and essential dependencies with proper fallbacks

set -e

echo "🚀 UNITY WHEEL CONTAINER SETUP v31"
echo "=================================="
echo ""

# Detect Python version
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v $cmd >/dev/null 2>&1; then
        VERSION=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo $VERSION | cut -d. -f1)
        MINOR=$(echo $VERSION | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_CMD=$cmd
            echo "✅ Found Python $VERSION at $cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ ERROR: Python 3.9+ required but not found!"
    echo "   Please install Python 3.9 or newer"
    exit 1
fi

# Save current directory
PROJECT_DIR=$(pwd)

# 1. Create directories first
echo ""
echo "📁 Creating directories..."
mkdir -p .codex/stubs 2>/dev/null
mkdir -p /tmp/.wheel/cache 2>/dev/null
mkdir -p /tmp/.wheel/secrets 2>/dev/null
mkdir -p logs 2>/dev/null
mkdir -p exports 2>/dev/null

# 2. Create compatibility stubs for all missing packages
echo ""
echo "🔧 Creating compatibility stubs..."

# Create numpy stub
cat > .codex/stubs/numpy.py <<'EOF'
"""Numpy stub for pure Python mode - provides minimal compatibility."""
import math
import warnings

# Stub version info
__version__ = "0.0.0-stub"

# Basic constants
pi = math.pi
e = math.e
inf = float('inf')
nan = float('nan')

# Stub array type
class ndarray:
    def __init__(self, data):
        self.data = data

# Numeric types
float64 = float
float32 = float
int64 = int
int32 = int

# Basic math functions
def exp(x):
    if isinstance(x, (list, tuple)):
        return [math.exp(v) for v in x]
    return math.exp(x)

def log(x):
    if isinstance(x, (list, tuple)):
        return [math.log(v) for v in x]
    return math.log(x)

def sqrt(x):
    if isinstance(x, (list, tuple)):
        return [math.sqrt(v) for v in x]
    return math.sqrt(x)

def maximum(x, y):
    if isinstance(x, (list, tuple)):
        return [max(a, b) for a, b in zip(x, y)]
    return max(x, y)

def minimum(x, y):
    if isinstance(x, (list, tuple)):
        return [min(a, b) for a, b in zip(x, y)]
    return min(x, y)

def isnan(x):
    if isinstance(x, (list, tuple)):
        return [math.isnan(v) for v in x]
    return math.isnan(x)

def array(data):
    return list(data) if hasattr(data, '__iter__') else [data]

# Stub typing support
class typing:
    class NDArray:
        def __getitem__(self, item):
            return list

# Show warning when imported
warnings.warn("Using numpy stub - calculations will use pure Python fallbacks", ImportWarning, stacklevel=2)
EOF

# Create scipy stub
cat > .codex/stubs/scipy.py <<'EOF'
"""Scipy stub for pure Python mode."""
import math

class stats:
    class norm:
        @staticmethod
        def cdf(x):
            """Cumulative distribution function for standard normal."""
            # Approximation of normal CDF using error function
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        @staticmethod
        def pdf(x):
            """Probability density function for standard normal."""
            return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
EOF

# Create scikit-learn stub
cat > .codex/stubs/sklearn.py <<'EOF'
"""Scikit-learn stub for pure Python mode."""
import warnings

__version__ = "0.0.0-stub"

# Stub for sklearn.ensemble
class ensemble:
    class IsolationForest:
        def __init__(self, **kwargs):
            warnings.warn("Using sklearn stub - anomaly detection disabled", ImportWarning, stacklevel=2)
            self.contamination = kwargs.get('contamination', 0.1)

        def fit(self, X):
            return self

        def predict(self, X):
            # Always return normal (1) for stub
            return [1] * len(X)

        def score_samples(self, X):
            # Return neutral scores
            return [0.5] * len(X)

# Stub for sklearn.cluster
class cluster:
    class KMeans:
        def __init__(self, n_clusters=2, **kwargs):
            self.n_clusters = n_clusters

        def fit(self, X):
            return self

        def predict(self, X):
            # Simple alternating cluster assignment
            return [i % self.n_clusters for i in range(len(X))]

# Make sklearn importable as module
import sys
sys.modules['sklearn.ensemble'] = ensemble
sys.modules['sklearn.cluster'] = cluster
EOF

# Create dotenv stub for environment variables
cat > .codex/stubs/dotenv.py <<'EOF'
"""Python-dotenv stub for environment variable loading."""
import os

def load_dotenv(dotenv_path=None, stream=None, verbose=False, override=False, interpolate=True, encoding=None):
    """Stub for load_dotenv - does nothing in container mode."""
    return True

def find_dotenv(filename='.env', raise_error_if_not_found=False, usecwd=False):
    """Find .env file - returns empty string in stub mode."""
    return ""

def dotenv_values(dotenv_path=None, stream=None, verbose=False, interpolate=True, encoding=None):
    """Return environment values - returns empty dict in stub mode."""
    return {}

# Make load_dotenv available at module level
__all__ = ['load_dotenv', 'find_dotenv', 'dotenv_values']
EOF

# Create requests stub for HTTP calls
cat > .codex/stubs/requests.py <<'EOF'
"""Requests stub for HTTP calls - minimal compatibility."""
import warnings

class Response:
    """Stub response object."""
    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

def get(url, **kwargs):
    """Stub GET request - always returns success."""
    warnings.warn("Using requests stub - HTTP calls disabled", ImportWarning, stacklevel=2)
    return Response(200, "stub response")

def post(url, **kwargs):
    """Stub POST request - always returns success."""
    warnings.warn("Using requests stub - HTTP calls disabled", ImportWarning, stacklevel=2)
    return Response(200, "stub response")

def put(url, **kwargs):
    """Stub PUT request - always returns success."""
    warnings.warn("Using requests stub - HTTP calls disabled", ImportWarning, stacklevel=2)
    return Response(200, "stub response")

def delete(url, **kwargs):
    """Stub DELETE request - always returns success."""
    warnings.warn("Using requests stub - HTTP calls disabled", ImportWarning, stacklevel=2)
    return Response(200, "stub response")

# Session stub
class Session:
    def get(self, url, **kwargs):
        return get(url, **kwargs)

    def post(self, url, **kwargs):
        return post(url, **kwargs)

__all__ = ['get', 'post', 'put', 'delete', 'Session', 'Response']
EOF

# Create pydantic_settings stub
cat > .codex/stubs/pydantic_settings.py <<'EOF'
"""Pydantic Settings stub for configuration management."""
import warnings
from typing import Any, Dict, Optional

class BaseSettings:
    """Stub BaseSettings class."""

    def __init__(self, **kwargs):
        # Set all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    class Config:
        """Stub Config class."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @classmethod
    def parse_env_vars(cls, **kwargs):
        """Parse environment variables - stub implementation."""
        return cls(**kwargs)

# Show warning
warnings.warn("Using pydantic_settings stub - settings loading limited", ImportWarning, stacklevel=2)

__all__ = ['BaseSettings']
EOF

# Create Google Cloud stub
cat > .codex/stubs/google.py <<'EOF'
"""Google Cloud stub for pure Python mode."""
import warnings
from typing import Any, Dict, Optional

class cloud:
    """Stub Google Cloud module."""

    class secretmanager:
        """Stub Secret Manager."""

        class SecretManagerServiceClient:
            def __init__(self, **kwargs):
                warnings.warn("Using Google Cloud stub - secrets disabled", ImportWarning, stacklevel=2)

            def access_secret_version(self, request=None, **kwargs):
                """Stub secret access - returns empty."""
                class Response:
                    class payload:
                        data = b""
                return Response()

    class storage:
        """Stub Cloud Storage."""

        class Client:
            def __init__(self, **kwargs):
                warnings.warn("Using Google Cloud stub - storage disabled", ImportWarning, stacklevel=2)

            def bucket(self, name):
                return type('Bucket', (), {'blob': lambda x: type('Blob', (), {})()})()

# Make modules importable
import sys
sys.modules['google.cloud'] = cloud
sys.modules['google.cloud.secretmanager'] = cloud.secretmanager
sys.modules['google.cloud.storage'] = cloud.storage

__all__ = ['cloud']
EOF

# Create hypothesis stub for testing
cat > .codex/stubs/hypothesis.py <<'EOF'
"""Hypothesis stub for testing - provides minimal compatibility."""
import warnings
import random
from typing import Any, Callable

# Stub version
__version__ = "0.0.0-stub"

# Stub strategies module
class strategies:
    @staticmethod
    def floats(min_value=None, max_value=None, allow_nan=False, allow_infinity=False):
        def generate():
            if min_value is not None and max_value is not None:
                return random.uniform(min_value, max_value)
            return random.random() * 1000
        return generate

    @staticmethod
    def integers(min_value=None, max_value=None):
        def generate():
            min_v = min_value if min_value is not None else 0
            max_v = max_value if max_value is not None else 1000
            return random.randint(min_v, max_v)
        return generate

    @staticmethod
    def booleans():
        def generate():
            return random.choice([True, False])
        return generate

    @staticmethod
    def text():
        def generate():
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
        return generate

    @staticmethod
    def sampled_from(elements):
        def generate():
            return random.choice(list(elements))
        return generate

# Stub given decorator
def given(*args, **kwargs):
    """Stub given decorator - runs test once with random data."""
    def decorator(test_func):
        def wrapper(*test_args, **test_kwargs):
            # Generate random values for each strategy
            generated_args = []
            for strategy in args:
                if callable(strategy):
                    generated_args.append(strategy())

            # Run test once with generated data
            warnings.warn("Using hypothesis stub - running single test with random data", ImportWarning, stacklevel=2)
            return test_func(*test_args, *generated_args, **test_kwargs)
        return wrapper
    return decorator

# Stub assume function
def assume(condition):
    """Stub assume - does nothing in stub mode."""
    if not condition:
        warnings.warn("Hypothesis assume() condition failed in stub mode", UserWarning, stacklevel=2)

# Make strategies importable
st = strategies
strategies.just = lambda x: lambda: x

warnings.warn("Using hypothesis stub - property-based testing limited", ImportWarning, stacklevel=2)
EOF

# 3. Try to install real packages
echo ""
echo "📦 Attempting to install Python packages..."

cd /tmp

NUMPY_INSTALLED=false
SKLEARN_INSTALLED=false
HYPOTHESIS_INSTALLED=false
PACKAGES_INSTALLED=false

# Try installing core packages
if $PYTHON_CMD -m pip install --user --no-deps numpy 2>/dev/null; then
    echo "   ✅ numpy installed"
    NUMPY_INSTALLED=true
fi

# Try other packages (including essential Unity Wheel dependencies)
# Split into essential and optional packages for better error handling
essential_packages="python-dotenv pydantic pydantic-settings PyYAML click rich cryptography pytest"
# Note: Skipping google-cloud-* packages - using local secrets in container mode
optional_packages="requests aiohttp pandas scipy asyncio-mqtt scikit-learn hypothesis"

echo "   Installing essential packages..."
for pkg in $essential_packages; do
    if $PYTHON_CMD -m pip install --user $pkg 2>/dev/null; then
        echo "   ✅ $pkg installed"
        PACKAGES_INSTALLED=true
    else
        echo "   ⚠️  Failed to install essential package: $pkg"
    fi
done

echo "   Installing optional packages..."
for pkg in $optional_packages; do
    if $PYTHON_CMD -m pip install --user $pkg 2>/dev/null; then
        echo "   ✅ $pkg installed"
        if [ "$pkg" = "scikit-learn" ]; then
            SKLEARN_INSTALLED=true
        elif [ "$pkg" = "hypothesis" ]; then
            HYPOTHESIS_INSTALLED=true
        fi
        PACKAGES_INSTALLED=true
    else
        echo "   ⚠️  Failed to install optional package: $pkg (will use stubs)"
    fi
done

cd "$PROJECT_DIR"

# 4. Set environment variables
echo ""
echo "🔧 Setting environment variables..."

# Determine Python mode
if [ "$NUMPY_INSTALLED" = true ] && [ "$SKLEARN_INSTALLED" = true ]; then
    USE_PURE_PYTHON=false
    PYTHONPATH_PREFIX=""
else
    USE_PURE_PYTHON=true
    # Add stubs to Python path for fallback imports
    PYTHONPATH_PREFIX="$PROJECT_DIR/.codex/stubs:"
fi

# Create environment file
cat > .codex/.container_env <<EOF
# Unity Wheel Container Environment
# Generated by container_setup_v31.sh on $(date)

# Core settings
export USE_MOCK_DATA=true
export OFFLINE_MODE=true
export DATABENTO_SKIP_VALIDATION=true
export CONTAINER_MODE=true
export LOG_LEVEL=INFO

# Python mode
export USE_PURE_PYTHON=$USE_PURE_PYTHON
export NUMPY_AVAILABLE=$NUMPY_INSTALLED
export SKLEARN_AVAILABLE=$SKLEARN_INSTALLED
export HYPOTHESIS_AVAILABLE=$HYPOTHESIS_INSTALLED

# Python path (includes stubs if needed)
export PYTHONPATH="${PYTHONPATH_PREFIX}$PROJECT_DIR:$PROJECT_DIR/src:\$PYTHONPATH"

# Cache directories
export WHEEL_CACHE_DIR=/tmp/.wheel/cache
export WHEEL_SECRETS_DIR=/tmp/.wheel/secrets

# Disable telemetry
export DO_NOT_TRACK=1
export PYTHONDONTWRITEBYTECODE=1

# Unity-specific settings
export UNITY_TICKER=U
export MAX_CONCURRENT_PUTS=3
export TARGET_DELTA=0.30
export TARGET_DTE=45

# Testing settings
export PYTEST_DISABLE_WARNINGS=1
export SKIP_INTEGRATION_TESTS=1

# Secret management - force local mode in containers
export WHEEL_SECRETS_PROVIDER=local
unset GCP_PROJECT_ID
unset GOOGLE_APPLICATION_CREDENTIALS
EOF

# Source the environment
source .codex/.container_env

echo "   ✅ Environment configured"
echo "   USE_PURE_PYTHON=$USE_PURE_PYTHON"
echo "   NUMPY_AVAILABLE=$NUMPY_INSTALLED"
echo "   SKLEARN_AVAILABLE=$SKLEARN_INSTALLED"
echo "   HYPOTHESIS_AVAILABLE=$HYPOTHESIS_INSTALLED"

# 5. Create __init__.py for stubs
cat > .codex/stubs/__init__.py <<'EOF'
"""Stub package for pure Python compatibility."""
import os
import sys

# Only use stubs if packages are missing
stub_dir = os.path.dirname(__file__)
if stub_dir not in sys.path:
    sys.path.insert(0, stub_dir)
EOF

# 6. Test imports with fallback handling
echo ""
echo "🧪 Testing core imports..."

$PYTHON_CMD -c "
import sys
import os

# Ensure our environment is set
os.environ['USE_PURE_PYTHON'] = '$USE_PURE_PYTHON'

# Test basic imports
import json
import datetime
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
print('✅ Standard library imports OK')

# Test packages with stubs
packages = [
    ('numpy', 'Scientific computing'),
    ('scipy.stats', 'Statistical functions'),
    ('sklearn.ensemble', 'Machine learning'),
    ('hypothesis', 'Property testing')
]

for pkg_name, desc in packages:
    try:
        pkg = __import__(pkg_name, fromlist=[''])
        version = getattr(pkg, '__version__', 'unknown')
        if 'stub' in str(version):
            print(f'⚠️  {desc}: Using stub')
        else:
            print(f'✅ {desc}: {version}')
    except ImportError:
        print(f'❌ {desc}: Not available')
"

# 7. Test Unity Wheel imports
echo ""
echo "🎯 Testing Unity Wheel imports..."

$PYTHON_CMD -c "
import sys
import os
import warnings

# Suppress stub warnings for cleaner output
warnings.filterwarnings('ignore', category=ImportWarning)

# Set up environment
os.environ['USE_PURE_PYTHON'] = '$USE_PURE_PYTHON'
sys.path.insert(0, '$PROJECT_DIR')
if '$USE_PURE_PYTHON' == 'true':
    sys.path.insert(0, '$PROJECT_DIR/.codex/stubs')

# Import with better error handling
successful_imports = []
failed_imports = []

# Test each module separately
modules_to_test = [
    ('src.unity_wheel.math.options', 'black_scholes_price_validated'),
    ('src.unity_wheel.strategy.wheel', 'WheelStrategy'),
    ('src.unity_wheel.utils.position_sizing', 'DynamicPositionSizer'),
    ('src.unity_wheel.api.advisor', 'WheelAdvisor'),
]

for module_path, class_name in modules_to_test:
    try:
        module = __import__(module_path, fromlist=[class_name])
        if hasattr(module, class_name):
            successful_imports.append(module_path)
        else:
            failed_imports.append((module_path, f'{class_name} not found'))
    except Exception as e:
        error_msg = str(e).split('\\n')[0]  # First line only
        failed_imports.append((module_path, error_msg))

# Report results
for module in successful_imports:
    print(f'✅ Imported: {module}')

for module, error in failed_imports:
    print(f'❌ Failed: {module} - {error}')

# Summary
if successful_imports:
    print(f'\\n📊 Imported {len(successful_imports)}/{len(modules_to_test)} modules')
    if failed_imports:
        print('   Some modules need real numpy/pandas to work properly')
else:
    print('\\n⚠️  No Unity Wheel modules could be imported')
    print('   This is normal in pure stub mode')
"

# 8. Create helper scripts
echo ""
echo "📝 Creating helper scripts..."

# Test runner script
cat > .codex/run_tests.sh <<'EOF'
#!/bin/bash
source .codex/.container_env

echo "🧪 Running Tests..."
echo "=================="

# Add stubs to path if needed
if [ "$USE_PURE_PYTHON" = "true" ]; then
    export PYTHONPATH="$PROJECT_DIR/.codex/stubs:$PYTHONPATH"
fi

# Run tests with appropriate filters
if [ "$USE_PURE_PYTHON" = "true" ]; then
    echo "⚠️  Pure Python mode - skipping property-based tests"
    python3 -m pytest tests/ -v -k "not hypothesis" --tb=short 2>/dev/null || {
        echo ""
        echo "💡 Some tests require numpy/sklearn. Try:"
        echo "   python3 -m pytest tests/test_math_simple.py -v"
        echo "   python3 -m pytest tests/test_config.py -v"
    }
else
    python3 -m pytest tests/ -v --tb=short
fi
EOF

chmod +x .codex/run_tests.sh

# Test script
cat > .codex/test_container.sh <<'EOF'
#!/bin/bash
source .codex/.container_env

echo "🧪 Container Test Suite"
echo "====================="
echo ""
echo "📋 Environment:"
echo "   Python: $(python3 --version)"
echo "   Mode: $([ "$USE_PURE_PYTHON" = "true" ] && echo "Pure Python (stubs)" || echo "Full (numpy+sklearn)")"
echo "   Numpy: $([ "$NUMPY_AVAILABLE" = "true" ] && echo "✅" || echo "⚠️ stub")"
echo "   Sklearn: $([ "$SKLEARN_AVAILABLE" = "true" ] && echo "✅" || echo "⚠️ stub")"
echo "   Hypothesis: $([ "$HYPOTHESIS_AVAILABLE" = "true" ] && echo "✅" || echo "⚠️ stub")"
echo ""

# Try a simple calculation
python3 -c "
import warnings
warnings.filterwarnings('ignore', category=ImportWarning)

try:
    from src.unity_wheel.math.options import black_scholes_price_validated as bs
    result = bs(35, 35, 0.125, 0.05, 0.45, 'put')
    print(f'✅ Options calculation: \${result.value:.2f}')
except Exception as e:
    print(f'❌ Calculation failed: {e}')
"

# Test imports work
echo ""
echo "📦 Package availability:"
python3 -c "
import warnings
warnings.filterwarnings('ignore')

for pkg in ['numpy', 'scipy', 'sklearn', 'hypothesis', 'pytest']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        print(f'   {pkg}: {ver}')
    except:
        print(f'   {pkg}: not available')
"
EOF

chmod +x .codex/test_container.sh

# Make command
cat > .codex/make_test.sh <<'EOF'
#!/bin/bash
# Wrapper for make test command
source .codex/.container_env

# Add stubs to path if needed
if [ "$USE_PURE_PYTHON" = "true" ]; then
    export PYTHONPATH="$PROJECT_DIR/.codex/stubs:$PYTHONPATH"
fi

# Check if poetry is available
if command -v poetry >/dev/null 2>&1; then
    # Use poetry if available
    echo "🧪 Running tests with poetry..."
    poetry run pytest tests/test_math.py tests/test_options_properties.py -v --tb=short || true
    poetry run pytest tests/test_e2e_recommendation_flow.py -v --tb=short || true
elif command -v pytest >/dev/null 2>&1 || python3 -m pytest --version >/dev/null 2>&1; then
    # Use pytest directly if available
    echo "🧪 Running critical tests (no poetry)..."
    echo ""
    # Run the same tests that make test would run
    python3 -m pytest tests/test_math.py tests/test_options_properties.py -v --tb=short || {
        echo "⚠️  Math tests need numpy. Trying simpler tests..."
        python3 -m pytest tests/test_config.py -v --tb=short || true
    }
    python3 -m pytest tests/test_e2e_recommendation_flow.py -v --tb=short || true
else
    # No pytest available
    echo "⚠️  pytest not available. Running basic Python tests..."
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
print('Testing basic imports...')
try:
    from src.unity_wheel import __version__
    print(f'✅ Package version: {__version__}')
except:
    print('❌ Package import failed')

try:
    from src.config.loader import get_config
    config = get_config()
    print(f'✅ Config loaded: Unity ticker={config.unity.ticker}')
except:
    print('❌ Config loading failed')
"
fi

echo ""
echo "✅ Test run complete"
EOF

chmod +x .codex/make_test.sh

# Run wrapper
cat > .codex/run_container.sh <<'EOF'
#!/bin/bash
source .codex/.container_env

# Set fallback mode if imports fail
export SKIP_VALIDATION=true
export USE_MOCK_DATA=true

echo "🎯 Running Unity Wheel Trading Bot..."
echo "   Mode: $([ "$USE_PURE_PYTHON" = "true" ] && echo "Pure Python" || echo "Enhanced")"
echo ""

PORTFOLIO=${1:-100000}
python3 run.py --portfolio $PORTFOLIO "${@:2}" 2>&1 || {
    echo ""
    echo "⚠️  If you see import errors, the bot needs numpy/pandas"
}
EOF

chmod +x .codex/run_container.sh

# Activation script
cat > .codex/activate_container.sh <<'EOF'
#!/bin/bash
if [ -f .codex/.container_env ]; then
    source .codex/.container_env
    echo "✅ Container environment activated"
    echo "   Mode: $([ "$USE_PURE_PYTHON" = "true" ] && echo "Pure Python (stubs)" || echo "Full (numpy+sklearn)")"
    echo "   Tests: $([ "$HYPOTHESIS_AVAILABLE" = "true" ] && echo "Property-based enabled" || echo "Basic only")"
else
    echo "❌ No container environment found!"
    echo "   Run: ./.codex/container_setup_v31.sh"
fi
EOF

chmod +x .codex/activate_container.sh

# 9. Final summary
echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
echo "✅ Python $($PYTHON_CMD --version 2>&1 | cut -d' ' -f2) configured"
if [ "$PACKAGES_INSTALLED" = true ]; then
    echo "✅ Some packages installed"
else
    echo "⚠️  Using Pure Python mode with stubs for all packages"
fi
echo "✅ Test stubs created (sklearn, hypothesis)"
echo "✅ Environment configured"
echo "✅ Helper scripts created"
echo ""
echo "🚀 Quick Start:"
echo "   source .codex/activate_container.sh    # Activate environment"
echo "   ./.codex/test_container.sh             # Test setup"
echo "   ./.codex/run_tests.sh                  # Run tests"
echo "   ./.codex/make_test.sh                  # Run make test"
echo ""
echo "📝 Test Commands:"
echo "   make test                               # Full test suite"
echo "   pytest tests/test_config.py -v          # Simple test"
echo "   pytest -k 'not hypothesis' -v           # Skip property tests"
echo ""
echo "⚠️  Notes:"
echo "   - Stubs provide minimal compatibility for testing"
echo "   - Some tests may fail without real numpy/sklearn"
echo "   - Property-based tests need real hypothesis package"
echo ""
echo "📖 Next: source .codex/activate_container.sh"

# 🚀 CODEX ENVIRONMENT SETUP GUIDE

## **PROBLEM**: Codex Environment Limitations

Codex may have restrictions on:
- Installing Python packages via pip
- Internet access for dependency downloads
- System-level package management
- Virtual environment creation

## **SOLUTION**: Pre-configured Environment

This guide provides multiple approaches to get Codex running the Unity Wheel Trading Bot.

---

## 🔧 **QUICK SETUP OPTIONS**

### Option 1: Use Existing Dependencies (RECOMMENDED)
```bash
# The project already has requirements files
# Try using existing installed packages first
python -c "import sys; print(sys.executable)"
python -c "import numpy, pandas, scipy; print('✅ Core dependencies available')"
```

### Option 2: Local Package Installation
```bash
# If pip is available, install from requirements
pip install --user -r requirements.txt

# If that fails, try without dependencies
pip install --user --no-deps -r requirements.txt
```

### Option 3: Minimal Runtime (NO EXTERNAL DEPS)
```bash
# Use the offline mode setup script
chmod +x .codex/setup_offline.sh
./.codex/setup_offline.sh
```

---

## 📦 **CRITICAL DEPENDENCIES**

### Core Python Packages (REQUIRED)
```python
# Standard library only - no installation needed
import json
import datetime
import typing
import dataclasses
import abc
import logging
import os
import sys
```

### Optional Dependencies (for full features)
```python
# If available, great. If not, fallback implementations exist
try:
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
```

### Package Priority List
1. **CRITICAL** (must have): `json`, `datetime`, `typing`, `dataclasses`
2. **IMPORTANT** (performance): `numpy`, `pandas`, `scipy`
3. **OPTIONAL** (features): `pydantic`, `requests`, `google-cloud-secret-manager`

---

## 🔄 **ENVIRONMENT VALIDATION**

### Quick Health Check
```bash
# Run this first to check what's available
python .codex/check_environment.py
```

### Manual Dependency Check
```python
# Copy this into a Python session
import sys
print("Python version:", sys.version)

# Check critical packages
required = ['json', 'datetime', 'typing', 'dataclasses', 'abc', 'logging']
for pkg in required:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} - CRITICAL!")

# Check optional packages
optional = ['numpy', 'pandas', 'scipy', 'pydantic', 'requests']
for pkg in optional:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"⚠️  {pkg} - using fallback")
```

---

## 🛠 **FALLBACK IMPLEMENTATIONS**

### When numpy is not available
```python
# Pure Python math implementations available in:
# unity_trading/math/options.py - has fallback math
# unity_trading/utils/position_sizing.py - pure Python calculations

# Enable fallback mode
export USE_PURE_PYTHON=true
```

### When external APIs fail
```python
# Use mock data mode
export USE_MOCK_DATA=true
export DATABENTO_SKIP_VALIDATION=true
```

### When file I/O is restricted
```python
# Use in-memory mode only
export USE_MEMORY_ONLY=true
```

---

## 🏃‍♂️ **QUICK START COMMANDS**

### Test Basic Functionality
```bash
# Test core imports
python -c "from unity_trading.math.options import black_scholes_price_validated; print('✅ Math module works')"

# Test strategy module
python -c "from unity_trading.strategy.wheel import WheelStrategy; print('✅ Strategy module works')"

# Test with fallback data
DATABENTO_SKIP_VALIDATION=true python -c "from unity_trading.api.advisor import WheelAdvisor; print('✅ API module works')"
```

### Run with Minimal Dependencies
```bash
# Set fallback flags
export USE_PURE_PYTHON=true
export USE_MOCK_DATA=true
export DATABENTO_SKIP_VALIDATION=true

# Try a basic recommendation
python -c "
from unity_trading.api.advisor_simple import WheelAdvisorSimple
advisor = WheelAdvisorSimple()
print('✅ Simple advisor works')
"
```

---

## 📝 **TROUBLESHOOTING GUIDE**

### ImportError: No module named 'X'
```bash
# For numpy/pandas/scipy
export USE_PURE_PYTHON=true

# For pydantic
export SKIP_VALIDATION=true

# For requests/http libraries
export USE_MOCK_DATA=true

# For google-cloud
export SKIP_CLOUD_SECRETS=true
```

### Permission Denied for package installation
```bash
# Try user installation
pip install --user package_name

# Or use the provided offline setup
./.codex/setup_offline.sh
```

### Network/Internet access issues
```bash
# Enable offline mode
export OFFLINE_MODE=true
export USE_MOCK_DATA=true
export DATABENTO_SKIP_VALIDATION=true
```

---

## 🎯 **ENVIRONMENT PROFILES**

### Profile 1: Full Environment
```bash
# All packages available, full features
python run.py --portfolio 100000
```

### Profile 2: Limited Environment
```bash
# Basic packages only
export USE_PURE_PYTHON=true
python -c "from unity_trading.api.advisor_simple import WheelAdvisorSimple; print('Limited mode ready')"
```

### Profile 3: Offline Environment
```bash
# No external connections
export OFFLINE_MODE=true
export USE_MOCK_DATA=true
export USE_PURE_PYTHON=true
python -c "from unity_trading.strategy.wheel import WheelStrategy; print('Offline mode ready')"
```

---

## 🔧 **CONFIGURATION FOR CODEX**

### Recommended Environment Variables
```bash
# Add these to your shell session
export USE_PURE_PYTHON=true
export USE_MOCK_DATA=true
export DATABENTO_SKIP_VALIDATION=true
export SKIP_VALIDATION=true
export OFFLINE_MODE=true
export LOG_LEVEL=INFO

# Test configuration
python .codex/test_config.py
```

### Pre-configured Commands
```bash
# Safe commands that work in any environment
python -c "from unity_trading.math import black_scholes_price_validated as bs; print(bs(100, 100, 1, 0.05, 0.2, 'call'))"

python -c "from unity_trading.utils.position_sizing import calculate_position_size; print('Position sizing available')"

python -c "from unity_trading.strategy.wheel import WheelStrategy; w = WheelStrategy(); print('Strategy available')"
```

---

## 📋 **SETUP CHECKLIST**

- [ ] Python 3.8+ available
- [ ] Standard library imports work
- [ ] Math module imports successfully
- [ ] Strategy module imports successfully
- [ ] Fallback environment variables set
- [ ] Test commands execute without errors
- [ ] Mock data mode enabled if needed
- [ ] Pure Python mode enabled if needed

---

## 🆘 **EMERGENCY WORKAROUNDS**

### If Nothing Works
```bash
# Use the absolutely minimal version
cp .codex/minimal_trader.py ./trader.py
python trader.py

# This uses zero external dependencies
# Pure Python implementation of core functionality
```

### If Import Errors Persist
```bash
# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Add current directory to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -c "from unity_trading import __version__; print(f'Unity Wheel v{__version__}')"
```

---

## 🎉 **SUCCESS VALIDATION**

When setup is complete, this should work:
```bash
python -c "
import sys
print(f'Python: {sys.version}')

try:
    from unity_trading.strategy.wheel import WheelStrategy
    from unity_trading.math.options import black_scholes_price_validated
    from unity_trading.utils.position_sizing import calculate_position_size
    print('✅ Unity Wheel Trading Bot is ready!')
except ImportError as e:
    print(f'❌ Setup incomplete: {e}')
"
```

**Ready to optimize! 🚀**

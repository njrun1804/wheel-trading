# Import Path Standardization Guide

## Overview

This document defines the standardized import path conventions for the Unity Wheel Trading Bot codebase to ensure consistency, maintainability, and prevent import errors.

## Standardized Import Patterns

### 1. Core Application Modules

```python
# Configuration
from src.config.loader import get_config
from src.config.schema import WheelConfig

# Math and Options
from src.unity_wheel.math.options import black_scholes_price_validated, calculate_all_greeks
from src.unity_wheel.math.vectorized_options import vectorized_black_scholes

# Strategy Components  
from src.unity_wheel.strategy.wheel import WheelStrategy
from src.unity_wheel.api.advisor import WheelAdvisor

# Risk Management
from src.unity_wheel.risk.analytics import RiskAnalytics
from src.unity_wheel.risk.limits import RiskLimits

# Data Providers
from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.data_providers.schwab.client import SchwabClient
```

### 2. Utility Modules

```python
# Logging
from src.unity_wheel.utils.logging import StructuredLogger

# Performance and Memory
from src.unity_wheel.utils.performance_cache import cached, get_cache_manager
from src.unity_wheel.utils.memory_optimizer import MemoryMonitor

# Trading Calendar
from src.unity_wheel.utils.trading_calendar import is_trading_day, get_next_expiry

# Validation
from src.unity_wheel.utils.validate import validate_market_data
```

### 3. Models and Data Types

```python
# Core Models
from src.unity_wheel.models.position import Position
from src.unity_wheel.models.account import Account
from src.unity_wheel.models.greeks import Greeks

# Databento Types
from src.unity_wheel.data_providers.databento.types import (
    OptionChain,
    OptionQuote, 
    InstrumentDefinition
)
```

### 4. Authentication and Secrets

```python
# Secrets Management
from src.unity_wheel.secrets.manager import SecretManager
from src.unity_wheel.secrets.integration import get_schwab_credentials

# Authentication
from src.unity_wheel.auth.client_v2 import AuthClient
```

## Import Path Rules

### 1. Always Use Absolute Imports

❌ **Incorrect:**
```python
from ..utils import logging
from .client import DatabentoClient
```

✅ **Correct:**
```python
from src.unity_wheel.utils.logging import StructuredLogger
from src.unity_wheel.data_providers.databento.client import DatabentoClient
```

### 2. Import Specific Classes/Functions

❌ **Incorrect:**
```python
import src.unity_wheel.math.options
```

✅ **Correct:**
```python
from src.unity_wheel.math.options import black_scholes_price_validated
```

### 3. Group Imports Logically

```python
# Standard library imports
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
import pytest

# Local application imports
from src.config.loader import get_config
from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.math.options import black_scholes_price_validated
```

### 4. Avoid Circular Imports

- Use dependency injection instead of direct imports
- Move shared dependencies to separate modules
- Use type hints with `TYPE_CHECKING` for circular references

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.unity_wheel.strategy.wheel import WheelStrategy
```

## Fixed Import Issues

### 1. ✅ Databento Unity Integration

**Previous (Broken):**
```python
from src.unity_wheel.data_providers.databento.unity_utils import (
    chain,
    get_wheel_candidates,
    spot
)
```

**Fixed:**
```python
from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration
from src.unity_wheel.data_providers.databento.client import DatabentoClient
from src.unity_wheel.data_providers.databento.types import OptionChain, OptionQuote
```

### 2. ✅ Test Module Imports

**Previous (Broken):**
```python
from unity_wheel.utils.databento_unity import API
```

**Fixed:**
```python
from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration
```

## Module Directory Structure Reference

```
src/
├── config/
│   ├── loader.py           # get_config()
│   ├── schema.py           # WheelConfig, config schemas
│   └── minimal.py          # MinimalWheelConfig
├── unity_wheel/
│   ├── math/
│   │   ├── options.py      # black_scholes_price_validated, calculate_all_greeks
│   │   └── vectorized_options.py  # vectorized calculations
│   ├── strategy/
│   │   └── wheel.py        # WheelStrategy
│   ├── api/
│   │   ├── advisor.py      # WheelAdvisor
│   │   └── types.py        # API types
│   ├── risk/
│   │   ├── analytics.py    # RiskAnalytics
│   │   └── limits.py       # RiskLimits
│   ├── data_providers/
│   │   ├── databento/
│   │   │   ├── integration.py  # DatabentoIntegration
│   │   │   ├── client.py       # DatabentoClient
│   │   │   └── types.py        # OptionChain, OptionQuote
│   │   └── schwab/
│   │       └── client.py   # SchwabClient
│   ├── utils/
│   │   ├── logging.py      # StructuredLogger
│   │   ├── performance_cache.py  # cached, MemoryAwareLRUCache
│   │   ├── memory_optimizer.py   # MemoryMonitor
│   │   └── trading_calendar.py   # is_trading_day
│   ├── models/
│   │   ├── position.py     # Position
│   │   ├── account.py      # Account
│   │   └── greeks.py       # Greeks
│   ├── secrets/
│   │   ├── manager.py      # SecretManager
│   │   └── integration.py  # get_schwab_credentials
│   └── auth/
│       └── client_v2.py    # AuthClient
```

## Common Import Anti-Patterns to Avoid

### 1. ❌ Importing Non-Existent Modules

```python
# These modules don't exist:
from src.unity_wheel.data_providers.databento.unity_utils import chain
from src.unity_wheel.utils.databento_unity import API
```

### 2. ❌ Wildcard Imports

```python
from src.unity_wheel.math.options import *
```

### 3. ❌ Mixing Relative and Absolute Imports

```python
from src.unity_wheel.math.options import black_scholes_price_validated
from ..utils import logging  # Don't mix!
```

### 4. ❌ Importing Internal Implementation Details

```python
# Don't import private methods or internals
from src.unity_wheel.strategy.wheel import _calculate_position_size
```

## Testing Import Standards

### Test File Imports

```python
# Standard test imports
import pytest
from unittest.mock import Mock, patch, AsyncMock

# Import the actual classes/functions being tested
from src.unity_wheel.data_providers.databento.integration import DatabentoIntegration
from src.unity_wheel.math.options import black_scholes_price_validated

# Test-specific utilities
from tests.conftest import mock_market_data
```

### Async Test Support

```python
# For testing async methods
@pytest.mark.asyncio
async def test_async_method():
    result = await some_async_function()
    assert result is not None
```

## IDE Configuration

### VS Code settings.json

```json
{
    "python.analysis.extraPaths": ["src"],
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black"
}
```

### PyCharm

- Mark `src` directory as "Sources Root"
- Enable absolute imports in Code Style > Python > Imports

## Validation Commands

### Check for Import Issues

```bash
# Find broken imports
python -m py_compile src/**/*.py

# Check with mypy
mypy src/

# Run import validation
python -c "
import sys
sys.path.insert(0, 'src')
from unity_wheel.math.options import black_scholes_price_validated
print('✅ Imports working correctly')
"
```

### Test Import Standardization

```bash
# Run tests to verify imports
pytest tests/test_databento_unity.py -v

# Check all test files
pytest --collect-only
```

## Migration Checklist

When updating import paths:

- [ ] Update all import statements to use absolute paths
- [ ] Verify the imported modules actually exist
- [ ] Update corresponding test files
- [ ] Run import validation commands
- [ ] Update documentation examples
- [ ] Check for circular import issues

## Benefits of Standardized Imports

1. **Consistency**: All team members use the same import patterns
2. **Maintainability**: Easy to refactor and move modules
3. **IDE Support**: Better autocomplete and navigation
4. **Error Prevention**: Catch import errors early
5. **Performance**: Faster import resolution
6. **Testing**: Easier to mock and test components

## Contact

For questions about import standards or to report import issues:
- Create an issue in the project repository
- Tag `@import-standards` in pull requests
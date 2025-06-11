# Migration Guide - January 2025 Architectural Updates

This guide helps you migrate from the old architecture patterns to the new, cleaner implementation.

## Overview of Changes

The January 2025 update includes major architectural improvements:
1. Centralized configuration management
2. Proper dependency injection
3. Consolidated adaptive logic
4. Clear async/sync boundaries
5. Cleaned file structure

## Migration Steps

### 1. Configuration Access

**Old Pattern:**
```python
from src.config.loader import get_config, get_config_loader

config = get_config()
loader = get_config_loader()
```

**New Pattern:**
```python
from src.config import get_config, get_config_service

# Simple usage (same as before)
config = get_config()

# Advanced usage with service
service = get_config_service()
health = service.get_health_report()
service.reload()  # Reload configuration
```

### 2. Adaptive System Imports

**Old Pattern:**
```python
from src.unity_wheel.strategy.adaptive_base import AdaptiveBase
from src.unity_wheel.risk.regime_detector import RegimeDetector
from src.unity_wheel.analytics.dynamic_optimizer import DynamicOptimizer
```

**New Pattern:**
```python
# All adaptive imports from one module
from src.unity_wheel.adaptive import (
    AdaptiveBase,
    RegimeDetector,
    DynamicOptimizer,
    get_volatility_tier,
    should_trade_unity
)
```

### 3. Dependency Injection

**Old Pattern (Lazy Imports):**
```python
# In advisor.py
def _get_market_validator():
    global _market_validator
    if _market_validator is None:
        from ..data_providers.base import get_market_validator
        _market_validator = get_market_validator()
    return _market_validator
```

**New Pattern (Dependency Injection):**
```python
from src.unity_wheel.api.dependencies import create_dependencies
from src.unity_wheel.api.advisor import WheelAdvisor

# Create advisor with custom dependencies
deps = create_dependencies(
    market_validator=MyCustomValidator(),
    wheel_parameters=WheelParameters(target_delta=0.40)
)
advisor = WheelAdvisor(dependencies=deps)
```

### 4. Testing with Mocks

**Old Pattern:**
```python
# Hard to mock due to lazy imports
advisor = WheelAdvisor()
# No easy way to inject mocks
```

**New Pattern:**
```python
from unittest.mock import Mock
from src.unity_wheel.api.dependencies import create_dependencies

# Easy mock injection
mock_validator = Mock()
mock_validator.validate.return_value = Mock(is_valid=True)

deps = create_dependencies(market_validator=mock_validator)
advisor = WheelAdvisor(dependencies=deps)
```

### 5. Async/Sync Data Access

**Old Pattern:**
```python
# Mixed async/sync without clear boundaries
data = await databento_client.get_data()  # Sometimes async
data = databento_client.get_data()        # Sometimes sync
```

**New Pattern:**
```python
# Clear interfaces
from src.unity_wheel.data_providers.base.interfaces import AsyncDataProvider

# Async provider with sync wrappers
class DatabentoClient(AsyncDataProvider):
    async def get_option_chain(self, symbol):  # Async method
        ...

    # Auto-generated sync wrapper
    # get_option_chain_sync() is available

# Use cached provider for performance
from src.unity_wheel.data_providers.base.interfaces import CachedDataProvider
cached = CachedDataProvider(databento_client)
data = cached.get_option_chain("U")  # Cached sync access
```

### 6. Import Updates

**Update these imports in your code:**

| Old Import | New Import |
|------------|------------|
| `from src.unity_wheel.risk.regime_detector import RegimeDetector` | `from src.unity_wheel.adaptive import RegimeDetector` |
| `from src.unity_wheel.analytics.dynamic_optimizer import DynamicOptimizer` | `from src.unity_wheel.adaptive import DynamicOptimizer` |
| `from src.config.loader import get_config` | `from src.config import get_config` |
| `from src.unity_wheel.risk.limits import RiskLimits` | `from src.unity_wheel.risk.limits import TradingLimits` |

## File Structure Changes

### Removed Files/Directories
- All files with " 2" suffix (duplicates)
- `ml_engine/`, `risk_engine/`, `strategy_engine/` directories
- `Documents/com~apple~CloudDocs/` nested structure

### Moved Files
- `src/unity_wheel/risk/regime_detector.py` → `src/unity_wheel/adaptive/regime_detector.py`
- `src/unity_wheel/analytics/dynamic_optimizer.py` → `src/unity_wheel/adaptive/dynamic_optimizer.py`

### New Files
- `src/config/service.py` - ConfigurationService singleton
- `src/unity_wheel/api/dependencies.py` - Dependency injection
- `src/unity_wheel/data_providers/base/interfaces.py` - Async/sync interfaces
- `docs/async_sync_boundaries.md` - Architecture documentation

## Benefits of Migration

1. **Better Testing**: Easy to inject mocks and test in isolation
2. **No Circular Imports**: Dependency injection eliminates circular dependencies
3. **Performance**: ConfigurationService provides cached access
4. **Clear Architecture**: Obvious boundaries between async I/O and sync calculations
5. **Maintainability**: All adaptive logic in one place

## Quick Checklist

- [ ] Update all imports to new locations
- [ ] Replace lazy imports with dependency injection
- [ ] Use ConfigurationService for config access
- [ ] Update tests to use dependency injection
- [ ] Remove references to deleted files
- [ ] Test your code with the new architecture

## Need Help?

- See `ARCHITECTURE_IMPROVEMENTS_SUMMARY.md` for detailed changes
- Check `docs/async_sync_boundaries.md` for async/sync patterns
- Review `src/unity_wheel/adaptive/README.md` for adaptive system usage
- Look at test files for examples of the new patterns

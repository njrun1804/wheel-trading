# Unity Wheel Trading Bot - File Structure Optimization Plan

## Overview

This plan optimizes the file structure for better discoverability by Claude Code CLI and improved maintainability. The reorganization focuses on:
- Clear module boundaries
- Consistent naming patterns
- Logical grouping of related functionality
- Reduced file path ambiguity

## Priority 1: Critical Moves (Improves Discoverability)

### 1.1 Move Root-Level Files

These files at the project root make it harder to understand the codebase structure:

```bash
# Move monitoring scripts to proper module
mkdir -p src/unity_wheel/monitoring/scripts
git mv src/unity_wheel/monitoring/scripts/live_monitor.py src/unity_wheel/monitoring/scripts/live_monitor.py
git mv src/unity_wheel/monitoring/scripts/data_quality_monitor.py src/unity_wheel/monitoring/scripts/data_quality_monitor.py
git mv src/unity_wheel/monitoring/scripts/daily_health_check.py src/unity_wheel/monitoring/scripts/daily_health_check.py

# Move diagnostics to monitoring
git mv src/unity_wheel/monitoring/diagnostics.py src/unity_wheel/monitoring/diagnostics.py

# Move validate to utils
git mv src/unity_wheel/utils/validate.py src/unity_wheel/utils/validate.py

# Move adaptive.py to strategy (already have adaptive/ dir there)
git mv src/unity_wheel/strategy/adaptive_base.py src/unity_wheel/strategy/adaptive_base.py
```

### 1.2 Create CLI Module for Entry Points

```bash
# Create CLI module for main entry points
mkdir -p src/unity_wheel/cli
git mv run.py src/unity_wheel/cli/run_legacy.py
git mv run.py src/unity_wheel/cli/run.py

# Create simple entry point scripts at root
cat > run.py << 'EOF'
#!/usr/bin/env python3
"""Main entry point for Unity Wheel Trading Bot."""
from src.unity_wheel.cli.run import main

if __name__ == "__main__":
    main()
EOF

chmod +x run.py
```

## Priority 2: Consolidate Data Providers

### 2.1 Unify Data Provider Structure

Currently data providers are scattered. Create unified structure:

```bash
# Create unified data providers structure
mkdir -p src/unity_wheel/data_providers/{databento,fred,schwab,base}

# Move Databento
git mv src/unity_wheel/data_providers/databento/* src/unity_wheel/data_providers/databento/
rmdir src/unity_wheel/databento

# Move FRED
git mv src/unity_wheel/data_providers/base/fred*.py src/unity_wheel/data_providers/fred/
git mv src/unity_wheel/data_providers/base/*.py src/unity_wheel/data_providers/base/
rmdir src/unity_wheel/data

# Move Schwab data modules
git mv src/unity_wheel/schwab/data_fetcher.py src/unity_wheel/data_providers/schwab/fetcher.py
git mv src/unity_wheel/schwab/data_ingestion.py src/unity_wheel/data_providers/schwab/ingestion.py

# Move Unity-specific data utils
git mv src/unity_wheel/utils/databento_unity.py src/unity_wheel/data_providers/databento/unity_utils.py
```

### 2.2 Create Common Data Interface

```python
# src/unity_wheel/data_providers/base/interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

class DataProvider(ABC):
    """Common interface for all data providers."""

    @abstractmethod
    async def get_price_history(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical price data."""
        pass

    @abstractmethod
    async def get_option_chain(self, symbol: str, expiration: datetime) -> Dict:
        """Get option chain data."""
        pass
```

## Priority 3: Resolve Naming Conflicts

### 3.1 Rename Duplicate Files

Multiple `client.py`, `exceptions.py`, `storage.py`, and `types.py` files cause confusion:

```bash
# Option 1: Use descriptive names
git mv src/unity_wheel/auth/client.py src/unity_wheel/auth/auth_client.py
git mv src/unity_wheel/schwab/client.py src/unity_wheel/schwab/schwab_client.py
git mv src/unity_wheel/data_providers/databento/client.py src/unity_wheel/data_providers/databento/databento_client.py

# Update imports in all files
find . -name "*.py" -exec sed -i '' 's/from \.client import/from \.auth_client import/g' {} \;
# (Repeat for other renames)
```

## Priority 4: Consolidate Storage/Caching

### 4.1 Unify Storage Module

```bash
# Move all storage-related code to unified module
mkdir -p src/unity_wheel/storage/{adapters,cache}

# Move storage implementations
git mv src/unity_wheel/storage/*.py src/unity_wheel/storage/adapters/
git mv src/unity_wheel/auth/storage.py src/unity_wheel/storage/auth_storage.py
git mv src/unity_wheel/auth/cache.py src/unity_wheel/storage/cache/auth_cache.py
git mv src/unity_wheel/utils/cache.py src/unity_wheel/storage/cache/general_cache.py
```

## Priority 5: Organize Tools Directory

### 5.1 Restructure Tools

```bash
# Create clear tool categories
mkdir -p tools/{data_collection,analysis,diagnostics,setup}

# Move data collection scripts
git mv tools/analysis/pull_*.py tools/data_collection/
git mv tools/analysis/fetch_*.py tools/data_collection/
git mv tools/data/*.py tools/data_collection/
rmdir tools/data

# Move diagnostic tools
git mv tools/debug/* tools/diagnostics/
git mv tools/verification/* tools/diagnostics/
rmdir tools/debug tools/verification

# Move setup tools
git mv scripts/setup-secrets.py tools/setup/
git mv scripts/test-secrets.py tools/setup/
```

## File Structure After Optimization

```
wheel-trading/
├── run.py                          # Simple entry point
├── src/
│   └── unity_wheel/
│       ├── __init__.py
│       ├── cli/                    # Command-line interface
│       │   ├── run.py              # Main entry point
│       │   └── run_legacy.py       # Legacy entry point
│       ├── analytics/              # Analysis modules
│       │   ├── market/             # Market analysis
│       │   ├── portfolio/          # Portfolio analysis
│       │   └── optimization/       # Optimization algorithms
│       ├── api/                    # External API interface
│       ├── auth/                   # Authentication (renamed files)
│       ├── data_providers/         # Unified data access
│       │   ├── base/               # Common interfaces
│       │   ├── databento/          # Databento integration
│       │   ├── fred/               # FRED integration
│       │   └── schwab/             # Schwab data only
│       ├── math/                   # Mathematical functions
│       ├── models/                 # Data models
│       ├── monitoring/             # All monitoring/diagnostics
│       │   ├── performance.py
│       │   ├── dashboard.py
│       │   ├── diagnostics.py
│       │   └── scripts/            # Monitoring scripts
│       ├── schwab/                 # Schwab broker interface
│       ├── secrets/                # Secret management
│       ├── storage/                # Unified storage/caching
│       │   ├── adapters/           # Storage adapters
│       │   └── cache/              # Caching implementations
│       ├── strategy/               # Trading strategies
│       │   ├── wheel.py
│       │   ├── adaptive_base.py
│       │   └── adaptive/           # Adaptive strategies
│       └── utils/                  # General utilities
├── tests/                          # Tests (consider organizing later)
├── tools/                          # Development tools
│   ├── data_collection/            # Data fetching scripts
│   ├── analysis/                   # Analysis scripts
│   ├── diagnostics/                # Debug/verification tools
│   └── setup/                      # Setup and configuration
└── scripts/                        # Shell scripts for operations

```

## Benefits for Claude Code CLI

1. **Clear Module Boundaries**: Each module has a single responsibility
2. **Reduced Ambiguity**: No duplicate filenames in different locations
3. **Logical Grouping**: Related functionality is co-located
4. **Predictable Structure**: Easy to guess where functionality lives
5. **Shallow Nesting**: Most files are only 3-4 levels deep

## Implementation Steps

1. **Create backup branch**: `git checkout -b file-structure-optimization`
2. **Run housekeeping check**: `./scripts/housekeeping.sh --quick`
3. **Execute Priority 1 moves** (most important for discoverability)
4. **Update imports** using automated tools
5. **Run tests** to ensure nothing broke
6. **Execute remaining priorities** based on time/need
7. **Update CLAUDE.md** with new structure

## Automated Import Update Script

```python
#!/usr/bin/env python3
"""Update imports after file moves."""
import os
import re
from pathlib import Path

# Define mappings of old imports to new imports
IMPORT_MAPPINGS = {
    'from src.unity_wheel.databento': 'from src.unity_wheel.data_providers.databento',
    'from ..databento': 'from ..data_providers.databento',
    'from .client import': 'from .auth_client import',
    # Add more mappings as needed
}

def update_imports(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()

    for old, new in IMPORT_MAPPINGS.items():
        content = re.sub(old, new, content)

    with open(file_path, 'w') as f:
        f.write(content)

# Find all Python files and update imports
for py_file in Path('src').rglob('*.py'):
    update_imports(py_file)
```

## Notes

- Some moves may require updating configuration files
- The housekeeping script may need updates to reflect new structure
- Consider updating pre-commit hooks to enforce structure
- Update documentation to reflect new organization

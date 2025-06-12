# Codebase Structure Analysis

## Directory Size Analysis

### 1. Largest Code Directories (Token Budget Bottlenecks)

**Top 5 by Size:**
1. **data_providers/** - 444K, 33 files
   - Heavy import usage (17 imports in databento/client.py)
   - Multiple provider integrations
   - Significant file I/O operations

2. **risk/** - 316K, 18 files
   - Complex analytics (15 imports in analytics.py)
   - Multiple risk management modules
   - Database connections

3. **analytics/** - 164K, 12 files
   - Decision engine complexity (13 imports)
   - Performance tracking with 8 DB connections
   - Heavy computational modules

4. **utils/** - 148K, 13 files
   - Shared utilities across codebase
   - Moderate file operations

5. **monitoring/** - 148K, 10 files
   - Diagnostics heavy on imports (16)
   - Performance monitoring
   - Live monitoring scripts

### 2. File Patterns for .claudeignore

**Cache and Temporary Files (1,683+ files found):**
- `__pycache__/`
- `*.pyc`
- `*.pyo`
- `*.egg-info/`
- `.DS_Store`

**Large Data Directories:**
- `data/cache/` (79MB)
- `data/archive_*/` (30MB+)
- `data/unity-options/` (11MB)
- `*.duckdb` (except wheel_trading_master.duckdb)

**Build and Test Artifacts:**
- `.pytest_cache/`
- `.coverage`
- `htmlcov/`
- `*.log`
- `*.tmp`
- `*.bak`

**Scripts Directory Candidates:**
- `scripts/*-mcp-*.sh` (numerous MCP-related scripts)
- `scripts/start-claude-*.sh` (multiple variants)

### 3. File Descriptor Usage Patterns

**High File I/O Files:**
1. **observability/dashboard.py** - 12 file operations + 4 DB connections
2. **secrets/manager.py** - 8 file operations
3. **storage/cache/auth_cache.py** - 5 file operations

**Database Connection Heavy:**
1. **analytics/performance_tracker.py** - 8 connections
2. **mcp/unified_cache.py** - 7 connections
3. **observability/dashboard.py** - 4 connections

**Network Connection Files:**
1. **auth/oauth.py** - 10 network operations
2. **auth/auth_client.py** - 7 network operations

## Recommendations

### For .claudeignore:
```
# Cache and temporary files
__pycache__/
*.pyc
*.pyo
*.egg-info/
.DS_Store
.pytest_cache/
.coverage
htmlcov/

# Large data files
data/cache/
data/archive_*/
data/unity-options/
*.duckdb
!data/wheel_trading_master.duckdb

# Logs and temporary files
*.log
*.tmp
*.bak
*.swp

# Build artifacts
build/
dist/
*.egg

# MCP-related scripts (numerous duplicates)
scripts/*-mcp-*.sh
scripts/start-claude-*.sh
!scripts/start-claude.sh

# Test data
test_data/
tests/fixtures/large_*.json
```

### Token Budget Optimization:
1. **Focus on core modules:** strategy/, api/, risk/
2. **Exclude data_providers/** unless specifically working on integrations
3. **Minimize monitoring/** and observability/** inclusion
4. **Use targeted file reads instead of directory-wide context**

### File Descriptor Management:
1. **Connection pooling needed for:**
   - DuckDB connections in analytics modules
   - OAuth/auth client connections
   - Data provider API connections

2. **Cache file operations in:**
   - secrets/manager.py
   - storage/cache modules
   - observability/dashboard.py
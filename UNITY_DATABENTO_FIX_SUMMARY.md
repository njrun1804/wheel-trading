# Unity Options Data Access Fix Summary

## Problem
Unity (U) options queries were failing in Databento due to incorrect symbol format and exchange specification.

## Solution Implemented

### 1. Symbol Format Handling (databento/client.py)
- Added multiple symbol format attempts for Unity:
  - `U.OPT` (standard format) - tried first
  - `U     *` (Unity with 5 spaces for OCC compatibility)
  - `U` (raw symbol as fallback)
- Implemented caching of successful formats to avoid retries

### 2. Exchange Correction
- Unity trades on NYSE American (XAMER), not NASDAQ
- Updated `_get_underlying_price()` to use correct dataset:
  - `XAMER.BASIC` for Unity
  - Falls back to `EQUS.MINI` (composite) if XAMER fails

### 3. Error Handling Improvements
- Better detection of subscription errors vs. symbol format issues
- Graceful handling with `DATABENTO_SKIP_VALIDATION=true` environment variable
- Clear error messages guiding users on resolution

### 4. Query Pattern Caching
- Added `_symbol_format_cache` to remember successful query patterns
- Reduces API calls and improves performance

## Key Changes

### databento/client.py:
```python
# Before: Single format attempt
symbols = [f"{underlying}.OPT"]

# After: Multiple format attempts with caching
if underlying == "U":
    symbol_formats = [
        (["U.OPT"], SType.PARENT),    # Standard format first
        (["U     *"], SType.PARENT),  # Unity with 5 spaces
        (["U"], SType.RAW_SYMBOL),    # Raw symbol
    ]
```

## Testing
Run the test script to verify Unity options access:
```bash
python tools/verification/test_unity_databento_fix.py
```

## Usage
If Unity options still fail due to subscription limitations:
```bash
export DATABENTO_SKIP_VALIDATION=true
python run.py --portfolio 100000
```

The system will skip Databento validation and use fallback data sources.

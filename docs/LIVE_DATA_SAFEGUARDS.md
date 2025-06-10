> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Live Data Safeguards

This document describes all the safeguards in place to ensure the Unity Wheel Trading Bot ONLY uses real market data from Databento, never mock or placeholder values.

## Critical Changes Made

### 1. Removed ALL Fallback Mechanisms

- **DATABENTO_SKIP_VALIDATION removed**: The system now FAILS HARD if it can't get real data
- **No default volatility**: Live monitor fails instead of using 0.77 fallback
- **No default prices**: Health checks fail instead of using test values
- **No placeholder option prices**: Position sizing requires real premiums

### 2. Fail-Fast Architecture

Every component now fails immediately if real data is unavailable:

```python
# Example from databento/client.py
raise ValueError(
    f"CRITICAL: Could not retrieve real options data for {underlying}. "
    f"Cannot proceed without market data."
)
```

### 3. Live Data Validator

New validation module at `src/unity_wheel/data_providers/validation/` that:

- Checks environment for test/mock indicators
- Validates prices are in realistic ranges
- Detects suspicious round numbers
- Ensures data freshness (<5 minutes old)
- Validates option chains have realistic bid/ask spreads

### 4. Required Real Data in Critical Functions

#### Position Sizing (wheel.py)
```python
if option_price <= 0:
    raise ValueError(f"Invalid option price: {option_price}. Must use real market data.")
```

#### Market Data Fetch (run.py)
```python
# CRITICAL: Validate this is real market data, not mock/dummy data
validate_market_data(market_snapshot)
```

### 5. Validation Scripts

Two new scripts for paranoid verification:

1. **scripts/validate-live-data-only.sh**
   - Checks environment variables
   - Validates no DATABENTO_SKIP_VALIDATION
   - Ensures API keys are set
   - Tests actual data fetch (optional)

2. **scripts/check-data-sources.sh**
   - Scans codebase for hardcoded values
   - Detects mock data patterns
   - Finds external data libraries

## Running Validation

Before each trading session:

```bash
# Quick validation
./scripts/validate-live-data-only.sh

# Full validation with test fetch
./scripts/validate-live-data-only.sh --test-fetch

# Check codebase for issues
./scripts/check-data-sources.sh
```

## Safeguards at Each Layer

### 1. Environment Level
- No DATABENTO_SKIP_VALIDATION allowed
- DATABENTO_API_KEY required
- No test/mock environment variables

### 2. Data Fetch Level
- DatabentoClient fails without real API response
- No empty option chains allowed
- Timestamp must be recent (<5 minutes)

### 3. Validation Level
- Prices must be positive and non-round
- Unity price must be in [15-60] range
- Volatility must be calculated from real data
- Option spreads must be realistic

### 4. Strategy Level
- Position sizing requires real option premium
- No placeholder values (1.0) allowed
- All calculations validate inputs

### 5. Monitoring Level
- Health checks fail without real data
- Live monitor calculates volatility fresh
- No hardcoded fallback values

## What Happens If Data Is Unavailable

The system will:

1. **Log a CRITICAL error** with details
2. **Raise an exception** that stops execution
3. **Exit with non-zero code** to prevent silent failures
4. **Never proceed** with placeholder/mock data

## Unity-Specific Validations

- Price range: $15-60 (historical bounds)
- Volatility range: 0.30-2.0 (typical for Unity)
- Earnings awareness: Skip trades near earnings
- Round number detection: Warns on suspiciously round values

## Paranoid Checks Before Trading

1. Run validation script: `./scripts/validate-live-data-only.sh --test-fetch`
2. Check no mock patterns: `./scripts/check-data-sources.sh`
3. Verify in market hours: Script warns if outside 9:30-4:00 ET
4. Test actual fetch: Confirms real Unity price from Databento
5. Review logs: Look for any WARNING or CRITICAL messages

## Emergency Override

There is NO emergency override. The system is designed to fail safely rather than proceed with bad data. If you need test/development mode, use a separate test environment with clearly marked test configuration.

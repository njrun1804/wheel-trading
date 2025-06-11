# Single Account Unity Trading Bot

This project has been simplified to work with a **single Schwab margin account only**.

## Key Changes

### ❌ REMOVED
- Multi-account support
- IRA account handling
- Account type detection
- Portfolio aggregation across accounts
- Graceful error handling for missing data

### ✅ ADDED
- Hard failures when ANY data is missing (program exits)
- Single account manager (`SingleAccountManager`)
- Simplified advisor (`SimpleWheelAdvisor`)
- Direct validation with immediate termination on errors

## Usage

### Basic Example
```python
from src.unity_wheel.api import SimpleWheelAdvisor
from src.unity_wheel.portfolio import SingleAccountManager

# Create managers
account_manager = SingleAccountManager()
advisor = SimpleWheelAdvisor()

# Parse account - DIES if any field missing
account = account_manager.parse_account(schwab_account_data)

# Validate buying power - DIES if insufficient
account_manager.validate_buying_power(required_amount, account)

# Get recommendation - DIES if any market data missing
recommendation = advisor.advise_with_fills(market_snapshot, account_data)
```

### Required Account Data Structure
ALL fields are required. Missing ANY field causes immediate program termination.

```python
account_data = {
    "securitiesAccount": {
        "accountNumber": "12345678",  # Optional but logged
        "type": "MARGIN",             # Informational only
        "currentBalances": {
            "liquidationValue": 150000,   # REQUIRED
            "cashBalance": 50000,         # REQUIRED
            "buyingPower": 100000,        # REQUIRED
            "marginBuyingPower": 100000   # Defaults to buyingPower
        },
        "positions": [  # Can be empty list
            {
                "instrument": {
                    "symbol": "U",           # REQUIRED
                    "assetType": "EQUITY"    # Used for type detection
                },
                "quantity": 1000,            # REQUIRED
                "averagePrice": 35.00,
                "marketValue": 35000
            }
        ]
    }
}
```

### Hard Failure Functions

#### `die(message: str) -> NoReturn`
Exits the program with error code 1. Used throughout when data is missing or invalid.

#### Account Manager Validations
```python
# All of these EXIT THE PROGRAM if validation fails:
manager.validate_buying_power(50000, account)  # Dies if BP < 50k
manager.validate_position_limits(10000, account)  # Dies if would exceed limits
```

#### Fill Model Validations
```python
# These cause immediate program termination:
fill_model.estimate_fill_price(bid=0, ask=2, size=5, is_opening=True)  # Dies: Invalid bid
fill_model.estimate_fill_price(bid=2, ask=1, size=5, is_opening=True)  # Dies: bid > ask
fill_model.estimate_fill_price(bid=2, ask=2.1, size=-5, is_opening=True)  # Dies: Invalid size
```

## Implementation Details

### SingleAccountManager
- Location: `src/unity_wheel/portfolio/single_account.py`
- Parses single Schwab account data
- Dies on ANY missing required field
- Validates buying power and position limits with hard exits
- Tracks Unity exposure (shares, puts, calls)

### SimpleWheelAdvisor
- Location: `examples/advisor_simple.py`
- Extends base WheelAdvisor
- Integrates Unity fill model
- Dies if option data missing
- Dies if bid/ask prices missing
- Validates account constraints before recommendations

### Unity Fill Model (Updated)
- Location: `src/unity_wheel/execution/unity_fill_model.py`
- Dies on invalid bid/ask spreads
- Dies on invalid position sizes
- No error recovery - immediate termination

## Testing

Tests verify the hard failure behavior:

```bash
pytest tests/test_single_account.py -v

# Test examples:
# - test_die_on_no_data - Verifies program exits with no data
# - test_die_on_missing_cash_balance - Exits if cash balance missing
# - test_die_on_insufficient_buying_power - Exits if BP too low
```

## Migration from Multi-Account

If you were using the multi-account aggregator:

### Before (Multi-Account)
```python
aggregator = PortfolioAggregator()
portfolio, confidence = aggregator.aggregate_positions(all_accounts)
# Handled missing data gracefully
```

### After (Single Account)
```python
manager = SingleAccountManager()
account = manager.parse_account(account_data)
# Dies immediately if data missing
```

## Philosophy

This implementation follows a **fail-fast** philosophy:
- No partial data handling
- No fallback values
- No error recovery
- Clear, immediate failures

This ensures you always have complete, valid data or the program stops immediately.

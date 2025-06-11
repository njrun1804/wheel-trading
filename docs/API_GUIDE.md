# API Guide for External Tools

This guide is for AI assistants (Claude Code CLI, GitHub Copilot, etc.) integrating with the wheel trading bot.

## Primary Entry Points

### 1. Command Line Interface
```bash
python run.py [options]
```

Options:
- `-p, --portfolio-value` - Portfolio value (required)
- `-a, --account-id` - Account ID (optional)
- `--mode` - Operation mode: `recommend` (default), `diagnose`
- `--config` - Path to config file

### 2. Python API
```python
from src.unity_wheel.api.advisor import UnityWheelAdvisor

advisor = UnityWheelAdvisor()
recommendation = advisor.get_recommendation(
    portfolio_value=100000,
    account_id="12345678"
)
```

## Key Modules

### Trading Strategy (`src.unity_wheel.strategy.wheel`)
```python
from src.unity_wheel.strategy.wheel import WheelStrategy

strategy = WheelStrategy(config)
strikes = strategy.select_put_strikes(underlying_price, volatility)
```

### Risk Management (`src.unity_wheel.risk.analytics`)
```python
from src.unity_wheel.risk.analytics import RiskAnalytics

analytics = RiskAnalytics()
risk_metrics = analytics.calculate_portfolio_risk(positions)
```

### Options Math (`src.unity_wheel.math.options`)
```python
from src.unity_wheel.math.options import calculate_greeks

greeks = calculate_greeks(
    spot_price=540.0,
    strike_price=530.0,
    time_to_expiry=30/365,
    volatility=0.18,
    risk_free_rate=0.05
)
```

## Data Access

### DuckDB Connection
```python
import duckdb

conn = duckdb.connect('data/wheel_trading_master.duckdb')
df = conn.execute("SELECT * FROM unity_options WHERE symbol = 'U'").df()
```

### Authentication
All authentication is handled automatically via environment variables:
- `UNITY_ACCOUNT_ID`
- `UNITY_USERNAME`
- `UNITY_CLIENT_ID`
- `UNITY_CLIENT_SECRET`

## Testing Integration

### Run Tests
```bash
# Quick tests
pytest -v -m "not slow"

# Specific module
pytest tests/test_wheel.py -v

# With coverage
pytest --cov=src/unity_wheel
```

### Mock External Services
```python
from unittest.mock import patch

@patch('src.unity_wheel.auth.client_v2.AuthClient')
def test_with_mock_auth(mock_auth):
    mock_auth.return_value.get_access_token.return_value = "fake_token"
    # Your test code
```

## Error Handling

All errors inherit from base exceptions:
```python
from src.unity_wheel.auth.exceptions import (
    AuthError,
    TokenExpiredError,
    RateLimitError
)
```

## Configuration

### Override Settings
```python
import os

# Override via environment
os.environ['WHEEL_STRATEGY__DELTA_TARGET'] = '0.25'

# Or via config dict
config = {
    'strategy': {'delta_target': 0.25},
    'risk': {'max_position_size': 0.15}
}
```

## Best Practices

1. **Always use type hints** - The codebase relies on them
2. **Mock external calls** - Don't hit real APIs in tests
3. **Use existing patterns** - Check similar code first
4. **Test locally** - Run pytest before commits
5. **Handle auth failures** - Token refresh is automatic

## Common Tasks

### Get Current Positions
```python
from src.unity_wheel.portfolio.single_account import SingleAccountPortfolio

portfolio = SingleAccountPortfolio(account_id)
positions = portfolio.get_positions()
```

### Calculate Position Size
```python
from src.unity_wheel.utils.position_sizing import PositionSizer

sizer = PositionSizer()
contracts = sizer.calculate_optimal_contracts(
    portfolio_value=100000,
    target_delta=0.30
)
```

### Market Data Access
```python
from src.unity_wheel.data_providers.databento import DatabentoProvider

provider = DatabentoProvider()
options_chain = provider.get_options_chain(symbol='U')
```

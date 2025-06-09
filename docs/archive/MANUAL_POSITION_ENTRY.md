# Manual Position Entry Workaround

Since Schwab OAuth is broken, you can manually enter your positions for recommendations.

## How to Use

1. **Edit your positions** in `my_positions.yaml`:
   ```yaml
   account:
     cash: 50000.00  # Your actual cash balance

   positions:
     stocks:
       - symbol: "AAPL"
         quantity: 100
         cost_basis: 220.50

     options:
       - symbol: "AAPL"
         position_type: "covered_call"
         strike: 230
         expiry: "2025-07-18"
         quantity: -1
         premium_collected: 3.50
         underlying_symbol: "AAPL"
   ```

2. **Get recommendations**:
   ```bash
   python get_recommendation.py
   ```

## Position Types Supported

- **Stocks**: Long stock positions
- **Covered Calls**: Short calls against stock
- **Cash-Secured Puts**: Short puts (coming soon)

## Example Entry

For a covered call position:
- You own 100 shares of AAPL at $220.50
- You sold 1 AAPL $230 call for $3.50
- Expiring July 18, 2025

```yaml
positions:
  stocks:
    - symbol: "AAPL"
      quantity: 100
      cost_basis: 220.50

  options:
    - symbol: "AAPL"
      position_type: "covered_call"
      strike: 230
      expiry: "2025-07-18"
      quantity: -1  # negative = short
      premium_collected: 3.50
      underlying_symbol: "AAPL"
```

## While Waiting for Schwab

This manual system will:
- Track your positions
- Calculate Greeks and risk metrics
- Provide roll recommendations
- Suggest new positions based on your cash

Update `my_positions.yaml` whenever you make trades!

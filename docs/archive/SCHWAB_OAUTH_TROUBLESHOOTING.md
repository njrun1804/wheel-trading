> **Disclaimer**
> This is a recommendation system only. No automated trading execution (recommendations only). Always verify recommendations before placing trades. All outputs are for informational purposes only and do not constitute financial advice.

# Schwab OAuth Troubleshooting

## Error: "We are unable to complete your request"

This error occurs on Schwab's side. Common causes and solutions:

### 1. Check Your App Status

Log into [Schwab Developer Portal](https://developer.schwab.com):
- Is your app **Active** or **Pending**?
- If Pending, you need to wait for approval (usually 1-2 business days)
- Check the "Status" column in your apps list

### 2. Verify Redirect URI

Your app's redirect URI must **EXACTLY** match:
```
http://localhost:8182/callback
```

Common mistakes:
- ❌ `https://localhost:8182/callback` (https vs http)
- ❌ `http://localhost:8182/` (missing callback)
- ❌ `http://localhost:8182` (missing /callback)
- ✅ `http://localhost:8182/callback` (correct)

### 3. Check App Permissions

In Schwab Developer Portal:
- Click on your app
- Check "Scopes" or "Permissions"
- Ensure these are enabled:
  - Accounts Read
  - Market Data Read

### 4. App Configuration Issues

Your app might need:
- **App Key**: `OB1E7lHclAbxMCFplibjuDvQlOnmwkEN`
- **App Secret**: `iIO3BpwOwIn9eLtD`
- **App Name**: Should not contain special characters
- **App Type**: Should be "Personal" or "Individual"

### 5. Alternative: Use Schwab's Test Flow

While waiting for app approval, you can test the setup:

```python
# Test with mock data instead of real OAuth
python example_risk_analytics.py
python example_config_usage.py
```

### 6. Contact Schwab Support

If your app shows as Active but still fails:
1. Email: developer@schwab.com
2. Include:
   - Your App Key (not secret)
   - The exact error message
   - Time of attempt

### 7. Temporary Workaround

While resolving OAuth, you can still use:
- ✅ Databento market data features
- ✅ FRED economic indicators
- ✅ Risk analytics calculations
- ✅ Greeks and options math
- ❌ Real-time Schwab positions (requires OAuth)

## Quick Test Commands

Test what's working:
```bash
# Test market data connection
python test_fred_simple.py

# Test risk analytics
python example_risk_analytics.py

# Test options math
python -c "from src.unity_wheel.math import black_scholes_price_validated; print(black_scholes_price_validated(100, 100, 1, 0.05, 0.2, 'call'))"
```

## Next Steps

1. **Check app status** in Schwab Developer Portal
2. **Verify redirect URI** matches exactly
3. **Wait for approval** if app is pending
4. **Use other features** while waiting

The bot has many features that work without Schwab OAuth!

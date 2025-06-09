# Data Fix Quick Start Guide

## Current Status
- **Data Health Score: 50/100 (POOR)**
- **Critical Issue:** No options data available
- **Unity Price Data:** 7 days stale
- **Position Data:** Not being collected

## Immediate Actions (Do These Now)

### Step 1: Update Unity Prices (2 min)
```bash
cd /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading
python tools/analysis/pull_unity_prices.py
```

### Step 2: Set Schwab Credentials (1 min)
```bash
# Add to your ~/.zshrc or ~/.bash_profile
export SCHWAB_CLIENT_ID="your_client_id_here"
export SCHWAB_CLIENT_SECRET="your_client_secret_here"

# Reload shell
source ~/.zshrc
```

### Step 3: Test Schwab Connection (2 min)
```bash
python tools/verification/schwab_status_check.py
```

### Step 4: Fetch Options Data (3 min)
```bash
python tools/data/fetch_unity_options.py
```

### Step 5: Run Full Data Refresh (5 min)
```bash
./scripts/refresh_data.sh
```

### Step 6: Monitor Data Quality (ongoing)
```bash
# In a new terminal
python monitor_data_quality.py
```

## Verification Checklist

After running the above steps, verify:

- [ ] Unity prices updated (check date in monitor)
- [ ] Options data loaded (should show contracts > 0)
- [ ] Schwab connection working
- [ ] Data health score improved (target: >80)

## Daily Maintenance

### Morning Routine (6:30 AM)
1. Run data refresh: `./scripts/refresh_data.sh`
2. Check diagnostics: `python run_aligned.py --diagnose`
3. Review recommendations: `python run_aligned.py --portfolio 100000`

### Set Up Automation (Optional)
```bash
# Add to crontab
crontab -e

# Add these lines:
# Morning refresh at 6:30 AM on weekdays
30 6 * * 1-5 cd /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading && ./scripts/refresh_data.sh

# Afternoon options update at 2:30 PM
30 14 * * 1-5 cd /Users/mikeedwards/Library/Mobile\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading && python tools/data/fetch_unity_options.py
```

## Troubleshooting

### If Schwab connection fails:
1. Check credentials are set: `echo $SCHWAB_CLIENT_ID`
2. Verify OAuth app status at developer.schwab.com
3. Run: `python tools/verification/verify_app_status.py`

### If options data is empty:
1. Ensure market is open
2. Check Unity is tradeable: `python -c "from src.unity_wheel.schwab.client import SchwabClient; ..."`
3. Try manual fetch with debug: `python tools/data/fetch_unity_options.py --debug`

### If data refresh errors:
1. Check logs: `tail -f logs/data_refresh_*.log`
2. Run components individually to isolate issue
3. Verify disk space: `df -h ~/.wheel_trading/`

## Expected Results

After successful setup:
- Unity prices: Current (0-1 days old)
- Options data: 100+ contracts available
- Greeks: Calculated for all options
- Health score: 80+ (GOOD)
- Recommendations: Valid wheel strategy suggestions

## Next Steps

Once data is flowing:
1. Test wheel recommendations: `python run_aligned.py --portfolio 100000`
2. Review adaptive system: `python run_aligned.py --adaptive`
3. Export metrics: `python run_aligned.py --export-metrics`
4. Set up performance tracking

## Support

- Check system status: `python run_aligned.py --diagnose`
- View logs: `ls -la logs/`
- Monitor live: `python monitor_data_quality.py`
- Full documentation: See `DATA_REMEDIATION_PLAN.md`

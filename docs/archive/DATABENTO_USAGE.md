# Databento Data Integration Usage Guide

## Overview

The wheel trading system now integrates with Databento for real options data using Google Secret Manager for secure API key storage.

## Data Architecture

### 1. **API Key Management**
- Databento API key is stored in Google Secret Manager as `databento_api_key`
- Automatically retrieved via `get_databento_api_key()` from `src.unity_wheel.secrets.integration`
- No need to set environment variables

### 2. **Data Storage**
- **Local Storage**: SQLite for structured data, Parquet files for options chains
  - Location: `data/databento/`
  - Retention: 30 days of tick data
- **Cloud Storage** (Optional): 
  - Google Cloud Storage via `GCS_BUCKET` environment variable
  - BigQuery via `BQ_PROJECT` environment variable

### 3. **Data Requirements for Wheel Strategy**

Based on our analysis, we store:
- **Option Chains**: Strikes within 20% of spot price (reduces data by 80%)
- **Target Options**: 25-35 delta puts, 30-60 DTE
- **Data Points**: Bid/ask/mid prices, volume, open interest, implied volatility
- **Update Frequency**: Daily for active positions, weekly for monitoring

## Usage Instructions

### 1. Pull Current Data

```bash
# Fetch latest Unity options data
python pull_databento_data.py
```

This script will:
- Connect to Databento using Google Secret Manager
- Find suitable wheel candidates (30 delta, 30-60 DTE)
- Display top options with expected returns
- Store data locally in Parquet format
- Save recommendations to `data/recommendations/`

### 2. Generate Recommendation with Live Data

```bash
# Generate recommendation using real data
python run_databento_recommendation.py --portfolio 100000
```

Options:
- `--portfolio`: Portfolio value for position sizing (default: 100000)
- `--format`: Output format - "text" or "json" (default: text)

### 3. Scheduled Data Updates

For continuous monitoring, set up a cron job:

```bash
# Add to crontab (runs at 4:30 PM ET on weekdays)
30 16 * * 1-5 cd /path/to/wheel-trading && python pull_databento_data.py
```

## Data Flow

1. **Secret Manager** → Databento API Key
2. **Databento API** → Option chains for Unity
3. **Data Validation** → Quality checks (spreads, liquidity, arbitrage)
4. **Storage** → Local Parquet files + optional cloud
5. **Integration** → Convert to MarketSnapshot for WheelAdvisor
6. **Recommendation** → Actionable trading advice

## Example Output

```
🔄 Initializing Databento data pull for U...
✅ Databento client initialized

🔍 Searching for U wheel candidates...
   Target delta: 0.30
   DTE range: 30-60 days
   Min premium: 1%

📊 Found 15 suitable options:

--------------------------------------------------------------------------------
  Strike  DTE    Bid    Ask    Mid     IV  Delta   Return
--------------------------------------------------------------------------------
   32.50   45   1.85   1.95   1.90  65.2% -0.298    4.8%
   30.00   45   1.20   1.30   1.25  68.1% -0.251    3.9%
   35.00   45   2.65   2.75   2.70  62.4% -0.342    5.1%
   ...

💾 Results saved to: data/recommendations/wheel_candidates_20240608_163045.json

✅ Data Quality Check:
   Confidence Score: 95%
   Bid-Ask Spreads: ✓
   Liquidity: ✓
   Tradeable: ✓

💾 Option chain data stored successfully

✅ Data pull complete
```

## Storage Estimates

For Unity (U) with our filtering:
- **Daily Storage**: ~50 MB (tick data)
- **Monthly Storage**: ~500 MB (aggregated)
- **Annual Storage**: ~6 GB
- **Monthly Cost**: <$1 (including API and storage)

## Integration with Existing System

The data integrates seamlessly with the existing wheel advisor:

1. **DatentoMarketSnapshotBuilder** converts Databento data to WheelAdvisor format
2. **WheelAdvisor** uses real Greeks and IV for recommendations
3. **Risk calculations** use actual market data instead of estimates

## Troubleshooting

### API Key Issues
```bash
# Check if secret exists
gcloud secrets list | grep databento

# Manually set secret if needed
echo -n "your-api-key" | gcloud secrets create databento_api_key --data-file=-
```

### No Data Found
- Check market hours (options data updates during trading hours)
- Verify Unity has liquid options in the 30-60 DTE range
- Check Databento subscription includes OPRA data

### Storage Issues
- Ensure `data/databento/` directory is writable
- Check disk space for local storage
- Verify GCS/BigQuery credentials if using cloud storage
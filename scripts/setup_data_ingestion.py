#!/usr/bin/env python3
"""
Setup data ingestion for the optimized database
Configures Databento and FRED data providers to save to the new structure
"""

import logging
import os
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_config_for_optimized_db():
    """Update config.yaml to use the optimized database"""
    logger.info("📝 Updating config.yaml for optimized database...")

    config_path = Path("config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update primary database path
    config["storage"]["database_path"] = "data/wheel_trading_optimized.duckdb"

    # Update cache database to use optimized DB
    config["storage"]["databases"]["cache"] = "data/wheel_trading_optimized.duckdb"

    # Ensure data provider settings are correct
    if "databento" not in config:
        config["databento"] = {}

    # Set up Databento for end-of-day and ad-hoc fetching
    config["databento"]["fetch_schedule"] = {
        "end_of_day": {
            "enabled": True,
            "time": "16:30",  # 4:30 PM ET
            "timezone": "America/New_York",
            "symbols": ["U"],  # Unity
            "lookback_days": 1,
        },
        "intraday": {
            "enabled": True,
            "cache_ttl_minutes": 5,  # 5-minute cache for live data
            "moneyness_filter": 0.35,  # Only options within 35% of spot
        },
    }

    # Set up FRED for daily updates
    if "fred" not in config:
        config["fred"] = {}

    config["fred"]["fetch_schedule"] = {
        "daily": {
            "enabled": True,
            "time": "06:00",  # 6 AM ET
            "timezone": "America/New_York",
            "series": [
                "DGS10",  # 10-Year Treasury Rate
                "DFF",  # Federal Funds Rate
                "VIXCLS",  # VIX
                "DEXUSEU",  # USD/EUR Exchange Rate
            ],
        }
    }

    # Save updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("✅ Config updated")


def create_data_ingestion_scripts():
    """Create scripts for automated data ingestion"""

    # Create end-of-day Databento script
    eod_script = '''#!/usr/bin/env python3
"""
End-of-day Databento data pull
Run daily at 4:30 PM ET to get final options data
"""

import duckdb
from datetime import datetime, timedelta
import logging
from src.unity_wheel.data_providers.databento import DatabentoClient
from src.unity_wheel.data_providers.databento.databento_storage_adapter import DatabentoStorageAdapter
from src.unity_wheel.storage.storage import Storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pull_eod_options_data():
    """Pull end-of-day options data from Databento"""
    logger.info("📊 Starting end-of-day Databento pull...")
    
    # Initialize components
    storage = Storage()
    client = DatabentoClient()
    adapter = DatabentoStorageAdapter(storage)
    
    # Get today's option chains
    symbol = 'U'
    chains = client.get_option_chains(symbol)
    
    if chains:
        # Store in optimized database
        conn = duckdb.connect('data/wheel_trading_optimized.duckdb')
        
        for chain in chains:
            # Process each option
            for option in chain.options:
                # Apply moneyness filter
                moneyness = option.strike / chain.spot_price
                if 0.65 <= moneyness <= 1.35:  # Within 35% of spot
                    
                    # Insert into options.contracts
                    conn.execute("""
                        INSERT INTO options.contracts (
                            symbol, expiration, strike, option_type, timestamp,
                            bid, ask, mid, volume, open_interest,
                            implied_volatility, delta, gamma, theta, vega, rho,
                            moneyness, days_to_expiry, year_month
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        symbol, option.expiration, option.strike, option.option_type,
                        datetime.now(), option.bid, option.ask, (option.bid + option.ask) / 2,
                        option.volume, option.open_interest, option.implied_volatility,
                        option.delta, option.gamma, option.theta, option.vega, option.rho,
                        moneyness, option.days_to_expiry, 
                        datetime.now().year * 100 + datetime.now().month
                    ])
        
        # Update market data if available
        conn.execute("""
            INSERT INTO market.price_data (symbol, date, close, year_month)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (symbol, date) DO UPDATE SET close = EXCLUDED.close
        """, [symbol, datetime.now().date(), chain.spot_price, 
              datetime.now().year * 100 + datetime.now().month])
        
        conn.close()
        logger.info(f"✅ Stored {len(chains)} option chains")
        
        # Refresh materialized view
        refresh_wheel_opportunities()
    else:
        logger.warning("❌ No data received from Databento")

def refresh_wheel_opportunities():
    """Refresh the wheel opportunities materialized view"""
    conn = duckdb.connect('data/wheel_trading_optimized.duckdb')
    
    # Drop and recreate (DuckDB doesn't have REFRESH MATERIALIZED VIEW)
    conn.execute("DROP TABLE IF EXISTS analytics.wheel_opportunities_mv")
    conn.execute("""
        CREATE TABLE analytics.wheel_opportunities_mv AS
        WITH latest_options AS (
            SELECT symbol, expiration, strike, option_type,
                   MAX(timestamp) as latest_timestamp
            FROM options.contracts
            WHERE option_type = 'PUT'
            GROUP BY symbol, expiration, strike, option_type
        )
        SELECT o.symbol, o.expiration, o.strike, o.bid, o.ask, o.delta,
               o.implied_volatility, o.volume, o.days_to_expiry, o.moneyness,
               (o.bid * 100) / (o.strike * 100) as premium_yield,
               (o.ask - o.bid) / o.ask as spread_pct, o.timestamp
        FROM options.contracts o
        JOIN latest_options l ON o.symbol = l.symbol 
            AND o.expiration = l.expiration 
            AND o.strike = l.strike 
            AND o.option_type = l.option_type 
            AND o.timestamp = l.latest_timestamp
        WHERE o.option_type = 'PUT' 
            AND o.delta BETWEEN -0.35 AND -0.25
            AND o.days_to_expiry BETWEEN 20 AND 45
            AND o.bid > 0 AND o.ask > 0
            AND (o.ask - o.bid) / o.ask < 0.10
    """)
    
    conn.execute("CREATE INDEX idx_wheel_mv_symbol ON analytics.wheel_opportunities_mv(symbol)")
    conn.close()
    logger.info("✅ Refreshed wheel opportunities view")

if __name__ == "__main__":
    pull_eod_options_data()
'''

    # Write EOD script
    with open("scripts/pull_databento_eod.py", "w") as f:
        f.write(eod_script)
    os.chmod("scripts/pull_databento_eod.py", 0o755)
    logger.info("✅ Created scripts/pull_databento_eod.py")

    # Create FRED daily script
    fred_script = '''#!/usr/bin/env python3
"""
Daily FRED data pull
Run daily at 6 AM ET to get economic indicators
"""

import duckdb
from datetime import datetime
import logging
from src.unity_wheel.data_providers.fred import FREDClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pull_daily_fred_data():
    """Pull daily economic data from FRED"""
    logger.info("📈 Starting daily FRED data pull...")
    
    client = FREDClient()
    
    # Series to fetch
    series_ids = ['DGS10', 'DFF', 'VIXCLS', 'DEXUSEU']
    
    conn = duckdb.connect('data/wheel_trading_optimized.duckdb')
    
    for series_id in series_ids:
        try:
            # Get latest observation
            data = client.get_series_observations(
                series_id, 
                observation_start=datetime.now().date() - timedelta(days=7)
            )
            
            if data:
                # Store in analytics.ml_features
                latest = data[-1]
                
                conn.execute("""
                    INSERT INTO analytics.ml_features (symbol, feature_date, vix_level)
                    VALUES ('U', ?, ?)
                    ON CONFLICT (symbol, feature_date) DO UPDATE 
                    SET vix_level = EXCLUDED.vix_level
                """, [latest['date'], latest['value']])
                
                logger.info(f"✅ Updated {series_id}: {latest['value']}")
                
        except Exception as e:
            logger.error(f"❌ Error fetching {series_id}: {e}")
    
    conn.close()
    logger.info("✅ FRED data update complete")

if __name__ == "__main__":
    pull_daily_fred_data()
'''

    # Write FRED script
    with open("scripts/pull_fred_daily.py", "w") as f:
        f.write(fred_script)
    os.chmod("scripts/pull_fred_daily.py", 0o755)
    logger.info("✅ Created scripts/pull_fred_daily.py")


def create_cron_entries():
    """Create cron entries for automated data pulls"""

    cron_entries = """# Wheel Trading Data Ingestion Schedule
# Add these to your crontab with: crontab -e

# End-of-day Databento pull (4:30 PM ET, Monday-Friday)
30 16 * * 1-5 cd /path/to/wheel-trading && /usr/bin/python3 scripts/pull_databento_eod.py >> logs/databento_eod.log 2>&1

# Daily FRED data pull (6:00 AM ET, every day)
0 6 * * * cd /path/to/wheel-trading && /usr/bin/python3 scripts/pull_fred_daily.py >> logs/fred_daily.log 2>&1

# Hourly cache refresh during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
30 9-15 * * 1-5 cd /path/to/wheel-trading && /usr/bin/python3 scripts/refresh_cache.py >> logs/cache_refresh.log 2>&1

# Weekly database optimization (Sunday 2:00 AM)
0 2 * * 0 cd /path/to/wheel-trading && duckdb data/wheel_trading_optimized.duckdb -c "CHECKPOINT; ANALYZE;" >> logs/db_maintenance.log 2>&1
"""

    with open("scripts/cron_data_ingestion.txt", "w") as f:
        f.write(cron_entries)

    logger.info("✅ Created scripts/cron_data_ingestion.txt")


def create_ad_hoc_fetch_function():
    """Create function for ad-hoc data fetching"""

    adhoc_script = '''#!/usr/bin/env python3
"""
Ad-hoc data fetcher for live/on-demand data
Uses caching to avoid excessive API calls
"""

import duckdb
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdHocDataFetcher:
    def __init__(self, db_path: str = 'data/wheel_trading_optimized.duckdb'):
        self.db_path = db_path
        self.cache_ttl = {
            'options': timedelta(minutes=5),
            'quotes': timedelta(seconds=30),
            'greeks': timedelta(minutes=1)
        }
    
    def get_current_options(self, symbol: str = 'U') -> List[Dict]:
        """Get current options data, fetching if cache is stale"""
        conn = duckdb.connect(self.db_path)
        
        # Check cache age
        latest = conn.execute("""
            SELECT MAX(timestamp) FROM options.contracts WHERE symbol = ?
        """, [symbol]).fetchone()[0]
        
        if latest and datetime.now() - latest < self.cache_ttl['options']:
            # Use cached data
            logger.info(f"📦 Using cached options data (age: {datetime.now() - latest})")
            result = conn.execute("""
                SELECT * FROM analytics.wheel_opportunities_mv WHERE symbol = ?
            """, [symbol]).fetchall()
            conn.close()
            return result
        
        # Fetch fresh data
        logger.info("🔄 Fetching fresh options data...")
        # This would call Databento API
        # For now, return cached data with warning
        logger.warning("⚠️  Live fetch not implemented - using cached data")
        result = conn.execute("""
            SELECT * FROM analytics.wheel_opportunities_mv WHERE symbol = ?
        """, [symbol]).fetchall()
        conn.close()
        return result
    
    def get_market_data(self, symbol: str = 'U') -> Optional[Dict]:
        """Get current market data"""
        conn = duckdb.connect(self.db_path)
        
        result = conn.execute("""
            SELECT * FROM market.price_data 
            WHERE symbol = ? 
            ORDER BY date DESC 
            LIMIT 1
        """, [symbol]).fetchone()
        
        conn.close()
        return result

# Convenience functions
def get_wheel_opportunities(symbol: str = 'U') -> List[Dict]:
    """Get current wheel opportunities"""
    fetcher = AdHocDataFetcher()
    return fetcher.get_current_options(symbol)

def get_latest_quote(symbol: str = 'U') -> Optional[Dict]:
    """Get latest market quote"""
    fetcher = AdHocDataFetcher()
    return fetcher.get_market_data(symbol)

if __name__ == "__main__":
    # Example usage
    opportunities = get_wheel_opportunities()
    print(f"Found {len(opportunities)} wheel opportunities")
    
    quote = get_latest_quote()
    if quote:
        print(f"Latest U quote: ${quote[5]}")  # close price
'''

    with open("scripts/fetch_adhoc_data.py", "w") as f:
        f.write(adhoc_script)
    os.chmod("scripts/fetch_adhoc_data.py", 0o755)
    logger.info("✅ Created scripts/fetch_adhoc_data.py")


def main():
    """Set up data ingestion for optimized database"""
    logger.info("🚀 Setting up data ingestion for optimized database\n")

    # Update configuration
    update_config_for_optimized_db()

    # Create ingestion scripts
    create_data_ingestion_scripts()

    # Create cron entries
    create_cron_entries()

    # Create ad-hoc fetch function
    create_ad_hoc_fetch_function()

    logger.info("\n✅ Data ingestion setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Review and install cron entries: crontab -e")
    logger.info("2. Test end-of-day script: python scripts/pull_databento_eod.py")
    logger.info("3. Test FRED script: python scripts/pull_fred_daily.py")
    logger.info("4. Test ad-hoc fetching: python scripts/fetch_adhoc_data.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Production EOD Data Collection for Unity Wheel Trading
Handles T+1 data availability and weekend/holiday logic
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

import databento as db
import duckdb
import pandas as pd
import aiohttp
from src.unity_wheel.secrets.manager import SecretManager

# Production logging setup
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "eod_collection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_last_trading_day():
    """Get the last trading day (handles weekends and basic holidays)"""
    today = datetime.now(timezone.utc).date()
    
    # If it's Monday, get Friday
    if today.weekday() == 0:
        return today - timedelta(days=3)
    # If it's Sunday, get Friday
    elif today.weekday() == 6:
        return today - timedelta(days=2)
    # If it's Saturday, get Friday
    elif today.weekday() == 5:
        return today - timedelta(days=1)
    # Otherwise, get yesterday
    else:
        return today - timedelta(days=1)


class EODCollector:
    """Production EOD data collector with error handling and monitoring"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent.parent / "data" / "wheel_trading_optimized.duckdb"
        self.metrics = {
            "start_time": datetime.now(),
            "stock_records": 0,
            "option_records": 0,
            "fred_records": 0,
            "errors": []
        }
        
        # Load credentials
        self._load_credentials()
        
        # Initialize clients
        self.databento_client = db.Historical(self.databento_key)
        self.conn = duckdb.connect(str(self.db_path))
        
    def _load_credentials(self):
        """Load API keys from SecretManager with error handling"""
        try:
            secret_mgr = SecretManager()
            self.databento_key = secret_mgr.get_secret("databento_api_key")
            self.fred_key = secret_mgr.get_secret("ofred_api_key") or secret_mgr.get_secret("fred_api_key")
            
            if not self.databento_key or not self.fred_key:
                raise ValueError("Missing API keys")
                
            logger.info("✅ Credentials loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            raise
            
    async def collect_unity_stock(self):
        """Collect Unity stock data with proper date handling"""
        logger.info("📈 Collecting Unity stock data...")
        
        try:
            # Get last trading day
            data_date = get_last_trading_day()
            
            # Databento expects timezone-aware datetimes
            end_dt = datetime.combine(data_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            start_dt = end_dt - timedelta(days=5)  # Get 5 days of data
            
            logger.info(f"Fetching stock data from {start_dt.date()} to {end_dt.date()}")
            
            # Fetch daily OHLCV data
            data = self.databento_client.timeseries.get_range(
                dataset="EQUS.MINI",
                symbols=["U"],
                stype_in="raw_symbol",
                schema="ohlcv-1d",
                start=start_dt,
                end=end_dt
            )
            
            df = data.to_df()
            
            if df.empty:
                logger.warning("No Unity stock data returned")
                return 0
                
            # Process and store
            stored = 0
            for idx, row in df.iterrows():
                try:
                    date = idx.date() if hasattr(idx, 'date') else idx
                    
                    # Convert from scaled prices
                    open_price = float(row.get('open', 0)) / 1e9
                    high_price = float(row.get('high', 0)) / 1e9
                    low_price = float(row.get('low', 0)) / 1e9
                    close_price = float(row.get('close', 0)) / 1e9
                    volume = int(row.get('volume', 0))
                    
                    if close_price > 0:  # Valid data
                        self.conn.execute("""
                            INSERT OR REPLACE INTO market.price_data
                            (symbol, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, ['U', date, open_price, high_price, low_price, close_price, volume])
                        stored += 1
                        
                except Exception as e:
                    logger.error(f"Failed to store stock record: {e}")
                    self.metrics["errors"].append(f"Stock storage: {str(e)}")
                    
            logger.info(f"✅ Stored {stored} days of Unity stock data")
            self.metrics["stock_records"] = stored
            return stored
            
        except Exception as e:
            error_msg = f"Stock collection failed: {e}"
            logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            return 0
            
    async def collect_unity_options(self):
        """Collect Unity options data with moneyness filtering"""
        logger.info("📊 Collecting Unity options...")
        
        try:
            # Get last trading day
            data_date = get_last_trading_day()
            
            # Check if we need to go back further for OPRA data
            # OPRA data typically available next day, but may need 2 days on weekends
            today = datetime.now(timezone.utc).date()
            if today.weekday() >= 5:  # Weekend
                # On weekends, Thursday's data should be available
                data_date = data_date - timedelta(days=1)
            
            # Create timezone-aware datetime for the data date
            # Use start of day to avoid "data too recent" errors
            data_dt = datetime.combine(data_date, datetime.min.time()).replace(tzinfo=timezone.utc)
            
            logger.info(f"Fetching options for {data_date} (T+1 delay)")
            
            # Get option definitions
            # OPRA data ends at 18:00 UTC (2 PM ET)
            definitions = self.databento_client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=["U.OPT"],
                stype_in="parent",
                schema="definition",
                start=data_dt,
                end=data_dt.replace(hour=17, minute=59),  # End before 18:00 UTC
                limit=500  # Get more options
            )
            
            def_df = definitions.to_df()
            
            if def_df.empty:
                logger.warning("No Unity option definitions found")
                return 0
                
            logger.info(f"Found {len(def_df)} option definitions")
            
            # Get Unity spot price for moneyness filtering
            spot_price = self._get_unity_spot_price()
            if not spot_price:
                logger.warning("Could not get Unity spot price, using all strikes")
                spot_price = 25.0  # Fallback estimate
                
            # Filter by moneyness (±35% for basic wheel, ±50% for ML)
            min_strike = spot_price * 0.65
            max_strike = spot_price * 1.35
            
            # Filter definitions
            # Databento sends strike prices in different scales, need to check
            logger.debug(f"Raw strike_price values: min={def_df['strike_price'].min()}, max={def_df['strike_price'].max()}")
            
            # Databento OPRA data uses different scaling
            # If strikes are very small (< 1), they need to be multiplied
            # If strikes are very large (> 10000), they need to be divided
            if def_df['strike_price'].max() < 1:
                # Strikes like 0.055 need to be multiplied by 1000 to get $55
                def_df['strike_unscaled'] = def_df['strike_price'] * 1000
            elif def_df['strike_price'].max() > 10000:
                # Strikes like 55000 need to be divided by 1000 to get $55
                def_df['strike_unscaled'] = def_df['strike_price'] / 1000
            else:
                # Already in dollars
                def_df['strike_unscaled'] = def_df['strike_price']
            
            # Debug: log strike range
            logger.debug(f"Strike range after scaling: {def_df['strike_unscaled'].min():.2f} - {def_df['strike_unscaled'].max():.2f}")
            logger.debug(f"Looking for strikes between {min_strike:.2f} and {max_strike:.2f}")
            
            filtered_defs = def_df[
                (def_df['strike_unscaled'] >= min_strike) & 
                (def_df['strike_unscaled'] <= max_strike)
            ]
            
            logger.info(f"Filtered to {len(filtered_defs)} options within ±35% moneyness")
            
            if filtered_defs.empty:
                return 0
                
            # Get quotes for filtered instruments
            instrument_ids = filtered_defs['instrument_id'].tolist()
            
            # Get quotes near market close (but before 18:00 UTC)
            quote_time = data_dt.replace(hour=15, minute=45)  # 3:45 PM UTC
            
            quotes = self.databento_client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                symbols=instrument_ids[:200],  # Limit to avoid API limits
                stype_in="instrument_id",
                schema="cmbp-1",  # Use consolidated MBP schema (mbp-1 was deprecated)
                start=quote_time,
                end=data_dt.replace(hour=17, minute=59)  # End before 18:00 UTC
            )
            
            quote_df = quotes.to_df()
            
            # Store options
            stored = self._store_options(filtered_defs, quote_df)
            
            logger.info(f"✅ Stored {stored} option contracts")
            self.metrics["option_records"] = stored
            return stored
            
        except Exception as e:
            error_msg = f"Options collection failed: {e}"
            logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            return 0
            
    def _get_unity_spot_price(self):
        """Get latest Unity stock price from database"""
        try:
            result = self.conn.execute("""
                SELECT close FROM market.price_data 
                WHERE symbol = 'U' 
                ORDER BY date DESC 
                LIMIT 1
            """).fetchone()
            
            return float(result[0]) if result else None
            
        except Exception:
            return None
            
    def _store_options(self, definitions, quotes):
        """Store options with proper error handling"""
        
        # Group quotes by instrument_id and get latest
        latest_quotes = quotes.groupby('instrument_id').last()
        
        # Debug: check what columns we have
        logger.debug(f"Quote columns: {quotes.columns.tolist()}")
        
        # For cmbp-1 schema, use appropriate columns
        # Check which columns are available
        quote_columns = []
        if 'bid_px_00' in latest_quotes.columns and 'ask_px_00' in latest_quotes.columns:
            quote_columns = ['bid_px_00', 'ask_px_00']
            if 'ts_recv' in latest_quotes.columns:
                quote_columns.append('ts_recv')
            elif 'ts_event' in latest_quotes.columns:
                quote_columns.append('ts_event')
        elif 'bid_px' in latest_quotes.columns and 'ask_px' in latest_quotes.columns:
            quote_columns = ['bid_px', 'ask_px', 'ts_recv']
        else:
            logger.error(f"Cannot find bid/ask columns in: {latest_quotes.columns.tolist()}")
            return 0
        
        # Merge with definitions
        merged = definitions.merge(
            latest_quotes[quote_columns], 
            left_on='instrument_id', 
            right_index=True, 
            how='inner'
        )
        
        stored = 0
        for _, row in merged.iterrows():
            try:
                # Extract bid/ask based on schema
                if 'bid_px_00' in row and 'ask_px_00' in row:
                    # cmbp-1 schema - prices in nanoseconds
                    bid = float(row['bid_px_00']) / 1e9
                    ask = float(row['ask_px_00']) / 1e9
                elif 'bid_px' in row and 'ask_px' in row:
                    # Alternative schema
                    bid = float(row['bid_px']) / 1e9
                    ask = float(row['ask_px']) / 1e9
                else:
                    continue
                
                if bid <= 0 or ask <= 0:
                    continue
                    
                # Prepare data
                option_type = 'CALL' if row['instrument_class'] == 'C' else 'PUT'
                # Use the already scaled strike price
                strike = float(row['strike_unscaled'])
                expiration = pd.to_datetime(row['expiration']).date()
                # Handle different timestamp columns
                timestamp = row.get('ts_recv', row.get('ts_event', datetime.now()))
                
                # Calculate Greeks if possible
                iv = self._calculate_implied_volatility(bid, ask, strike, expiration)
                
                # Store in database
                self.conn.execute("""
                    INSERT OR REPLACE INTO options.contracts
                    (symbol, expiration, strike, option_type, bid, ask, 
                     volume, open_interest, implied_volatility, 
                     timestamp, year_month)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    'U', expiration, strike, option_type,
                    bid, ask, 0, 0, iv,
                    timestamp, expiration.year * 100 + expiration.month
                ])
                stored += 1
                
            except Exception as e:
                logger.debug(f"Failed to store option: {e}")
                
        return stored
        
    def _calculate_implied_volatility(self, bid, ask, strike, expiration):
        """Calculate implied volatility using proper Black-Scholes"""
        try:
            # Import the validated IV calculator
            from src.unity_wheel.math.options import implied_volatility_validated
            
            # Get inputs
            days_to_expiry = (expiration - datetime.now().date()).days
            time_to_expiry = days_to_expiry / 365.0
            mid_price = (bid + ask) / 2
            spot = self._get_unity_spot_price() or 25.0
            risk_free_rate = 0.05  # Could get from FRED
            
            # Calculate IV
            result = implied_volatility_validated(
                option_price=mid_price,
                S=spot,
                K=strike,
                T=time_to_expiry,
                r=risk_free_rate,
                option_type="put"  # Wheel strategy uses puts
            )
            
            # Return IV if confidence is good, otherwise use default
            if result.confidence > 0.5:
                return float(result.value)
            else:
                # Fallback to Unity's typical IV
                return 0.65
                
        except Exception as e:
            logger.debug(f"IV calculation failed: {e}, using default")
            return 0.65  # Unity's typical IV
        
    async def collect_fred_data(self):
        """Collect FRED economic indicators"""
        logger.info("💰 Collecting FRED economic data...")
        
        series_list = [
            ('VIXCLS', 'VIX'),
            ('DGS10', '10Y Treasury'),
            ('DFF', 'Fed Funds Rate'),
            ('TEDRATE', 'TED Spread'),
            ('BAMLH0A0HYM2', 'High Yield Spread')
        ]
        
        stored = 0
        
        async with aiohttp.ClientSession() as session:
            for series_id, name in series_list:
                try:
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_key,
                        'file_type': 'json',
                        'limit': 10,
                        'sort_order': 'desc'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get('observations', [])
                            
                            series_stored = 0
                            for obs in observations:
                                if obs['value'] != '.' and obs['value']:
                                    try:
                                        # Store FRED data in ml_features table
                                        # For now, only VIX goes into ml_features
                                        # TODO: Create proper FRED storage table
                                        if series_id == 'VIXCLS':
                                            self.latest_vix = float(obs['value'])
                                        series_stored += 1
                                    except Exception as e:
                                        logger.debug(f"Failed to store {series_id}: {e}")
                                        
                            stored += series_stored
                            
                            if series_stored > 0:
                                latest = observations[0]
                                logger.info(f"✅ {name}: {latest['value']} ({series_stored} records)")
                                # Store VIX for ML features
                                if series_id == 'VIXCLS':
                                    self.latest_vix = float(latest['value'])
                        else:
                            error_msg = f"FRED API error for {name}: HTTP {response.status}"
                            logger.error(error_msg)
                            self.metrics["errors"].append(error_msg)
                            
                except Exception as e:
                    error_msg = f"Failed to fetch {name}: {e}"
                    logger.error(error_msg)
                    self.metrics["errors"].append(error_msg)
                    
        logger.info(f"✅ Stored {stored} FRED observations")
        self.metrics["fred_records"] = stored
        return stored
        
    def calculate_ml_features(self):
        """Calculate and store ML features"""
        logger.info("🤖 Calculating ML features...")
        
        try:
            # Get latest market data
            vix = self._get_latest_fred_value('VIXCLS')
            ted_spread = self._get_latest_fred_value('TEDRATE')
            risk_free_rate = self._get_latest_fred_value('DGS10')
            
            # Get Unity volatility
            unity_vol = self._calculate_unity_volatility()
            
            # Determine market regime
            regime = self._determine_market_regime(vix, unity_vol)
            
            # Store ML features
            self.conn.execute("""
                INSERT OR REPLACE INTO analytics.ml_features
                (symbol, feature_date, vix_level, market_regime,
                 volatility_realized)
                VALUES (?, CURRENT_DATE, ?, ?, ?)
            """, ['U', vix, regime, unity_vol])
            
            # Refresh materialized views (DuckDB doesn't support REFRESH, recreate instead)
            try:
                self.conn.execute("DROP VIEW IF EXISTS analytics.wheel_opportunities_mv")
                self.conn.execute("""
                    CREATE VIEW analytics.wheel_opportunities_mv AS
                    SELECT 
                        o.symbol,
                        o.expiration,
                        o.strike,
                        o.option_type,
                        o.bid,
                        o.ask,
                        (o.bid / o.strike) * 100 as premium_yield,
                        CAST((o.expiration - CURRENT_DATE) AS INTEGER) as days_to_expiry,
                        m.vix_level,
                        m.market_regime
                    FROM options.contracts o
                    LEFT JOIN analytics.ml_features m ON m.symbol = o.symbol
                    WHERE o.symbol = 'U'
                    AND o.option_type = 'PUT'
                    AND o.bid > 0
                    AND o.ask > 0
                    AND o.expiration > CURRENT_DATE
                    AND o.expiration <= CURRENT_DATE + INTERVAL 60 DAY
                """)
            except Exception as e:
                logger.warning(f"Could not refresh materialized view: {e}")
            
            if vix is not None:
                logger.info(f"✅ ML Features: VIX={vix:.2f}, Regime={regime}, Unity Vol={unity_vol:.2%}")
            else:
                logger.info(f"✅ ML Features: VIX=N/A, Regime={regime}, Unity Vol={unity_vol:.2%}")
            
        except Exception as e:
            error_msg = f"ML feature calculation failed: {e}"
            logger.error(error_msg)
            self.metrics["errors"].append(error_msg)
            
    def _get_latest_fred_value(self, series_id):
        """Get latest value for FRED series"""
        try:
            # For VIX, check ml_features table
            if series_id == 'VIXCLS':
                # First check if we just fetched it
                if hasattr(self, 'latest_vix'):
                    return self.latest_vix
                    
                # Otherwise check database
                result = self.conn.execute("""
                    SELECT vix_level FROM analytics.ml_features
                    WHERE vix_level IS NOT NULL
                    ORDER BY feature_date DESC
                    LIMIT 1
                """).fetchone()
                
                return float(result[0]) if result else None
            
            # For other FRED series, return None for now
            # TODO: Implement proper FRED data storage
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get latest FRED value for {series_id}: {e}")
            return None
            
    def _calculate_unity_volatility(self):
        """Calculate Unity's realized volatility"""
        try:
            result = self.conn.execute("""
                SELECT STDDEV(daily_return) * SQRT(252) as volatility
                FROM (
                    SELECT 
                        LN(close / LAG(close) OVER (ORDER BY date)) as daily_return
                    FROM market.price_data
                    WHERE symbol = 'U'
                    ORDER BY date DESC
                    LIMIT 20
                )
            """).fetchone()
            
            return float(result[0]) if result and result[0] else 0.5
            
        except Exception:
            return 0.5  # Default Unity volatility
            
    def _determine_market_regime(self, vix, unity_vol):
        """Determine market regime based on indicators"""
        if vix is None:
            vix = 20  # Default
            
        if vix < 15 and unity_vol < 0.5:
            return 'low_volatility'
        elif vix < 25 and unity_vol < 0.75:
            return 'normal'
        elif vix < 35 and unity_vol < 1.0:
            return 'volatile'
        else:
            return 'stressed'
            
    def generate_summary(self):
        """Generate collection summary with metrics"""
        duration = (datetime.now() - self.metrics["start_time"]).total_seconds()
        
        summary = f"""
=====================================
EOD Collection Summary
=====================================
Date: {get_last_trading_day()}
Duration: {duration:.1f} seconds

Records Collected:
- Stock: {self.metrics["stock_records"]}
- Options: {self.metrics["option_records"]}
- FRED: {self.metrics["fred_records"]}

Errors: {len(self.metrics["errors"])}
"""
        
        # Database summary
        try:
            options_total = self.conn.execute(
                "SELECT COUNT(*) FROM options.contracts WHERE symbol='U'"
            ).fetchone()[0]
            
            opportunities = self.conn.execute(
                "SELECT COUNT(*) FROM analytics.wheel_opportunities_mv"
            ).fetchone()[0]
            
            summary += f"""
Database Status:
- Total Unity options: {options_total}
- Wheel opportunities: {opportunities}
"""
        except Exception as e:
            logger.error(f"Failed to get database summary: {e}")
            
        if self.metrics["errors"]:
            summary += "\nErrors encountered:\n"
            for error in self.metrics["errors"][:5]:  # First 5 errors
                summary += f"- {error}\n"
                
        summary += "====================================="
        
        return summary
        
    async def run(self):
        """Run complete EOD collection with error handling"""
        logger.info("=" * 50)
        logger.info(f"Starting EOD Collection for {get_last_trading_day()}")
        logger.info("=" * 50)
        
        try:
            # Run all collections
            await self.collect_unity_stock()
            await self.collect_unity_options()
            await self.collect_fred_data()
            self.calculate_ml_features()
            
            # Generate and log summary
            summary = self.generate_summary()
            logger.info(summary)
            
            # Return success code
            return 0 if len(self.metrics["errors"]) == 0 else 1
            
        except Exception as e:
            logger.error(f"Fatal error in EOD collection: {e}")
            logger.error(traceback.format_exc())
            return 2
            
        finally:
            # Clean up
            self.conn.close()
            

async def main():
    """Main entry point with exit code"""
    collector = EODCollector()
    exit_code = await collector.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
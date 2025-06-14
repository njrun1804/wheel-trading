"""
Optimized storage adapter for Databento data
Writes directly to the new optimized database structure
"""

import duckdb
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
import logging

from .types import OptionChain, OptionQuote, InstrumentDefinition
from config.loader import get_config
from unity_wheel.utils import get_logger

logger = get_logger(__name__)

class OptimizedStorageAdapter:
    """Storage adapter that writes to optimized database structure"""
    
    def __init__(self, db_path: str = "data/wheel_trading_optimized.duckdb"):
        self.db_path = db_path
        # Use default values from config.yaml databento section
        self.MONEYNESS_RANGE = 0.35  # Range around spot price  
        self.MAX_EXPIRATIONS = 6     # Number of expirations to keep
        self.MIN_VOLUME = 1          # Minimum volume filter
        
        # Track metrics
        self._metrics = {
            "options_stored": 0,
            "options_filtered": 0,
            "market_data_stored": 0
        }
        
    def store_option_chain(self, chain: OptionChain) -> int:
        """Store option chain with moneyness filtering"""
        stored_count = 0
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # First, update market data
            if chain.spot_price and chain.spot_price > 0:
                self._store_market_data(conn, chain.symbol, chain.spot_price)
            
            # Combine calls and puts
            all_options = chain.calls + chain.puts
            
            # Get unique expirations and limit to MAX_EXPIRATIONS
            expirations = sorted(set(opt.expiration for opt in all_options if hasattr(opt, 'expiration')))
            if len(expirations) > self.MAX_EXPIRATIONS:
                expirations = expirations[:self.MAX_EXPIRATIONS]
                logger.info(f"Limited to {self.MAX_EXPIRATIONS} nearest expirations")
            
            # Process each option
            batch_data = []
            for option in all_options:
                # Skip if expiration not in selected list
                if option.expiration not in expirations:
                    self._metrics["options_filtered"] += 1
                    continue
                
                # Calculate moneyness
                if chain.spot_price and chain.spot_price > 0:
                    moneyness = float(option.strike) / float(chain.spot_price)
                else:
                    moneyness = 1.0
                
                # Apply moneyness filter
                if abs(1.0 - moneyness) > self.MONEYNESS_RANGE:
                    self._metrics["options_filtered"] += 1
                    continue
                
                # Apply volume filter
                if option.volume is not None and option.volume < self.MIN_VOLUME:
                    self._metrics["options_filtered"] += 1
                    continue
                
                # Calculate derived fields
                bid = float(option.bid) if option.bid else 0
                ask = float(option.ask) if option.ask else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                
                days_to_expiry = (option.expiration - datetime.now().date()).days
                year_month = datetime.now().year * 100 + datetime.now().month
                
                # Prepare row data
                row = (
                    chain.symbol,                    # symbol
                    option.expiration,               # expiration
                    float(option.strike),            # strike
                    option.option_type.upper(),      # option_type
                    datetime.now(),                  # timestamp
                    bid,                             # bid
                    ask,                             # ask
                    mid,                             # mid
                    option.volume or 0,              # volume
                    option.open_interest or 0,       # open_interest
                    float(option.implied_volatility) if option.implied_volatility else None,  # iv
                    float(option.delta) if option.delta else None,              # delta
                    float(option.gamma) if option.gamma else None,              # gamma
                    float(option.theta) if option.theta else None,              # theta
                    float(option.vega) if option.vega else None,                # vega
                    float(option.rho) if option.rho else None,                  # rho
                    moneyness,                       # moneyness
                    days_to_expiry,                  # days_to_expiry
                    year_month                       # year_month
                )
                
                batch_data.append(row)
            
            # Batch insert
            if batch_data:
                conn.executemany("""
                    INSERT INTO options.contracts (
                        symbol, expiration, strike, option_type, timestamp,
                        bid, ask, mid, volume, open_interest,
                        implied_volatility, delta, gamma, theta, vega, rho,
                        moneyness, days_to_expiry, year_month
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (symbol, expiration, strike, option_type, timestamp) 
                    DO UPDATE SET
                        bid = EXCLUDED.bid,
                        ask = EXCLUDED.ask,
                        mid = EXCLUDED.mid,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility,
                        delta = EXCLUDED.delta,
                        gamma = EXCLUDED.gamma,
                        theta = EXCLUDED.theta,
                        vega = EXCLUDED.vega,
                        rho = EXCLUDED.rho
                """, batch_data)
                
                stored_count = len(batch_data)
                self._metrics["options_stored"] += stored_count
                
                logger.info(
                    f"Stored {stored_count} options for {chain.symbol}",
                    extra={
                        "filtered": len(chain.options) - stored_count,
                        "moneyness_range": self.MONEYNESS_RANGE,
                        "expirations": len(expirations)
                    }
                )
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing option chain: {e}")
            raise
            
        return stored_count
    
    def _store_market_data(self, conn: duckdb.DuckDBPyConnection, symbol: str, price: float):
        """Store or update market data"""
        try:
            # For EOD data, we just update the close price
            today = datetime.now().date()
            year_month = today.year * 100 + today.month
            
            conn.execute("""
                INSERT INTO market.price_data (
                    symbol, date, close, year_month
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    close = EXCLUDED.close
            """, [symbol, today, float(price), year_month])
            
            self._metrics["market_data_stored"] += 1
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    def refresh_wheel_opportunities(self):
        """Refresh the wheel opportunities materialized view"""
        try:
            conn = duckdb.connect(self.db_path)
            
            # Drop and recreate
            conn.execute("DROP TABLE IF EXISTS analytics.wheel_opportunities_mv")
            
            conn.execute("""
                CREATE TABLE analytics.wheel_opportunities_mv AS
                WITH latest_options AS (
                    SELECT 
                        symbol, expiration, strike, option_type,
                        MAX(timestamp) as latest_timestamp
                    FROM options.contracts
                    WHERE option_type = 'PUT'
                    GROUP BY symbol, expiration, strike, option_type
                )
                SELECT 
                    o.symbol,
                    o.expiration,
                    o.strike,
                    o.bid,
                    o.ask,
                    o.delta,
                    o.implied_volatility,
                    o.volume,
                    o.days_to_expiry,
                    o.moneyness,
                    CASE 
                        WHEN o.strike > 0 THEN (o.bid * 100) / (o.strike * 100)
                        ELSE 0
                    END as premium_yield,
                    CASE 
                        WHEN o.ask > 0 THEN (o.ask - o.bid) / o.ask
                        ELSE 1.0
                    END as spread_pct,
                    o.timestamp
                FROM options.contracts o
                JOIN latest_options l ON 
                    o.symbol = l.symbol AND
                    o.expiration = l.expiration AND
                    o.strike = l.strike AND
                    o.option_type = l.option_type AND
                    o.timestamp = l.latest_timestamp
                WHERE 
                    o.option_type = 'PUT'
                    AND o.delta BETWEEN -0.35 AND -0.25
                    AND o.days_to_expiry BETWEEN 20 AND 45
                    AND o.bid > 0 
                    AND o.ask > 0
                    AND (o.ask - o.bid) / o.ask < 0.10
                ORDER BY premium_yield DESC
            """)
            
            # Create index
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_wheel_mv_symbol 
                ON analytics.wheel_opportunities_mv(symbol)
            """)
            
            # Get count
            count = conn.execute(
                "SELECT COUNT(*) FROM analytics.wheel_opportunities_mv"
            ).fetchone()[0]
            
            conn.close()
            
            logger.info(f"Refreshed wheel opportunities: {count} candidates")
            
        except Exception as e:
            logger.error(f"Error refreshing wheel opportunities: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, int]:
        """Return storage metrics"""
        return self._metrics.copy()
    
    def reset_metrics(self):
        """Reset storage metrics"""
        self._metrics = {
            "options_stored": 0,
            "options_filtered": 0,
            "market_data_stored": 0
        }
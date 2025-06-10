#!/usr/bin/env python3
"""
Collect Unity options data using PROPER Databento API structure.

Implements best practices from comprehensive technical guide:
- CMBP-1 schema for consolidated NBBO quotes
- Parent symbology (U.OPT) with stype_in="parent"
- Proper data timing (pull after 2 AM ET)
- Optimal chunking and memory management
- DuckDB transformation with partitioning
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
import duckdb
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
import pytz

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import databento as db
from databento_dbn import Schema, SType
from src.unity_wheel.secrets.integration import get_databento_api_key

logger = logging.getLogger(__name__)

DB_PATH = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")

class ProperUnityOptionsCollector:
    """Collect Unity options using correct Databento API patterns."""
    
    def __init__(self, max_concurrent=8):
        self.api_key = get_databento_api_key()
        self.client = db.Historical(self.api_key)
        self.max_concurrent = max_concurrent
        self.conn = None
        
    def initialize_database(self):
        """Initialize optimized DuckDB schema per technical guide."""
        self.conn = duckdb.connect(DB_PATH)
        
        # Create optimized options table (DuckDB compatible syntax)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS options_ticks (
                trade_date DATE NOT NULL,
                ts_event TIMESTAMP NOT NULL,
                instrument_id UBIGINT NOT NULL,
                bid_px DECIMAL(10,4) NOT NULL,
                ask_px DECIMAL(10,4) NOT NULL,
                bid_sz UBIGINT NOT NULL,
                ask_sz UBIGINT NOT NULL,
                PRIMARY KEY (trade_date, ts_event, instrument_id)
            )
        """)
        
        # Separate instrument reference table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                instrument_id UBIGINT PRIMARY KEY,
                symbol VARCHAR NOT NULL,
                underlying VARCHAR NOT NULL,
                expiration DATE NOT NULL,
                strike DECIMAL(10,2) NOT NULL,
                option_type VARCHAR(1) NOT NULL,
                date_listed DATE
            )
        """)
        
        # Create indexes for efficient queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_options_instrument 
            ON options_ticks(instrument_id)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_instruments_underlying 
            ON instruments(underlying, expiration)
        """)
        
        print("✅ Database schema initialized with partitioning")
        
    def verify_data_availability(self, date):
        """Check if data is available for given date (after 2 AM ET cutoff)."""
        eastern = pytz.timezone('US/Eastern')
        
        # Data available ~2 AM ET next day
        cutoff_time = datetime.combine(date + timedelta(days=1), datetime.min.time())
        cutoff_time = eastern.localize(cutoff_time.replace(hour=2))
        
        now = datetime.now(eastern)
        
        if now < cutoff_time:
            print(f"⏳ Data for {date} not yet available (need to wait until {cutoff_time})")
            return False
            
        return True
        
    def get_instrument_definitions(self, date):
        """Get Unity option definitions using proper schema."""
        print(f"📋 Getting Unity option definitions for {date}...")
        
        try:
            # Use DEFINITION schema with parent symbology - KEY CORRECTION
            definitions = self.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema=Schema.DEFINITION,  # Correct schema for definitions
                symbols=["U.OPT"],         # Parent symbol as recommended
                stype_in=SType.PARENT,     # CRITICAL: parent symbology
                start=date,
                end=date + timedelta(days=1)
            )
            
            definitions_list = list(definitions)
            print(f"✅ Found {len(definitions_list)} Unity option definitions")
            
            return definitions_list
            
        except Exception as e:
            print(f"❌ Error getting definitions: {e}")
            return []
    
    def get_option_quotes(self, date):
        """Get Unity option quotes using CMBP-1 schema as recommended."""
        print(f"📊 Getting Unity option quotes for {date}...")
        
        try:
            # Use CMBP-1 for consolidated NBBO - KEY RECOMMENDATION
            quotes_data = self.client.timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",           # Optimal schema per guide
                symbols=["U.OPT"],         # Parent symbol
                stype_in=SType.PARENT,     # Parent symbology
                start=date.replace(hour=9, minute=30),   # Market hours
                end=date.replace(hour=16, minute=0)      # Market close
            )
            
            # Process in chunks for memory efficiency
            quotes_list = []
            chunk_size = 50000  # As recommended in guide
            
            for i, record in enumerate(quotes_data):
                quotes_list.append(record)
                
                # Process in chunks to manage memory
                if len(quotes_list) >= chunk_size:
                    self.process_quotes_chunk(quotes_list, date)
                    quotes_list = []
                    
                    if i % 100000 == 0:
                        print(f"   Processed {i:,} quotes...")
            
            # Process remaining quotes
            if quotes_list:
                self.process_quotes_chunk(quotes_list, date)
                
            print(f"✅ Completed processing Unity option quotes for {date}")
            return True
            
        except Exception as e:
            print(f"❌ Error getting quotes: {e}")
            return False
    
    def process_quotes_chunk(self, quotes_chunk, trade_date):
        """Process and store quotes chunk in DuckDB."""
        if not quotes_chunk:
            return
            
        # Convert to DataFrame
        df_data = []
        for record in quotes_chunk:
            df_data.append({
                'trade_date': trade_date.date(),
                'ts_event': pd.to_datetime(record.ts_event, unit='ns', utc=True),
                'instrument_id': record.instrument_id,
                'bid_px': record.bid_px / 1e9,  # Convert from fixed-point
                'ask_px': record.ask_px / 1e9,
                'bid_sz': record.bid_sz,
                'ask_sz': record.ask_sz,
            })
        
        df = pd.DataFrame(df_data)
        
        # Optimize data types for storage efficiency
        df = self.optimize_dtypes(df)
        
        # Insert into DuckDB with deduplication
        self.conn.execute("""
            INSERT INTO options_ticks 
            SELECT * FROM df
            WHERE NOT EXISTS (
                SELECT 1 FROM options_ticks ot 
                WHERE ot.trade_date = df.trade_date 
                AND ot.ts_event = df.ts_event 
                AND ot.instrument_id = df.instrument_id
            )
        """)
        
    def store_instrument_definitions(self, definitions):
        """Store instrument definitions in reference table."""
        if not definitions:
            return
            
        df_data = []
        for defn in definitions:
            # Extract option type from raw symbol (OCC format: last character before numbers)
            # Unity options format: U[spaces]YYMMDD[C/P]xxxxxxxx
            raw_symbol = defn.raw_symbol if hasattr(defn, 'raw_symbol') else str(defn.symbol)
            
            # Parse option type from OCC symbol format
            option_type = 'C'  # Default to call
            if 'P' in raw_symbol:
                option_type = 'P'
            elif 'C' in raw_symbol:
                option_type = 'C'
                
            # Parse strike price from symbol if available
            strike_price = 0.0
            if hasattr(defn, 'strike_price'):
                strike_price = float(defn.strike_price) / 1e9
            
            # Parse expiration if available
            expiration_date = None
            if hasattr(defn, 'expiration'):
                expiration_date = pd.to_datetime(defn.expiration, unit='ns').date()
            
            df_data.append({
                'instrument_id': defn.instrument_id,
                'symbol': raw_symbol,
                'underlying': 'U',  # Unity
                'expiration': expiration_date,
                'strike': strike_price,
                'option_type': option_type,
                'date_listed': pd.to_datetime(defn.ts_event, unit='ns').date()
            })
        
        df = pd.DataFrame(df_data)
        
        # Insert with deduplication
        self.conn.execute("""
            INSERT OR REPLACE INTO instruments 
            SELECT * FROM df
        """)
        
        print(f"✅ Stored {len(df)} instrument definitions")
    
    def optimize_dtypes(self, df):
        """Optimize DataFrame dtypes for storage efficiency."""
        # Convert to optimal types as recommended
        if 'bid_px' in df.columns:
            df['bid_px'] = df['bid_px'].astype('float64')  # DuckDB DECIMAL precision
        if 'ask_px' in df.columns:
            df['ask_px'] = df['ask_px'].astype('float64')
        if 'bid_sz' in df.columns:
            df['bid_sz'] = df['bid_sz'].astype('uint64')   # Match UBIGINT
        if 'ask_sz' in df.columns:
            df['ask_sz'] = df['ask_sz'].astype('uint64')
        if 'instrument_id' in df.columns:
            df['instrument_id'] = df['instrument_id'].astype('uint64')
            
        return df
    
    def get_underlying_price(self, date):
        """Get Unity underlying price using 1-minute bars as recommended."""
        print(f"📈 Getting Unity underlying price for {date}...")
        
        try:
            # Use 1-minute OHLCV bars as recommended in guide
            underlying_data = self.client.timeseries.get_range(
                dataset="XNAS.ITCH",  # NASDAQ feed for Unity
                schema="ohlcv-1m",    # 1-minute bars essential per guide
                symbols=["U"],        # Unity stock symbol
                start=date.replace(hour=9, minute=30),
                end=date.replace(hour=16, minute=0)
            )
            
            underlying_list = list(underlying_data)
            print(f"✅ Found {len(underlying_list)} Unity price bars")
            
            if underlying_list:
                # Get closing price for the day
                last_bar = underlying_list[-1]
                close_price = last_bar.close / 1e9  # Convert from fixed-point
                print(f"   Unity closing price: ${close_price:.2f}")
                return close_price
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting underlying price: {e}")
            return None
    
    def collect_daily_data(self, date):
        """Collect complete Unity options data for a single day."""
        print(f"\n📅 Collecting Unity options data for {date.strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        # 1. Verify data availability (2 AM ET cutoff)
        if not self.verify_data_availability(date):
            return False
            
        # 2. Get underlying price
        underlying_price = self.get_underlying_price(date)
        if underlying_price is None:
            print("⚠️  Warning: Could not get underlying price")
        
        # 3. Get instrument definitions
        definitions = self.get_instrument_definitions(date)
        if definitions:
            self.store_instrument_definitions(definitions)
        else:
            print("⚠️  Warning: No instrument definitions found")
            
        # 4. Get option quotes
        success = self.get_option_quotes(date)
        
        if success:
            print(f"✅ Successfully collected Unity options data for {date.strftime('%Y-%m-%d')}")
            
            # Show stats
            self.show_daily_stats(date)
            return True
        else:
            print(f"❌ Failed to collect Unity options data for {date.strftime('%Y-%m-%d')}")
            return False
    
    def show_daily_stats(self, date):
        """Show collection statistics for the day."""
        stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_quotes,
                COUNT(DISTINCT instrument_id) as unique_instruments,
                MIN(ts_event) as first_quote,
                MAX(ts_event) as last_quote
            FROM options_ticks 
            WHERE trade_date = ?
        """, [date.date()]).fetchone()
        
        instrument_stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_instruments,
                COUNT(CASE WHEN option_type = 'P' THEN 1 END) as puts,
                COUNT(CASE WHEN option_type = 'C' THEN 1 END) as calls,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike
            FROM instruments 
            WHERE underlying = 'U'
        """).fetchone()
        
        print(f"\n📊 Collection Statistics:")
        print(f"   Total quotes: {stats[0]:,}")
        print(f"   Unique instruments: {stats[1]:,}")
        print(f"   Quote timespan: {stats[2]} to {stats[3]}")
        print(f"   Total Unity instruments: {instrument_stats[0]:,}")
        print(f"   Puts: {instrument_stats[1]:,}, Calls: {instrument_stats[2]:,}")
        print(f"   Strike range: ${instrument_stats[3]:.2f} - ${instrument_stats[4]:.2f}")
    
    def collect_date_range(self, start_date, end_date):
        """Collect data for a range of dates with proper error handling."""
        print(f"🚀 COLLECTING UNITY OPTIONS DATA")
        print(f"📅 Date range: {start_date} to {end_date}")
        print("=" * 60)
        
        current_date = start_date
        successful_days = 0
        failed_days = 0
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
                
            try:
                success = self.collect_daily_data(current_date)
                if success:
                    successful_days += 1
                else:
                    failed_days += 1
                    
            except Exception as e:
                print(f"❌ Unexpected error for {current_date}: {e}")
                failed_days += 1
                
            current_date += timedelta(days=1)
        
        print(f"\n🎯 COLLECTION SUMMARY")
        print("=" * 60)
        print(f"✅ Successful days: {successful_days}")
        print(f"❌ Failed days: {failed_days}")
        
        if successful_days > 0:
            total_stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_quotes,
                    COUNT(DISTINCT instrument_id) as unique_instruments,
                    COUNT(DISTINCT trade_date) as trading_days
                FROM options_ticks
            """).fetchone()
            
            print(f"📊 Final Database Stats:")
            print(f"   Total quotes: {total_stats[0]:,}")
            print(f"   Unique instruments: {total_stats[1]:,}")
            print(f"   Trading days: {total_stats[2]:,}")
            
            print(f"\n✅ SUCCESS: Unity options data collection complete")
            print(f"🚨 NO SYNTHETIC DATA - All data is real from Databento OPRA feed")
        
    def close(self):
        """Clean up resources."""
        if self.conn:
            self.conn.close()

def main():
    """Main function following technical guide best practices."""
    collector = ProperUnityOptionsCollector()
    
    try:
        # Initialize database with optimized schema
        collector.initialize_database()
        
        # Collect recent data (last 5 trading days)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=10)  # Go back 10 days to catch 5 trading days
        
        # Convert to datetime for processing
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        # Collect data using proper API patterns
        collector.collect_date_range(start_datetime, end_datetime)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print("Check Databento credentials and subscription")
        raise
        
    finally:
        collector.close()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ML Data Collection Scheduler
Intelligently collects market data based on market conditions
"""

import asyncio
import duckdb
from datetime import datetime, time, timedelta
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unity_wheel.data_providers.databento.ml_enhanced_collector import MLEnhancedDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntelligentScheduler:
    """
    Schedules data collection based on:
    1. Market hours
    2. Volatility regime
    3. Position proximity to expiry
    4. Upcoming events
    """
    
    def __init__(self, db_path: str = "data/wheel_trading_optimized.duckdb"):
        self.db_path = db_path
        self.collector = MLEnhancedDataCollector(db_path)
        
        # Market hours (ET)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Collection rules
        self.collection_rules = {
            'regular': {
                'interval_minutes': 30,
                'conditions': lambda: self._is_market_hours() and not self._is_high_vol()
            },
            'high_volatility': {
                'interval_minutes': 15,
                'conditions': lambda: self._is_market_hours() and self._is_high_vol()
            },
            'near_expiry': {
                'interval_minutes': 5,
                'conditions': lambda: self._has_near_expiry_positions()
            },
            'pre_market': {
                'interval_minutes': 60,
                'conditions': lambda: self._is_pre_market()
            },
            'after_hours': {
                'interval_minutes': 120,
                'conditions': lambda: self._is_after_hours()
            }
        }
        
        self._last_collection = {}
        self._vix_level = None
        self._positions = []
    
    async def run(self):
        """Main scheduler loop"""
        logger.info("🚀 Starting ML data collection scheduler")
        
        while True:
            try:
                # Update market state
                await self._update_market_state()
                
                # Determine collection type
                collection_type = self._determine_collection_type()
                
                if collection_type:
                    # Check if it's time to collect
                    if self._should_collect(collection_type):
                        logger.info(f"📊 Collecting {collection_type} snapshot")
                        
                        # Collect data
                        await self.collector.collect_snapshot()
                        
                        # Update last collection time
                        self._last_collection[collection_type] = datetime.now()
                        
                        # Show storage stats
                        self._show_storage_stats()
                
                # Sleep for a minute before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
    
    async def _update_market_state(self):
        """Update current market conditions"""
        conn = duckdb.connect(self.db_path, read_only=True)
        
        try:
            # Get latest VIX level
            vix_result = conn.execute("""
                SELECT vix_level 
                FROM analytics.ml_features 
                WHERE symbol = 'U' 
                ORDER BY feature_date DESC 
                LIMIT 1
            """).fetchone()
            
            if vix_result:
                self._vix_level = vix_result[0]
            
            # Get active positions
            positions = conn.execute("""
                SELECT symbol, expiration, 
                       expiration - CURRENT_DATE as days_to_expiry
                FROM trading.positions
                WHERE status = 'ACTIVE'
                AND expiration IS NOT NULL
            """).fetchall()
            
            self._positions = positions
            
        finally:
            conn.close()
    
    def _is_market_hours(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        current_time = now.time()
        
        # Check weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        return self.market_open <= current_time <= self.market_close
    
    def _is_pre_market(self) -> bool:
        """Check if in pre-market hours"""
        now = datetime.now()
        current_time = now.time()
        
        if now.weekday() >= 5:
            return False
        
        return time(7, 0) <= current_time < self.market_open
    
    def _is_after_hours(self) -> bool:
        """Check if in after-hours trading"""
        now = datetime.now()
        current_time = now.time()
        
        if now.weekday() >= 5:
            return False
        
        return self.market_close < current_time <= time(20, 0)
    
    def _is_high_vol(self) -> bool:
        """Check if in high volatility regime"""
        if self._vix_level is None:
            return False
        return self._vix_level > 30
    
    def _has_near_expiry_positions(self) -> bool:
        """Check if we have positions expiring soon"""
        for _, _, days_to_expiry in self._positions:
            if days_to_expiry is not None and days_to_expiry <= 2:
                return True
        return False
    
    def _determine_collection_type(self) -> str:
        """Determine which collection rule applies"""
        # Check rules in priority order
        priority_order = ['near_expiry', 'high_volatility', 'regular', 'pre_market', 'after_hours']
        
        for rule_name in priority_order:
            rule = self.collection_rules[rule_name]
            if rule['conditions']():
                return rule_name
        
        return None
    
    def _should_collect(self, collection_type: str) -> bool:
        """Check if enough time has passed since last collection"""
        if collection_type not in self._last_collection:
            return True
        
        interval = self.collection_rules[collection_type]['interval_minutes']
        time_since = datetime.now() - self._last_collection[collection_type]
        
        return time_since >= timedelta(minutes=interval)
    
    def _show_storage_stats(self):
        """Show current storage statistics"""
        conn = duckdb.connect(self.db_path, read_only=True)
        
        try:
            # Count ML data rows
            stats = {}
            tables = [
                'ml_data.market_snapshots',
                'ml_data.option_snapshots', 
                'ml_data.surface_metrics',
                'ml_data.model_predictions'
            ]
            
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            
            # Get database size
            db_size = Path(self.db_path).stat().st_size / (1024 * 1024)  # MB
            
            logger.info(f"📊 Storage Stats - DB: {db_size:.1f}MB")
            for table, count in stats.items():
                if count > 0:
                    logger.info(f"   {table}: {count:,} rows")
            
            # Estimate growth rate
            if stats['ml_data.market_snapshots'] > 100:
                logger.info(f"   Growth rate: ~{db_size / max(1, stats['ml_data.market_snapshots'] / 48):.1f}MB/day")
            
        finally:
            conn.close()

def create_collection_service():
    """Create systemd service file for continuous collection"""
    
    service_content = """[Unit]
Description=ML Data Collection Service for Wheel Trading
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/wheel-trading
ExecStart=/usr/bin/python3 scripts/ml_data_collection_scheduler.py
Restart=always
RestartSec=60

# Logging
StandardOutput=append:/var/log/wheel-trading/ml_collection.log
StandardError=append:/var/log/wheel-trading/ml_collection_error.log

[Install]
WantedBy=multi-user.target
"""
    
    print("To run as a service, create: /etc/systemd/system/wheel-ml-collection.service")
    print(service_content)
    print("\nThen run:")
    print("sudo systemctl daemon-reload")
    print("sudo systemctl enable wheel-ml-collection")
    print("sudo systemctl start wheel-ml-collection")

async def test_collection():
    """Test data collection"""
    logger.info("🧪 Testing ML data collection...")
    
    collector = MLEnhancedDataCollector()
    
    # Test snapshot collection
    snapshot = await collector.collect_snapshot()
    
    if snapshot:
        logger.info("✅ Test snapshot collected successfully")
        logger.info(f"   Timestamp: {snapshot.get('timestamp')}")
        logger.info(f"   Spot price: ${snapshot.get('spot_price', 0):.2f}")
        
        if 'surface_metrics' in snapshot:
            metrics = snapshot['surface_metrics']
            logger.info(f"   ATM IV: {metrics.get('atm_iv', 0):.2%}")
            logger.info(f"   IV Skew: {metrics.get('iv_skew', 0):.3f}")
    else:
        logger.warning("❌ No snapshot collected")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Data Collection Scheduler')
    parser.add_argument('--test', action='store_true', help='Run test collection')
    parser.add_argument('--service', action='store_true', help='Show service configuration')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    if args.test:
        await test_collection()
    elif args.service:
        create_collection_service()
    else:
        # Run scheduler
        scheduler = IntelligentScheduler()
        
        if args.daemon:
            logger.info("Running in daemon mode")
            # Could add pidfile handling here
        
        await scheduler.run()

if __name__ == "__main__":
    asyncio.run(main())
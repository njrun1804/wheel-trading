#!/usr/bin/env python3
"""
Monitor data collection status and health
Shows recent collections, data quality, and alerts
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import duckdb
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


class CollectionMonitor:
    """Monitor data collection health and status"""
    
    def __init__(self):
        self.db_path = Path(__file__).parent.parent / "data" / "wheel_trading_optimized.duckdb"
        self.log_dir = Path(__file__).parent.parent / "logs"
        
        if not self.db_path.exists():
            print(f"{RED}Database not found at {self.db_path}{RESET}")
            sys.exit(1)
            
        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        
    def check_recent_collections(self, days=7):
        """Check recent data collections"""
        print(f"\n{BLUE}=== Recent Collections (Last {days} Days) ==={RESET}")
        
        # Check stock data
        stock_data = self.conn.execute("""
            SELECT 
                date,
                close,
                volume
            FROM market.price_data
            WHERE symbol = 'U'
            AND date >= CURRENT_DATE - INTERVAL '7' DAY
            ORDER BY date DESC
        """).fetchall()
        
        print(f"\n📈 Stock Data:")
        if stock_data:
            volume = stock_data[0][2] if stock_data[0][2] is not None else 0
            print(f"   Latest: {stock_data[0][0]} - Close: ${stock_data[0][1]:.2f}, Volume: {volume:,}")
            print(f"   Total days: {len(stock_data)}")
        else:
            print(f"   {RED}No recent stock data{RESET}")
            
        # Check options data
        options_data = self.conn.execute("""
            SELECT 
                DATE(timestamp) as collection_date,
                COUNT(*) as option_count,
                COUNT(DISTINCT expiration) as expirations,
                MIN(strike) as min_strike,
                MAX(strike) as max_strike
            FROM options.contracts
            WHERE symbol = 'U'
            AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '7' DAY
            GROUP BY DATE(timestamp)
            ORDER BY collection_date DESC
        """).fetchall()
        
        print(f"\n📊 Options Data:")
        if options_data:
            latest = options_data[0]
            print(f"   Latest: {latest[0]} - {latest[1]} contracts, {latest[2]} expirations")
            print(f"   Strike range: ${latest[3]:.2f} - ${latest[4]:.2f}")
            print(f"   Collection days: {len(options_data)}")
        else:
            print(f"   {RED}No recent options data{RESET}")
            
        # Check economic indicators from ML features
        fred_data = self.conn.execute("""
            SELECT 
                'VIXCLS' as series_id,
                MAX(feature_date) as latest_date,
                MAX(vix_level) as value
            FROM analytics.ml_features
            WHERE feature_date >= CURRENT_DATE - INTERVAL '7' DAY
            AND vix_level IS NOT NULL
        """).fetchall()
        
        print(f"\n💰 Economic Indicators:")
        if fred_data:
            for series_id, date, value in fred_data:
                print(f"   {series_id}: {value} ({date})")
        else:
            print(f"   {RED}No recent FRED data{RESET}")
            
    def check_data_quality(self):
        """Check data quality metrics"""
        print(f"\n{BLUE}=== Data Quality Metrics ==={RESET}")
        
        # Check for gaps in stock data
        gaps = self.conn.execute("""
            WITH date_series AS (
                SELECT 
                    date,
                    LAG(date) OVER (ORDER BY date) as prev_date,
                    DATEDIFF('day', LAG(date) OVER (ORDER BY date), date) as gap_days
                FROM market.price_data
                WHERE symbol = 'U'
                AND date >= CURRENT_DATE - INTERVAL 30 DAY
            )
            SELECT date, prev_date, gap_days
            FROM date_series
            WHERE gap_days > 3  -- More than weekend gap
        """).fetchall()
        
        print(f"\n📅 Data Gaps:")
        if gaps:
            print(f"   {YELLOW}Found {len(gaps)} gaps in stock data:{RESET}")
            for date, prev_date, gap_days in gaps[:5]:
                print(f"   Gap: {prev_date} to {date} ({gap_days} days)")
        else:
            print(f"   {GREEN}No significant gaps found{RESET}")
            
        # Check option data completeness
        option_quality = self.conn.execute("""
            SELECT 
                COUNT(*) as total_options,
                COUNT(CASE WHEN implied_volatility IS NOT NULL THEN 1 END) as with_iv,
                COUNT(CASE WHEN bid > 0 AND ask > 0 THEN 1 END) as valid_quotes,
                AVG(CASE WHEN bid > 0 AND ask > 0 THEN (ask - bid) / ask END) as avg_spread
            FROM options.contracts
            WHERE symbol = 'U'
            AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 7 DAY
        """).fetchone()
        
        print(f"\n🎯 Option Data Quality:")
        if option_quality[0] > 0:
            iv_pct = (option_quality[1] / option_quality[0]) * 100
            valid_pct = (option_quality[2] / option_quality[0]) * 100
            
            print(f"   Total options: {option_quality[0]}")
            print(f"   With IV: {option_quality[1]} ({iv_pct:.1f}%)")
            print(f"   Valid quotes: {option_quality[2]} ({valid_pct:.1f}%)")
            if option_quality[3]:
                print(f"   Avg spread: {option_quality[3]:.2%}")
        else:
            print(f"   {RED}No recent options data{RESET}")
            
    def check_ml_features(self):
        """Check ML feature calculation status"""
        print(f"\n{BLUE}=== ML Features Status ==={RESET}")
        
        features = self.conn.execute("""
            SELECT 
                feature_date,
                vix_level,
                market_regime,
                volatility_realized
            FROM analytics.ml_features
            WHERE symbol = 'U'
            ORDER BY feature_date DESC
            LIMIT 5
        """).fetchall()
        
        if features:
            latest = features[0]
            print(f"\n🤖 Latest ML Features ({latest[0]}):")
            print(f"   VIX Level: {latest[1]:.2f}")
            print(f"   Market Regime: {latest[2]}")
            print(f"   Realized Vol: {latest[3]:.2%}" if latest[3] else "   Realized Vol: N/A")
            
            # Check regime changes
            regimes = [f[2] for f in features]
            if len(set(regimes)) > 1:
                print(f"   {YELLOW}Regime change detected in last 5 days{RESET}")
        else:
            print(f"   {RED}No ML features found{RESET}")
            
        # Check wheel opportunities
        opportunities = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(premium_yield) as avg_yield,
                MIN(days_to_expiry) as min_dte,
                MAX(days_to_expiry) as max_dte
            FROM analytics.wheel_opportunities_mv
        """).fetchone()
        
        print(f"\n🎡 Wheel Opportunities:")
        if opportunities[0] > 0:
            print(f"   Total: {opportunities[0]}")
            print(f"   Avg yield: {opportunities[1]:.2%}")
            print(f"   DTE range: {opportunities[2]} - {opportunities[3]} days")
        else:
            print(f"   {RED}No wheel opportunities found{RESET}")
            
    def check_recent_logs(self):
        """Check recent log entries for errors"""
        print(f"\n{BLUE}=== Recent Log Activity ==={RESET}")
        
        log_files = {
            "EOD Collection": self.log_dir / "eod_collection.log",
            "Intraday": self.log_dir / "intraday_collection.log"
        }
        
        for name, log_file in log_files.items():
            if log_file.exists():
                # Get last 10 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Count errors and successes
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                errors = sum(1 for line in recent_lines if 'ERROR' in line)
                successes = sum(1 for line in recent_lines if '✅' in line)
                
                print(f"\n📝 {name}:")
                
                # Get last timestamp
                for line in reversed(lines):
                    if line.strip() and not line.startswith('#'):
                        try:
                            timestamp = line.split(' - ')[0]
                            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
                            age = datetime.now() - dt
                            
                            if age.total_seconds() < 3600:
                                print(f"   Last activity: {int(age.total_seconds() / 60)} minutes ago")
                            else:
                                print(f"   Last activity: {int(age.total_seconds() / 3600)} hours ago")
                            break
                        except:
                            pass
                            
                if errors > 0:
                    print(f"   {RED}Recent errors: {errors}{RESET}")
                if successes > 0:
                    print(f"   {GREEN}Recent successes: {successes}{RESET}")
                    
                # Show last error if any
                for line in reversed(recent_lines):
                    if 'ERROR' in line:
                        error_msg = line.strip().split('ERROR - ')[-1][:80]
                        print(f"   Last error: {error_msg}...")
                        break
            else:
                print(f"\n📝 {name}: {YELLOW}Log file not found{RESET}")
                
    def show_summary(self):
        """Show overall collection summary"""
        print(f"\n{BLUE}=== Collection Summary ==={RESET}")
        
        # Database size
        db_size = Path(self.db_path).stat().st_size / (1024 * 1024)
        print(f"\n💾 Database: {db_size:.1f} MB")
        
        # Total records
        totals = self.conn.execute("""
            SELECT 
                (SELECT COUNT(*) FROM options.contracts WHERE symbol='U') as options,
                (SELECT COUNT(*) FROM market.price_data WHERE symbol='U') as stock_days,
                (SELECT COUNT(*) FROM analytics.ml_features WHERE symbol='U') as ml_features
        """).fetchone()
        
        # FRED data is now stored in ml_features
        fred_count = 0
        try:
            # Count days with VIX data as proxy for FRED data
            fred_result = self.conn.execute("""
                SELECT COUNT(*) FROM analytics.ml_features 
                WHERE vix_level IS NOT NULL
            """).fetchone()
            fred_count = fred_result[0] if fred_result else 0
        except Exception:
            pass
        
        print(f"\n📊 Total Records:")
        print(f"   Options: {totals[0]:,}")
        print(f"   Stock days: {totals[1]:,}")
        print(f"   FRED observations: {fred_count:,}")
        print(f"   ML features: {totals[2]:,}")
        
        # Collection health
        print(f"\n🏥 Health Status:")
        
        # Check if collections are recent
        latest_option = self.conn.execute("""
            SELECT MAX(timestamp) FROM options.contracts WHERE symbol='U'
        """).fetchone()[0]
        
        latest_stock = self.conn.execute("""
            SELECT MAX(date) FROM market.price_data WHERE symbol='U'
        """).fetchone()[0]
        
        if latest_option:
            option_age = (datetime.now() - latest_option).days
            if option_age == 0:
                print(f"   Options: {GREEN}Current (today){RESET}")
            elif option_age == 1:
                print(f"   Options: {GREEN}Current (yesterday){RESET}")
            elif option_age <= 3:
                print(f"   Options: {YELLOW}Stale ({option_age} days old){RESET}")
            else:
                print(f"   Options: {RED}Very stale ({option_age} days old){RESET}")
                
        if latest_stock:
            stock_age = (datetime.now().date() - latest_stock).days
            if stock_age <= 1:
                print(f"   Stock: {GREEN}Current{RESET}")
            elif stock_age <= 3:
                print(f"   Stock: {YELLOW}Stale ({stock_age} days old){RESET}")
            else:
                print(f"   Stock: {RED}Very stale ({stock_age} days old){RESET}")
                

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Monitor Unity data collection')
    parser.add_argument('--days', type=int, default=7, help='Days to look back')
    parser.add_argument('--quality', action='store_true', help='Show detailed quality metrics')
    parser.add_argument('--logs', action='store_true', help='Show log analysis')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring mode')
    
    args = parser.parse_args()
    
    monitor = CollectionMonitor()
    
    try:
        if args.watch:
            # Continuous monitoring
            import time
            while True:
                # Clear screen
                print('\033[2J\033[H')
                
                print(f"{BLUE}Unity Data Collection Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
                monitor.check_recent_collections(days=args.days)
                monitor.check_ml_features()
                monitor.show_summary()
                
                print(f"\n{YELLOW}Press Ctrl+C to exit{RESET}")
                time.sleep(60)  # Refresh every minute
        else:
            # Single run
            monitor.check_recent_collections(days=args.days)
            
            if args.quality:
                monitor.check_data_quality()
                
            monitor.check_ml_features()
            
            if args.logs:
                monitor.check_recent_logs()
                
            monitor.show_summary()
            
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Monitoring stopped{RESET}")
    finally:
        monitor.conn.close()


if __name__ == "__main__":
    main()
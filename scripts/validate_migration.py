#!/usr/bin/env python3
"""
Validate the data migration and test performance
"""

import time

import duckdb


def run_performance_tests():
    """Test query performance on new database"""
    print("🔍 Running performance validation tests...")

    old_db = "data/wheel_trading_optimized.duckdb"
    new_db = "data/wheel_trading_optimized.duckdb"

    # Test queries
    queries = {
        "wheel_candidates": {
            "old": """
                SELECT symbol, strike, delta, bid, ask 
                FROM options_data 
                WHERE option_type = 'PUT' 
                AND delta BETWEEN -0.35 AND -0.25
                AND bid > 0 AND ask > 0
                LIMIT 100
            """,
            "new": """
                SELECT symbol, strike, delta, bid, ask 
                FROM analytics.wheel_opportunities_mv
                LIMIT 100
            """,
        },
        "recent_market": {
            "old": """
                SELECT * FROM market_data 
                WHERE date >= CURRENT_DATE - 30
                ORDER BY date DESC
            """,
            "new": """
                SELECT * FROM market.price_data 
                WHERE date >= CURRENT_DATE - 30
                ORDER BY date DESC
            """,
        },
        "options_lookup": {
            "old": """
                SELECT * FROM options_data
                WHERE symbol = 'U' 
                AND expiration >= CURRENT_DATE
                ORDER BY expiration, strike
            """,
            "new": """
                SELECT * FROM options.contracts
                WHERE symbol = 'U' 
                AND expiration >= CURRENT_DATE
                ORDER BY expiration, strike
            """,
        },
    }

    for query_name, query_pair in queries.items():
        print(f"\n📊 Testing: {query_name}")

        # Test old database
        try:
            old_conn = duckdb.connect(old_db, read_only=True)
            start = time.time()
            old_result = old_conn.execute(query_pair["old"]).fetchall()
            old_time = (time.time() - start) * 1000
            old_conn.close()
            print(f"  Old DB: {len(old_result)} rows in {old_time:.1f}ms")
        except Exception as e:
            print(f"  Old DB: Error - {e}")
            old_time = None
            old_result = []

        # Test new database
        try:
            new_conn = duckdb.connect(new_db, read_only=True)
            start = time.time()
            new_result = new_conn.execute(query_pair["new"]).fetchall()
            new_time = (time.time() - start) * 1000
            new_conn.close()
            print(f"  New DB: {len(new_result)} rows in {new_time:.1f}ms")

            if old_time and new_time:
                speedup = old_time / new_time
                print(f"  Speedup: {speedup:.1f}x")

        except Exception as e:
            print(f"  New DB: Error - {e}")

    print("\n✅ Performance tests complete")


def validate_data_integrity():
    """Validate data was migrated correctly"""
    print("\n🔍 Validating data integrity...")

    conn = duckdb.connect("data/wheel_trading_optimized.duckdb", read_only=True)

    # Check market data
    print("\n📊 Market Data Validation:")
    market_stats = conn.execute(
        """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT symbol) as symbols,
            MIN(date) as min_date,
            MAX(date) as max_date,
            AVG(volatility_20d) as avg_vol_20d,
            COUNT(CASE WHEN daily_return IS NOT NULL THEN 1 END) as returns_calculated
        FROM market.price_data
    """
    ).fetchone()

    print(f"  Total rows: {market_stats[0]}")
    print(f"  Symbols: {market_stats[1]}")
    print(f"  Date range: {market_stats[2]} to {market_stats[3]}")
    print(
        f"  Avg 20d volatility: {market_stats[4]:.2%}"
        if market_stats[4]
        else "  Avg 20d volatility: N/A"
    )
    print(f"  Returns calculated: {market_stats[5]}/{market_stats[0]}")

    # Check options data
    print("\n📈 Options Data Validation:")
    options_stats = conn.execute(
        """
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT symbol) as symbols,
            COUNT(DISTINCT expiration) as expirations,
            AVG(CASE WHEN bid > 0 AND ask > 0 THEN (ask - bid) / ask ELSE NULL END) as avg_spread,
            COUNT(CASE WHEN delta IS NOT NULL THEN 1 END) as greeks_count,
            MIN(timestamp) as min_timestamp,
            MAX(timestamp) as max_timestamp
        FROM options.contracts
    """
    ).fetchone()

    print(f"  Total rows: {options_stats[0]}")
    print(f"  Symbols: {options_stats[1]}")
    print(f"  Expirations: {options_stats[2]}")
    print(
        f"  Avg spread: {options_stats[3]:.2%}"
        if options_stats[3]
        else "  Avg spread: N/A"
    )
    print(f"  Greeks populated: {options_stats[4]}/{options_stats[0]}")
    print(f"  Timestamp range: {options_stats[5]} to {options_stats[6]}")

    # Check wheel opportunities
    print("\n🎯 Wheel Opportunities Validation:")
    wheel_stats = conn.execute(
        """
        SELECT 
            COUNT(*) as total_opps,
            AVG(premium_yield) as avg_yield,
            AVG(spread_pct) as avg_spread,
            MIN(delta) as min_delta,
            MAX(delta) as max_delta,
            AVG(days_to_expiry) as avg_dte
        FROM analytics.wheel_opportunities_mv
    """
    ).fetchone()

    print(f"  Total opportunities: {wheel_stats[0]}")
    print(
        f"  Avg premium yield: {wheel_stats[1]:.2%}"
        if wheel_stats[1]
        else "  Avg premium yield: N/A"
    )
    print(
        f"  Avg spread: {wheel_stats[2]:.2%}" if wheel_stats[2] else "  Avg spread: N/A"
    )
    print(f"  Delta range: {wheel_stats[3]:.2f} to {wheel_stats[4]:.2f}")
    print(
        f"  Avg days to expiry: {wheel_stats[5]:.0f}"
        if wheel_stats[5]
        else "  Avg days to expiry: N/A"
    )

    # Check indexes
    print("\n🏎️  Index Usage:")
    indexes = conn.execute(
        """
        SELECT table_name, index_name
        FROM duckdb_indexes()
        ORDER BY table_name, index_name
    """
    ).fetchall()

    for table, index in indexes:
        print(f"  {table}: {index}")

    conn.close()
    print("\n✅ Data integrity validation complete")


def check_hardware_optimization():
    """Check if hardware optimizations are working"""
    print("\n⚡ Checking hardware optimization...")

    conn = duckdb.connect("data/wheel_trading_optimized.duckdb", read_only=True)

    # Check settings
    settings = conn.execute(
        """
        SELECT 
            current_setting('memory_limit') as memory,
            current_setting('threads') as threads,
            current_setting('checkpoint_threshold') as checkpoint
    """
    ).fetchone()

    print(f"  Memory limit: {settings[0]}")
    print(f"  Threads: {settings[1]}")
    print(f"  Checkpoint: {settings[2]}")

    # Run a parallel scan test
    print("\n  Testing parallel scan performance...")
    start = time.time()
    result = conn.execute(
        """
        SELECT 
            symbol, 
            COUNT(*) as options,
            AVG(bid) as avg_bid,
            AVG(implied_volatility) as avg_iv
        FROM options.contracts
        GROUP BY symbol
    """
    ).fetchall()
    duration = time.time() - start

    print(f"  Aggregated {len(result)} symbols in {duration:.2f}s")
    print(f"  Performance: {len(result)/duration:.0f} symbols/sec")

    conn.close()
    print("\n✅ Hardware optimization check complete")


def main():
    """Run all validations"""
    print("🚀 Starting migration validation\n")

    run_performance_tests()
    validate_data_integrity()
    check_hardware_optimization()

    print("\n✨ Validation complete!")


if __name__ == "__main__":
    main()

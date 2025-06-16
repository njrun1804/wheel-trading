#!/usr/bin/env python3
"""
Create optimized DuckDB schema with M4 Pro hardware acceleration
Simplified version without partial indexes (not supported in DuckDB)
"""

import os

import duckdb


def create_optimized_database():
    """Create the optimized database with hardware-specific settings"""

    db_path = "data/wheel_trading_optimized.duckdb"

    # Remove if exists
    if os.path.exists(db_path):
        print(f"⚠️  Removing existing {db_path}")
        os.remove(db_path)

    print(f"🚀 Creating optimized database at {db_path}")
    conn = duckdb.connect(db_path)

    # Configure for M4 Pro hardware acceleration
    print("⚡ Configuring for M4 Pro (12 cores, 24GB RAM)...")
    conn.execute("SET memory_limit='8GB'")  # Use 8GB (33% of 24GB)
    conn.execute("SET threads TO 10")  # Use 10 cores (leaving 2 for system)

    # Create schemas
    print("📁 Creating schemas...")
    for schema in ["market", "options", "trading", "analytics", "system"]:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    # Create market data table
    print("📊 Creating market data tables...")
    conn.execute(
        """
        CREATE TABLE market.price_data (
            symbol VARCHAR NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            daily_return DECIMAL(8,6),
            volatility_20d DECIMAL(8,6),
            volatility_60d DECIMAL(8,6),
            year_month INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """
    )

    # Create options table
    print("📈 Creating options tables...")
    conn.execute(
        """
        CREATE TABLE options.contracts (
            symbol VARCHAR NOT NULL,
            expiration DATE NOT NULL,
            strike DECIMAL(10,2) NOT NULL,
            option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
            timestamp TIMESTAMP NOT NULL,
            bid DECIMAL(10,4),
            ask DECIMAL(10,4),
            mid DECIMAL(10,4),
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility DECIMAL(8,6),
            delta DECIMAL(6,4),
            gamma DECIMAL(8,6),
            theta DECIMAL(10,4),
            vega DECIMAL(10,4),
            rho DECIMAL(10,4),
            moneyness DECIMAL(6,4),
            days_to_expiry INTEGER,
            year_month INTEGER,
            PRIMARY KEY (symbol, expiration, strike, option_type, timestamp)
        )
    """
    )

    # Create indexes without WHERE clauses
    print("🏎️  Creating performance indexes...")

    # Options indexes
    conn.execute(
        """
        CREATE INDEX idx_wheel_candidates ON options.contracts(
            symbol, option_type, delta, expiration, timestamp DESC
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX idx_options_lookup ON options.contracts(
            symbol, expiration, strike, option_type, timestamp DESC
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX idx_recent_options ON options.contracts(
            timestamp DESC, symbol
        )
    """
    )

    # Market data indexes
    conn.execute(
        """
        CREATE INDEX idx_market_symbol_date ON market.price_data(
            symbol, date DESC
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX idx_market_volatility ON market.price_data(
            date DESC, symbol, volatility_20d
        )
    """
    )

    # Trading tables
    print("💼 Creating trading tables...")
    conn.execute(
        """
        CREATE TABLE trading.positions (
            position_id VARCHAR PRIMARY KEY,
            account_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            position_type VARCHAR(20) NOT NULL,
            quantity INTEGER NOT NULL,
            entry_price DECIMAL(10,2) NOT NULL,
            entry_date TIMESTAMP NOT NULL,
            expiration DATE,
            strike DECIMAL(10,2),
            option_type VARCHAR(4),
            current_price DECIMAL(10,2),
            unrealized_pnl DECIMAL(12,2),
            status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
            position_delta DECIMAL(8,4),
            position_theta DECIMAL(10,2),
            margin_required DECIMAL(12,2),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX idx_active_positions ON trading.positions(
            status, account_id, symbol
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX idx_expiring_positions ON trading.positions(
            status, expiration, symbol
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE trading.decisions (
            decision_id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            decision_type VARCHAR(20) NOT NULL,
            recommendation JSON NOT NULL,
            executed BOOLEAN DEFAULT FALSE,
            execution_price DECIMAL(10,2),
            execution_timestamp TIMESTAMP,
            confidence_score DECIMAL(4,3),
            expected_return DECIMAL(8,4),
            risk_score DECIMAL(4,3)
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX idx_pending_decisions ON trading.decisions(
            executed, symbol, timestamp DESC
        )
    """
    )

    # Analytics tables
    print("🧮 Creating analytics tables...")
    conn.execute(
        """
        CREATE TABLE analytics.ml_features (
            symbol VARCHAR NOT NULL,
            feature_date DATE NOT NULL,
            returns_5d DECIMAL(8,6),
            returns_20d DECIMAL(8,6),
            volatility_realized DECIMAL(8,6),
            volatility_garch DECIMAL(8,6),
            put_call_ratio DECIMAL(8,4),
            iv_rank DECIMAL(4,3),
            iv_percentile DECIMAL(4,3),
            term_structure_slope DECIMAL(8,6),
            vix_level DECIMAL(6,2),
            market_regime VARCHAR(20),
            rsi_14d DECIMAL(4,2),
            bollinger_position DECIMAL(4,3),
            PRIMARY KEY (symbol, feature_date)
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX idx_ml_recent ON analytics.ml_features(
            feature_date DESC, symbol
        )
    """
    )

    # System tables
    print("⚙️  Creating system tables...")
    conn.execute("USE system")
    conn.execute(
        """
        CREATE TABLE migration_log (
            migration_id INTEGER PRIMARY KEY,
            table_name VARCHAR NOT NULL,
            rows_migrated BIGINT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            status VARCHAR(20),
            error_message TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE SEQUENCE migration_seq START 1
    """
    )
    conn.execute("USE main")  # Switch back

    # Create views for common queries
    print("👁️  Creating optimized views...")
    conn.execute(
        """
        CREATE VIEW analytics.wheel_opportunities AS
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
            (o.bid * 100) / (o.strike * 100) as premium_yield,
            (o.ask - o.bid) / NULLIF(o.ask, 0) as spread_pct
        FROM options.contracts o
        WHERE 
            o.option_type = 'PUT'
            AND o.delta BETWEEN -0.35 AND -0.25
            AND o.days_to_expiry BETWEEN 20 AND 45
            AND o.bid > 0 
            AND o.ask > 0
            AND o.timestamp = (
                SELECT MAX(timestamp) 
                FROM options.contracts o2 
                WHERE o2.symbol = o.symbol 
                AND o2.expiration = o.expiration 
                AND o2.strike = o.strike 
                AND o2.option_type = o.option_type
            )
        ORDER BY premium_yield DESC
    """
    )

    print("✅ Schema creation complete!")

    # Show configuration
    result = conn.execute(
        """
        SELECT 
            current_setting('memory_limit') as memory_limit,
            current_setting('threads') as threads
    """
    ).fetchone()

    print("\n🔧 Configuration:")
    print(f"   Memory Limit: {result[0]}")
    print(f"   Threads: {result[1]}")

    # Show created objects
    tables = conn.execute(
        """
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'main')
        ORDER BY table_schema, table_name
    """
    ).fetchall()

    print(f"\n📊 Created {len(tables)} tables:")
    for schema, table in tables:
        print(f"   {schema}.{table}")

    conn.close()
    print(f"\n✨ Database created successfully at {db_path}")


if __name__ == "__main__":
    create_optimized_database()

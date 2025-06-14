#!/usr/bin/env python3
"""
ETL script to sync data from operational to analytical database.
Focuses on preparing data for backtesting and daily recommendations.
"""

import sys
from pathlib import Path

import duckdb

from unity_wheel.config.unified_config import get_config
config = get_config()



def create_etl_pipeline():
    """Create ETL pipeline to prepare unified database for backtesting."""

    # Database paths
    operational_db = Path.home() / "data/wheel_trading_optimized.duckdb"
    analytical_db = Path("data/unified_wheel_trading.duckdb")

    if not analytical_db.exists():
        print(f"Analytical database not found at {analytical_db}")
        return False

    print(f"Connecting to analytical database: {analytical_db}")
    conn = duckdb.connect(str(analytical_db))

    try:
        # 1. Calculate returns for all securities
        print("\n=== Calculating Returns ===")
        conn.execute(
            """
            -- Add returns column if missing
            ALTER TABLE market_data ADD COLUMN IF NOT EXISTS returns DOUBLE;
        """
        )

        # Calculate returns using window functions
        conn.execute(
            """
            UPDATE market_data m1
            SET returns = (
                SELECT (m1.close - prev.close) / prev.close
                FROM (
                    SELECT symbol, date, close,
                           LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close
                    FROM market_data
                    WHERE close IS NOT NULL AND close > 0
                ) prev
                WHERE prev.symbol = m1.symbol
                AND prev.date = m1.date
                AND prev.prev_close IS NOT NULL
            )
            WHERE m1.close IS NOT NULL AND m1.close > 0
        """
        )

        updated = conn.execute(
            """
            SELECT COUNT(*) FROM market_data
            WHERE returns IS NOT NULL
        """
        ).fetchone()[0]
        print(f"Calculated returns for {updated:,} records")

        # 2. Create backtest-optimized tables
        print("\n=== Creating Backtest Tables ===")

        # Create price history view for stocks only
        conn.execute(
            """
            CREATE OR REPLACE VIEW stock_price_history AS
            SELECT
                symbol,
                date,
                open,
                high,
                low,
                close,
                volume,
                returns,
                -- Rolling calculations for risk metrics
                AVG(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                ) as avg_return_250d,
                STDDEV(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                ) * SQRT(252) as volatility_250d,
                STDDEV(returns) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) * SQRT(252) as volatility_20d
            FROM market_data
            WHERE data_type = 'stock'
            AND close IS NOT NULL
            ORDER BY symbol, date
        """
        )

        # Create options chain history view
        conn.execute(
            """
            CREATE OR REPLACE VIEW options_history AS
            SELECT
                md.symbol as option_symbol,
                om.underlying,
                md.date,
                om.expiration,
                om.strike,
                om.option_type,
                md.open,
                md.high,
                md.low,
                md.close as premium,
                md.volume,
                -- Days to expiration
                DATEDIFF('day', md.date, om.expiration) as dte,
                -- Moneyness
                s.close as spot_price,
                om.strike / s.close as moneyness
            FROM market_data md
            JOIN options_metadata om ON md.symbol = om.symbol
            LEFT JOIN market_data s ON om.underlying = s.symbol AND md.date = s.date
            WHERE md.data_type = 'option'
            AND s.data_type = 'stock'
            AND md.close IS NOT NULL
            AND md.close > 0
        """
        )

        # 3. Create pre-calculated features for backtesting
        print("\n=== Creating Feature Tables ===")

        conn.execute(
            """
            CREATE OR REPLACE TABLE backtest_features AS
            SELECT
                s.date,
                s.symbol,
                s.close as stock_price,
                s.returns,
                s.volume,
                s.volatility_20d,
                s.volatility_250d,
                -- VaR calculation (5th percentile of returns)
                QUANTILE(s2.returns, 0.05) OVER (
                    PARTITION BY s.symbol
                    ORDER BY s.date
                    ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
                ) as var_95,
                -- Risk-free rate from FRED data
                COALESCE(ei.value, 0.05) as risk_free_rate,
                -- VIX if available
                vix.value as vix
            FROM stock_price_history s
            LEFT JOIN economic_indicators ei
                ON s.date = ei.date
                AND ei.indicator = 'DGS3MO'
            LEFT JOIN economic_indicators vix
                ON s.date = vix.date
                AND vix.indicator = 'VIXCLS'
            LEFT JOIN stock_price_history s2
                ON s.symbol = s2.symbol
                AND s2.date <= s.date
                AND s2.date >= s.date - INTERVAL '250 days'
            WHERE s.symbol = config.trading.symbol  -- Focus on Unity for now
        """
        )

        features = conn.execute("SELECT COUNT(*) FROM backtest_features").fetchone()[0]
        print(f"Created {features:,} feature records for backtesting")

        # 4. Create indexes for performance
        print("\n=== Creating Performance Indexes ===")

        indexes = [
            # Primary indexes for time series queries
            "CREATE INDEX IF NOT EXISTS idx_market_symbol_date ON market_data(symbol, date)",
            "CREATE INDEX IF NOT EXISTS idx_market_date_type ON market_data(date, data_type)",
            # Options metadata indexes
            "CREATE INDEX IF NOT EXISTS idx_options_meta_exp ON options_metadata(expiration)",
            "CREATE INDEX IF NOT EXISTS idx_options_meta_strike ON options_metadata(underlying, strike)",
            # Backtest features index
            "CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_features(date)",
            # Economic indicators index
            "CREATE INDEX IF NOT EXISTS idx_econ_ind_date ON economic_indicators(indicator, date)",
        ]

        for idx_sql in indexes:
            conn.execute(idx_sql)
            idx_name = idx_sql.split("idx_")[1].split(" ")[0]
            print(f"  Created index: {idx_name}")

        # 5. Create helper views for daily recommendations
        print("\n=== Creating Helper Views ===")

        # Latest market snapshot
        conn.execute(
            """
            CREATE OR REPLACE VIEW latest_market_snapshot AS
            SELECT
                s.symbol,
                s.close as stock_price,
                s.volatility_20d,
                s.volatility_250d,
                s.avg_return_250d * 252 as annual_return,
                rf.rate as risk_free_rate,
                v.value as vix,
                s.date as last_update
            FROM stock_price_history s
            LEFT JOIN current_risk_free_rate rf ON 1=1
            LEFT JOIN current_vix v ON 1=1
            WHERE s.date = (SELECT MAX(date) FROM stock_price_history WHERE symbol = s.symbol)
        """
        )

        # Available strikes with liquidity
        conn.execute(
            """
            CREATE OR REPLACE VIEW liquid_option_strikes AS
            SELECT DISTINCT
                om.underlying,
                om.expiration,
                om.strike,
                om.option_type,
                oh.premium,
                oh.volume,
                oh.dte,
                oh.moneyness
            FROM options_history oh
            JOIN options_metadata om ON oh.option_symbol = om.symbol
            WHERE oh.date >= CURRENT_DATE - INTERVAL '5 days'
            AND oh.volume > 0
            AND oh.premium > 0.01
            ORDER BY om.underlying, om.expiration, om.strike
        """
        )

        # 6. Prepare backtest results table
        print("\n=== Preparing Backtest Results Table ===")

        conn.execute(
            """
            -- Ensure backtest_results table has proper schema
            DROP TABLE IF EXISTS backtest_results;

            CREATE TABLE backtest_results (
                run_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol VARCHAR,
                start_date DATE,
                end_date DATE,
                initial_capital DOUBLE,
                final_capital DOUBLE,
                total_return DOUBLE,
                annualized_return DOUBLE,
                sharpe_ratio DOUBLE,
                max_drawdown DOUBLE,
                win_rate DOUBLE,
                total_trades INT,
                parameters JSON,  -- Strategy parameters used
                metrics JSON,     -- Detailed metrics
                positions JSON    -- All positions taken
            )
        """
        )

        # 7. Final statistics
        print("\n=== Database Statistics ===")
        stats = conn.execute(
            """
            SELECT
                'Unity Stock Data' as dataset,
                COUNT(*) as records,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM market_data
            WHERE symbol = config.trading.symbol AND data_type = 'stock'

            UNION ALL

            SELECT
                'Unity Options Data',
                COUNT(*),
                MIN(date),
                MAX(date)
            FROM market_data
            WHERE symbol LIKE 'U %' AND data_type = 'option'

            UNION ALL

            SELECT
                'Backtest Features',
                COUNT(*),
                MIN(date),
                MAX(date)
            FROM backtest_features
        """
        ).fetchall()

        for dataset, records, start, end in stats:
            print(f"  {dataset:<20} {records:>8,} records  ({start} to {end})")

        # 8. Vacuum and analyze
        print("\n=== Optimizing Database ===")
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
        print("Database optimized for backtesting")

        conn.close()
        return True

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        conn.close()
        return False


if __name__ == "__main__":
    success = create_etl_pipeline()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""Verify Unity data is properly stored and test risk calculations."""

import os
from datetime import datetime, timedelta

import duckdb
import numpy as np

from unity_wheel.config.unified_config import get_config
config = get_config()


DB_PATH = os.path.expanduser(config.storage.database_path)


def main():
    print("🔍 Verifying Unity Historical Data")
    print("=" * 60)

    # Connect to DuckDB
    conn = duckdb.connect(DB_PATH)

    # Check data summary
    summary = conn.execute(
        """
        SELECT
            symbol,
            COUNT(*) as days,
            MIN(date) as start_date,
            MAX(date) as end_date,
            AVG(returns) * 252 as annual_return,
            STDDEV(returns) * SQRT(252) as annual_vol,
            MIN(close) as min_price,
            MAX(close) as max_price,
            MIN(returns) as worst_day,
            MAX(returns) as best_day
        FROM price_history
        WHERE symbol = config.trading.symbol
        GROUP BY symbol
    """
    ).fetchone()

    if summary:
        symbol, days, start, end, ret, vol, min_p, max_p, worst, best = summary

        print(f"\n📊 Unity Price Data Summary:")
        print(f"   Days stored: {days}")
        print(f"   Date range: {start} to {end}")
        print(f"   Price range: ${min_p:.2f} - ${max_p:.2f}")
        print(f"\n📈 Return Statistics:")
        print(f"   Annual return: {ret*100:.1f}%")
        print(f"   Annual volatility: {vol*100:.1f}%")
        print(f"   Worst day: {worst*100:.1f}%")
        print(f"   Best day: {best*100:.1f}%")

        # Show recent prices
        print(f"\n📅 Recent 10 Days:")
        recent = conn.execute(
            """
            SELECT date, open, high, low, close, volume, returns
            FROM price_history
            WHERE symbol = config.trading.symbol
            ORDER BY date DESC
            LIMIT 10
        """
        ).fetchall()

        print(
            f"{'Date':>12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12} {'Return':>8}"
        )
        print("-" * 80)
        for row in recent:
            date, o, h, l, c, v, r = row
            print(
                f"{str(date):>12} {o:>8.2f} {h:>8.2f} {l:>8.2f} {c:>8.2f} {v:>12,} {r*100:>7.1f}%"
            )

        # Calculate risk metrics
        print(f"\n💡 Risk Metrics (250-day):")

        # Get returns for VaR calculation
        returns = conn.execute(
            """
            SELECT returns
            FROM price_history
            WHERE symbol = config.trading.symbol
            AND date >= CURRENT_DATE - INTERVAL '250 days'
            ORDER BY date
        """
        ).fetchall()

        if returns:
            returns = np.array([float(r[0]) for r in returns if r[0] is not None])

            # VaR calculations
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)

            # CVaR (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= var_95])

            # Sharpe ratio (assuming 5% risk-free rate)
            rf_daily = 0.05 / 252
            sharpe = (np.mean(returns) - rf_daily) / np.std(returns) * np.sqrt(252)

            print(f"   VaR (95%): {var_95*100:.1f}% daily")
            print(f"   VaR (99%): {var_99*100:.1f}% daily")
            print(f"   CVaR (95%): {cvar_95*100:.1f}% daily")
            print(f"   Sharpe Ratio: {sharpe:.2f}")

        # Check if we have both FRED and price data
        print(f"\n🔗 Cross-Database Check:")

        # Check FRED tables
        fred_check = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name IN ('fred_series', 'fred_observations')
        """
        ).fetchone()[0]

        if fred_check > 0:
            print(f"   ✅ FRED tables exist")

            # Check for VIX data
            vix_count = (
                conn.execute(
                    """
                SELECT COUNT(*)
                FROM fred_observations
                WHERE series_id = 'VIXCLS'
            """
                ).fetchone()[0]
                if fred_check
                else 0
            )

            if vix_count > 0:
                print(f"   ✅ VIX data available ({vix_count} days)")
        else:
            print(f"   ❌ FRED tables not found")

        print(f"\n✅ Unity data is properly stored and ready for risk calculations!")

    else:
        print("❌ No Unity data found in database")

    conn.close()


if __name__ == "__main__":
    main()

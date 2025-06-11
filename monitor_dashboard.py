#!/usr/bin/env python3
"""Simple monitoring dashboard - run in terminal."""

import os
import time
from datetime import datetime
from pathlib import Path

import duckdb
import yaml


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def get_current_metrics():
    metrics = {}

    # Get market data
    db_path = Path("data/unified_wheel_trading.duckdb")
    if db_path.exists():
        conn = duckdb.connect(str(db_path), read_only=True)

        # Current volatility
        vol = conn.execute(
            """
            SELECT volatility_20d, stock_price
            FROM backtest_features_clean
            WHERE symbol = 'U'
            ORDER BY date DESC
            LIMIT 1
        """
        ).fetchone()

        if vol:
            metrics["volatility"] = vol[0]
            metrics["unity_price"] = vol[1]

        conn.close()

    # Get positions
    positions_file = Path("my_positions.yaml")
    if positions_file.exists():
        with open(positions_file) as f:
            data = yaml.safe_load(f) or {}

        open_positions = [p for p in data.get("positions", []) if p.get("status") == "open"]
        metrics["open_puts"] = len(open_positions)
        metrics["cash_available"] = data.get("cash_available", 0)

    return metrics


def display_dashboard():
    while True:
        clear_screen()
        metrics = get_current_metrics()

        print("=" * 60)
        print(f"UNITY WHEEL TRADING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        print(f"\nMARKET DATA:")
        print(f"  Unity Price:  ${metrics.get('unity_price', 0):.2f}")
        print(f"  Volatility:   {metrics.get('volatility', 0):.1%}")

        # Volatility indicator
        vol = metrics.get("volatility", 0)
        if vol > 1.20:
            vol_status = "🔴 EXTREME - STOP TRADING"
        elif vol > 1.00:
            vol_status = "🟠 VERY HIGH - REDUCE SIZE"
        elif vol > 0.80:
            vol_status = "🟡 HIGH - BE CAUTIOUS"
        elif vol > 0.60:
            vol_status = "🟢 ELEVATED - NORMAL OPS"
        else:
            vol_status = "🟢 LOW - INCREASE SIZE"
        print(f"  Status:       {vol_status}")

        print(f"\nPOSITION STATUS:")
        print(f"  Open Puts:    {metrics.get('open_puts', 0)}/3")
        print(f"  Cash Free:    ${metrics.get('cash_available', 0):,.0f}")

        print(f"\nRECOMMENDED PARAMETERS:")
        if vol > 0.80:
            print(f"  Delta:        0.40")
            print(f"  DTE:          21-30")
            print(f"  Position:     10%")
        else:
            print(f"  Delta:        0.35")
            print(f"  DTE:          30-45")
            print(f"  Position:     15-20%")

        print("\n" + "=" * 60)
        print("Press Ctrl+C to exit | Refreshing in 30s...")

        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            break


if __name__ == "__main__":
    display_dashboard()

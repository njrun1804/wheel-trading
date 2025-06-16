#!/usr/bin/env python3
"""
Simple on-demand position assessment for Unity Wheel Trading.
Run this anytime to check current market conditions and get recommendations.
"""

from datetime import datetime
from pathlib import Path

import duckdb
import yaml

from unity_wheel.config.unified_config import get_config

config = get_config()


def check_current_market():
    """Check current Unity market conditions."""

    db_path = Path("data/unified_wheel_trading.duckdb")
    if not db_path.exists():
        print("❌ Database not found!")
        return None

    conn = duckdb.connect(str(db_path), read_only=True)

    # Get latest market data
    market_data = conn.execute(
        """
        SELECT
            date,
            stock_price,
            volatility_20d,
            volume
        FROM backtest_features_clean
        WHERE symbol = config.trading.symbol
        ORDER BY date DESC
        LIMIT 1
    """
    ).fetchone()

    conn.close()

    if market_data:
        return {
            "date": market_data[0],
            "price": market_data[1],
            "volatility": market_data[2],
            "volume": market_data[3],
        }
    return None


def check_current_positions():
    """Check current open positions."""

    positions_file = Path("my_positions.yaml")
    if not positions_file.exists():
        return []

    with open(positions_file) as f:
        data = yaml.safe_load(f)

    open_positions = [p for p in data.get("positions", []) if p.get("status") == "open"]

    return open_positions


def assess_trading_conditions():
    """Assess current trading conditions and provide recommendations."""

    print("=" * 60)
    print("UNITY WHEEL TRADING - POSITION ASSESSMENT")
    print("=" * 60)
    print(f"Assessment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Get market data
    market = check_current_market()
    if not market:
        print("❌ Could not retrieve market data")
        return

    print("📊 CURRENT MARKET CONDITIONS")
    print("-" * 30)
    print(f"Unity Price: ${market['price']:.2f}")
    print(f"Volatility: {market['volatility']:.1%}")
    print(f"Volume: {market['volume']:,}")
    print(f"Data Date: {market['date']}")

    # Check volatility regime
    vol = market["volatility"]
    print("\n🎯 VOLATILITY REGIME")
    print("-" * 30)

    if vol > 1.20:
        print("⛔ EXTREME VOLATILITY (>120%)")
        print("Action: STOP TRADING - Circuit breaker activated")
        regime = "extreme"
    elif vol > 1.00:
        print("🔴 VERY HIGH VOLATILITY (100-120%)")
        print("Action: Trade with extreme caution")
        regime = "very_high"
    elif vol > 0.80:
        print("🟡 HIGH VOLATILITY (80-100%)")
        print("Action: Reduce position sizes")
        regime = "high"
    elif vol > 0.60:
        print("🟢 ELEVATED VOLATILITY (60-80%)")
        print("Action: Normal trading with standard parameters")
        regime = "elevated"
    else:
        print("🔵 LOW VOLATILITY (<60%)")
        print("Action: Consider increasing position sizes")
        regime = "low"

    # Check current positions
    positions = check_current_positions()
    print("\n📈 CURRENT POSITIONS")
    print("-" * 30)
    print(f"Open Puts: {len(positions)}")

    if positions:
        total_risk = 0
        for i, pos in enumerate(positions):
            risk = pos["strike"] * pos["contracts"] * 100
            total_risk += risk
            print(f"\nPosition {i+1}:")
            print(f"  Strike: ${pos['strike']}")
            print(f"  Expiration: {pos['expiration']}")
            print(f"  Contracts: {pos['contracts']}")
            print(f"  Risk: ${risk:,.0f}")

        print(f"\nTotal Risk: ${total_risk:,.0f}")
    else:
        print("No open positions")

    # Trading recommendations based on regime
    print("\n💡 RECOMMENDATIONS")
    print("-" * 30)

    if regime == "extreme":
        print("1. DO NOT open new positions")
        print("2. Consider closing existing positions")
        print("3. Wait for volatility < 100%")
    elif regime == "very_high":
        print("1. Maximum 1 position (5% of portfolio)")
        print("2. Use far OTM strikes (Delta 0.15)")
        print("3. Short DTE only (14-21 days)")
    elif regime == "high":
        print("1. Maximum 2 positions (10% total)")
        print("2. Delta target: 0.25-0.30")
        print("3. DTE: 21-30 days")
        print("4. Take profits at 25%")
    elif regime == "elevated":
        print("1. Standard parameters apply")
        print("2. Delta target: 0.35-0.40")
        print("3. DTE: 30-45 days")
        print("4. Position size: 15-20%")
    else:
        print("1. Opportunity to be aggressive")
        print("2. Delta target: 0.40-0.45")
        print("3. DTE: 45-60 days")
        print("4. Position size: up to 25%")

    # Earnings check
    print("\n📅 EARNINGS CHECK")
    print("-" * 30)

    # Unity earnings dates (update quarterly)
    earnings_dates = [
        datetime(2025, 8, 7),  # Q2 2025
        datetime(2025, 11, 6),  # Q3 2025
    ]

    today = datetime.now()
    days_to_earnings = 999

    for earnings_date in earnings_dates:
        if earnings_date > today:
            days_to_earnings = (earnings_date - today).days
            break

    print(f"Days to next earnings: {days_to_earnings}")

    if days_to_earnings <= 7:
        print("⚠️  EARNINGS BLACKOUT - No new positions!")
    elif days_to_earnings <= 14:
        print("⚠️  Earnings approaching - consider reducing exposure")
    else:
        print("✅ Safe distance from earnings")

    # Summary
    print("\n📋 SUMMARY")
    print("-" * 30)

    can_trade = vol <= 1.20 and days_to_earnings > 7

    if can_trade:
        if len(positions) < 2:
            print("✅ OK TO TRADE - Consider opening new position")
            print(
                f"   Recommended: Delta {0.40 if vol < 0.80 else 0.30}, DTE {30 if vol < 0.80 else 21}"
            )
        else:
            print("⚠️  At position limit - manage existing positions")
    else:
        print("🛑 DO NOT TRADE - Conditions unfavorable")

    print("\n" + "=" * 60)
    print("Run 'python run.py -p 100000' for detailed recommendation")


if __name__ == "__main__":
    assess_trading_conditions()

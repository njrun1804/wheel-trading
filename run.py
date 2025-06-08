#!/usr/bin/env python3
"""Simple CLI runner for wheel trading decisions."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from src.config import get_settings
from src.wheel import WheelStrategy


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(message)s" if not verbose else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_str,
        stream=sys.stdout
    )


def main() -> None:
    """Run wheel trading decision engine."""
    parser = argparse.ArgumentParser(description="Wheel Trading Strategy Decision Engine")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--ticker", default="U", help="Ticker to analyze (default: U)")
    parser.add_argument("--portfolio", type=float, default=100000, help="Portfolio value")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print(f"\n🎯 Wheel Trading Decision Engine")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Initialize strategy
    settings = get_settings()
    wheel = WheelStrategy()
    
    # Demo decision (will be replaced with real broker data)
    print(f"\n📊 Analyzing {args.ticker}")
    print(f"💰 Portfolio: ${args.portfolio:,.0f}")
    print(f"🎯 Target Delta: {settings.wheel_delta_target}")
    print(f"📆 Target DTE: {settings.days_to_expiry_target} days")
    
    # Example strikes for Unity (will come from broker API)
    current_price = 35.50
    strikes = [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0]
    
    print(f"\n💹 Current Price: ${current_price}")
    
    # Find optimal put
    optimal_put = wheel.find_optimal_put_strike(
        current_price=current_price,
        available_strikes=strikes,
        volatility=0.65,  # Unity typical IV
        days_to_expiry=settings.days_to_expiry_target,
    )
    
    if optimal_put:
        contracts = wheel.calculate_position_size(args.ticker, current_price, args.portfolio)
        print(f"\n✅ RECOMMENDATION: Sell {contracts} {args.ticker} ${optimal_put}P")
        print(f"   Expiry: {settings.days_to_expiry_target} days")
        print(f"   Max Risk: ${optimal_put * 100 * contracts:,.0f}")
    
    print("\n" + "=" * 50)
    print("Note: Using demo data. Connect broker for live recommendations.\n")


if __name__ == "__main__":
    main()
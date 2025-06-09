#!/usr/bin/env python3
"""Simple CLI runner for wheel trading decisions using unity_wheel."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from src.config import get_settings
from src.unity_wheel.diagnostics import SelfDiagnostics
from src.unity_wheel.strategy import WheelParameters, WheelStrategy


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = (
        "%(message)s" if not verbose else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(level=level, format=format_str, stream=sys.stdout)


def main() -> None:
    """Run wheel trading decision engine."""
    # Deprecation warning
    print("\n⚠️  WARNING: run.py is deprecated!")
    print("   Please use 'python run_aligned.py' for the v2.0 autonomous system.")
    print("   run.py will be removed in a future version.\n")

    parser = argparse.ArgumentParser(description="Unity Wheel Trading Strategy Decision Engine")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--ticker", default="U", help="Ticker to analyze (default: U)")
    parser.add_argument("--portfolio", type=float, default=100000, help="Portfolio value")
    parser.add_argument("--diagnose", action="store_true", help="Run self-diagnostics")

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Run diagnostics if requested
    if args.diagnose:
        print("\n🔍 Running Self-Diagnostics...")
        print("=" * 50)
        diag = SelfDiagnostics()
        if diag.run_all_checks():
            print("✅ All diagnostics passed!")
        else:
            print("❌ Diagnostics failed - check errors above")
            sys.exit(1)
        print("=" * 50 + "\n")

    print(f"\n🎯 Unity Wheel Trading Decision Engine")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Initialize strategy with parameters
    settings = get_settings()
    wheel_params = WheelParameters(
        target_delta=settings.wheel_delta_target,
        target_dte=settings.days_to_expiry_target,
        max_position_size=settings.max_position_size,
    )
    wheel = WheelStrategy(wheel_params)

    # Demo decision (will be replaced with real broker data)
    print(f"\n📊 Analyzing {args.ticker}")
    print(f"💰 Portfolio: ${args.portfolio:,.0f}")
    print(f"🎯 Target Delta: {wheel_params.target_delta}")
    print(f"📆 Target DTE: {wheel_params.target_dte} days")

    # Example strikes for Unity (will come from broker API)
    current_price = 35.50
    strikes = [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0]

    print(f"\n💹 Current Price: ${current_price}")

    # Find optimal put with validation
    strike_rec = wheel.find_optimal_put_strike(
        current_price=current_price,
        available_strikes=strikes,
        volatility=0.65,  # Unity typical IV
        days_to_expiry=wheel_params.target_dte,
        portfolio_value=args.portfolio,
    )

    if strike_rec and strike_rec.confidence >= 0.75:
        contracts, size_confidence = wheel.calculate_position_size(
            strike_price=strike_rec.strike,
            portfolio_value=args.portfolio,
        )

        print(
            f"\n✅ RECOMMENDATION: Sell {contracts} {args.ticker} ${strike_rec.strike:.2f}P @ ${strike_rec.premium:.2f}"
        )
        print(f"   Delta: {strike_rec.delta:.3f}")
        print(f"   Probability ITM: {strike_rec.probability_itm:.1%}")
        print(f"   Expiry: {wheel_params.target_dte} days")
        print(f"   Max Risk: ${strike_rec.strike * 100 * contracts:,.0f}")
        print(f"   Confidence: {strike_rec.confidence:.0%}")
        print(f"   Reason: {strike_rec.reason}")
    else:
        if strike_rec:
            print(f"\n⚠️  No recommendation - confidence too low ({strike_rec.confidence:.0%})")
        else:
            print(f"\n⚠️  No suitable strikes found")

    print("\n" + "=" * 50)
    print("Note: Using demo data. Connect broker for live recommendations.\n")


if __name__ == "__main__":
    main()

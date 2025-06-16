#!/usr/bin/env python3
"""Unified entry point for wheel trading with all enhancements."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


async def main():
    """Run unified wheel trading system."""
    # Import after path setup
    # Parse command line arguments
    import argparse

    from src.unity_wheel.analytics.enhanced_integration import EnhancedWheelSystem

    parser = argparse.ArgumentParser(
        description="Unity Wheel Trading Bot - Unified System"
    )
    parser.add_argument(
        "-p",
        "--portfolio",
        type=float,
        default=100000,
        help="Portfolio value (default: 100000)",
    )
    parser.add_argument(
        "-s", "--symbol", type=str, default="U", help="Stock symbol (default: U)"
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Run system diagnostics"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Show performance metrics"
    )

    args = parser.parse_args()

    if args.diagnose:
        # Run diagnostics
        print("🔍 Running System Diagnostics...")
        from src.unity_wheel.monitoring.diagnostics import SelfDiagnostics

        diag = SelfDiagnostics()
        success = diag.run_all_checks()
        print(diag.report(format="text"))
        return 0 if success else 1

    # Initialize enhanced system
    print("🚀 Initializing Enhanced Wheel System...")
    print(f"   Portfolio: ${args.portfolio:,.0f}")
    print(f"   Symbol: {args.symbol}")

    system = EnhancedWheelSystem(portfolio_value=args.portfolio)

    try:
        # Generate recommendation
        print("\n📊 Generating Enhanced Recommendation...")
        recommendation = await system.generate_enhanced_recommendation(
            symbol=args.symbol
        )

        # Display results
        print(f"\n{'='*60}")
        print(f"🎯 RECOMMENDATION: {recommendation.action}")
        print(f"{'='*60}")

        if recommendation.action == "SELL_PUT":
            print(f"Symbol: {recommendation.symbol}")
            print(f"Strike: ${recommendation.strike:.2f}")
            print(f"Contracts: {recommendation.contracts}")
            print(f"Premium: ${recommendation.premium:.2f}")
            print(f"Expected Return: {recommendation.expected_return:.2%}")
            print(f"Confidence: {recommendation.confidence:.1%}")

            print("\n📈 Risk Metrics:")
            for key, value in recommendation.risk_metrics.items():
                if isinstance(value, float):
                    if "probability" in key or "allocation" in key:
                        print(f"   {key}: {value:.1%}")
                    elif key.startswith("var") or key.endswith("loss"):
                        print(f"   {key}: ${value:,.0f}")
                    else:
                        print(f"   {key}: {value:.4f}")

            print("\n🔧 Optimization Details:")
            for key, value in recommendation.optimization_details.items():
                print(f"   {key}: {value}")

        else:
            print(f"Reason: {recommendation.symbol}")

        if args.performance:
            print("\n📊 Performance Metrics:")
            metrics = system.get_performance_metrics()
            for key, value in metrics.items():
                print(f"   {key}: {value}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

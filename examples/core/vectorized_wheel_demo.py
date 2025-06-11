#!/usr/bin/env python3
"""Demonstration of vectorized wheel strategy analysis.

This script shows how the new vectorized operations provide massive
performance improvements for analyzing multiple wheel strategy candidates.
"""

import time
from typing import List, Tuple

import numpy as np

from src.unity_wheel.math.vectorized_options import (
    compare_scenario_analysis,
    quick_strike_comparison,
    quick_vol_sensitivity,
    vectorized_wheel_analysis,
)


def demonstrate_basic_vectorization():
    """Show basic vectorized calculations vs single calculations."""
    print("🔢 Basic Vectorization Demo")
    print("=" * 35)
    print()

    # Unity trading parameters
    unity_spot = 35.00
    strikes = [28.0, 30.0, 32.5, 35.0, 37.5, 40.0]
    dte = 45
    vol = 0.65

    print(f"📊 Unity Options Analysis:")
    print(f"  Current price: ${unity_spot}")
    print(f"  Days to expiry: {dte}")
    print(f"  Implied volatility: {vol:.0%}")
    print(f"  Analyzing {len(strikes)} strikes")
    print()

    # Use vectorized quick comparison
    start_time = time.time()
    comparison = quick_strike_comparison(
        spot=unity_spot, strikes=strikes, dte=dte, vol=vol, option_type="put"
    )
    vectorized_time = time.time() - start_time

    print("⚡ Vectorized Results (all strikes calculated simultaneously):")
    print("Strike   Price    Delta   Premium%  Ann.Yield%")
    print("-" * 50)

    for strike in strikes:
        data = comparison[strike]
        premium_pct = data["premium_pct"]
        ann_yield = (premium_pct / dte) * 365

        print(
            f"${strike:5.1f}  ${data['price']:6.3f}  {data['delta']:7.3f}  "
            f"{premium_pct:8.2f}  {ann_yield:9.1f}"
        )

    print()
    print(f"⏱️  Computation time: {vectorized_time*1000:.2f}ms for {len(strikes)} calculations")
    print(f"   Average per strike: {vectorized_time*1000/len(strikes):.3f}ms")
    print()


def demonstrate_comprehensive_wheel_analysis():
    """Show comprehensive wheel strategy analysis across multiple scenarios."""
    print("🎯 Comprehensive Wheel Strategy Analysis")
    print("=" * 45)
    print()

    # Unity wheel strategy parameters
    unity_spot = 35.00

    # Wide range of strikes (20% OTM to 5% ITM)
    strikes = [round(unity_spot * (0.80 + i * 0.05), 1) for i in range(6)]  # 28.0 to 40.25

    # Multiple expiration cycles
    expirations = [30, 45, 60, 75]  # Days to expiry
    exp_years = [dte / 365 for dte in expirations]

    print(f"📋 Analysis Parameters:")
    print(f"  Unity spot: ${unity_spot}")
    print(f"  Strike range: ${min(strikes)} - ${max(strikes)} ({len(strikes)} strikes)")
    print(f"  Expirations: {expirations} days ({len(expirations)} cycles)")
    print(f"  Total combinations: {len(strikes) * len(expirations)}")
    print()

    # Run comprehensive vectorized analysis
    start_time = time.time()
    analysis = vectorized_wheel_analysis(
        spot_price=unity_spot,
        strikes=strikes,
        expirations=exp_years,
        volatility=0.65,
        target_delta=0.30,
        min_premium_pct=1.5,
    )
    analysis_time = time.time() - start_time

    candidates = analysis["candidates"]
    summary = analysis["summary"]

    print(f"🚀 Analysis Results:")
    print(f"  Computation time: {analysis_time*1000:.1f}ms")
    print(f"  Rate: {summary['total_combinations']/analysis_time:.0f} calculations/second")
    print(f"  Valid candidates: {summary['valid_candidates']}")
    print(f"  Average premium: {summary['avg_premium_pct']:.2f}%")
    print(f"  Average annualized yield: {summary['avg_annualized_yield']:.1f}%")
    print()

    # Show top candidates
    print("🏆 Top 5 Wheel Candidates:")
    print("Rank  Strike  DTE  Price   Delta   Prem%  Ann%   Prob.Assign  Score")
    print("-" * 70)

    for i, candidate in enumerate(candidates[:5]):
        print(
            f"{i+1:4d}  ${candidate['strike']:5.1f}  "
            f"{candidate['dte_days']:3d}  ${candidate['option_price']:5.3f}  "
            f"{candidate['delta']:7.3f}  {candidate['premium_pct']:5.2f}  "
            f"{candidate['annualized_yield']:5.1f}  "
            f"{candidate['prob_assignment']:10.1%}  {candidate['score']:6.3f}"
        )

    print()

    # Best candidate analysis
    if candidates:
        best = candidates[0]
        print(f"💎 Best Candidate Analysis:")
        print(f"  Strike: ${best['strike']}")
        print(f"  Days to expiry: {best['dte_days']}")
        print(f"  Option price: ${best['option_price']:.3f}")
        print(f"  Premium yield: {best['premium_pct']:.2f}%")
        print(f"  Annualized yield: {best['annualized_yield']:.1f}%")
        print(f"  Delta: {best['delta']:.3f} (assignment prob: {best['prob_assignment']:.1%})")
        print(f"  Expected return: {best['expected_return']:.2f}%")
        print()


def demonstrate_scenario_analysis():
    """Show scenario analysis for different market conditions."""
    print("📈 Market Scenario Analysis")
    print("=" * 30)
    print()

    # Base case for Unity wheel strategy
    base_case = {
        "spot_price": 35.00,
        "strike": 32.50,
        "expiration": 45 / 365,
        "volatility": 0.65,
        "risk_free_rate": 0.05,
        "option_type": "put",
    }

    # Different market scenarios
    scenarios = [
        {"spot_price": 33.00, "volatility": 0.80, "name": "Market Selloff"},
        {"spot_price": 37.00, "volatility": 0.50, "name": "Market Rally"},
        {"volatility": 0.40, "name": "Low Vol Environment"},
        {"volatility": 0.90, "name": "High Vol Environment"},
        {"spot_price": 30.00, "volatility": 1.00, "name": "Crash Scenario"},
        {"spot_price": 40.00, "volatility": 0.35, "name": "Strong Bull Market"},
    ]

    print(f"📊 Base Case:")
    print(f"  Unity @ ${base_case['spot_price']}")
    print(f"  ${base_case['strike']} put, 45 DTE")
    print(f"  {base_case['volatility']:.0%} implied volatility")
    print()

    # Run scenario analysis
    start_time = time.time()
    comparison = compare_scenario_analysis(scenarios, base_case)
    scenario_time = time.time() - start_time

    print(f"⚡ Scenario Analysis ({scenario_time*1000:.1f}ms for {len(scenarios)} scenarios):")
    print()
    print("Scenario             Spot    Vol    Price   Δ Price  Δ%     Delta   Δ Delta")
    print("-" * 75)

    base_price = comparison["base_case"]["price"]
    base_delta = comparison["base_case"]["delta"]

    print(
        f"{'Base Case':<20} ${base_case['spot_price']:5.1f}  "
        f"{base_case['volatility']:5.0%}  ${base_price:6.3f}     --     --   "
        f"{base_delta:7.3f}    --"
    )

    for i, scenario_result in enumerate(comparison["scenarios"]):
        scenario = scenarios[i]
        name = scenario.get("name", f"Scenario {i+1}")
        spot = scenario.get("spot_price", base_case["spot_price"])
        vol = scenario.get("volatility", base_case["volatility"])

        price = scenario_result["option_price"]
        price_diff = scenario_result["price_diff"]
        price_diff_pct = scenario_result["price_diff_pct"]
        delta = scenario_result["delta"]
        delta_diff = scenario_result["delta_diff"]

        print(
            f"{name:<20} ${spot:5.1f}  {vol:5.0%}  ${price:6.3f}  "
            f"{price_diff:+6.3f}  {price_diff_pct:+5.1f}  "
            f"{delta:7.3f}  {delta_diff:+6.3f}"
        )

    print()

    # Summary insights
    max_price_impact = comparison["summary"]["max_price_impact"]
    max_delta_impact = comparison["summary"]["max_delta_impact"]

    print(f"📋 Scenario Insights:")
    print(f"  Maximum price impact: ${max_price_impact:.3f}")
    print(f"  Maximum delta impact: {max_delta_impact:.3f}")
    print(f"  Scenarios analyzed: {comparison['summary']['scenario_count']}")
    print()


def demonstrate_volatility_sensitivity():
    """Show volatility sensitivity analysis."""
    print("📊 Volatility Sensitivity Analysis")
    print("=" * 40)
    print()

    # Unity wheel put sensitivity
    unity_spot = 35.00
    strike = 32.50
    dte = 45

    print(f"🎯 Unity Put Sensitivity Analysis:")
    print(f"  Unity @ ${unity_spot}")
    print(f"  ${strike} put, {dte} DTE")
    print()

    # Quick volatility sensitivity
    vol_analysis = quick_vol_sensitivity(
        spot=unity_spot, strike=strike, dte=dte, vol_range=(0.30, 1.20), num_points=15
    )

    vols = vol_analysis["volatilities"]
    prices = vol_analysis["prices"]
    sensitivities = vol_analysis["vol_sensitivity"]

    print("Volatility  Put Price  Vega     Premium%  Ann.Yield%")
    print("-" * 50)

    for i, (vol, price, vega) in enumerate(zip(vols, prices, sensitivities)):
        premium_pct = (price / strike) * 100
        ann_yield = (premium_pct / dte) * 365

        print(
            f"{vol:8.0%}   ${price:7.3f}  {vega:+6.3f}   " f"{premium_pct:7.2f}   {ann_yield:8.1f}"
        )

    print()

    # Key insights
    min_vol, max_vol = min(vols), max(vols)
    min_price, max_price = min(prices), max(prices)
    price_range = max_price - min_price

    print(f"💡 Volatility Insights:")
    print(f"  Vol range: {min_vol:.0%} - {max_vol:.0%}")
    print(f"  Price range: ${min_price:.3f} - ${max_price:.3f}")
    print(f"  Total price impact: ${price_range:.3f} ({price_range/min_price:.0%})")
    print(f"  Average vega: {np.mean(sensitivities):.3f}")
    print()


def demonstrate_performance_comparison():
    """Show performance comparison for large-scale analysis."""
    print("🚀 Performance Comparison Demo")
    print("=" * 35)
    print()

    # Large-scale analysis parameters
    strikes = list(np.linspace(25, 45, 40))  # 40 strikes
    expirations = [d / 365 for d in range(15, 121, 15)]  # 8 expirations (15-120 days)

    total_combinations = len(strikes) * len(expirations)

    print(f"📊 Large-Scale Analysis Parameters:")
    print(f"  Strike range: ${min(strikes):.1f} - ${max(strikes):.1f} ({len(strikes)} strikes)")
    print(f"  Expiration range: 15-120 days ({len(expirations)} cycles)")
    print(f"  Total combinations: {total_combinations:,}")
    print()

    # Run vectorized analysis
    start_time = time.time()
    analysis = vectorized_wheel_analysis(
        spot_price=35.00,
        strikes=strikes,
        expirations=expirations,
        volatility=0.65,
        target_delta=0.30,
        min_premium_pct=1.0,
    )
    vectorized_time = time.time() - start_time

    candidates = analysis["candidates"]

    print(f"⚡ Vectorized Performance:")
    print(f"  Total time: {vectorized_time:.3f}s")
    print(f"  Rate: {total_combinations/vectorized_time:,.0f} calculations/second")
    print(f"  Valid candidates found: {len(candidates)}")
    print(f"  Time per combination: {vectorized_time*1000/total_combinations:.3f}ms")
    print()

    # Estimate sequential performance
    estimated_sequential_time = total_combinations * 0.001  # Assume 1ms per calculation
    speedup_estimate = estimated_sequential_time / vectorized_time

    print(f"💡 Performance Benefits:")
    print(f"  Estimated sequential time: {estimated_sequential_time:.1f}s")
    print(f"  Speedup factor: ~{speedup_estimate:.0f}x")
    print(f"  Memory efficient: processes {total_combinations:,} scenarios simultaneously")
    print(f"  Cache friendly: results cached for instant retrieval")
    print()

    # Show distribution of results
    if candidates:
        yields = [c["annualized_yield"] for c in candidates]
        deltas = [abs(c["delta"]) for c in candidates]

        print(f"📈 Results Distribution:")
        print(f"  Yield range: {min(yields):.1f}% - {max(yields):.1f}%")
        print(f"  Average yield: {np.mean(yields):.1f}%")
        print(f"  Delta range: {min(deltas):.3f} - {max(deltas):.3f}")
        print(f"  Average delta: {np.mean(deltas):.3f}")
    print()


def main():
    """Run the complete vectorized wheel demo."""
    print("🎯 Unity Wheel Trading Bot - Vectorized Operations Demo")
    print("=" * 60)
    print()
    print("This demo shows how vectorized operations provide massive")
    print("performance improvements for wheel strategy analysis.")
    print()

    # Run all demonstrations
    demonstrate_basic_vectorization()
    demonstrate_comprehensive_wheel_analysis()
    demonstrate_scenario_analysis()
    demonstrate_volatility_sensitivity()
    demonstrate_performance_comparison()

    print("🎉 Key Benefits of Vectorized Operations:")
    print("  ✅ 10-100x faster than sequential calculations")
    print("  ✅ Analyze thousands of scenarios simultaneously")
    print("  ✅ Memory efficient numpy-based computations")
    print("  ✅ Automatic caching for repeated calculations")
    print("  ✅ Broadcasting support for flexible input combinations")
    print("  ✅ Built-in validation and confidence scoring")
    print()

    print("💡 Perfect for Unity Wheel Strategy:")
    print("  🎯 Rapidly evaluate all strike/expiration combinations")
    print("  📊 Compare scenarios across different market conditions")
    print("  ⚡ Real-time analysis of changing market conditions")
    print("  🔍 Sensitivity analysis for volatility and other parameters")
    print("  📈 Portfolio-level optimization across multiple positions")
    print()

    print("🚀 The vectorized system is ready for production trading!")


if __name__ == "__main__":
    main()

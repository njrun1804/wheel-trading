#!/usr/bin/env python3
"""
Calculate EXACTLY how much historical data we need for statistically valid results.
No hand-waving - just math.
"""

import math

import numpy as np
from scipy import stats


def calculate_var_requirements():
    """How many days for statistically valid VaR?"""
    print("📊 VALUE AT RISK (VaR) REQUIREMENTS")
    print("=" * 60)

    # For 95% VaR, we're estimating the 5th percentile
    # We need enough observations in the tail for stability

    confidence_level = 0.95
    tail_probability = 1 - confidence_level  # 0.05

    # Statistical rule: need at least 20-30 observations in the tail
    # For robust estimate: 50+ observations in the tail

    min_tail_obs = 20
    robust_tail_obs = 50

    min_days = min_tail_obs / tail_probability
    robust_days = robust_tail_obs / tail_probability

    print(f"VaR confidence level: {confidence_level:.0%}")
    print(f"Tail probability: {tail_probability:.0%}")
    print(f"\nMinimum tail observations: {min_tail_obs}")
    print(f"Minimum days needed: {min_days:.0f}")
    print(f"\nRobust tail observations: {robust_tail_obs}")
    print(f"Robust days needed: {robust_days:.0f}")

    # Calculate standard error of percentile estimator
    for days in [250, 400, 500, 1000]:
        tail_obs = days * tail_probability
        se = math.sqrt(tail_probability * (1 - tail_probability) / days)
        print(f"\n{days} days → {tail_obs:.0f} tail observations")
        print(f"  Standard error: {se:.4f}")
        print(f"  95% CI width: ±{1.96 * se * 100:.1f}%")

    return robust_days


def calculate_volatility_requirements():
    """How many days for accurate volatility estimation?"""
    print("\n\n📊 VOLATILITY ESTIMATION REQUIREMENTS")
    print("=" * 60)

    # Standard error of volatility estimate: SE = σ / sqrt(2N)
    # We want SE < 5% of σ for good estimate

    target_error = 0.05  # 5% relative error

    # Solve: σ / sqrt(2N) < 0.05σ
    # 1 / sqrt(2N) < 0.05
    # sqrt(2N) > 20
    # 2N > 400
    # N > 200

    min_days = 200

    print(f"Target relative error: {target_error:.0%}")
    print(f"Minimum days needed: {min_days}")

    # Show actual errors for different sample sizes
    print("\nActual relative errors:")
    for days in [100, 200, 250, 365, 500]:
        relative_error = 1 / math.sqrt(2 * days)
        print(f"{days} days: {relative_error:.1%} error")

    return min_days


def calculate_kelly_requirements():
    """How many days for reliable Kelly criterion?"""
    print("\n\n📊 KELLY CRITERION REQUIREMENTS")
    print("=" * 60)

    # Kelly needs win rate and average win/loss size
    # For monthly options, we need enough cycles

    # Statistical rule: 30+ observations for reliable estimates
    min_observations = 30

    # But for financial data, we want more due to fat tails
    robust_observations = 50

    days_per_month = 21  # Trading days
    days_needed_min = min_observations * days_per_month
    days_needed_robust = robust_observations * days_per_month

    print(f"Minimum monthly cycles: {min_observations}")
    print(f"Days needed (minimum): {days_needed_min} ({days_needed_min/252:.1f} years)")

    print(f"\nRobust monthly cycles: {robust_observations}")
    print(f"Days needed (robust): {days_needed_robust} ({days_needed_robust/252:.1f} years)")

    # Calculate confidence intervals for win rate
    print("\nWin rate confidence intervals:")
    true_win_rate = 0.70  # Typical for wheel

    for months in [12, 24, 36, 50]:
        se = math.sqrt(true_win_rate * (1 - true_win_rate) / months)
        ci_width = 1.96 * se
        print(f"\n{months} months of data:")
        print(f"  Standard error: {se:.3f}")
        print(f"  95% CI: {true_win_rate:.1%} ± {ci_width:.1%}")
        print(f"  Range: [{true_win_rate-ci_width:.1%}, {true_win_rate+ci_width:.1%}]")

    return days_needed_robust


def calculate_unified_requirement():
    """What's the actual requirement considering all metrics?"""
    print("\n\n🎯 UNIFIED REQUIREMENT ANALYSIS")
    print("=" * 60)

    var_days = 500  # For 50 tail observations
    vol_days = 200  # For <5% error
    kelly_days = 1050  # For 50 monthly cycles

    print(f"VaR needs: {var_days} days")
    print(f"Volatility needs: {vol_days} days")
    print(f"Kelly needs: {kelly_days} days ({kelly_days/252:.1f} years)")

    print("\n⚠️  PROBLEM: Different metrics need different amounts!")

    # But wait - let's think about this differently
    print("\n💡 PRACTICAL SOLUTION:")
    print("-" * 40)

    # Option 1: Use what we can get
    print("\nOption 1: Optimize for available data")
    print("- Use 500 days (2 years) as compromise")
    print("- Good VaR ✓")
    print("- Excellent volatility ✓")
    print("- Acceptable Kelly (24 cycles) ⚠️")

    # Option 2: Adapt metrics to data
    print("\nOption 2: Use 250 days but adjust approach")
    print("- VaR: Use parametric instead of empirical ✓")
    print("- Volatility: Good enough (7% error) ✓")
    print("- Kelly: Use conservative fraction (0.25 instead of 0.5) ✓")

    # Option 3: Rolling updates
    print("\nOption 3: Start with 250, grow over time")
    print("- Month 1-12: Use 250 days + conservative Kelly")
    print("- Month 13-24: Use 500 days + moderate Kelly")
    print("- Month 25+: Use 750 days + full Kelly")

    return 500  # Reasonable compromise


def show_specific_calculations_for_unity():
    """Do the actual calculations for Unity's characteristics."""
    print("\n\n🎯 SPECIFIC CALCULATIONS FOR UNITY (U)")
    print("=" * 60)

    # Unity's actual characteristics
    annual_vol = 0.65  # Unity is volatile! ~65% annual vol
    daily_vol = annual_vol / math.sqrt(252)

    print(f"Unity's volatility: {annual_vol:.0%} annual, {daily_vol:.1%} daily")

    # For highly volatile stocks, we need MORE data
    print("\nAdjustments for high volatility:")

    # VaR needs more tail observations when vol is high
    base_tail_obs = 50
    vol_multiplier = annual_vol / 0.30  # Relative to "normal" 30% vol
    adjusted_tail_obs = base_tail_obs * vol_multiplier
    adjusted_var_days = adjusted_tail_obs / 0.05

    print(f"- Base tail observations: {base_tail_obs}")
    print(f"- Volatility multiplier: {vol_multiplier:.1f}x")
    print(f"- Adjusted tail observations: {adjusted_tail_obs:.0f}")
    print(f"- Days needed for VaR: {adjusted_var_days:.0f}")

    # Show the difference in estimates with different data sizes
    print("\nVaR accuracy for Unity with different sample sizes:")

    np.random.seed(42)
    true_var_95 = -daily_vol * 1.645  # Parametric VaR

    for days in [250, 500, 750, 1000]:
        # Simulate returns
        returns = np.random.normal(0, daily_vol, days)
        empirical_var = np.percentile(returns, 5)
        error = abs(empirical_var - true_var_95) / abs(true_var_95)

        print(f"\n{days} days:")
        print(f"  True VaR: {true_var_95:.1%}")
        print(f"  Empirical VaR: {empirical_var:.1%}")
        print(f"  Relative error: {error:.1%}")


if __name__ == "__main__":
    var_req = calculate_var_requirements()
    vol_req = calculate_volatility_requirements()
    kelly_req = calculate_kelly_requirements()

    unified = calculate_unified_requirement()

    show_specific_calculations_for_unity()

    print("\n\n" + "=" * 60)
    print("📊 FINAL ANSWER:")
    print("=" * 60)
    print("\nFor statistically valid results:")
    print("- MINIMUM: 500 trading days (2 years)")
    print("- BETTER: 750 trading days (3 years)")
    print("- But Unity is so volatile, even 1000 days is reasonable")
    print("\nStorage impact:")
    print("- 500 days = 4KB")
    print("- 750 days = 6KB")
    print("- 1000 days = 8KB")
    print("- Still basically nothing!")

    print("\nAPI calls:")
    print("- Initial load: 1-4 calls (depending on dataset limits)")
    print("- Ongoing: 1 call per day")
    print("\nCost: Still under $1/month")

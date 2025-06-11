#!/usr/bin/env python3
"""
Validation A: Statistical Pitfalls and Data Quality Checks
Guards against look-ahead bias, survivorship bias, and overfitting.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def check_lookahead_bias():
    """Verify no future data leaks into historical decisions."""

    print("=== LOOK-AHEAD BIAS CHECK ===\n")

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Check option data timestamps
    suspicious = conn.execute(
        """
        WITH option_timing AS (
            SELECT
                om.symbol,
                om.expiration,
                om.strike,
                md.date as price_date,
                DATEDIFF('day', md.date, om.expiration) as dte,
                CASE
                    WHEN md.date > om.expiration THEN 1
                    ELSE 0
                END as future_leak
            FROM market_data md
            JOIN options_metadata om ON md.symbol = om.symbol
            WHERE om.underlying = 'U'
            AND om.option_type = 'P'
            AND md.data_type = 'option'
            LIMIT 10000
        )
        SELECT
            COUNT(*) as total_records,
            SUM(future_leak) as leaked_records,
            MIN(dte) as min_dte,
            MAX(dte) as max_dte
        FROM option_timing
    """
    ).fetchone()

    total, leaked, min_dte, max_dte = suspicious

    print(f"1. Option Chain Timing Analysis:")
    print(f"   Total option records checked: {total:,}")
    print(f"   Records with future data: {leaked:,}")
    print(f"   Leak rate: {leaked/total*100:.2f}%")
    print(f"   DTE range: {min_dte} to {max_dte} days")

    if leaked > 0:
        print("   ⚠️  WARNING: Future data detected in option chains!")
    else:
        print("   ✅ No look-ahead bias detected")

    # Check for impossible strike prices
    print("\n2. Strike Price Validation:")

    impossible_strikes = conn.execute(
        """
        SELECT
            COUNT(*) as count,
            om.strike,
            s.close as spot_price,
            ABS(om.strike - s.close) / s.close as distance_pct
        FROM options_metadata om
        JOIN market_data s ON om.underlying = s.symbol
            AND s.data_type = 'stock'
            AND s.date = (
                SELECT MAX(date) FROM market_data
                WHERE symbol = om.underlying
                AND data_type = 'stock'
                AND date <= om.expiration - INTERVAL '7' DAY
            )
        WHERE om.underlying = 'U'
        AND om.option_type = 'P'
        AND ABS(om.strike - s.close) / s.close > 0.50  -- More than 50% away
        GROUP BY om.strike, s.close
        ORDER BY distance_pct DESC
        LIMIT 5
    """
    ).fetchall()

    if impossible_strikes:
        print("   Found strikes >50% from spot:")
        for count, strike, spot, dist in impossible_strikes:
            print(
                f"     Strike ${strike:.2f}, Spot ${spot:.2f}, Distance {dist:.1%} ({count} records)"
            )
        print("   ⚠️  Suspicious strikes detected - possible data quality issue")
    else:
        print("   ✅ All strikes within reasonable range")

    conn.close()


def check_survivorship_bias():
    """Verify we're not missing delisted/expired data."""

    print("\n\n=== SURVIVORSHIP BIAS CHECK ===\n")

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Check data continuity
    gaps = conn.execute(
        """
        WITH daily_data AS (
            SELECT
                date,
                LAG(date) OVER (ORDER BY date) as prev_date,
                DATEDIFF('day', LAG(date) OVER (ORDER BY date), date) as gap_days
            FROM backtest_features
            WHERE symbol = 'U'
            ORDER BY date
        )
        SELECT
            prev_date,
            date,
            gap_days
        FROM daily_data
        WHERE gap_days > 5  -- More than a week
        ORDER BY gap_days DESC
    """
    ).fetchall()

    print("1. Data Continuity Check:")
    if gaps:
        print(f"   Found {len(gaps)} gaps in price history:")
        for prev, curr, gap in gaps[:5]:
            print(f"     {prev} to {curr}: {gap} days missing")
        print("   ⚠️  Data gaps detected - possible survivorship bias")
    else:
        print("   ✅ No significant gaps in price history")

    # Check option chain completeness
    print("\n2. Option Chain Completeness:")

    chain_coverage = conn.execute(
        """
        WITH strike_analysis AS (
            SELECT
                DATE_TRUNC('month', md.date) as month,
                COUNT(DISTINCT om.strike) as unique_strikes,
                MIN(om.strike) as min_strike,
                MAX(om.strike) as max_strike,
                AVG(s.close) as avg_spot
            FROM market_data md
            JOIN options_metadata om ON md.symbol = om.symbol
            JOIN market_data s ON md.date = s.date
                AND s.symbol = 'U'
                AND s.data_type = 'stock'
            WHERE om.underlying = 'U'
            AND om.option_type = 'P'
            AND md.data_type = 'option'
            GROUP BY DATE_TRUNC('month', md.date)
            ORDER BY month DESC
            LIMIT 12
        )
        SELECT * FROM strike_analysis
    """
    ).fetchall()

    if chain_coverage:
        print("   Monthly Strike Analysis (Last 12 months):")
        print("   Month    | Strikes | Min    | Max    | Avg Spot | Coverage")
        print("   ---------|---------|--------|--------|----------|----------")

        for month, strikes, min_s, max_s, avg_spot in chain_coverage[:6]:
            if avg_spot > 0:
                coverage = (max_s - min_s) / avg_spot
                print(
                    f"   {month.strftime('%Y-%m')} | {strikes:>7} | ${min_s:>5.0f} | ${max_s:>5.0f} | ${avg_spot:>7.1f} | {coverage:>7.1%}"
                )

        # Overall assessment
        avg_strikes = sum(row[1] for row in chain_coverage) / len(chain_coverage)
        if avg_strikes > 20:
            print("   ✅ Good option chain coverage")
        else:
            print("   ⚠️  Limited option chain coverage")

    conn.close()


def check_regime_overfitting():
    """Test if regime detection is robust or overfitted."""

    print("\n\n=== REGIME OVERFITTING CHECK ===\n")

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Load returns
    returns_data = conn.execute(
        """
        SELECT returns
        FROM backtest_features
        WHERE symbol = 'U'
        AND returns IS NOT NULL
        ORDER BY date
    """
    ).fetchdf()

    returns = returns_data["returns"].values

    print("1. Cross-Validation of Regime Count:")

    # Test different numbers of regimes
    n_regimes_options = [2, 3, 4, 5]
    cv_scores = []

    for n_regimes in n_regimes_options:
        # Create features
        returns_series = pd.Series(returns)
        features = pd.DataFrame(
            {
                "vol_5d": returns_series.rolling(5).std() * np.sqrt(252),
                "vol_20d": returns_series.rolling(20).std() * np.sqrt(252),
                "vol_60d": returns_series.rolling(60).std() * np.sqrt(252),
                "abs_return": np.abs(returns_series),
            }
        ).dropna()

        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=False)
        fold_scores = []

        for train_idx, test_idx in kf.split(features):
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]

            gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=42)
            gmm.fit(X_train)
            score = gmm.score(X_test)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        cv_scores.append((n_regimes, avg_score))
        print(f"   {n_regimes} regimes: Log-likelihood = {avg_score:.2f}")

    # Find elbow
    best_n = max(cv_scores, key=lambda x: x[1])[0]
    print(f"\n   Optimal regime count: {best_n}")

    # Test regime stability
    print("\n2. Regime Stability Test (Expanding Window):")

    window_starts = [0, len(returns) // 3, 2 * len(returns) // 3]
    regime_boundaries = []

    for start in window_starts:
        subset_returns = returns[start:]

        # Fit model
        returns_series = pd.Series(subset_returns)
        features = pd.DataFrame(
            {
                "vol_20d": returns_series.rolling(20).std() * np.sqrt(252),
                "abs_return": np.abs(returns_series),
            }
        ).dropna()

        if len(features) > 100:
            gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
            labels = gmm.fit_predict(features)

            # Find volatility of each regime
            regime_vols = []
            for regime in range(3):
                mask = labels == regime
                if np.sum(mask) > 10:
                    regime_vol = np.std(subset_returns[features.index[mask]]) * np.sqrt(252)
                    regime_vols.append(regime_vol)

            regime_boundaries.append(sorted(regime_vols))
            print(
                f"   Window {start//len(returns)*100:.0f}%-100%: Regime vols = {[f'{v:.0%}' for v in sorted(regime_vols)]}"
            )

    # Check consistency
    if len(regime_boundaries) >= 2:
        max_drift = max(
            abs(regime_boundaries[i][j] - regime_boundaries[0][j])
            for i in range(1, len(regime_boundaries))
            for j in range(min(len(regime_boundaries[0]), len(regime_boundaries[i])))
        )

        print(f"\n   Maximum regime boundary drift: {max_drift:.1%}")
        if max_drift > 0.20:
            print("   ⚠️  Regime boundaries unstable - possible overfitting")
        else:
            print("   ✅ Regime boundaries stable across time")

    conn.close()


def check_multiple_comparisons():
    """Apply false discovery rate control to parameter optimization."""

    print("\n\n=== MULTIPLE COMPARISONS CHECK ===\n")

    # Load previous optimization results
    results_file = Path("parameter_sensitivity_results.csv")

    if not results_file.exists():
        print("   No optimization results found. Run advanced_parameter_analysis.py first.")
        return

    results = pd.read_csv(results_file)

    # Group by regime and count tests
    print("1. Number of Parameter Combinations Tested:")
    for regime in results["regime"].unique():
        regime_data = results[results["regime"] == regime]
        n_tests = len(regime_data)
        print(f"   {regime}: {n_tests} combinations")

    # Simulate p-values based on Sharpe ratios
    # Null hypothesis: Sharpe = 0
    results["p_value"] = 1 - stats.norm.cdf(results["sharpe_ratio"] * np.sqrt(252))

    # Apply Benjamini-Hochberg correction
    from statsmodels.stats.multitest import multipletests

    print("\n2. False Discovery Rate Control (α = 0.05):")

    for regime in results["regime"].unique():
        regime_data = results[results["regime"] == regime]

        # Get p-values
        p_values = regime_data["p_value"].values

        # Apply FDR control
        rejected, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

        # Find significant results
        significant = regime_data[rejected].nlargest(5, "sharpe_ratio")

        print(f"\n   {regime} - Significant parameter sets after FDR control:")
        if len(significant) > 0:
            for _, row in significant.iterrows():
                print(
                    f"     Delta={row['delta']:.2f}, DTE={row['dte']}, "
                    f"Size={row['position_size']:.0%}, Sharpe={row['sharpe_ratio']:.2f}"
                )
        else:
            print("     None survive FDR correction!")

    print("\n   Note: Many 'optimal' parameters may be false positives without FDR control")


def main():
    """Run all statistical validation checks."""

    print("VALIDATION SUITE A: STATISTICAL PITFALLS")
    print("=" * 60)

    # 1. Look-ahead bias
    check_lookahead_bias()

    # 2. Survivorship bias
    check_survivorship_bias()

    # 3. Regime overfitting
    check_regime_overfitting()

    # 4. Multiple comparisons
    check_multiple_comparisons()

    print("\n" + "=" * 60)
    print("✅ Statistical validation complete")
    print("\nNext: Run validation_b_additional_tests.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test regime-aware risk calculations for Unity.
Handles volatility regime changes to avoid skewing risk metrics.
"""

import os
import warnings

import duckdb
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

DB_PATH = os.path.expanduser("~/.wheel_trading/cache/wheel_cache.duckdb")


class RegimeAwareRiskAnalyzer:
    """Risk analyzer that detects and adapts to volatility regimes."""

    def __init__(self, lookback_days=250):
        self.lookback_days = lookback_days
        self.regime_model = None

    def detect_volatility_regimes(self, returns):
        """
        Detect volatility regimes using Gaussian Mixture Model.
        Returns regime labels and probabilities.
        """
        # Calculate rolling volatility features
        vol_5d = pd.Series(returns).rolling(5).std() * np.sqrt(252)
        vol_20d = pd.Series(returns).rolling(20).std() * np.sqrt(252)
        vol_60d = pd.Series(returns).rolling(60).std() * np.sqrt(252)

        # Create feature matrix (skip NaN values)
        features = pd.DataFrame(
            {
                "vol_5d": vol_5d,
                "vol_20d": vol_20d,
                "vol_60d": vol_60d,
                "abs_return": np.abs(returns),
            }
        ).dropna()

        # Fit GMM with 2-3 regimes
        best_bic = np.inf
        best_n = 2

        for n_components in [2, 3]:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(features)
            bic = gmm.bic(features)

            if bic < best_bic:
                best_bic = bic
                best_n = n_components
                self.regime_model = gmm

        # Get regime labels
        regimes = self.regime_model.predict(features)

        # Sort regimes by average volatility (0 = low vol, 1 = high vol, etc.)
        regime_vols = []
        for i in range(best_n):
            regime_mask = regimes == i
            avg_vol = features.loc[features.index[regime_mask], "vol_20d"].mean()
            regime_vols.append((i, avg_vol))

        regime_vols.sort(key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(regime_vols)}

        # Remap regimes
        regimes_sorted = np.array([regime_mapping[r] for r in regimes])

        return features.index, regimes_sorted, self.regime_model.predict_proba(features)

    def calculate_regime_aware_var(self, returns, confidence=0.95):
        """
        Calculate VaR considering volatility regimes.
        """
        # Detect regimes
        regime_idx, regimes, regime_probs = self.detect_volatility_regimes(returns)

        # Get current regime (last observation)
        current_regime = regimes[-1]
        current_regime_prob = regime_probs[-1, current_regime]

        # Calculate VaR for each regime
        regime_vars = {}
        regime_stats = {}

        for regime in range(regimes.max() + 1):
            # Get returns for this regime
            regime_mask = regimes == regime
            regime_returns = returns[regime_idx[regime_mask]]

            if len(regime_returns) > 20:  # Need minimum data
                # Calculate regime-specific VaR
                regime_var = np.percentile(regime_returns, (1 - confidence) * 100)
                regime_mean = np.mean(regime_returns)
                regime_vol = np.std(regime_returns) * np.sqrt(252)

                regime_vars[regime] = regime_var
                regime_stats[regime] = {
                    "mean": regime_mean,
                    "volatility": regime_vol,
                    "count": len(regime_returns),
                    "recent_weight": np.sum(regime_mask[-60:]) / min(60, len(regime_mask)),
                }

        return {
            "current_regime": current_regime,
            "regime_probability": current_regime_prob,
            "regime_vars": regime_vars,
            "regime_stats": regime_stats,
        }

    def calculate_ewma_var(self, returns, lambda_param=0.94, confidence=0.95):
        """
        Calculate Exponentially Weighted Moving Average VaR.
        More weight on recent observations.
        """
        n = len(returns)
        weights = np.array([(1 - lambda_param) * lambda_param**i for i in range(n - 1, -1, -1)])
        weights = weights / weights.sum()

        # Sort returns with weights
        sorted_idx = np.argsort(returns)
        sorted_returns = returns[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Find VaR using cumulative weights
        cumsum_weights = np.cumsum(sorted_weights)
        var_idx = np.searchsorted(cumsum_weights, 1 - confidence)

        if var_idx < len(sorted_returns):
            var_ewma = sorted_returns[var_idx]
        else:
            var_ewma = sorted_returns[-1]

        return var_ewma

    def calculate_conditional_var(self, returns, vol_threshold=None):
        """
        Calculate VaR conditional on high volatility periods.
        """
        # Calculate rolling volatility
        vol_20d = pd.Series(returns).rolling(20).std() * np.sqrt(252)

        if vol_threshold is None:
            # Use 75th percentile as high vol threshold
            vol_threshold = np.nanpercentile(vol_20d, 75)

        # Get returns during high volatility periods
        high_vol_mask = vol_20d > vol_threshold
        high_vol_returns = returns[high_vol_mask[~np.isnan(high_vol_mask)]]

        if len(high_vol_returns) > 20:
            cvar_95 = np.percentile(high_vol_returns, 5)
            cvar_99 = np.percentile(high_vol_returns, 1)
        else:
            # Fallback to all data
            cvar_95 = np.percentile(returns, 5)
            cvar_99 = np.percentile(returns, 1)

        return {
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "vol_threshold": vol_threshold,
            "high_vol_days": np.sum(high_vol_mask[~np.isnan(high_vol_mask)]),
            "total_days": len(returns),
        }


def main():
    """Test regime-aware risk calculations."""

    print("🔬 Regime-Aware Risk Analysis for Unity")
    print("=" * 60)

    # Connect to database
    conn = duckdb.connect(DB_PATH)

    # Get Unity returns
    returns_data = conn.execute(
        """
        SELECT date, returns
        FROM price_history
        WHERE symbol = 'U'
        AND returns IS NOT NULL
        ORDER BY date
    """
    ).fetchall()

    dates = [r[0] for r in returns_data]
    returns = np.array([float(r[1]) for r in returns_data])

    print("\n📊 Data Summary:")
    print(f"   Total days: {len(returns)}")
    print(f"   Date range: {dates[0]} to {dates[-1]}")

    # Initialize analyzer
    analyzer = RegimeAwareRiskAnalyzer()

    # 1. Regime Detection
    print("\n🎯 Volatility Regime Analysis:")
    regime_results = analyzer.calculate_regime_aware_var(returns)

    current_regime = regime_results["current_regime"]
    print(
        f"\n   Current Regime: {current_regime} (probability: {regime_results['regime_probability']:.1%})"
    )

    print("\n   Regime Statistics:")
    for regime, stats in regime_results["regime_stats"].items():
        regime_name = (
            ["Low Vol", "Medium Vol", "High Vol"][regime] if regime < 3 else f"Regime {regime}"
        )
        print(f"\n   {regime_name}:")
        print(f"      Days: {stats['count']} ({stats['count']/len(returns)*100:.1f}% of total)")
        print(f"      Volatility: {stats['volatility']*100:.1f}%")
        print(f"      Daily return: {stats['mean']*100:.2f}%")
        print(f"      Recent weight: {stats['recent_weight']*100:.1f}% of last 60 days")
        print(f"      VaR (95%): {regime_results['regime_vars'].get(regime, 0)*100:.1f}%")

    # 2. EWMA VaR (recent data weighted)
    print("\n📈 Exponentially Weighted VaR:")
    ewma_var_95 = analyzer.calculate_ewma_var(returns, lambda_param=0.94, confidence=0.95)
    ewma_var_99 = analyzer.calculate_ewma_var(returns, lambda_param=0.94, confidence=0.99)

    print(f"   EWMA VaR (95%): {ewma_var_95*100:.1f}%")
    print(f"   EWMA VaR (99%): {ewma_var_99*100:.1f}%")

    # 3. Conditional VaR (high volatility periods)
    print("\n⚠️  Conditional VaR (High Volatility Periods):")
    cvar_results = analyzer.calculate_conditional_var(returns)

    print(f"   Volatility threshold: {cvar_results['vol_threshold']*100:.1f}%")
    print(
        f"   High vol days: {cvar_results['high_vol_days']} ({cvar_results['high_vol_days']/cvar_results['total_days']*100:.1f}%)"
    )
    print(f"   Conditional VaR (95%): {cvar_results['cvar_95']*100:.1f}%")
    print(f"   Conditional VaR (99%): {cvar_results['cvar_99']*100:.1f}%")

    # 4. Recommendation based on current regime
    print("\n💡 Risk Management Recommendation:")

    if current_regime == 0:  # Low volatility
        print("   ✅ Low volatility regime - Standard position sizing")
        print("   - Can use standard VaR: {:.1f}%".format(regime_results["regime_vars"][0] * 100))
        print("   - Kelly fraction: 0.50 (half-Kelly)")
    elif current_regime == 1:  # Medium volatility
        print("   ⚠️  Medium volatility regime - Reduce position size")
        print(
            "   - Use regime-specific VaR: {:.1f}%".format(regime_results["regime_vars"][1] * 100)
        )
        print("   - Kelly fraction: 0.33 (third-Kelly)")
    else:  # High volatility
        print("   🚨 High volatility regime - Defensive positioning")
        print("   - Use conditional VaR: {:.1f}%".format(cvar_results["cvar_95"] * 100))
        print("   - Kelly fraction: 0.25 (quarter-Kelly)")
        print("   - Consider reducing DTE targets")

    # 5. Compare with naive approach
    print("\n📊 Comparison with Naive Approach:")
    naive_var_95 = np.percentile(returns, 5)

    print(f"   Naive VaR (all data equally): {naive_var_95*100:.1f}%")
    print(f"   Current regime VaR: {regime_results['regime_vars'][current_regime]*100:.1f}%")
    print(f"   EWMA VaR (recent weighted): {ewma_var_95*100:.1f}%")

    diff = abs(ewma_var_95 - naive_var_95) / abs(naive_var_95) * 100
    print(
        f"\n   Difference: {diff:.1f}% - {'Significant!' if diff > 20 else 'Moderate' if diff > 10 else 'Minor'}"
    )

    conn.close()


if __name__ == "__main__":
    main()

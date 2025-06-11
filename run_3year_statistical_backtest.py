#!/usr/bin/env python3
"""
Statistically rigorous 3-year backtest with regime detection.
Implements proper statistical methods for regime identification and analysis.
"""

import asyncio
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from src.unity_wheel.backtesting import WheelBacktester
from src.unity_wheel.risk.regime_detector import RegimeDetector
from src.unity_wheel.storage import Storage
from src.unity_wheel.strategy import WheelParameters

warnings.filterwarnings("ignore", category=RuntimeWarning)


def calculate_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling Sharpe ratio with proper annualization."""
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    return rolling_mean / rolling_std


def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio vs benchmark."""
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    return (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0


def markov_regime_detection(returns: np.ndarray, n_regimes: int = 3) -> dict:
    """
    Implement Markov regime switching model for volatility regimes.
    Returns regime assignments and transition matrix.
    """
    # Create feature matrix with multiple volatility measures
    returns_series = pd.Series(returns)

    features = pd.DataFrame(
        {
            "vol_5d": returns_series.rolling(5).std() * np.sqrt(252),
            "vol_20d": returns_series.rolling(20).std() * np.sqrt(252),
            "vol_60d": returns_series.rolling(60).std() * np.sqrt(252),
            "abs_return": np.abs(returns_series),
            "squared_return": returns_series**2,
            "garch_proxy": returns_series.rolling(5).apply(lambda x: np.sum(x**2)),
        }
    ).dropna()

    # Fit Gaussian Mixture Model with full covariance
    gmm = GaussianMixture(
        n_components=n_regimes, covariance_type="full", max_iter=1000, n_init=10, random_state=42
    )

    # Standardize features
    features_scaled = (features - features.mean()) / features.std()

    # Fit model and get regime assignments
    regime_labels = gmm.fit_predict(features_scaled)
    regime_probs = gmm.predict_proba(features_scaled)

    # Calculate transition matrix
    transitions = np.zeros((n_regimes, n_regimes))
    for i in range(len(regime_labels) - 1):
        transitions[regime_labels[i], regime_labels[i + 1]] += 1

    # Normalize rows to get probabilities
    for i in range(n_regimes):
        if transitions[i].sum() > 0:
            transitions[i] /= transitions[i].sum()

    # Calculate regime statistics
    regime_stats = {}
    for regime in range(n_regimes):
        mask = regime_labels == regime
        regime_returns = returns[features.index[mask]]

        regime_stats[regime] = {
            "mean_return": np.mean(regime_returns) * 252,
            "volatility": np.std(regime_returns) * np.sqrt(252),
            "skewness": stats.skew(regime_returns),
            "kurtosis": stats.kurtosis(regime_returns),
            "var_95": np.percentile(regime_returns, 5),
            "cvar_95": np.mean(regime_returns[regime_returns <= np.percentile(regime_returns, 5)]),
            "count": np.sum(mask),
            "percentage": np.sum(mask) / len(regime_labels),
        }

    return {
        "labels": regime_labels,
        "probabilities": regime_probs,
        "transition_matrix": transitions,
        "statistics": regime_stats,
        "features": features,
        "model": gmm,
    }


async def run_statistical_3year_analysis():
    """Run statistically rigorous 3-year analysis."""

    print("=== STATISTICAL 3-YEAR WHEEL STRATEGY ANALYSIS ===\n")

    # 1. Load complete dataset
    print("1. DATA LOADING AND VALIDATION")
    print("-" * 60)

    db_path = Path("data/unified_wheel_trading.duckdb")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Load complete dataset with proper handling
    full_data = conn.execute(
        """
        SELECT
            bf.date,
            bf.stock_price as close,
            bf.returns,
            bf.volatility_20d,
            bf.volatility_250d,
            bf.volume,
            bf.var_95,
            bf.risk_free_rate,
            COALESCE(md.high, bf.stock_price) as high,
            COALESCE(md.low, bf.stock_price) as low
        FROM backtest_features bf
        LEFT JOIN market_data md
            ON bf.date = md.date
            AND bf.symbol = md.symbol
            AND md.data_type = 'stock'
        WHERE bf.symbol = 'U'
        AND bf.returns IS NOT NULL
        ORDER BY bf.date
    """
    ).fetchdf()

    # Convert to proper types
    full_data["date"] = pd.to_datetime(full_data["date"])
    full_data.set_index("date", inplace=True)

    # Handle any None values in volatility columns
    for col in ["volatility_20d", "volatility_250d"]:
        full_data[col] = pd.to_numeric(full_data[col], errors="coerce")

    print(f"  Loaded {len(full_data)} days of data")
    print(f"  Date range: {full_data.index.min()} to {full_data.index.max()}")
    print(f"  Years: {(full_data.index.max() - full_data.index.min()).days / 365.25:.1f}")

    # Data quality check
    print(f"\n  Data Quality:")
    print(f"  - Returns completeness: {(~full_data['returns'].isna()).mean():.1%}")
    print(f"  - Vol 20d completeness: {(~full_data['volatility_20d'].isna()).mean():.1%}")
    print(f"  - Vol 250d completeness: {(~full_data['volatility_250d'].isna()).mean():.1%}")

    # 2. Statistical properties
    print("\n2. STATISTICAL PROPERTIES OF RETURNS")
    print("-" * 60)

    returns = full_data["returns"].dropna().values

    # Calculate comprehensive statistics
    stats_dict = {
        "Mean (annualized)": np.mean(returns) * 252,
        "Volatility (annualized)": np.std(returns) * np.sqrt(252),
        "Skewness": stats.skew(returns),
        "Kurtosis": stats.kurtosis(returns),
        "Jarque-Bera statistic": stats.jarque_bera(returns)[0],
        "Jarque-Bera p-value": stats.jarque_bera(returns)[1],
        "VaR 95%": np.percentile(returns, 5),
        "CVaR 95%": np.mean(returns[returns <= np.percentile(returns, 5)]),
        "Max daily gain": np.max(returns),
        "Max daily loss": np.min(returns),
        "Positive days": (returns > 0).mean(),
    }

    print("\n  Return Distribution Analysis:")
    for key, value in stats_dict.items():
        if isinstance(value, float):
            if "p-value" in key:
                print(f"  {key:<25}: {value:.2e}")
            elif "%" in key or "Positive" in key:
                print(f"  {key:<25}: {value:.1%}")
            else:
                print(f"  {key:<25}: {value:.4f}")
        else:
            print(f"  {key:<25}: {value}")

    # Test for normality
    if stats_dict["Jarque-Bera p-value"] < 0.01:
        print("\n  ⚠️  Returns are NOT normally distributed (p < 0.01)")
        print("     Heavy tails detected - increased gap risk")

    # 3. Markov Regime Detection
    print("\n3. MARKOV REGIME SWITCHING ANALYSIS")
    print("-" * 60)

    regime_results = markov_regime_detection(returns, n_regimes=3)

    # Sort regimes by volatility
    regime_order = sorted(
        regime_results["statistics"].keys(),
        key=lambda x: regime_results["statistics"][x]["volatility"],
    )

    print("\n  Detected Volatility Regimes:")
    print("  Regime | Vol    | Return | Skew   | Kurt  | VaR 95% | Days | %")
    print("  -------|--------|--------|--------|-------|---------|------|----")

    regime_names = {regime_order[0]: "Low", regime_order[1]: "Med", regime_order[2]: "High"}

    for i, regime in enumerate(regime_order):
        s = regime_results["statistics"][regime]
        print(
            f"  {regime_names[regime]:<6} | {s['volatility']:>5.0%} | {s['mean_return']:>6.0%} | "
            f"{s['skewness']:>6.2f} | {s['kurtosis']:>5.1f} | {s['var_95']:>7.2%} | "
            f"{s['count']:>4} | {s['percentage']:>3.0%}"
        )

    # Transition matrix
    print("\n  Regime Transition Matrix (daily probabilities):")
    print("  From\\To |  Low  |  Med  | High")
    print("  ---------|-------|-------|------")

    trans = regime_results["transition_matrix"]
    for i, from_regime in enumerate(regime_order):
        row = f"  {regime_names[from_regime]:<8} |"
        for j, to_regime in enumerate(regime_order):
            row += f" {trans[from_regime, to_regime]:>5.1%} |"
        print(row)

    # Expected regime duration
    print("\n  Expected Regime Duration:")
    for i, regime in enumerate(regime_order):
        if trans[regime, regime] < 1.0:
            duration = 1 / (1 - trans[regime, regime])
            print(f"  {regime_names[regime]}: {duration:.1f} days")

    # Save results for next steps
    return full_data, regime_results, regime_names, regime_order


if __name__ == "__main__":
    asyncio.run(run_statistical_3year_analysis())

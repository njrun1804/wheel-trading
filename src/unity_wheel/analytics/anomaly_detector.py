"""
from __future__ import annotations

Anomaly detection system for market conditions.
Identifies when current market deviates from historical norms.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from src.config.loader import get_config

from ..utils import get_logger, timed_operation

logger = get_logger(__name__)


class AnomalyType(NamedTuple):
    """Type of market anomaly detected."""

    name: str
    severity: float  # 0-1, higher is more severe
    description: str
    recommended_action: str


@dataclass
class MarketAnomaly:
    """Detected market anomaly with details."""

    anomaly_type: AnomalyType
    timestamp: datetime
    metrics: dict[str, float]
    z_score: float
    percentile: float
    historical_frequency: float  # How often this occurs
    confidence: float


class AnomalyDetector:
    """Detects unusual market conditions that require strategy adjustment."""

    def __init__(
        self,
        symbol: str = None,
        lookback_days: int = 500,
        sensitivity: float = 2.5,  # Z-score threshold
    ):
        if symbol is None:
            config = get_config()
            symbol = config.unity.ticker
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.sensitivity = sensitivity

        # Anomaly detectors
        self.isolation_forest = IsolationForest(
            contamination=0.05, random_state=42  # Expect 5% anomalies
        )
        self.is_fitted = False

        # Define anomaly types
        self.ANOMALY_TYPES = {
            "extreme_volatility": AnomalyType(
                "extreme_volatility",
                0.9,
                "Volatility beyond 3 standard deviations",
                "Reduce position size significantly",
            ),
            "volume_spike": AnomalyType(
                "volume_spike",
                0.7,
                "Volume 5x above average",
                "Check for news, consider avoiding",
            ),
            "price_gap": AnomalyType(
                "price_gap",
                0.8,
                "Price gap > 10% from previous close",
                "Wait for stability before entering",
            ),
            "correlation_break": AnomalyType(
                "correlation_break",
                0.6,
                "Unity decorrelated from tech sector",
                "Review sector-specific risks",
            ),
            "option_flow_unusual": AnomalyType(
                "option_flow_unusual",
                0.5,
                "Unusual options activity detected",
                "Monitor for potential catalyst",
            ),
            "iv_price_divergence": AnomalyType(
                "iv_price_divergence",
                0.7,
                "IV not reflecting price movement",
                "Potential mispricing opportunity",
            ),
            "liquidity_drought": AnomalyType(
                "liquidity_drought",
                0.8,
                "Bid-ask spreads abnormally wide",
                "Reduce size or avoid trading",
            ),
        }

    @timed_operation(threshold_ms=100)
    def detect_anomalies(
        self,
        current_data: dict[str, float],
        historical_data: pd.DataFrame,
        option_data: dict | None = None,
    ) -> list[MarketAnomaly]:
        """
        Detect all types of anomalies in current market data.

        Args:
            current_data: Current market metrics
            historical_data: Historical price/volume data
            option_data: Current option chain data

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # 1. Price/Volume anomalies
        price_anomalies = self._detect_price_anomalies(current_data, historical_data)
        anomalies.extend(price_anomalies)

        # 2. Volatility anomalies
        vol_anomalies = self._detect_volatility_anomalies(current_data, historical_data)
        anomalies.extend(vol_anomalies)

        # 3. Correlation anomalies
        corr_anomalies = self._detect_correlation_anomalies(
            current_data, historical_data
        )
        anomalies.extend(corr_anomalies)

        # 4. Options anomalies (if data available)
        if option_data:
            option_anomalies = self._detect_option_anomalies(
                option_data, historical_data
            )
            anomalies.extend(option_anomalies)

        # 5. Multi-factor anomalies using ML
        ml_anomalies = self._detect_ml_anomalies(current_data, historical_data)
        anomalies.extend(ml_anomalies)

        # Sort by severity
        anomalies.sort(key=lambda x: x.anomaly_type.severity, reverse=True)

        # Log anomalies
        if anomalies:
            logger.warning(
                f"Detected {len(anomalies)} market anomalies",
                extra={
                    "anomalies": [a.anomaly_type.name for a in anomalies],
                    "max_severity": max(a.anomaly_type.severity for a in anomalies),
                },
            )

        return anomalies

    def _detect_price_anomalies(
        self, current: dict[str, float], historical: pd.DataFrame
    ) -> list[MarketAnomaly]:
        """Detect price-related anomalies."""
        anomalies = []

        # 1. Check for price gaps
        if "open" in current and "prev_close" in current:
            gap = abs(current["open"] - current["prev_close"]) / current["prev_close"]

            if gap > 0.10:  # 10% gap
                historical_gaps = self._calculate_historical_gaps(historical)
                z_score = (gap - historical_gaps.mean()) / (
                    historical_gaps.std() + 1e-6
                )
                percentile = stats.percentileofscore(historical_gaps, gap)

                anomaly = MarketAnomaly(
                    anomaly_type=self.ANOMALY_TYPES["price_gap"],
                    timestamp=datetime.now(),
                    metrics={
                        "gap_size": gap,
                        "direction": np.sign(current["open"] - current["prev_close"]),
                    },
                    z_score=z_score,
                    percentile=percentile,
                    historical_frequency=(historical_gaps > 0.10).mean(),
                    confidence=0.95,
                )
                anomalies.append(anomaly)

        # 2. Check for volume spikes
        if "volume" in current:
            avg_volume = historical["volume"].mean()
            volume_ratio = current["volume"] / (avg_volume + 1e-6)

            if volume_ratio > 5.0:  # 5x average
                volume_ratios = (
                    historical["volume"] / historical["volume"].rolling(20).mean()
                )
                z_score = (volume_ratio - volume_ratios.mean()) / (
                    volume_ratios.std() + 1e-6
                )
                percentile = stats.percentileofscore(
                    volume_ratios.dropna(), volume_ratio
                )

                anomaly = MarketAnomaly(
                    anomaly_type=self.ANOMALY_TYPES["volume_spike"],
                    timestamp=datetime.now(),
                    metrics={
                        "volume_ratio": volume_ratio,
                        "actual_volume": current["volume"],
                    },
                    z_score=z_score,
                    percentile=percentile,
                    historical_frequency=(volume_ratios > 5.0).mean(),
                    confidence=0.90,
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_volatility_anomalies(
        self, current: dict[str, float], historical: pd.DataFrame
    ) -> list[MarketAnomaly]:
        """Detect volatility-related anomalies."""
        anomalies = []

        if "realized_vol" in current:
            # Historical volatility distribution
            historical_vols = historical["returns"].rolling(20).std() * np.sqrt(252)
            current_vol = current["realized_vol"]

            # Z-score
            z_score = (current_vol - historical_vols.mean()) / (
                historical_vols.std() + 1e-6
            )
            percentile = stats.percentileofscore(historical_vols.dropna(), current_vol)

            if abs(z_score) > 3.0:  # 3 standard deviations
                anomaly = MarketAnomaly(
                    anomaly_type=self.ANOMALY_TYPES["extreme_volatility"],
                    timestamp=datetime.now(),
                    metrics={
                        "volatility": current_vol,
                        "historical_mean": historical_vols.mean(),
                    },
                    z_score=z_score,
                    percentile=percentile,
                    historical_frequency=(
                        abs(historical_vols - historical_vols.mean())
                        > 3 * historical_vols.std()
                    ).mean(),
                    confidence=0.95,
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_correlation_anomalies(
        self, current: dict[str, float], historical: pd.DataFrame
    ) -> list[MarketAnomaly]:
        """Detect correlation breaks with market/sector."""
        anomalies = []

        # Would need QQQ/tech sector data for real implementation
        # Simplified version
        if "market_correlation" in current:
            typical_correlation = 0.7  # Unity typically 70% correlated with tech
            current_corr = current["market_correlation"]

            if abs(current_corr - typical_correlation) > 0.3:
                anomaly = MarketAnomaly(
                    anomaly_type=self.ANOMALY_TYPES["correlation_break"],
                    timestamp=datetime.now(),
                    metrics={
                        "correlation": current_corr,
                        "typical": typical_correlation,
                    },
                    z_score=(current_corr - typical_correlation) / 0.15,
                    percentile=20 if current_corr < typical_correlation else 80,
                    historical_frequency=0.10,  # 10% of time
                    confidence=0.70,
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_option_anomalies(
        self, option_data: dict, historical: pd.DataFrame
    ) -> list[MarketAnomaly]:
        """Detect anomalies in options data."""
        anomalies = []

        # 1. IV vs realized vol divergence
        if "implied_vol" in option_data and "realized_vol" in option_data:
            iv = option_data["implied_vol"]
            rv = option_data["realized_vol"]

            iv_premium = iv / (rv + 0.01) - 1

            if abs(iv_premium) > 0.50:  # IV 50% different from RV
                anomaly = MarketAnomaly(
                    anomaly_type=self.ANOMALY_TYPES["iv_price_divergence"],
                    timestamp=datetime.now(),
                    metrics={"iv": iv, "rv": rv, "premium": iv_premium},
                    z_score=iv_premium / 0.20,  # Assume 20% std
                    percentile=90 if iv_premium > 0 else 10,
                    historical_frequency=0.15,
                    confidence=0.80,
                )
                anomalies.append(anomaly)

        # 2. Liquidity issues
        if "avg_spread" in option_data:
            typical_spread = 0.05  # 5% typical for Unity options
            current_spread = option_data["avg_spread"]

            if current_spread > typical_spread * 3:
                anomaly = MarketAnomaly(
                    anomaly_type=self.ANOMALY_TYPES["liquidity_drought"],
                    timestamp=datetime.now(),
                    metrics={"spread": current_spread, "typical": typical_spread},
                    z_score=(current_spread - typical_spread) / (typical_spread * 0.5),
                    percentile=95,
                    historical_frequency=0.05,
                    confidence=0.85,
                )
                anomalies.append(anomaly)

        # 3. Unusual option flow
        if "put_call_ratio" in option_data:
            pcr = option_data["put_call_ratio"]

            if pcr > 2.0 or pcr < 0.3:  # Extreme put/call ratios
                anomaly = MarketAnomaly(
                    anomaly_type=self.ANOMALY_TYPES["option_flow_unusual"],
                    timestamp=datetime.now(),
                    metrics={"put_call_ratio": pcr},
                    z_score=(pcr - 1.0) / 0.3,
                    percentile=95 if pcr > 2.0 else 5,
                    historical_frequency=0.10,
                    confidence=0.75,
                )
                anomalies.append(anomaly)

        return anomalies

    def _detect_ml_anomalies(
        self, current: dict[str, float], historical: pd.DataFrame
    ) -> list[MarketAnomaly]:
        """Use ML to detect multi-factor anomalies."""
        anomalies = []

        # Prepare features
        features = self._prepare_ml_features(current)

        if features is not None and self.is_fitted:
            # Predict anomaly
            anomaly_score = self.isolation_forest.decision_function([features])[0]
            is_anomaly = self.isolation_forest.predict([features])[0] == -1

            if is_anomaly:
                # Create generic ML anomaly
                anomaly = MarketAnomaly(
                    anomaly_type=AnomalyType(
                        "ml_multi_factor",
                        0.6,
                        "Multiple factors indicate unusual conditions",
                        "Review all metrics before trading",
                    ),
                    timestamp=datetime.now(),
                    metrics={"anomaly_score": anomaly_score},
                    z_score=abs(anomaly_score) * 3,  # Approximate
                    percentile=95,
                    historical_frequency=0.05,
                    confidence=0.70,
                )
                anomalies.append(anomaly)

        return anomalies

    def _prepare_ml_features(self, current: dict[str, float]) -> np.ndarray | None:
        """Prepare feature vector for ML anomaly detection."""
        feature_names = [
            "returns",
            "volume_ratio",
            "realized_vol",
            "price_momentum",
            "rsi",
            "spread",
        ]

        features = []
        for name in feature_names:
            if name in current:
                features.append(current[name])
            else:
                return None  # Missing required features

        return np.array(features)

    def fit_ml_detector(self, historical_data: pd.DataFrame) -> None:
        """Fit ML anomaly detector on historical data."""
        # Prepare historical features
        features_list = []

        for i in range(20, len(historical_data)):
            window = historical_data.iloc[i - 20 : i]
            current = historical_data.iloc[i]

            features = {
                "returns": current["returns"],
                "volume_ratio": current["volume"] / window["volume"].mean(),
                "realized_vol": window["returns"].std() * np.sqrt(252),
                "price_momentum": (current["close"] - window.iloc[0]["close"])
                / window.iloc[0]["close"],
                "rsi": self._calculate_rsi(window["close"]),
                "spread": 0.02,  # Placeholder
            }

            feature_vector = self._prepare_ml_features(features)
            if feature_vector is not None:
                features_list.append(feature_vector)

        if len(features_list) > 100:
            X = np.array(features_list)
            self.isolation_forest.fit(X)
            self.is_fitted = True
            logger.info(f"Fitted anomaly detector on {len(X)} samples")

    def _calculate_historical_gaps(self, historical: pd.DataFrame) -> pd.Series:
        """Calculate historical gap sizes."""
        gaps = abs(historical["open"] - historical["close"].shift(1)) / historical[
            "close"
        ].shift(1)
        return gaps.dropna()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if len(rsi) > 0 else 50.0

    def generate_anomaly_report(self, anomalies: list[MarketAnomaly]) -> list[str]:
        """Generate human-readable anomaly report."""
        report = ["=== MARKET ANOMALY REPORT ===", ""]

        if not anomalies:
            report.append("✅ No significant anomalies detected")
            return report

        report.append(f"⚠️  {len(anomalies)} anomalies detected:")
        report.append("")

        for anomaly in anomalies:
            severity_emoji = "🔴" if anomaly.anomaly_type.severity > 0.7 else "🟡"

            report.append(f"{severity_emoji} {anomaly.anomaly_type.name.upper()}")
            report.append(f"   Severity: {anomaly.anomaly_type.severity:.1f}/1.0")
            report.append(f"   Description: {anomaly.anomaly_type.description}")
            report.append(f"   Z-Score: {anomaly.z_score:.1f}")
            report.append(f"   Percentile: {anomaly.percentile:.0f}%")
            report.append(
                f"   Historical frequency: {anomaly.historical_frequency:.1%}"
            )
            report.append(f"   ACTION: {anomaly.anomaly_type.recommended_action}")
            report.append("")

        # Overall recommendation
        max_severity = max(a.anomaly_type.severity for a in anomalies)

        if max_severity > 0.8:
            report.append("🚨 OVERALL: AVOID TRADING - Multiple severe anomalies")
        elif max_severity > 0.6:
            report.append("⚠️  OVERALL: REDUCE SIZE - Elevated risk conditions")
        else:
            report.append("✓ OVERALL: PROCEED WITH CAUTION - Monitor conditions")

        return report

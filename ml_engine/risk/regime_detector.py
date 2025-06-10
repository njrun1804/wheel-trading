"""
Volatility regime detection for sophisticated risk management.
Prevents risk skewing from mixing different market regimes.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeInfo:
    """Information about a volatility regime."""

    regime_id: int
    name: str
    volatility: float
    var_95: float
    var_99: float
    days_count: int
    recent_weight: float  # Percentage in last 60 days

    @property
    def kelly_fraction(self) -> float:
        """Recommended Kelly fraction for this regime."""
        if self.volatility < 0.40:  # 40% vol
            return 0.50  # Half-Kelly
        elif self.volatility < 0.80:  # 80% vol
            return 0.33  # Third-Kelly
        else:
            return 0.25  # Quarter-Kelly


class RegimeDetector:
    """Detects and tracks volatility regimes."""

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model: Optional[GaussianMixture] = None
        self.regime_info: Dict[int, RegimeInfo] = {}

    def fit(self, returns: np.ndarray) -> None:
        """Fit regime model to historical returns."""

        # Create volatility features
        returns_series = pd.Series(returns)
        features = pd.DataFrame(
            {
                "vol_5d": returns_series.rolling(5).std() * np.sqrt(252),
                "vol_20d": returns_series.rolling(20).std() * np.sqrt(252),
                "vol_60d": returns_series.rolling(60).std() * np.sqrt(252),
                "abs_return": np.abs(returns),
            }
        ).dropna()

        # Fit Gaussian Mixture Model
        self.model = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full", random_state=42
        )
        self.model.fit(features)

        # Get regime assignments
        regimes = self.model.predict(features)

        # Calculate regime statistics
        regime_stats = []
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[features.index[mask]]

            if len(regime_returns) > 20:
                stats = {
                    "id": regime,
                    "volatility": np.std(regime_returns) * np.sqrt(252),
                    "var_95": np.percentile(regime_returns, 5),
                    "var_99": np.percentile(regime_returns, 1),
                    "count": len(regime_returns),
                    "recent_weight": np.sum(mask[-60:]) / min(60, len(mask)),
                }
                regime_stats.append(stats)

        # Sort by volatility and create RegimeInfo objects
        regime_stats.sort(key=lambda x: x["volatility"])

        for i, stats in enumerate(regime_stats):
            name = ["Low Vol", "Medium Vol", "High Vol"][i] if i < 3 else f"Regime {i}"
            self.regime_info[i] = RegimeInfo(
                regime_id=i,
                name=name,
                volatility=stats["volatility"],
                var_95=stats["var_95"],
                var_99=stats["var_99"],
                days_count=stats["count"],
                recent_weight=stats["recent_weight"],
            )

        logger.info(
            "Regime detection complete",
            n_regimes=len(self.regime_info),
            regimes=[r.name for r in self.regime_info.values()],
        )

    def get_current_regime(self, returns: np.ndarray) -> Tuple[RegimeInfo, float]:
        """Get current regime and confidence."""

        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create features for recent data
        returns_series = pd.Series(returns)
        features = pd.DataFrame(
            {
                "vol_5d": returns_series.rolling(5).std() * np.sqrt(252),
                "vol_20d": returns_series.rolling(20).std() * np.sqrt(252),
                "vol_60d": returns_series.rolling(60).std() * np.sqrt(252),
                "abs_return": np.abs(returns),
            }
        ).dropna()

        # Get last observation
        last_features = features.iloc[-1:].values

        # Predict regime and get probability
        regime_probs = self.model.predict_proba(last_features)[0]
        current_regime_id = np.argmax(regime_probs)
        confidence = regime_probs[current_regime_id]

        # Map to sorted regime
        vol_sorted = sorted(self.regime_info.values(), key=lambda r: r.volatility)
        current_regime = vol_sorted[current_regime_id]

        return current_regime, confidence

    def calculate_regime_adjusted_var(
        self, returns: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate VaR adjusted for current regime."""

        current_regime, regime_confidence = self.get_current_regime(returns)

        # Get regime-specific VaR
        if confidence_level == 0.95:
            regime_var = current_regime.var_95
        elif confidence_level == 0.99:
            regime_var = current_regime.var_99
        else:
            # Interpolate for other confidence levels
            regime_var = np.percentile(
                returns[-current_regime.days_count :], (1 - confidence_level) * 100
            )

        # Calculate EWMA VaR for comparison
        ewma_var = self._calculate_ewma_var(returns, confidence_level)

        # Blend based on regime confidence
        final_var = regime_confidence * regime_var + (1 - regime_confidence) * ewma_var

        return {
            "var": final_var,
            "regime_var": regime_var,
            "ewma_var": ewma_var,
            "regime": current_regime.name,
            "regime_confidence": regime_confidence,
            "kelly_fraction": current_regime.kelly_fraction,
        }

    def _calculate_ewma_var(
        self, returns: np.ndarray, confidence_level: float, lambda_param: float = 0.94
    ) -> float:
        """Calculate EWMA VaR."""
        n = len(returns)
        weights = np.array([(1 - lambda_param) * lambda_param**i for i in range(n - 1, -1, -1)])
        weights = weights / weights.sum()

        # Sort returns with weights
        sorted_idx = np.argsort(returns)
        sorted_returns = returns[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Find VaR using cumulative weights
        cumsum_weights = np.cumsum(sorted_weights)
        var_idx = np.searchsorted(cumsum_weights, 1 - confidence_level)

        if var_idx < len(sorted_returns):
            return sorted_returns[var_idx]
        else:
            return sorted_returns[-1]

"""
Implied Volatility surface analyzer for options strategy optimization.
Provides IV rank, percentile, term structure, and skew analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d

from ..config.loader import get_config
from ..models.position import PositionType
from ..utils import get_logger, timed_operation, with_recovery
from ..utils.recovery import RecoveryStrategy

logger = get_logger(__name__)


class IVMetrics(NamedTuple):
    """Comprehensive IV metrics for decision making."""

    current_iv: float
    iv_rank: float  # 0-100 percentile over lookback period
    iv_percentile: float  # % of days below current IV
    iv_zscore: float  # Standard deviations from mean
    term_structure: Dict[int, float]  # DTE -> IV
    put_call_skew: float  # 25 delta put IV / 25 delta call IV
    vol_of_vol: float  # Volatility of implied volatility
    mean_reversion_speed: float  # Half-life in days
    regime: str  # 'contango', 'backwardation', 'flat'
    confidence: float


@dataclass
class SkewMetrics:
    """Option skew analysis."""

    skew_25delta: float  # (25d put - 25d call) / ATM
    skew_10delta: float  # (10d put - 10d call) / ATM
    risk_reversal: float  # 25d call - 25d put
    butterfly: float  # 0.5*(25d put + 25d call) - ATM
    skew_slope: float  # Change in IV per 10 delta points
    crash_indicator: float  # 0-1 score of crash protection demand


class IVSurfaceAnalyzer:
    """Analyzes implied volatility patterns for trading decisions."""

    def __init__(self, lookback_days: int = 252, min_history: int = 30):  # 1 year for IV rank
        self.lookback_days = lookback_days
        self.min_history = min_history
        self.iv_history: Dict[str, pd.DataFrame] = {}

    @timed_operation(threshold_ms=50)
    def analyze_iv_surface(self, option_chain: Dict, symbol: str = None) -> IVMetrics:
        """
        Analyze current IV surface and historical patterns.

        Args:
            option_chain: Current option chain data
            symbol: Underlying symbol

        Returns:
            Comprehensive IV metrics
        """
        if symbol is None:
            config = get_config()
            symbol = config.unity.ticker
        logger.info(f"Analyzing IV surface for {symbol}")

        # Extract current ATM IV
        current_iv = self._get_atm_iv(option_chain)

        # Calculate IV rank and percentile
        iv_rank, iv_percentile = self._calculate_iv_rank(symbol, current_iv)

        # Calculate z-score
        iv_zscore = self._calculate_iv_zscore(symbol, current_iv)

        # Analyze term structure
        term_structure, regime = self._analyze_term_structure(option_chain)

        # Calculate skew metrics
        skew_metrics = self._calculate_skew(option_chain)

        # Calculate volatility of volatility
        vol_of_vol = self._calculate_vol_of_vol(symbol)

        # Estimate mean reversion
        mean_reversion_speed = self._estimate_mean_reversion(symbol)

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(symbol, option_chain)

        metrics = IVMetrics(
            current_iv=current_iv,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            iv_zscore=iv_zscore,
            term_structure=term_structure,
            put_call_skew=skew_metrics.skew_25delta,
            vol_of_vol=vol_of_vol,
            mean_reversion_speed=mean_reversion_speed,
            regime=regime,
            confidence=confidence,
        )

        # Log key insights
        self._log_insights(metrics)

        return metrics

    def update_iv_history(self, symbol: str, date: datetime, iv: float, dte: int = 30) -> None:
        """Update historical IV data for a symbol."""
        if symbol not in self.iv_history:
            self.iv_history[symbol] = pd.DataFrame()

        # Add new data point
        new_data = pd.DataFrame({"date": [date], "iv": [iv], "dte": [dte]})

        self.iv_history[symbol] = pd.concat([self.iv_history[symbol], new_data], ignore_index=True)

        # Keep only lookback period
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        self.iv_history[symbol] = self.iv_history[symbol][
            self.iv_history[symbol]["date"] >= cutoff_date
        ]

    def _get_atm_iv(self, option_chain: Dict) -> float:
        """Extract at-the-money implied volatility."""
        spot = option_chain.get("spot_price", 0)

        # Find closest strike to spot
        puts = option_chain.get("puts", [])
        if not puts:
            return 0.30  # Default fallback

        atm_put = min(puts, key=lambda x: abs(x["strike"] - spot))
        return atm_put.get("implied_volatility", 0.30)

    def _calculate_iv_rank(self, symbol: str, current_iv: float) -> Tuple[float, float]:
        """Calculate IV rank and percentile."""
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < self.min_history:
            return 50.0, 50.0  # Default to middle

        historical_ivs = self.iv_history[symbol]["iv"].values

        # IV Rank: (Current - Min) / (Max - Min) * 100
        min_iv = historical_ivs.min()
        max_iv = historical_ivs.max()

        if max_iv - min_iv > 0:
            iv_rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
        else:
            iv_rank = 50.0

        # IV Percentile: % of days with IV below current
        iv_percentile = stats.percentileofscore(historical_ivs, current_iv)

        return np.clip(iv_rank, 0, 100), np.clip(iv_percentile, 0, 100)

    def _calculate_iv_zscore(self, symbol: str, current_iv: float) -> float:
        """Calculate how many standard deviations IV is from mean."""
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < self.min_history:
            return 0.0

        historical_ivs = self.iv_history[symbol]["iv"].values
        mean_iv = historical_ivs.mean()
        std_iv = historical_ivs.std()

        if std_iv > 0:
            return (current_iv - mean_iv) / std_iv
        return 0.0

    def _analyze_term_structure(self, option_chain: Dict) -> Tuple[Dict[int, float], str]:
        """Analyze IV term structure across expirations."""
        term_structure = {}

        # Group by expiration
        expirations = {}
        for put in option_chain.get("puts", []):
            exp = put.get("expiration")
            dte = put.get("dte", 0)
            if exp and dte > 0:
                if dte not in expirations:
                    expirations[dte] = []
                expirations[dte].append(put.get("implied_volatility", 0))

        # Average IV for each expiration
        for dte, ivs in expirations.items():
            if ivs:
                term_structure[dte] = np.mean(ivs)

        # Determine regime
        if len(term_structure) >= 2:
            dtes = sorted(term_structure.keys())
            front_iv = term_structure[dtes[0]]
            back_iv = term_structure[dtes[-1]]

            if front_iv > back_iv * 1.05:
                regime = "backwardation"  # Front month premium
            elif back_iv > front_iv * 1.05:
                regime = "contango"  # Normal state
            else:
                regime = "flat"
        else:
            regime = "unknown"

        return term_structure, regime

    def _calculate_skew(self, option_chain: Dict) -> SkewMetrics:
        """Calculate option skew metrics."""
        spot = option_chain.get("spot_price", 0)
        puts = option_chain.get("puts", [])
        calls = option_chain.get("calls", [])

        if not puts or not calls or spot == 0:
            return SkewMetrics(
                skew_25delta=0,
                skew_10delta=0,
                risk_reversal=0,
                butterfly=0,
                skew_slope=0,
                crash_indicator=0,
            )

        # Find 25-delta and 10-delta options
        put_25d = self._find_delta_option(puts, 0.25)
        call_25d = self._find_delta_option(calls, 0.25)
        put_10d = self._find_delta_option(puts, 0.10)
        call_10d = self._find_delta_option(calls, 0.10)
        atm_iv = self._get_atm_iv(option_chain)

        # Calculate skew metrics
        skew_25delta = 0
        if put_25d and call_25d and atm_iv > 0:
            skew_25delta = (put_25d["implied_volatility"] - call_25d["implied_volatility"]) / atm_iv

        skew_10delta = 0
        if put_10d and call_10d and atm_iv > 0:
            skew_10delta = (put_10d["implied_volatility"] - call_10d["implied_volatility"]) / atm_iv

        risk_reversal = 0
        if put_25d and call_25d:
            risk_reversal = call_25d["implied_volatility"] - put_25d["implied_volatility"]

        butterfly = 0
        if put_25d and call_25d and atm_iv > 0:
            butterfly = (
                0.5 * (put_25d["implied_volatility"] + call_25d["implied_volatility"]) - atm_iv
            )

        # Estimate skew slope
        skew_slope = self._calculate_skew_slope(puts)

        # Crash indicator (0-1)
        crash_indicator = self._calculate_crash_indicator(skew_25delta, skew_10delta, risk_reversal)

        return SkewMetrics(
            skew_25delta=skew_25delta,
            skew_10delta=skew_10delta,
            risk_reversal=risk_reversal,
            butterfly=butterfly,
            skew_slope=skew_slope,
            crash_indicator=crash_indicator,
        )

    def _find_delta_option(self, options: List[Dict], target_delta: float) -> Optional[Dict]:
        """Find option closest to target delta."""
        if not options:
            return None

        best_option = None
        min_diff = float("inf")

        for option in options:
            delta = abs(option.get("delta", 0))
            diff = abs(delta - target_delta)
            if diff < min_diff:
                min_diff = diff
                best_option = option

        return best_option if min_diff < 0.05 else None

    def _calculate_skew_slope(self, puts: List[Dict]) -> float:
        """Calculate IV change per 10 delta points."""
        if len(puts) < 3:
            return 0.0

        # Sort by delta
        sorted_puts = sorted(puts, key=lambda x: abs(x.get("delta", 0)))

        deltas = []
        ivs = []

        for put in sorted_puts:
            delta = abs(put.get("delta", 0))
            iv = put.get("implied_volatility", 0)
            if 0.05 < delta < 0.50 and iv > 0:
                deltas.append(delta)
                ivs.append(iv)

        if len(deltas) >= 3:
            # Linear regression
            z = np.polyfit(deltas, ivs, 1)
            return z[0] * 0.10  # IV change per 10 delta

        return 0.0

    def _calculate_crash_indicator(
        self, skew_25d: float, skew_10d: float, risk_reversal: float
    ) -> float:
        """Calculate crash protection demand indicator (0-1)."""
        # Higher put skew = more crash protection demand
        indicators = []

        # 25-delta skew contribution
        if skew_25d > 0.10:
            indicators.append(min(1.0, skew_25d / 0.30))

        # 10-delta skew contribution (more weight)
        if skew_10d > 0.15:
            indicators.append(min(1.0, skew_10d / 0.40))

        # Negative risk reversal contribution
        if risk_reversal < -0.05:
            indicators.append(min(1.0, abs(risk_reversal) / 0.20))

        if indicators:
            return np.mean(indicators)
        return 0.0

    def _calculate_vol_of_vol(self, symbol: str) -> float:
        """Calculate volatility of implied volatility."""
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 20:
            return 0.30  # Default

        # Get recent IV changes
        ivs = self.iv_history[symbol].sort_values("date")["iv"].values[-60:]

        if len(ivs) >= 20:
            # Daily IV returns
            iv_returns = np.diff(np.log(ivs))
            # Annualized volatility of IV
            vol_of_vol = np.std(iv_returns) * np.sqrt(252)
            return vol_of_vol

        return 0.30

    def _estimate_mean_reversion(self, symbol: str) -> float:
        """Estimate IV mean reversion speed (half-life in days)."""
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 60:
            return 30.0  # Default 30-day half-life

        ivs = self.iv_history[symbol].sort_values("date")["iv"].values

        if len(ivs) >= 60:
            # AR(1) model: IV(t) = alpha + beta * IV(t-1) + error
            iv_lag = ivs[:-1]
            iv_current = ivs[1:]

            # OLS regression
            beta = np.cov(iv_current, iv_lag)[0, 1] / np.var(iv_lag)

            # Half-life = -ln(2) / ln(beta)
            if 0 < beta < 1:
                half_life = -np.log(2) / np.log(beta)
                return min(60, max(5, half_life))  # Bound between 5-60 days

        return 30.0

    def _calculate_confidence(self, symbol: str, option_chain: Dict) -> float:
        """Calculate confidence in IV analysis."""
        confidence = 1.0

        # Historical data sufficiency
        if symbol in self.iv_history:
            data_points = len(self.iv_history[symbol])
            data_conf = min(1.0, data_points / self.lookback_days)
            confidence *= data_conf
        else:
            confidence *= 0.5

        # Option chain quality
        n_strikes = len(option_chain.get("puts", [])) + len(option_chain.get("calls", []))
        chain_conf = min(1.0, n_strikes / 20)
        confidence *= chain_conf

        return confidence

    def _log_insights(self, metrics: IVMetrics) -> None:
        """Log actionable insights from IV analysis."""
        insights = []

        # IV rank insights
        if metrics.iv_rank > 80:
            insights.append("HIGH_IV_RANK: Excellent for selling premium")
        elif metrics.iv_rank < 20:
            insights.append("LOW_IV_RANK: Poor for selling, consider buying")

        # Term structure insights
        if metrics.regime == "backwardation":
            insights.append("BACKWARDATION: Event risk priced in front month")
        elif metrics.regime == "contango":
            insights.append("CONTANGO: Normal term structure")

        # Skew insights
        if metrics.put_call_skew > 1.15:
            insights.append("HIGH_PUT_SKEW: Market pricing crash risk")

        # Mean reversion insights
        if metrics.iv_zscore > 2 and metrics.mean_reversion_speed < 20:
            insights.append("IV_EXTREME: Fast mean reversion expected")

        if insights:
            logger.info(
                "IV surface insights",
                extra={
                    "insights": insights,
                    "iv_rank": metrics.iv_rank,
                    "regime": metrics.regime,
                    "confidence": metrics.confidence,
                },
            )

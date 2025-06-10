"""Risk measurement system with self-monitoring and automatic recalibration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config.loader import get_config

from ..models.greeks import Greeks
from ..models.position import Position
from ..storage.cache.general_cache import cached
from ..utils import RecoveryStrategy, get_logger, timed_operation, with_recovery

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    """Risk severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Container for calculated risk metrics."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    kelly_fraction: float
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    portfolio_theta: float
    margin_requirement: float
    margin_utilization: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "kelly_fraction": self.kelly_fraction,
            "portfolio_delta": self.portfolio_delta,
            "portfolio_gamma": self.portfolio_gamma,
            "portfolio_vega": self.portfolio_vega,
            "portfolio_theta": self.portfolio_theta,
            "margin_requirement": self.margin_requirement,
            "margin_utilization": self.margin_utilization,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskLimitBreach:
    """Information about a risk limit breach."""

    metric: str
    current_value: float
    limit_value: float
    severity: RiskLevel
    timestamp: datetime
    recommendation: str


@dataclass
class RiskLimits:
    """Configuration-driven risk limits."""

    max_var_95: float = field(default=None)
    max_cvar_95: float = field(default=None)
    max_kelly_fraction: float = field(default=None)
    max_delta_exposure: float = field(default=None)
    max_gamma_exposure: float = field(default=None)
    max_vega_exposure: float = field(default=None)
    max_margin_utilization: float = field(default=None)

    # Dynamic scaling factors
    volatility_scalar: float = field(default=1.0)

    def __post_init__(self):
        """Initialize from config if values not provided."""
        config = get_config()

        # Use config values as defaults if not explicitly set
        if self.max_var_95 is None:
            self.max_var_95 = config.risk.limits.max_var_95
        if self.max_cvar_95 is None:
            self.max_cvar_95 = config.risk.limits.max_cvar_95
        if self.max_kelly_fraction is None:
            self.max_kelly_fraction = config.risk.limits.max_kelly_fraction
        if self.max_delta_exposure is None:
            self.max_delta_exposure = config.risk.greeks.max_delta_exposure
        if self.max_gamma_exposure is None:
            self.max_gamma_exposure = config.risk.greeks.max_gamma_exposure
        if self.max_vega_exposure is None:
            self.max_vega_exposure = config.risk.greeks.max_vega_exposure
        if self.max_margin_utilization is None:
            self.max_margin_utilization = config.risk.margin.max_utilization

    def scale_by_volatility(self, current_vol: float, baseline_vol: float = 0.15) -> None:
        """Scale limits based on current volatility regime."""
        self.volatility_scalar = baseline_vol / max(current_vol, 0.05)

        # Apply scaling to percentage-based limits
        self.max_var_95 *= self.volatility_scalar
        self.max_cvar_95 *= self.volatility_scalar

        logger.info(
            "Risk limits scaled by volatility",
            extra={
                "current_vol": current_vol,
                "baseline_vol": baseline_vol,
                "scalar": self.volatility_scalar,
            },
        )


class RiskAnalyzer:
    """Main risk analysis engine with self-monitoring."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        limits: Optional[RiskLimits] = None,
        history_file: Optional[Path] = None,
    ):
        """
        Initialize risk analyzer.

        Parameters
        ----------
        config : Dict, optional
            Configuration dictionary
        limits : RiskLimits, optional
            Risk limits configuration
        history_file : Path, optional
            Path to store historical accuracy data
        """
        self.config = config or {}
        self.limits = limits or RiskLimits()
        self.history_file = history_file or Path("risk_history.json")
        self.accuracy_tracker = AccuracyTracker(self.history_file)
        self._recalibration_needed = False

    @timed_operation(threshold_ms=10.0)
    @cached(ttl=timedelta(minutes=15))
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "parametric",
    ) -> Tuple[float, float]:
        """
        Calculate VaR with confidence score.

        Parameters
        ----------
        returns : np.ndarray
            Historical returns
        confidence_level : float
            Confidence level for VaR
        method : str
            'parametric', 'historical', or 'cornish-fisher'

        Returns
        -------
        Tuple[float, float]
            (VaR value, confidence score)
        """
        if len(returns) < 20:
            logger.warning("Insufficient data for reliable VaR calculation")
            return np.nan, 0.0

        # Sanity checks
        if np.any(np.isnan(returns)):
            logger.warning("NaN values in returns data")
            returns = returns[~np.isnan(returns)]

        if np.std(returns) == 0:
            logger.warning("Zero volatility in returns")
            return 0.0, 0.5

        confidence = 1.0

        if method == "parametric":
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)

            # Test for normality
            _, p_value = stats.jarque_bera(returns)
            if p_value < 0.05:
                confidence *= 0.8
                logger.info("Returns show non-normality, parametric VaR less reliable")

        elif method == "historical":
            var = -np.percentile(returns, (1 - confidence_level) * 100)

            # Confidence based on sample size
            if len(returns) < 250:
                confidence *= 0.9
            if len(returns) < 100:
                confidence *= 0.8

        elif method == "cornish-fisher":
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns, fisher=True)

            # Standard z-score
            z = stats.norm.ppf(1 - confidence_level)

            # Cornish-Fisher adjustment
            z_cf = z + (z**2 - 1) * skew / 6
            z_cf += (z**3 - 3 * z) * kurt / 24
            z_cf -= (2 * z**3 - 5 * z) * skew**2 / 36

            var = -(mean + z_cf * std)

            # Higher confidence due to adjustment for higher moments
            confidence *= 0.95

        # Sanity check on VaR value
        if var < 0:
            logger.warning("Negative VaR calculated, setting to 0")
            var = 0.0
            confidence *= 0.5

        if var > 1.0:  # More than 100% loss
            logger.warning("VaR exceeds 100%, capping at 100%")
            var = 1.0
            confidence *= 0.7

        logger.debug(
            "VaR calculation completed",
            extra={
                "method": method,
                "confidence_level": confidence_level,
                "var": var,
                "confidence": confidence,
            },
        )

        return var, confidence

    @timed_operation(threshold_ms=10.0)
    @cached(ttl=timedelta(minutes=15))
    @with_recovery(strategy=RecoveryStrategy.FALLBACK)
    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "parametric",
    ) -> Tuple[float, float]:
        """
        Calculate CVaR (Expected Shortfall) with confidence score.

        Returns
        -------
        Tuple[float, float]
            (CVaR value, confidence score)
        """
        var, var_confidence = self.calculate_var(returns, confidence_level, method)

        if np.isnan(var):
            return np.nan, 0.0

        confidence = var_confidence

        if method == "parametric":
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            alpha = 1 - confidence_level
            z = stats.norm.ppf(alpha)
            pdf_z = stats.norm.pdf(z)
            cvar = -mean + std * pdf_z / alpha

        elif method == "historical":
            # Get returns worse than VaR
            threshold = -var
            tail_returns = returns[returns <= threshold]

            if len(tail_returns) == 0:
                # Use worst return if no returns beyond VaR
                cvar = -np.min(returns)
                confidence *= 0.7
            else:
                cvar = -np.mean(tail_returns)

        elif method == "cornish-fisher":
            # Use adjusted CVaR formula with higher moments
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns, fisher=True)

            alpha = 1 - confidence_level
            z = stats.norm.ppf(alpha)
            pdf_z = stats.norm.pdf(z)

            # Base CVaR
            cvar = -mean + std * pdf_z / alpha

            # Adjust for skewness and kurtosis
            adjustment = 1 + (skew * z / 6) + (kurt * (z**2 - 1) / 24)
            cvar *= adjustment

        # CVaR should be >= VaR
        if cvar < var:
            logger.warning("CVaR less than VaR, adjusting")
            cvar = var * 1.1
            confidence *= 0.8

        logger.debug(
            "CVaR calculation completed",
            extra={
                "method": method,
                "confidence_level": confidence_level,
                "cvar": cvar,
                "confidence": confidence,
            },
        )

        return cvar, confidence

    @timed_operation(threshold_ms=1.0)
    @cached(ttl=timedelta(hours=1))
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        apply_half_kelly: bool = True,
    ) -> Tuple[float, float]:
        """
        Calculate Kelly criterion for position sizing.

        Returns
        -------
        Tuple[float, float]
            (Kelly fraction, confidence score)
        """
        confidence = 1.0

        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win rate: {win_rate}")
            return 0.0, 0.0

        if avg_win <= 0 or avg_loss <= 0:
            logger.warning("Invalid win/loss amounts")
            return 0.0, 0.0

        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss

        kelly = (p * b - q) / b

        # Apply constraints
        if kelly < 0:
            logger.info("Negative Kelly fraction - no edge")
            return 0.0, 0.9

        if kelly > 1:
            logger.warning("Kelly fraction exceeds 100%, capping")
            kelly = 1.0
            confidence *= 0.7

        # Apply half-Kelly for safety
        if apply_half_kelly:
            kelly /= 2

        # Further cap at configured maximum
        kelly = min(kelly, self.limits.max_kelly_fraction)

        # Adjust confidence based on sample size concerns
        # (In practice, would check actual sample size)
        if win_rate < 0.4 or win_rate > 0.7:
            confidence *= 0.9  # Extreme win rates less reliable

        logger.debug(
            "Kelly criterion calculated",
            extra={
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "kelly_fraction": kelly,
                "confidence": confidence,
            },
        )

        return kelly, confidence

    @timed_operation(threshold_ms=5.0)
    def aggregate_portfolio_greeks(
        self,
        positions: List[Tuple[Position, Greeks, float]],
    ) -> Tuple[Dict[str, float], float]:
        """
        Aggregate Greeks across portfolio.

        Parameters
        ----------
        positions : List[Tuple[Position, Greeks, float]]
            List of (position, greeks, underlying_price) tuples

        Returns
        -------
        Tuple[Dict[str, float], float]
            Aggregated Greeks and confidence score
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        total_rho = 0.0

        for position, greeks, underlying_price in positions:
            # Scale by position size and contract multiplier
            multiplier = 100 if position.position_type != "stock" else 1
            qty = position.quantity

            if greeks.delta is not None:
                total_delta += qty * multiplier * greeks.delta

            if greeks.gamma is not None:
                # Gamma in shares per $1 move
                total_gamma += qty * multiplier * greeks.gamma

            if greeks.vega is not None:
                # Vega in dollars per 1% volatility move
                total_vega += qty * multiplier * greeks.vega

            if greeks.theta is not None:
                # Theta in dollars per day
                total_theta += qty * multiplier * greeks.theta

            if greeks.rho is not None:
                # Rho in dollars per 1% rate move
                total_rho += qty * multiplier * greeks.rho

        aggregated = {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega,
            "theta": total_theta,
            "rho": total_rho,
            "delta_dollars": total_delta * underlying_price if positions else 0,
        }

        logger.debug(
            "Portfolio Greeks aggregated",
            extra={"greeks": aggregated, "position_count": len(positions)},
        )

        # Calculate confidence based on data quality
        confidence = 0.95 if positions else 0.0

        # Reduce confidence if any Greeks are missing
        missing_greeks = sum(
            1
            for pos, greeks, _ in positions
            if any(getattr(greeks, g) is None for g in ["delta", "gamma", "vega", "theta"])
        )
        if positions and missing_greeks > 0:
            confidence *= 1.0 - missing_greeks / len(positions) * 0.2

        return aggregated, confidence

    def estimate_margin_requirement(
        self,
        positions: List[Tuple[Position, float, float]],
    ) -> Tuple[float, float]:
        """
        Estimate total margin requirement.

        Parameters
        ----------
        positions : List[Tuple[Position, float, float]]
            List of (position, underlying_price, option_price) tuples

        Returns
        -------
        Tuple[float, float]
            Total margin requirement and confidence score
        """
        total_margin = 0.0

        for position, underlying_price, option_price in positions:
            if position.position_type == "put" and position.is_short:
                # Standard margin for naked puts
                strike = position.strike or 0
                otm_amount = max(underlying_price - strike, 0)

                method1 = 0.20 * underlying_price - otm_amount + option_price
                method2 = 0.10 * strike + option_price

                margin_per_contract = max(method1, method2) * 100
                total_margin += abs(position.quantity) * margin_per_contract

            elif position.position_type == "call" and position.is_short:
                # Naked calls require stock margin
                total_margin += abs(position.quantity) * 100 * underlying_price * 0.5

        # High confidence for standard margin calculations
        confidence = 0.90 if positions else 0.0

        # Slightly reduce confidence for complex positions
        complex_positions = sum(
            1 for pos, _, _ in positions if pos.position_type not in ["put", "call", "stock"]
        )
        if positions and complex_positions > 0:
            confidence *= 0.85

        return total_margin, confidence

    def check_limits(
        self,
        metrics: RiskMetrics,
        portfolio_value: float,
    ) -> List[RiskLimitBreach]:
        """
        Check current metrics against configured limits.

        Returns
        -------
        List[RiskLimitBreach]
            List of any limit breaches
        """
        breaches = []

        # VaR limit check
        var_pct = metrics.var_95 / portfolio_value if portfolio_value > 0 else 0
        if var_pct > self.limits.max_var_95:
            breaches.append(
                RiskLimitBreach(
                    metric="var_95",
                    current_value=var_pct,
                    limit_value=self.limits.max_var_95,
                    severity=(
                        RiskLevel.HIGH
                        if var_pct > self.limits.max_var_95 * 1.5
                        else RiskLevel.MEDIUM
                    ),
                    timestamp=datetime.now(timezone.utc),
                    recommendation="Reduce position sizes or hedge portfolio",
                )
            )

        # CVaR limit check
        cvar_pct = metrics.cvar_95 / portfolio_value if portfolio_value > 0 else 0
        if cvar_pct > self.limits.max_cvar_95:
            breaches.append(
                RiskLimitBreach(
                    metric="cvar_95",
                    current_value=cvar_pct,
                    limit_value=self.limits.max_cvar_95,
                    severity=(
                        RiskLevel.CRITICAL
                        if cvar_pct > self.limits.max_cvar_95 * 1.5
                        else RiskLevel.HIGH
                    ),
                    timestamp=datetime.now(timezone.utc),
                    recommendation="Immediately reduce tail risk exposure",
                )
            )

        # Greeks limits
        if abs(metrics.portfolio_delta) > self.limits.max_delta_exposure:
            breaches.append(
                RiskLimitBreach(
                    metric="delta",
                    current_value=metrics.portfolio_delta,
                    limit_value=self.limits.max_delta_exposure,
                    severity=RiskLevel.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                    recommendation="Adjust delta hedge to reduce directional risk",
                )
            )

        # Margin utilization
        if metrics.margin_utilization > self.limits.max_margin_utilization:
            breaches.append(
                RiskLimitBreach(
                    metric="margin_utilization",
                    current_value=metrics.margin_utilization,
                    limit_value=self.limits.max_margin_utilization,
                    severity=(
                        RiskLevel.HIGH if metrics.margin_utilization > 0.75 else RiskLevel.MEDIUM
                    ),
                    timestamp=datetime.now(timezone.utc),
                    recommendation="Reduce leverage or add capital",
                )
            )

        return breaches

    def generate_risk_report(
        self,
        metrics: RiskMetrics,
        breaches: List[RiskLimitBreach],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_value": portfolio_value,
            "metrics": metrics.to_dict(),
            "breaches": [
                {
                    "metric": b.metric,
                    "current": b.current_value,
                    "limit": b.limit_value,
                    "severity": b.severity.value,
                    "recommendation": b.recommendation,
                }
                for b in breaches
            ],
            "risk_score": self._calculate_risk_score(metrics, breaches, portfolio_value),
            "recommendations": self._generate_recommendations(metrics, breaches),
        }

        # Log critical breaches
        critical_breaches = [b for b in breaches if b.severity == RiskLevel.CRITICAL]
        if critical_breaches:
            logger.critical(
                "Critical risk limits breached",
                extra={"breaches": [b.metric for b in critical_breaches]},
            )

        return report

    def _calculate_risk_score(
        self,
        metrics: RiskMetrics,
        breaches: List[RiskLimitBreach],
        portfolio_value: float,
    ) -> float:
        """Calculate overall risk score (0-100)."""
        score = 0.0

        # Base score from VaR/CVaR
        var_contribution = min((metrics.var_95 / portfolio_value) * 1000, 40)
        cvar_contribution = min((metrics.cvar_95 / portfolio_value) * 1000, 30)
        score += var_contribution + cvar_contribution

        # Add breach penalties
        for breach in breaches:
            if breach.severity == RiskLevel.CRITICAL:
                score += 20
            elif breach.severity == RiskLevel.HIGH:
                score += 10
            elif breach.severity == RiskLevel.MEDIUM:
                score += 5

        # Cap at 100
        return min(score, 100)

    def _generate_recommendations(
        self,
        metrics: RiskMetrics,
        breaches: List[RiskLimitBreach],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Get unique recommendations from breaches
        for breach in breaches:
            if breach.recommendation not in recommendations:
                recommendations.append(breach.recommendation)

        # Add general recommendations based on metrics
        if metrics.portfolio_theta < -1000:
            recommendations.append("High theta decay - consider rolling positions")

        if abs(metrics.portfolio_vega) > 5000:
            recommendations.append("High vega exposure - vulnerable to volatility changes")

        if metrics.kelly_fraction < 0.05:
            recommendations.append(
                "Low Kelly fraction - consider increasing edge or reducing position"
            )

        return recommendations


class AccuracyTracker:
    """Track historical accuracy of risk predictions."""

    def __init__(self, history_file: Path):
        """Initialize accuracy tracker."""
        self.history_file = history_file
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load historical predictions."""
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                return json.load(f)
        return []

    def record_prediction(
        self,
        metric: str,
        predicted_value: float,
        confidence: float,
        timestamp: datetime,
    ) -> None:
        """Record a prediction for later validation."""
        self.history.append(
            {
                "metric": metric,
                "predicted": predicted_value,
                "confidence": confidence,
                "timestamp": timestamp.isoformat(),
                "actual": None,
                "validated": False,
            }
        )
        self._save_history()

    def validate_prediction(
        self,
        metric: str,
        actual_value: float,
        timestamp: datetime,
    ) -> None:
        """Validate a previous prediction with actual outcome."""
        # Find matching prediction
        for pred in self.history:
            if (
                pred["metric"] == metric
                and not pred["validated"]
                and abs((datetime.fromisoformat(pred["timestamp"]) - timestamp).total_seconds())
                < 86400
            ):

                pred["actual"] = actual_value
                pred["validated"] = True
                pred["error"] = abs(pred["predicted"] - actual_value)
                pred["relative_error"] = pred["error"] / max(abs(actual_value), 1e-10)

                logger.info(
                    "Prediction validated",
                    extra={
                        "metric": metric,
                        "predicted": pred["predicted"],
                        "actual": actual_value,
                        "error": pred["error"],
                    },
                )

        self._save_history()

    def get_accuracy_stats(self, metric: str, lookback_days: int = 30) -> Dict[str, float]:
        """Get accuracy statistics for a metric."""
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=lookback_days)

        relevant = [
            p
            for p in self.history
            if p["metric"] == metric
            and p["validated"]
            and datetime.fromisoformat(p["timestamp"]) > cutoff
        ]

        if not relevant:
            return {"count": 0, "mean_error": np.nan, "mean_relative_error": np.nan}

        errors = [p["error"] for p in relevant]
        rel_errors = [p["relative_error"] for p in relevant]

        return {
            "count": len(relevant),
            "mean_error": np.mean(errors),
            "mean_relative_error": np.mean(rel_errors),
            "std_error": np.std(errors),
            "max_error": np.max(errors),
        }

    def needs_recalibration(self, metric: str, threshold: float = 0.1) -> bool:
        """Check if a metric needs recalibration."""
        stats = self.get_accuracy_stats(metric)

        if stats["count"] < 10:
            return False  # Not enough data

        # Check if mean relative error exceeds threshold
        return stats["mean_relative_error"] > threshold

    def _save_history(self) -> None:
        """Save history to file."""
        # Keep only last 90 days
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=90)
        self.history = [p for p in self.history if datetime.fromisoformat(p["timestamp"]) > cutoff]

        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

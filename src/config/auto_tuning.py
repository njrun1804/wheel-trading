"""Configuration auto-tuning based on performance metrics and outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from unity_wheel.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParameterPerformance:
    """Performance metrics for a specific parameter value."""

    parameter_name: str
    parameter_value: Any
    sample_count: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_return: float = 0.0
    risk_adjusted_return: float = 0.0
    timestamps: list[datetime] = field(default_factory=list)


@dataclass
class TuningRecommendation:
    """Recommendation for parameter adjustment."""

    parameter_name: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    reasoning: str
    based_on_samples: int


class ConfigAutoTuner:
    """
    Automatic configuration tuning based on observed outcomes.

    Analyzes decision quality metrics and recommends parameter
    adjustments to improve performance.
    """

    # Parameters that can be auto-tuned
    TUNABLE_PARAMETERS = {
        "target_delta": {
            "type": "float",
            "min": 0.15,
            "max": 0.45,
            "step": 0.05,
            "description": "Target delta for put selection",
        },
        "target_dte": {
            "type": "int",
            "min": 21,
            "max": 60,
            "step": 7,
            "description": "Target days to expiration",
        },
        "min_premium_yield": {
            "type": "float",
            "min": 0.005,
            "max": 0.03,
            "step": 0.005,
            "description": "Minimum acceptable premium yield",
        },
        "roll_dte_threshold": {
            "type": "int",
            "min": 3,
            "max": 14,
            "step": 1,
            "description": "Days to expiry threshold for rolling",
        },
        "roll_delta_threshold": {
            "type": "float",
            "min": 0.60,
            "max": 0.85,
            "step": 0.05,
            "description": "Delta threshold for rolling positions",
        },
        "max_position_size": {
            "type": "float",
            "min": 0.05,
            "max": 0.30,
            "step": 0.05,
            "description": "Maximum position size as portfolio percentage",
        },
    }

    def __init__(
        self,
        history_file: Path = Path("tuning_history.json"),
        min_samples: int = 20,
        lookback_days: int = 30,
    ):
        """Initialize auto-tuner."""
        self.history_file = history_file
        self.min_samples = min_samples
        self.lookback_days = lookback_days

        # Performance tracking by parameter values
        self.parameter_performance: dict[str, dict[Any, ParameterPerformance]] = {}

        # Load historical data
        self._load_history()

    def record_outcome(
        self,
        decision_id: str,
        parameters_used: dict[str, Any],
        outcome: dict[str, Any],
    ) -> None:
        """
        Record the outcome of a decision with the parameters used.

        Parameters
        ----------
        decision_id : str
            Unique decision identifier
        parameters_used : dict
            Configuration parameters used for the decision
        outcome : dict
            Outcome metrics (confidence, return, success, etc.)
        """
        timestamp = datetime.now(UTC)

        for param_name, param_value in parameters_used.items():
            if param_name not in self.TUNABLE_PARAMETERS:
                continue

            # Initialize tracking if needed
            if param_name not in self.parameter_performance:
                self.parameter_performance[param_name] = {}

            if param_value not in self.parameter_performance[param_name]:
                self.parameter_performance[param_name][
                    param_value
                ] = ParameterPerformance(
                    parameter_name=param_name,
                    parameter_value=param_value,
                )

            # Update performance metrics
            perf = self.parameter_performance[param_name][param_value]
            perf.sample_count += 1
            perf.timestamps.append(timestamp)

            # Update rolling averages
            alpha = 0.1  # Exponential moving average factor

            success = outcome.get("success", False)
            perf.success_rate = (
                alpha * (1.0 if success else 0.0) + (1 - alpha) * perf.success_rate
            )

            confidence = outcome.get("confidence", 0.0)
            perf.avg_confidence = alpha * confidence + (1 - alpha) * perf.avg_confidence

            returns = outcome.get("realized_return", 0.0)
            perf.avg_return = alpha * returns + (1 - alpha) * perf.avg_return

            # Risk-adjusted return (simplified Sharpe-like metric)
            risk = outcome.get("risk_taken", 1.0)
            if risk > 0:
                risk_adj = returns / risk
                perf.risk_adjusted_return = (
                    alpha * risk_adj + (1 - alpha) * perf.risk_adjusted_return
                )

        # Save periodically
        if len(self.parameter_performance) > 0 and timestamp.minute % 5 == 0:
            self._save_history()

        logger.info(
            "Recorded decision outcome for tuning",
            extra={
                "decision_id": decision_id,
                "parameters": list(parameters_used.keys()),
                "outcome_success": outcome.get("success", False),
                "outcome_confidence": outcome.get("confidence", 0.0),
            },
        )

    def get_recommendations(
        self, current_config: dict[str, Any]
    ) -> list[TuningRecommendation]:
        """
        Get tuning recommendations based on performance data.

        Parameters
        ----------
        current_config : dict
            Current configuration values

        Returns
        -------
        List[TuningRecommendation]
            Sorted list of recommendations (best first)
        """
        recommendations = []
        cutoff_date = datetime.now(UTC) - timedelta(days=self.lookback_days)

        for param_name, _param_spec in self.TUNABLE_PARAMETERS.items():
            if param_name not in current_config:
                continue

            current_value = current_config[param_name]

            # Get performance data for different values
            if param_name not in self.parameter_performance:
                continue

            # Filter to recent data only
            recent_perfs = []
            for _value, perf in self.parameter_performance[param_name].items():
                recent_timestamps = [t for t in perf.timestamps if t > cutoff_date]
                if len(recent_timestamps) >= self.min_samples:
                    recent_perfs.append(perf)

            if len(recent_perfs) < 2:  # Need at least 2 values to compare
                continue

            # Find best performing value
            best_perf = max(recent_perfs, key=lambda p: self._calculate_score(p))
            current_perf = self.parameter_performance[param_name].get(current_value)

            if not current_perf or best_perf.parameter_value == current_value:
                continue

            # Calculate expected improvement
            current_score = self._calculate_score(current_perf)
            best_score = self._calculate_score(best_perf)
            improvement = (best_score - current_score) / max(abs(current_score), 0.01)

            if improvement > 0.05:  # At least 5% improvement
                # Generate recommendation
                rec = TuningRecommendation(
                    parameter_name=param_name,
                    current_value=current_value,
                    recommended_value=best_perf.parameter_value,
                    expected_improvement=improvement,
                    confidence=self._calculate_confidence(best_perf, current_perf),
                    reasoning=self._generate_reasoning(
                        param_name, current_perf, best_perf
                    ),
                    based_on_samples=best_perf.sample_count,
                )
                recommendations.append(rec)

        # Sort by expected improvement
        recommendations.sort(
            key=lambda r: r.expected_improvement * r.confidence, reverse=True
        )

        return recommendations

    def _calculate_score(self, perf: ParameterPerformance) -> float:
        """Calculate overall performance score for a parameter value."""
        # Weighted combination of metrics
        weights = {
            "success_rate": 0.3,
            "avg_confidence": 0.2,
            "avg_return": 0.3,
            "risk_adjusted_return": 0.2,
        }

        score = (
            weights["success_rate"] * perf.success_rate
            + weights["avg_confidence"] * perf.avg_confidence
            + weights["avg_return"] * min(perf.avg_return, 1.0)
            + weights["risk_adjusted_return"]
            * min(perf.risk_adjusted_return, 1.0)  # Cap at 100%
        )

        return score

    def _calculate_confidence(
        self,
        recommended: ParameterPerformance,
        current: ParameterPerformance,
    ) -> float:
        """Calculate confidence in the recommendation."""
        # Based on sample size and consistency
        sample_confidence = min(recommended.sample_count / 50.0, 1.0)

        # Based on score difference
        score_diff = abs(
            self._calculate_score(recommended) - self._calculate_score(current)
        )
        diff_confidence = min(score_diff * 2, 1.0)

        # Based on recency
        if recommended.timestamps:
            days_old = (datetime.now(UTC) - max(recommended.timestamps)).days
            recency_confidence = max(0, 1.0 - days_old / 30.0)
        else:
            recency_confidence = 0.5

        # Combined confidence
        confidence = (
            sample_confidence * 0.4 + diff_confidence * 0.4 + recency_confidence * 0.2
        )

        return min(max(confidence, 0.0), 1.0)

    def _generate_reasoning(
        self,
        param_name: str,
        current: ParameterPerformance,
        recommended: ParameterPerformance,
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        reasons = []

        # Success rate comparison
        if recommended.success_rate > current.success_rate:
            improvement = (recommended.success_rate - current.success_rate) * 100
            reasons.append(f"{improvement:.0f}% higher success rate")

        # Return comparison
        if recommended.avg_return > current.avg_return:
            improvement = (recommended.avg_return - current.avg_return) * 100
            reasons.append(f"{improvement:.1f}% better returns")

        # Risk-adjusted comparison
        if recommended.risk_adjusted_return > current.risk_adjusted_return:
            reasons.append("better risk-adjusted performance")

        # Parameter-specific insights
        if param_name == "target_delta":
            if recommended.parameter_value < current.parameter_value:
                reasons.append("more conservative strikes reduce assignment risk")
            else:
                reasons.append("higher premium collection with acceptable risk")
        elif param_name == "target_dte":
            if recommended.parameter_value > current.parameter_value:
                reasons.append("longer expiration provides more time premium")
            else:
                reasons.append("faster theta decay with manageable risk")

        return f"Based on {recommended.sample_count} recent decisions: " + ", ".join(
            reasons
        )

    def apply_recommendations(
        self,
        recommendations: list[TuningRecommendation],
        current_config: dict[str, Any],
        max_changes: int = 2,
        min_confidence: float = 0.7,
    ) -> dict[str, Any]:
        """
        Apply recommended changes to configuration.

        Parameters
        ----------
        recommendations : List[TuningRecommendation]
            Recommendations to consider
        current_config : dict
            Current configuration
        max_changes : int
            Maximum number of changes to apply at once
        min_confidence : float
            Minimum confidence required to apply change

        Returns
        -------
        dict
            Updated configuration
        """
        updated_config = current_config.copy()
        changes_applied = 0

        for rec in recommendations:
            if changes_applied >= max_changes:
                break

            if rec.confidence < min_confidence:
                continue

            # Apply the change
            old_value = updated_config.get(rec.parameter_name)
            updated_config[rec.parameter_name] = rec.recommended_value
            changes_applied += 1

            logger.info(
                f"Auto-tuning {rec.parameter_name}: {old_value} -> {rec.recommended_value}",
                extra={
                    "parameter": rec.parameter_name,
                    "old_value": old_value,
                    "new_value": rec.recommended_value,
                    "expected_improvement": rec.expected_improvement,
                    "confidence": rec.confidence,
                    "reasoning": rec.reasoning,
                },
            )

        return updated_config

    def get_parameter_trends(self, param_name: str) -> dict[str, Any]:
        """Get performance trends for a specific parameter."""
        if param_name not in self.parameter_performance:
            return {"error": "No data for parameter"}

        trends = {
            "parameter": param_name,
            "values_tested": {},
            "current_best": None,
            "trend_direction": "stable",
        }

        # Analyze each value
        perfs = []
        for value, perf in self.parameter_performance[param_name].items():
            if perf.sample_count >= 5:  # Minimum samples
                score = self._calculate_score(perf)
                trends["values_tested"][str(value)] = {
                    "score": score,
                    "samples": perf.sample_count,
                    "success_rate": perf.success_rate,
                    "avg_return": perf.avg_return,
                }
                perfs.append((value, score))

        if perfs:
            # Find best value
            best_value, best_score = max(perfs, key=lambda x: x[1])
            trends["current_best"] = best_value

            # Determine trend
            if len(perfs) >= 3:
                values = [p[0] for p in sorted(perfs)]
                scores = [p[1] for p in sorted(perfs)]

                # Simple linear regression to find trend
                if isinstance(values[0], int | float):
                    x = np.array(values)
                    y = np.array(scores)
                    if len(x) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        if abs(slope) > 0.01:
                            trends["trend_direction"] = (
                                "increasing" if slope > 0 else "decreasing"
                            )

        return trends

    def _save_history(self) -> None:
        """Save performance history to file."""
        try:
            history = {
                "last_updated": datetime.now(UTC).isoformat(),
                "parameter_performance": {},
            }

            # Convert to serializable format
            for param_name, perfs in self.parameter_performance.items():
                history["parameter_performance"][param_name] = {}
                for value, perf in perfs.items():
                    history["parameter_performance"][param_name][str(value)] = {
                        "sample_count": perf.sample_count,
                        "success_rate": perf.success_rate,
                        "avg_confidence": perf.avg_confidence,
                        "avg_return": perf.avg_return,
                        "risk_adjusted_return": perf.risk_adjusted_return,
                        "last_timestamp": (
                            max(perf.timestamps).isoformat()
                            if perf.timestamps
                            else None
                        ),
                    }

            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to save tuning history: {e}")

    def _load_history(self) -> None:
        """Load performance history from file."""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file) as f:
                history = json.load(f)

            # Convert back to objects
            for param_name, perfs in history.get("parameter_performance", {}).items():
                self.parameter_performance[param_name] = {}

                for value_str, perf_data in perfs.items():
                    # Convert value back to appropriate type
                    param_spec = self.TUNABLE_PARAMETERS.get(param_name, {})
                    if param_spec.get("type") == "float":
                        value = float(value_str)
                    elif param_spec.get("type") == "int":
                        value = int(value_str)
                    else:
                        value = value_str

                    perf = ParameterPerformance(
                        parameter_name=param_name,
                        parameter_value=value,
                        sample_count=perf_data["sample_count"],
                        success_rate=perf_data["success_rate"],
                        avg_confidence=perf_data["avg_confidence"],
                        avg_return=perf_data["avg_return"],
                        risk_adjusted_return=perf_data["risk_adjusted_return"],
                    )

                    # Restore approximate timestamps
                    if perf_data.get("last_timestamp"):
                        last_ts = datetime.fromisoformat(perf_data["last_timestamp"])
                        # Create synthetic timestamp history
                        for i in range(min(perf.sample_count, 10)):
                            ts = last_ts - timedelta(days=i)
                            perf.timestamps.append(ts)

                    self.parameter_performance[param_name][value] = perf

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to load tuning history: {e}")


# Global auto-tuner instance
_auto_tuner: ConfigAutoTuner | None = None


def get_auto_tuner() -> ConfigAutoTuner:
    """Get or create global auto-tuner instance."""
    global _auto_tuner
    if _auto_tuner is None:
        _auto_tuner = ConfigAutoTuner()
    return _auto_tuner

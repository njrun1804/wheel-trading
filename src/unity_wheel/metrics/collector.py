"""Metrics collection and decision quality tracking for autonomous operation."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..risk.analytics import RiskMetrics

logger = get_logger(__name__)


@dataclass
class DecisionMetrics:
    """Metrics for a single decision."""

    decision_id: str
    timestamp: datetime
    action: str
    confidence: float
    expected_return: float
    actual_return: Optional[float] = None
    risk_taken: float = 0.0
    execution_time_ms: float = 0.0
    features_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def return_error(self) -> Optional[float]:
        """Calculate prediction error if actual return available."""
        if self.actual_return is None:
            return None
        return abs(self.actual_return - self.expected_return)

    @property
    def confidence_calibration(self) -> Optional[float]:
        """
        Measure how well calibrated confidence was.
        Perfect calibration: high confidence = low error.
        """
        if self.return_error is None:
            return None

        # Normalize error to [0, 1] range
        normalized_error = min(self.return_error / abs(self.expected_return + 1e-6), 1.0)

        # Calibration score: 1.0 is perfect
        # High confidence (0.9) with low error (0.1) = good (0.8)
        # Low confidence (0.3) with high error (0.7) = good (0.6)
        # High confidence (0.9) with high error (0.7) = bad (0.2)
        return 1.0 - abs(self.confidence - (1.0 - normalized_error))


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""

    total_decisions: int = 0
    successful_decisions: int = 0
    average_confidence: float = 0.0
    average_execution_time_ms: float = 0.0
    total_expected_return: float = 0.0
    total_actual_return: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    average_return_error: float = 0.0
    confidence_calibration_score: float = 0.0
    decisions_per_hour: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_decisions": self.total_decisions,
            "success_rate": f"{self.win_rate:.1%}",
            "avg_confidence": f"{self.average_confidence:.1%}",
            "avg_execution_ms": round(self.average_execution_time_ms, 1),
            "expected_return": f"{self.total_expected_return:.2%}",
            "actual_return": f"{self.total_actual_return:.2%}",
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "avg_error": f"{self.average_return_error:.2%}",
            "confidence_calibration": round(self.confidence_calibration_score, 2),
            "decisions_per_hour": round(self.decisions_per_hour, 1),
        }


class MetricsCollector:
    """
    Collects and analyzes decision metrics for continuous improvement.
    Tracks decision quality, performance, and provides insights.
    """

    def __init__(
        self,
        window_size: int = 1000,
        persistence_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.persistence_path = persistence_path

        # Recent decisions window
        self.recent_decisions: Deque[DecisionMetrics] = deque(maxlen=window_size)

        # Historical aggregates
        self.hourly_metrics: Dict[str, PerformanceMetrics] = {}
        self.daily_metrics: Dict[str, PerformanceMetrics] = {}

        # Feature importance tracking
        self.feature_importance: Dict[str, float] = defaultdict(float)
        self.feature_usage: Dict[str, int] = defaultdict(int)

        # Action-specific metrics
        self.action_metrics: Dict[str, List[float]] = defaultdict(list)

        # Function timing metrics
        self.function_timings: Dict[str, List[float]] = defaultdict(list)

        # Cache statistics
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.cache_evictions: int = 0

        # Risk metrics history
        self.risk_history: Dict[str, List[float]] = defaultdict(list)

        # Load historical data if available
        if persistence_path and persistence_path.exists():
            self._load_from_disk()

    def record_decision(
        self,
        decision_id: str,
        action: str,
        confidence: float,
        expected_return: float,
        risk_taken: float = 0.0,
        execution_time_ms: float = 0.0,
        features_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a new decision."""
        decision = DecisionMetrics(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            action=action,
            confidence=confidence,
            expected_return=expected_return,
            risk_taken=risk_taken,
            execution_time_ms=execution_time_ms,
            features_used=features_used or [],
            metadata=metadata or {},
        )

        self.recent_decisions.append(decision)

        # Update feature usage
        for feature in decision.features_used:
            self.feature_usage[feature] += 1

        # Log high-confidence decisions
        if confidence > 0.8:
            logger.info(
                f"High confidence decision: {action}",
                extra={
                    "decision_id": decision_id,
                    "confidence": confidence,
                    "expected_return": expected_return,
                    "features": features_used,
                },
            )

    def update_decision_outcome(
        self,
        decision_id: str,
        actual_return: float,
    ) -> None:
        """Update decision with actual outcome."""
        # Find decision in recent history
        for decision in self.recent_decisions:
            if decision.decision_id == decision_id:
                decision.actual_return = actual_return

                # Update feature importance based on prediction accuracy
                error = decision.return_error
                if error is not None:
                    # Features that led to accurate predictions get higher importance
                    importance_delta = 1.0 / (1.0 + error)
                    for feature in decision.features_used:
                        self.feature_importance[feature] += importance_delta

                # Track action-specific performance
                self.action_metrics[decision.action].append(actual_return)

                logger.info(
                    f"Updated decision outcome",
                    extra={
                        "decision_id": decision_id,
                        "expected": decision.expected_return,
                        "actual": actual_return,
                        "error": error,
                    },
                )
                break

    def record_function_timing(self, operation: str, duration_ms: float) -> None:
        """Record execution time for a function."""
        self.function_timings[operation].append(duration_ms)

    def record_cache_hit(self) -> None:
        """Increment cache hit counter."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Increment cache miss counter."""
        self.cache_misses += 1

    def record_cache_eviction(self) -> None:
        """Increment cache eviction counter."""
        self.cache_evictions += 1

    def record_risk_metrics(self, metrics: "RiskMetrics") -> None:
        """Store risk metrics for distribution analysis."""
        for name, value in metrics.to_dict().items():
            if isinstance(value, (int, float)):
                self.risk_history[name].append(float(value))

    def get_function_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated timing statistics for recorded functions."""
        stats = {}
        for op, times in self.function_timings.items():
            arr = np.array(times)
            stats[op] = {
                "count": len(times),
                "avg_ms": float(np.mean(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
            }
        return stats

    def get_cache_summary(self) -> Dict[str, float]:
        """Return cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total else 0.0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "evictions": self.cache_evictions,
            "hit_rate": hit_rate,
        }

    def get_risk_distribution(self) -> Dict[str, Dict[str, float]]:
        """Return distribution stats for recorded risk metrics."""
        summary = {}
        for name, values in self.risk_history.items():
            arr = np.array(values)
            summary[name] = {
                "avg": float(np.mean(arr)),
                "p95": float(np.percentile(arr, 95)),
            }
        return summary

    def get_performance_summary(self, hours: Optional[int] = None) -> PerformanceMetrics:
        """Get performance summary for recent period."""
        # Filter decisions by time if specified
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            decisions = [d for d in self.recent_decisions if d.timestamp >= cutoff]
        else:
            decisions = list(self.recent_decisions)

        if not decisions:
            return PerformanceMetrics()

        # Calculate metrics
        total = len(decisions)
        with_outcomes = [d for d in decisions if d.actual_return is not None]

        if with_outcomes:
            returns = [d.actual_return for d in with_outcomes]
            successful = sum(1 for r in returns if r > 0)

            # Risk-adjusted returns
            returns_array = np.array(returns)
            sharpe = (
                np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                if np.std(returns_array) > 0
                else 0.0
            )

            # Calibration
            calibrations = [
                d.confidence_calibration
                for d in with_outcomes
                if d.confidence_calibration is not None
            ]
            avg_calibration = np.mean(calibrations) if calibrations else 0.0

        else:
            successful = 0
            sharpe = 0.0
            avg_calibration = 0.0

        # Time metrics
        if len(decisions) >= 2:
            time_span = (decisions[-1].timestamp - decisions[0].timestamp).total_seconds() / 3600
            decisions_per_hour = total / time_span if time_span > 0 else 0.0
        else:
            decisions_per_hour = 0.0

        return PerformanceMetrics(
            total_decisions=total,
            successful_decisions=successful,
            average_confidence=np.mean([d.confidence for d in decisions]),
            average_execution_time_ms=np.mean([d.execution_time_ms for d in decisions]),
            total_expected_return=sum(d.expected_return for d in decisions),
            total_actual_return=sum(d.actual_return for d in with_outcomes),
            sharpe_ratio=sharpe,
            win_rate=successful / len(with_outcomes) if with_outcomes else 0.0,
            average_return_error=(
                np.mean([d.return_error for d in with_outcomes if d.return_error is not None])
                if with_outcomes
                else 0.0
            ),
            confidence_calibration_score=avg_calibration,
            decisions_per_hour=decisions_per_hour,
        )

    def get_feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get most important features."""
        # Normalize by usage count
        normalized_importance = {
            feature: importance / max(self.feature_usage[feature], 1)
            for feature, importance in self.feature_importance.items()
        }

        # Sort by importance
        sorted_features = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)

        return sorted_features[:top_n]

    def get_action_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by action type."""
        analysis = {}

        for action, returns in self.action_metrics.items():
            if returns:
                returns_array = np.array(returns)
                analysis[action] = {
                    "count": len(returns),
                    "avg_return": float(np.mean(returns_array)),
                    "std_return": float(np.std(returns_array)),
                    "win_rate": float(np.sum(returns_array > 0) / len(returns_array)),
                    "max_return": float(np.max(returns_array)),
                    "min_return": float(np.min(returns_array)),
                }

        return analysis

    def identify_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in decision making."""
        patterns = []

        # Pattern 1: Confidence vs Success Rate
        decisions_by_confidence = defaultdict(list)
        for d in self.recent_decisions:
            if d.actual_return is not None:
                bucket = int(d.confidence * 10) / 10  # Round to nearest 0.1
                decisions_by_confidence[bucket].append(d.actual_return > 0)

        for confidence, outcomes in decisions_by_confidence.items():
            if len(outcomes) >= 10:  # Minimum sample size
                success_rate = sum(outcomes) / len(outcomes)
                expected_rate = confidence  # Ideally, confidence should match success rate

                if abs(success_rate - expected_rate) > 0.2:
                    patterns.append(
                        {
                            "type": "confidence_miscalibration",
                            "confidence_level": confidence,
                            "actual_success_rate": success_rate,
                            "expected_success_rate": expected_rate,
                            "sample_size": len(outcomes),
                            "recommendation": (
                                "Increase confidence"
                                if success_rate > expected_rate
                                else "Decrease confidence"
                            ),
                        }
                    )

        # Pattern 2: Time of day effects
        decisions_by_hour = defaultdict(list)
        for d in self.recent_decisions:
            if d.actual_return is not None:
                hour = d.timestamp.hour
                decisions_by_hour[hour].append(d.actual_return)

        for hour, returns in decisions_by_hour.items():
            if len(returns) >= 5:
                avg_return = np.mean(returns)
                overall_avg = np.mean(
                    [r for returns_list in decisions_by_hour.values() for r in returns_list]
                )

                if abs(avg_return - overall_avg) > 0.02:  # 2% difference
                    patterns.append(
                        {
                            "type": "time_of_day_bias",
                            "hour": hour,
                            "avg_return": avg_return,
                            "overall_avg": overall_avg,
                            "sample_size": len(returns),
                            "recommendation": (
                                f"{'Increase' if avg_return > overall_avg else 'Decrease'} "
                                f"activity at hour {hour}"
                            ),
                        }
                    )

        return patterns

    def generate_report(self) -> str:
        """Generate comprehensive metrics report."""
        lines = [
            "=" * 60,
            "DECISION METRICS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
        ]

        # Overall performance
        perf = self.get_performance_summary()
        lines.extend(
            [
                "OVERALL PERFORMANCE:",
                "-" * 30,
                f"Total Decisions: {perf.total_decisions}",
                f"Success Rate: {perf.win_rate:.1%}",
                f"Average Confidence: {perf.average_confidence:.1%}",
                f"Sharpe Ratio: {perf.sharpe_ratio:.2f}",
                f"Decisions/Hour: {perf.decisions_per_hour:.1f}",
                f"Avg Execution Time: {perf.average_execution_time_ms:.1f}ms",
                "",
            ]
        )

        # Confidence calibration
        lines.extend(
            [
                "CONFIDENCE CALIBRATION:",
                "-" * 30,
                f"Calibration Score: {perf.confidence_calibration_score:.2f} (1.0 = perfect)",
                f"Average Prediction Error: {perf.average_return_error:.2%}",
                "",
            ]
        )

        # Feature importance
        top_features = self.get_feature_importance(5)
        if top_features:
            lines.extend(
                [
                    "TOP FEATURES:",
                    "-" * 30,
                ]
            )
            for feature, importance in top_features:
                usage = self.feature_usage[feature]
                lines.append(f"  {feature}: {importance:.3f} (used {usage} times)")
            lines.append("")

        # Action analysis
        action_analysis = self.get_action_analysis()
        if action_analysis:
            lines.extend(
                [
                    "ACTION ANALYSIS:",
                    "-" * 30,
                ]
            )
            for action, stats in action_analysis.items():
                lines.extend(
                    [
                        f"  {action}:",
                        f"    Count: {stats['count']}",
                        f"    Win Rate: {stats['win_rate']:.1%}",
                        f"    Avg Return: {stats['avg_return']:.2%}",
                        f"    Volatility: {stats['std_return']:.2%}",
                    ]
                )
            lines.append("")

        # Patterns
        patterns = self.identify_patterns()
        if patterns:
            lines.extend(
                [
                    "IDENTIFIED PATTERNS:",
                    "-" * 30,
                ]
            )
            for i, pattern in enumerate(patterns[:5], 1):
                lines.extend(
                    [
                        f"  {i}. {pattern['type']}:",
                        f"     {pattern['recommendation']}",
                    ]
                )
            lines.append("")

        # Recent performance trend
        recent_24h = self.get_performance_summary(24)
        recent_1h = self.get_performance_summary(1)

        lines.extend(
            [
                "RECENT TRENDS:",
                "-" * 30,
                f"Last 1h: {recent_1h.total_decisions} decisions, {recent_1h.win_rate:.1%} success",
                f"Last 24h: {recent_24h.total_decisions} decisions, {recent_24h.win_rate:.1%} success",
                "",
                "=" * 60,
            ]
        )

        return "\n".join(lines)

    def _save_to_disk(self) -> None:
        """Save metrics to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "recent_decisions": [
                    {
                        "decision_id": d.decision_id,
                        "timestamp": d.timestamp.isoformat(),
                        "action": d.action,
                        "confidence": d.confidence,
                        "expected_return": d.expected_return,
                        "actual_return": d.actual_return,
                        "risk_taken": d.risk_taken,
                        "execution_time_ms": d.execution_time_ms,
                        "features_used": d.features_used,
                        "metadata": d.metadata,
                    }
                    for d in self.recent_decisions
                ],
                "feature_importance": dict(self.feature_importance),
                "feature_usage": dict(self.feature_usage),
                "action_metrics": dict(self.action_metrics),
            }

            with open(self.persistence_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.recent_decisions)} decisions to disk")

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to save metrics: {e}")

    def _load_from_disk(self) -> None:
        """Load metrics from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, "r") as f:
                data = json.load(f)

            # Restore recent decisions
            for d in data.get("recent_decisions", []):
                decision = DecisionMetrics(
                    decision_id=d["decision_id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    action=d["action"],
                    confidence=d["confidence"],
                    expected_return=d["expected_return"],
                    actual_return=d.get("actual_return"),
                    risk_taken=d.get("risk_taken", 0.0),
                    execution_time_ms=d.get("execution_time_ms", 0.0),
                    features_used=d.get("features_used", []),
                    metadata=d.get("metadata", {}),
                )
                self.recent_decisions.append(decision)

            # Restore aggregates
            self.feature_importance = defaultdict(float, data.get("feature_importance", {}))
            self.feature_usage = defaultdict(int, data.get("feature_usage", {}))
            self.action_metrics = defaultdict(list, data.get("action_metrics", {}))

            logger.info(f"Loaded {len(self.recent_decisions)} historical decisions")

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Failed to load metrics: {e}")


# Global metrics collector
metrics_collector = MetricsCollector()

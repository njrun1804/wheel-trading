"""
Simple performance tracker to learn from actual results.
Compares predictions to outcomes for continuous improvement.
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class TradeOutcome:
    """Record of a trade recommendation and its outcome."""

    timestamp: datetime
    action: str
    strike: float
    dte: int
    delta_target: float
    kelly_fraction: float
    confidence: float
    predicted_return: float
    actual_return: Optional[float] = None
    days_held: Optional[int] = None
    exit_reason: Optional[str] = None
    market_regime: Optional[str] = None
    warnings_count: int = 0


class PerformanceTracker:
    """Track predictions vs outcomes to improve the system."""

    def __init__(self, db_path: str = "~/.wheel_trading/performance.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the performance database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    action TEXT,
                    strike REAL,
                    dte INTEGER,
                    delta_target REAL,
                    kelly_fraction REAL,
                    confidence REAL,
                    predicted_return REAL,
                    actual_return REAL,
                    days_held INTEGER,
                    exit_reason TEXT,
                    market_regime TEXT,
                    warnings_count INTEGER,
                    metadata TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)
            """
            )

    def record_recommendation(self, recommendation: Dict) -> int:
        """Record a new trade recommendation."""
        trade = TradeOutcome(
            timestamp=datetime.now(),
            action=recommendation["action"],
            strike=recommendation.get("strike", 0),
            dte=recommendation.get("dte_target", 0),
            delta_target=recommendation.get("delta_target", 0),
            kelly_fraction=recommendation.get("kelly_fraction", 0),
            confidence=recommendation.get("confidence", 0),
            predicted_return=recommendation.get("expected_return", 0),
            market_regime=recommendation.get("market_regime", ""),
            warnings_count=len(recommendation.get("warnings", [])),
        )

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (
                    timestamp, action, strike, dte, delta_target,
                    kelly_fraction, confidence, predicted_return,
                    market_regime, warnings_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.timestamp,
                    trade.action,
                    trade.strike,
                    trade.dte,
                    trade.delta_target,
                    trade.kelly_fraction,
                    trade.confidence,
                    trade.predicted_return,
                    trade.market_regime,
                    trade.warnings_count,
                    json.dumps(recommendation),
                ),
            )

        logger.info(f"Recorded {trade.action} recommendation", trade_id=cursor.lastrowid)
        return cursor.lastrowid

    def update_outcome(self, trade_id: int, actual_return: float, days_held: int, exit_reason: str):
        """Update a trade with its actual outcome."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                UPDATE trades
                SET actual_return = ?, days_held = ?, exit_reason = ?
                WHERE id = ?
            """,
                (actual_return, days_held, exit_reason, trade_id),
            )

        logger.info(
            f"Updated trade {trade_id} outcome",
            actual_return=actual_return,
            exit_reason=exit_reason,
        )

    def get_performance_stats(self, days: int = 90) -> Dict:
        """Get performance statistics for recent trades."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Completed trades only
            completed = conn.execute(
                """
                SELECT * FROM trades
                WHERE actual_return IS NOT NULL
                AND timestamp > datetime('now', '-{} days')
            """.format(
                    days
                )
            ).fetchall()

        if not completed:
            return {"message": "No completed trades yet"}

        # Calculate statistics
        stats = {
            "total_trades": len(completed),
            "win_rate": sum(1 for t in completed if t[9] > 0) / len(completed),
            "avg_return": sum(t[9] for t in completed) / len(completed),
            "avg_days_held": sum(t[10] for t in completed) / len(completed),
            "prediction_accuracy": [],
            "confidence_calibration": {},
        }

        # Prediction accuracy by confidence bucket
        buckets = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in buckets:
            bucket_trades = [t for t in completed if low <= t[7] < high]
            if bucket_trades:
                actual_win_rate = sum(1 for t in bucket_trades if t[9] > 0) / len(bucket_trades)
                expected_win_rate = sum(t[7] for t in bucket_trades) / len(bucket_trades)
                stats["confidence_calibration"][f"{low:.0%}-{high:.0%}"] = {
                    "expected": expected_win_rate,
                    "actual": actual_win_rate,
                    "trades": len(bucket_trades),
                }

        return stats

    def get_regime_performance(self) -> Dict[str, Dict]:
        """Analyze performance by market regime."""
        with sqlite3.connect(str(self.db_path)) as conn:
            regimes = conn.execute(
                """
                SELECT market_regime,
                       COUNT(*) as trades,
                       AVG(actual_return) as avg_return,
                       AVG(CASE WHEN actual_return > 0 THEN 1.0 ELSE 0.0 END) as win_rate
                FROM trades
                WHERE actual_return IS NOT NULL
                GROUP BY market_regime
            """
            ).fetchall()

        return {
            regime[0]: {"trades": regime[1], "avg_return": regime[2], "win_rate": regime[3]}
            for regime in regimes
        }

    def suggest_improvements(self) -> List[str]:
        """Suggest parameter adjustments based on performance."""
        suggestions = []
        stats = self.get_performance_stats()

        if isinstance(stats, dict) and "win_rate" in stats:
            # Check confidence calibration
            for bucket, data in stats.get("confidence_calibration", {}).items():
                if abs(data["expected"] - data["actual"]) > 0.15:
                    if data["actual"] < data["expected"]:
                        suggestions.append(
                            f"Confidence too high in {bucket} range. "
                            f"Expected {data['expected']:.1%} win rate, got {data['actual']:.1%}. "
                            "Consider more conservative parameters."
                        )
                    else:
                        suggestions.append(
                            f"Confidence too low in {bucket} range. "
                            f"Expected {data['expected']:.1%} win rate, got {data['actual']:.1%}. "
                            "Consider more aggressive parameters."
                        )

            # Check regime performance
            regime_perf = self.get_regime_performance()
            for regime, perf in regime_perf.items():
                if perf["trades"] > 5 and perf["win_rate"] < 0.6:
                    suggestions.append(
                        f"Poor performance in {regime} regime ({perf['win_rate']:.1%} win rate). "
                        "Consider avoiding trades or reducing size in this regime."
                    )

        return suggestions

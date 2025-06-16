"""
from __future__ import annotations

Enhanced performance tracker for Unity wheel trading.
Tracks Unity-specific patterns and learns from actual results.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    actual_return: float | None = None
    days_held: int | None = None
    exit_reason: str | None = None
    market_regime: str | None = None
    warnings_count: int = 0
    # Unity-specific fields
    unity_price: float | None = None
    unity_volatility: float | None = None
    iv_rank: float | None = None
    near_earnings: bool | None = None
    days_to_earnings: int | None = None
    was_assigned: bool | None = None
    portfolio_drawdown: float | None = None
    volatility_tier: str | None = None  # 'low', 'normal', 'high', 'extreme'


@dataclass
class UnityOutcome:
    """Actual outcome for Unity wheel trades."""

    recommendation_id: int
    actual_pnl: float
    was_assigned: bool
    days_held: int
    exit_reason: str  # 'expired', 'rolled', 'assigned', 'closed_early'
    final_stock_price: float | None = None
    assignment_loss: float | None = None  # If assigned, unrealized loss


class PerformanceTracker:
    """Track predictions vs outcomes to improve the system."""

    def __init__(self, db_path: str = "data/wheel_trading_optimized.duckdb"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_vol_regime(self, volatility: float) -> str:
        """Determine volatility regime based on Unity's volatility."""
        if volatility < 0.40:
            return "low"
        elif volatility < 0.60:
            return "normal"
        elif volatility < 0.80:
            return "high"
        else:
            return "extreme"

    def near_earnings_window(self, days_to_earnings: int | None = None) -> bool:
        """Check if within earnings window (7 days)."""
        if days_to_earnings is None:
            return False
        return days_to_earnings <= 7

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
                    unity_price REAL,
                    unity_volatility REAL,
                    iv_rank REAL,
                    near_earnings BOOLEAN,
                    days_to_earnings INTEGER,
                    was_assigned BOOLEAN,
                    portfolio_drawdown REAL,
                    volatility_tier TEXT,
                    metadata TEXT
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_volatility_tier ON trades(volatility_tier)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_near_earnings ON trades(near_earnings)
            """
            )

    def record_recommendation(self, recommendation: dict) -> int:
        """Record a new trade recommendation."""
        # Extract Unity-specific fields
        unity_price = recommendation.get("current_price", 0)
        unity_volatility = recommendation.get("volatility", 0)
        iv_rank = recommendation.get("iv_rank", 0)
        days_to_earnings = recommendation.get("days_to_earnings")
        portfolio_drawdown = recommendation.get("portfolio_drawdown", 0)

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
            # Unity-specific
            unity_price=unity_price,
            unity_volatility=unity_volatility,
            iv_rank=iv_rank,
            near_earnings=self.near_earnings_window(days_to_earnings),
            days_to_earnings=days_to_earnings,
            portfolio_drawdown=portfolio_drawdown,
            volatility_tier=self.get_vol_regime(unity_volatility),
        )

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (
                    timestamp, action, strike, dte, delta_target,
                    kelly_fraction, confidence, predicted_return,
                    market_regime, warnings_count,
                    unity_price, unity_volatility, iv_rank,
                    near_earnings, days_to_earnings, portfolio_drawdown,
                    volatility_tier, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    trade.unity_price,
                    trade.unity_volatility,
                    trade.iv_rank,
                    trade.near_earnings,
                    trade.days_to_earnings,
                    trade.portfolio_drawdown,
                    trade.volatility_tier,
                    json.dumps(recommendation),
                ),
            )

        logger.info(
            f"Recorded {trade.action} recommendation", trade_id=cursor.lastrowid
        )
        return cursor.lastrowid

    def update_outcome(
        self,
        trade_id: int,
        actual_return: float,
        days_held: int,
        exit_reason: str,
        was_assigned: bool = False,
    ):
        """Update a trade with its actual outcome."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                UPDATE trades
                SET actual_return = ?, days_held = ?, exit_reason = ?, was_assigned = ?
                WHERE id = ?
            """,
                (actual_return, days_held, exit_reason, was_assigned, trade_id),
            )

        logger.info(
            f"Updated trade {trade_id} outcome",
            actual_return=actual_return,
            exit_reason=exit_reason,
            was_assigned=was_assigned,
        )

    def track_unity_recommendation(self, rec: dict, actual_outcome: UnityOutcome):
        """Track Unity-specific patterns with outcomes."""
        # Get current market conditions for tracking
        volatility = rec.get("volatility", 0)
        days_to_earnings = rec.get("days_to_earnings")

        # Record the outcome with Unity-specific patterns
        self.record_outcome(
            predicted_return=rec.get("expected_return", 0),
            actual_return=actual_outcome.actual_pnl,
            was_assigned=actual_outcome.was_assigned,
            volatility_regime=self.get_vol_regime(volatility),
            near_earnings=self.near_earnings_window(days_to_earnings),
        )

        # Update the database with the outcome
        self.update_outcome(
            trade_id=actual_outcome.recommendation_id,
            actual_return=actual_outcome.actual_pnl,
            days_held=actual_outcome.days_held,
            exit_reason=actual_outcome.exit_reason,
            was_assigned=actual_outcome.was_assigned,
        )

        # Log Unity-specific insights
        if actual_outcome.was_assigned:
            logger.info(
                "Unity position assigned",
                strike=rec.get("strike"),
                final_price=actual_outcome.final_stock_price,
                assignment_loss=actual_outcome.assignment_loss,
                volatility_tier=self.get_vol_regime(volatility),
            )

        return actual_outcome.recommendation_id

    def record_outcome(
        self,
        predicted_return: float,
        actual_return: float,
        was_assigned: bool,
        volatility_regime: str,
        near_earnings: bool,
    ):
        """Record outcome for analysis (in-memory tracking)."""
        # This is for immediate logging/analysis
        # The persistent storage happens via update_outcome
        outcome_data = {
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "was_assigned": was_assigned,
            "volatility_regime": volatility_regime,
            "near_earnings": near_earnings,
            "prediction_error": abs(predicted_return - actual_return),
            "timestamp": datetime.now(),
        }

        logger.info("Unity outcome recorded", **outcome_data)

    def get_performance_stats(self, days: int = 90) -> dict:
        """Get performance statistics for recent trades."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Completed trades only
            completed = conn.execute(
                """
                SELECT * FROM trades
                WHERE actual_return IS NOT NULL
                AND timestamp > datetime('now', '-' || ? || ' days')
            """,
                (days,),
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
                actual_win_rate = sum(1 for t in bucket_trades if t[9] > 0) / len(
                    bucket_trades
                )
                expected_win_rate = sum(t[7] for t in bucket_trades) / len(
                    bucket_trades
                )
                stats["confidence_calibration"][f"{low:.0%}-{high:.0%}"] = {
                    "expected": expected_win_rate,
                    "actual": actual_win_rate,
                    "trades": len(bucket_trades),
                }

        return stats

    def get_regime_performance(self) -> dict[str, dict]:
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
            regime[0]: {
                "trades": regime[1],
                "avg_return": regime[2],
                "win_rate": regime[3],
            }
            for regime in regimes
        }

    def suggest_improvements(self) -> list[str]:
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

    def get_unity_volatility_performance(self) -> dict[str, dict]:
        """Analyze Unity performance by volatility tier."""
        with sqlite3.connect(str(self.db_path)) as conn:
            vol_tiers = conn.execute(
                """
                SELECT volatility_tier,
                       COUNT(*) as trades,
                       AVG(actual_return) as avg_return,
                       AVG(CASE WHEN actual_return > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                       SUM(CASE WHEN was_assigned THEN 1 ELSE 0 END) as assignments,
                       AVG(unity_volatility) as avg_volatility
                FROM trades
                WHERE actual_return IS NOT NULL
                AND volatility_tier IS NOT NULL
                GROUP BY volatility_tier
                ORDER BY CASE volatility_tier
                    WHEN 'low' THEN 1
                    WHEN 'normal' THEN 2
                    WHEN 'high' THEN 3
                    WHEN 'extreme' THEN 4
                END
            """
            ).fetchall()

        return {
            tier[0]: {
                "trades": tier[1],
                "avg_return": tier[2],
                "win_rate": tier[3],
                "assignment_rate": tier[4] / tier[1] if tier[1] > 0 else 0,
                "avg_volatility": tier[5],
            }
            for tier in vol_tiers
        }

    def get_earnings_impact(self) -> dict[str, dict]:
        """Analyze impact of earnings proximity on outcomes."""
        with sqlite3.connect(str(self.db_path)) as conn:
            earnings_data = conn.execute(
                """
                SELECT near_earnings,
                       COUNT(*) as trades,
                       AVG(actual_return) as avg_return,
                       AVG(CASE WHEN actual_return > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                       SUM(CASE WHEN was_assigned THEN 1 ELSE 0 END) as assignments,
                       AVG(ABS(actual_return - predicted_return)) as avg_prediction_error
                FROM trades
                WHERE actual_return IS NOT NULL
                GROUP BY near_earnings
            """
            ).fetchall()

        return {
            "near_earnings": {
                "trades": earnings_data[0][1]
                if earnings_data and earnings_data[0][0]
                else 0,
                "avg_return": earnings_data[0][2]
                if earnings_data and earnings_data[0][0]
                else 0,
                "win_rate": earnings_data[0][3]
                if earnings_data and earnings_data[0][0]
                else 0,
                "assignment_rate": (
                    earnings_data[0][4] / earnings_data[0][1]
                    if earnings_data and earnings_data[0][0] and earnings_data[0][1] > 0
                    else 0
                ),
                "avg_prediction_error": (
                    earnings_data[0][5] if earnings_data and earnings_data[0][0] else 0
                ),
            },
            "normal": {
                "trades": (
                    earnings_data[1][1]
                    if len(earnings_data) > 1
                    else earnings_data[0][1]
                    if earnings_data and not earnings_data[0][0]
                    else 0
                ),
                "avg_return": (
                    earnings_data[1][2]
                    if len(earnings_data) > 1
                    else earnings_data[0][2]
                    if earnings_data and not earnings_data[0][0]
                    else 0
                ),
                "win_rate": (
                    earnings_data[1][3]
                    if len(earnings_data) > 1
                    else earnings_data[0][3]
                    if earnings_data and not earnings_data[0][0]
                    else 0
                ),
                "assignment_rate": (
                    earnings_data[1][4] / earnings_data[1][1]
                    if len(earnings_data) > 1 and earnings_data[1][1] > 0
                    else (
                        earnings_data[0][4] / earnings_data[0][1]
                        if earnings_data
                        and not earnings_data[0][0]
                        and earnings_data[0][1] > 0
                        else 0
                    )
                ),
                "avg_prediction_error": (
                    earnings_data[1][5]
                    if len(earnings_data) > 1
                    else earnings_data[0][5]
                    if earnings_data and not earnings_data[0][0]
                    else 0
                ),
            },
        }

    def get_unity_insights(self) -> list[str]:
        """Generate Unity-specific trading insights."""
        insights = []

        # Volatility performance
        vol_perf = self.get_unity_volatility_performance()
        for tier, stats in vol_perf.items():
            if stats["trades"] >= 5:
                if tier == "extreme" and stats["win_rate"] < 0.4:
                    insights.append(
                        f"⚠️ Poor performance in extreme volatility (>{stats['avg_volatility']:.0%}). "
                        f"Consider skipping trades when Unity volatility exceeds 80%."
                    )
                elif tier == "low" and stats["assignment_rate"] > 0.5:
                    insights.append(
                        f"📊 High assignment rate ({stats['assignment_rate']:.0%}) in low volatility. "
                        f"Consider more conservative strikes when Unity volatility <40%."
                    )

        # Earnings impact
        earnings_impact = self.get_earnings_impact()
        if earnings_impact["near_earnings"]["trades"] >= 3:
            if earnings_impact["near_earnings"]["avg_prediction_error"] > 0.10:
                insights.append(
                    f"📅 High prediction error near earnings "
                    f"({earnings_impact['near_earnings']['avg_prediction_error']:.1%}). "
                    f"Unity's earnings volatility is difficult to predict."
                )

        # Assignment patterns
        with sqlite3.connect(str(self.db_path)) as conn:
            assignment_data = conn.execute(
                """
                SELECT AVG(unity_price - strike) / strike as avg_moneyness,
                       COUNT(*) as count
                FROM trades
                WHERE was_assigned = 1
                AND unity_price IS NOT NULL
                AND strike > 0
            """
            ).fetchone()

            if assignment_data and assignment_data[1] >= 5:
                avg_moneyness = assignment_data[0]
                if avg_moneyness < -0.05:
                    insights.append(
                        f"💡 Assignments typically occur {abs(avg_moneyness):.1%} ITM. "
                        f"Consider rolling positions when Unity approaches strike - 5%."
                    )

        return insights

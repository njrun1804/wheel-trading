"""Test enhanced Unity performance tracker functionality."""

import os
import tempfile
from datetime import datetime, timedelta

import pytest
from unity_wheel.analytics.performance_tracker import PerformanceTracker, UnityOutcome


class TestPerformanceTracker:
    """Test Unity-specific performance tracking."""

    @pytest.fixture
    def tracker(self):
        """Create a temporary performance tracker."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tracker = PerformanceTracker(db_path=tmp.name)
            yield tracker
            os.unlink(tmp.name)

    def test_volatility_regime_classification(self, tracker):
        """Test volatility regime determination."""
        assert tracker.get_vol_regime(0.30) == "low"
        assert tracker.get_vol_regime(0.50) == "normal"
        assert tracker.get_vol_regime(0.70) == "high"
        assert tracker.get_vol_regime(0.90) == "extreme"

    def test_earnings_window_detection(self, tracker):
        """Test earnings window detection."""
        assert tracker.near_earnings_window(5) is True
        assert tracker.near_earnings_window(7) is True
        assert tracker.near_earnings_window(8) is False
        assert tracker.near_earnings_window(None) is False

    def test_unity_recommendation_recording(self, tracker):
        """Test recording Unity-specific recommendations."""
        rec = {
            "action": "SELL_PUT",
            "strike": 30.0,
            "dte_target": 45,
            "delta_target": 0.30,
            "kelly_fraction": 0.25,
            "confidence": 0.85,
            "expected_return": 0.02,
            "current_price": 35.0,
            "volatility": 0.65,
            "iv_rank": 75.0,
            "days_to_earnings": 10,
            "portfolio_drawdown": -0.05,
            "warnings": [],
        }

        trade_id = tracker.record_recommendation(rec)
        assert trade_id == 1

    def test_unity_outcome_tracking(self, tracker):
        """Test tracking Unity-specific outcomes."""
        # First record a recommendation
        rec = {
            "action": "SELL_PUT",
            "strike": 30.0,
            "expected_return": 0.02,
            "volatility": 0.65,
            "days_to_earnings": 10,
        }
        trade_id = tracker.record_recommendation(rec)

        # Create an outcome
        outcome = UnityOutcome(
            recommendation_id=trade_id,
            actual_pnl=150.0,
            was_assigned=False,
            days_held=21,
            exit_reason="expired",
            final_stock_price=32.5,
        )

        # Track the outcome
        result_id = tracker.track_unity_recommendation(rec, outcome)
        assert result_id == trade_id

    def test_unity_performance_analysis(self, tracker):
        """Test Unity-specific performance analysis."""
        # Record multiple recommendations with different volatility regimes
        recommendations = [
            # Low volatility trades
            {
                "action": "SELL_PUT",
                "strike": 30.0,
                "expected_return": 0.015,
                "volatility": 0.35,
                "days_to_earnings": 20,
                "confidence": 0.90,
            },
            # High volatility trades
            {
                "action": "SELL_PUT",
                "strike": 28.0,
                "expected_return": 0.025,
                "volatility": 0.75,
                "days_to_earnings": 15,
                "confidence": 0.70,
            },
            # Near earnings trade
            {
                "action": "SELL_PUT",
                "strike": 32.0,
                "expected_return": 0.03,
                "volatility": 0.55,
                "days_to_earnings": 5,
                "confidence": 0.60,
            },
        ]

        # Record recommendations and outcomes
        for i, rec in enumerate(recommendations):
            trade_id = tracker.record_recommendation(rec)

            # Simulate different outcomes
            if i == 0:  # Low vol - expired worthless (profit)
                outcome = UnityOutcome(
                    recommendation_id=trade_id,
                    actual_pnl=150.0,
                    was_assigned=False,
                    days_held=45,
                    exit_reason="expired",
                )
            elif i == 1:  # High vol - assigned (loss)
                outcome = UnityOutcome(
                    recommendation_id=trade_id,
                    actual_pnl=-200.0,
                    was_assigned=True,
                    days_held=30,
                    exit_reason="assigned",
                    final_stock_price=26.0,
                    assignment_loss=-200.0,
                )
            else:  # Near earnings - closed early
                outcome = UnityOutcome(
                    recommendation_id=trade_id,
                    actual_pnl=50.0,
                    was_assigned=False,
                    days_held=4,
                    exit_reason="closed_early",
                )

            tracker.track_unity_recommendation(rec, outcome)

        # Test volatility performance analysis
        vol_perf = tracker.get_unity_volatility_performance()
        assert "low" in vol_perf
        assert "high" in vol_perf
        assert vol_perf["low"]["win_rate"] == 1.0  # Low vol trade was profitable
        assert vol_perf["high"]["win_rate"] == 0.0  # High vol trade was a loss

        # Test earnings impact analysis
        earnings_impact = tracker.get_earnings_impact()
        assert "near_earnings" in earnings_impact
        assert "normal" in earnings_impact

        # Test insights generation
        insights = tracker.get_unity_insights()
        assert isinstance(insights, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

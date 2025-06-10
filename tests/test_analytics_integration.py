"""
Tests for integrated analytics components.
Ensures autonomous operation and proper error handling.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.unity_wheel.analytics import (
    AnomalyDetector,
    DynamicOptimizer,
    EventImpactAnalyzer,
    IntegratedDecisionEngine,
    IVSurfaceAnalyzer,
    MarketState,
    SeasonalityDetector,
    WheelRecommendation,
)


class TestDynamicOptimizer:
    """Test dynamic parameter optimization."""

    def test_optimization_bounds(self):
        """Test parameter bounds are respected."""
        optimizer = DynamicOptimizer("U")

        # Extreme market state
        extreme_state = MarketState(
            realized_volatility=1.5,  # 150% vol
            volatility_percentile=0.99,
            price_momentum=0.5,
            volume_ratio=10.0,
        )

        returns = np.random.normal(0, 0.05, 500)
        result = optimizer.optimize_parameters(extreme_state, returns)

        # Check bounds
        assert 0.10 <= result.delta_target <= 0.40
        assert 21 <= result.dte_target <= 49
        assert 0.0 <= result.kelly_fraction <= 0.50
        assert result.confidence_score >= 0.0

    def test_optimization_validation(self):
        """Test optimization validation catches issues."""
        optimizer = DynamicOptimizer("U")

        # Create result with negative objective
        result = optimizer.OptimizationResult(
            delta_target=0.25,
            dte_target=35,
            kelly_fraction=0.5,
            expected_cagr=-0.10,
            expected_cvar=-0.20,
            objective_value=-0.05,  # Negative!
            confidence_score=0.8,
            diagnostics={},
        )

        validation = optimizer.validate_optimization(result)

        assert validation["objective_positive"] is False
        assert validation["delta_in_range"] is True

    def test_continuous_adjustments(self):
        """Test smooth parameter transitions."""
        optimizer = DynamicOptimizer("U")

        vol_percentiles = [0.2, 0.4, 0.6, 0.8]
        deltas = []

        for vp in vol_percentiles:
            state = MarketState(
                realized_volatility=0.5 + 0.5 * vp,
                volatility_percentile=vp,
                price_momentum=0.0,
                volume_ratio=1.0,
            )

            returns = np.random.normal(0, 0.03, 500)
            result = optimizer.optimize_parameters(state, returns)
            deltas.append(result.delta_target)

        # Check monotonic decrease (higher vol = lower delta)
        for i in range(len(deltas) - 1):
            assert deltas[i] >= deltas[i + 1]


class TestIVSurfaceAnalyzer:
    """Test IV surface analysis."""

    def test_iv_rank_calculation(self):
        """Test IV rank calculation."""
        analyzer = IVSurfaceAnalyzer(lookback_days=252)

        # Add historical IV data
        for i in range(252):
            date = datetime.now() - timedelta(days=252 - i)
            iv = 0.30 + 0.20 * np.sin(i / 40)  # Oscillating IV
            analyzer.update_iv_history("U", date, iv)

        # Test IV rank
        current_iv = 0.50  # High IV
        iv_rank, iv_percentile = analyzer._calculate_iv_rank("U", current_iv)

        assert iv_rank > 90  # Should be high rank
        assert iv_percentile > 90

    def test_term_structure_detection(self):
        """Test term structure regime detection."""
        analyzer = IVSurfaceAnalyzer()

        # Mock option chain with backwardation
        chain = {
            "spot_price": 25.0,
            "puts": [
                {
                    "dte": 30,
                    "implied_volatility": 0.80,
                    "expiration": datetime.now() + timedelta(30),
                },
                {
                    "dte": 60,
                    "implied_volatility": 0.70,
                    "expiration": datetime.now() + timedelta(60),
                },
                {
                    "dte": 90,
                    "implied_volatility": 0.65,
                    "expiration": datetime.now() + timedelta(90),
                },
            ],
        }

        term_structure, regime = analyzer._analyze_term_structure(chain)

        assert regime == "backwardation"
        assert 30 in term_structure
        assert term_structure[30] > term_structure[60]


class TestEventAnalyzer:
    """Test event impact analysis."""

    def test_earnings_adjustment(self):
        """Test earnings event adjustments."""
        analyzer = EventImpactAnalyzer("U")

        # Add earnings in 7 days
        events = [{"type": "earnings", "date": datetime.now() + timedelta(days=7)}]

        analyzer.update_event_calendar(events)

        should_adjust, adjustments = analyzer.should_adjust_for_event(
            dte_target=45, current_iv_rank=40
        )

        assert should_adjust is True
        assert adjustments["size_adjustment"] < 1.0  # Reduced size
        assert adjustments["confidence"] < 1.0

    def test_event_too_close(self):
        """Test avoiding trades near events."""
        analyzer = EventImpactAnalyzer("U")

        # Earnings in 3 days
        events = [{"type": "earnings", "date": datetime.now() + timedelta(days=3)}]

        analyzer.update_event_calendar(events)

        should_adjust, adjustments = analyzer.should_adjust_for_event(45, 50)

        assert should_adjust is True
        assert adjustments["size_adjustment"] == 0.0  # No position


class TestAnomalyDetector:
    """Test anomaly detection."""

    def test_volume_spike_detection(self):
        """Test detection of volume anomalies."""
        detector = AnomalyDetector("U")

        # Historical data with normal volume
        dates = pd.date_range(end=datetime.now(), periods=100)
        historical = pd.DataFrame(
            {
                "date": dates,
                "volume": np.random.normal(1000000, 100000, 100),
                "returns": np.random.normal(0, 0.02, 100),
            }
        )

        # Current with volume spike
        current = {"volume": 6000000, "realized_vol": 0.77}  # 6x normal

        anomalies = detector.detect_anomalies(current, historical)

        # Should detect volume spike
        volume_anomalies = [a for a in anomalies if a.anomaly_type.name == "volume_spike"]
        assert len(volume_anomalies) > 0

    def test_ml_anomaly_detection(self):
        """Test ML-based anomaly detection."""
        detector = AnomalyDetector("U")

        # Create synthetic historical data
        n_samples = 500
        dates = pd.date_range(end=datetime.now(), periods=n_samples)

        historical = pd.DataFrame(
            {
                "date": dates,
                "returns": np.random.normal(0, 0.03, n_samples),
                "volume": np.random.lognormal(13, 0.5, n_samples),
                "open": 20 + np.random.normal(0, 2, n_samples),
                "high": 21 + np.random.normal(0, 2, n_samples),
                "low": 19 + np.random.normal(0, 2, n_samples),
                "close": 20 + np.random.normal(0, 2, n_samples),
            }
        )

        # Fit detector
        detector.fit_ml_detector(historical)

        assert detector.is_fitted is True


class TestSeasonalityDetector:
    """Test seasonality pattern detection."""

    def test_day_of_week_pattern(self):
        """Test detection of day-of-week effects."""
        detector = SeasonalityDetector("U")

        # Create data with Monday weakness
        dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq="B")
        returns = []

        for date in dates:
            if date.dayofweek == 0:  # Monday
                returns.append(np.random.normal(-0.002, 0.02))  # Negative bias
            else:
                returns.append(np.random.normal(0.001, 0.02))  # Positive bias

        data = pd.DataFrame({"returns": returns}, index=dates)

        patterns = detector.analyze_seasonality(data)

        # Should detect day of week pattern
        dow_patterns = [p for p in patterns if p.pattern_type == "day_of_week"]
        assert len(dow_patterns) > 0

    def test_earnings_cycle_pattern(self):
        """Test detection of quarterly patterns."""
        detector = SeasonalityDetector("U")

        # Create data with higher vol in earnings months
        dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq="B")
        returns = []

        for date in dates:
            if date.month in [2, 5, 8, 11]:  # Earnings months
                returns.append(np.random.normal(0, 0.05))  # Higher vol
            else:
                returns.append(np.random.normal(0, 0.02))  # Normal vol

        data = pd.DataFrame({"returns": returns}, index=dates)

        patterns = detector.analyze_seasonality(data)

        # Should detect earnings cycle
        earnings_patterns = [p for p in patterns if "earnings" in p.pattern_type]
        assert len(earnings_patterns) > 0


class TestIntegratedDecisionEngine:
    """Test integrated decision making."""

    @pytest.mark.asyncio
    async def test_full_recommendation_flow(self):
        """Test complete recommendation generation."""
        engine = IntegratedDecisionEngine("U", 100000)

        # Mock data
        current_prices = {
            "close": 25.0,
            "open": 24.5,
            "prev_close": 24.0,
            "volume": 2000000,
            "realized_vol": 0.77,
        }

        # Historical data
        dates = pd.date_range(end=datetime.now(), periods=500)
        historical = pd.DataFrame(
            {
                "date": dates,
                "open": 20 + np.random.normal(0, 2, 500),
                "high": 21 + np.random.normal(0, 2, 500),
                "low": 19 + np.random.normal(0, 2, 500),
                "close": 20 + np.random.normal(0, 2, 500),
                "volume": np.random.lognormal(13, 0.5, 500),
                "returns": np.random.normal(0, 0.03, 500),
            }
        )

        # Mock option chain
        option_chain = {
            "spot_price": 25.0,
            "puts": [
                {
                    "strike": 22.5,
                    "expiration": datetime.now() + timedelta(35),
                    "dte": 35,
                    "delta": -0.25,
                    "bid": 0.80,
                    "implied_volatility": 0.75,
                }
            ],
        }

        # Get recommendation
        recommendation = await engine.get_recommendation(
            current_prices=current_prices,
            historical_data=historical,
            option_chain=option_chain,
            current_positions=None,
            event_calendar=[],
        )

        # Validate recommendation
        assert isinstance(recommendation, WheelRecommendation)
        assert recommendation.action in ["SELL_PUT", "NO_TRADE", "ROLL", "CLOSE"]
        assert 0 <= recommendation.confidence <= 1
        assert recommendation.symbol == "U"

    @pytest.mark.asyncio
    async def test_no_trade_conditions(self):
        """Test conditions that should result in NO_TRADE."""
        engine = IntegratedDecisionEngine("U", 100000)

        # Extreme anomaly conditions
        current_prices = {
            "close": 25.0,
            "open": 20.0,  # 20% gap!
            "prev_close": 25.0,
            "volume": 10000000,  # 10x normal
            "realized_vol": 1.5,  # 150% vol
        }

        # Minimal historical data
        dates = pd.date_range(end=datetime.now(), periods=50)
        historical = pd.DataFrame(
            {
                "date": dates,
                "close": 25 + np.random.normal(0, 5, 50),  # High volatility
                "volume": 1000000 + np.random.normal(0, 100000, 50),
                "returns": np.random.normal(0, 0.10, 50),  # 10% daily vol!
            }
        )

        recommendation = await engine.get_recommendation(
            current_prices=current_prices,
            historical_data=historical,
            option_chain=None,
            current_positions=None,
            event_calendar=[],
        )

        # Should recommend no trade due to anomalies
        assert recommendation.action == "NO_TRADE"
        assert len(recommendation.warnings) > 0
        assert recommendation.confidence < 0.5

    def test_decision_report_generation(self):
        """Test human-readable report generation."""
        engine = IntegratedDecisionEngine("U", 100000)

        # Mock recommendation
        recommendation = WheelRecommendation(
            action="SELL_PUT",
            symbol="U",
            strike=22.5,
            expiration=datetime.now() + timedelta(35),
            contracts=4,
            delta_target=0.25,
            dte_target=35,
            position_size=9000,
            kelly_fraction=0.45,
            expected_return=0.02,
            expected_risk=0.10,
            objective_value=0.015,
            max_loss=9000,
            confidence=0.75,
            warnings=["High volatility regime"],
            adjustments={"volatility": "Reduced position size"},
            market_regime="high_vol_uptrend",
            iv_metrics=None,
            anomalies=[],
            active_patterns=["earnings_cycle"],
        )

        report = engine.generate_decision_report(recommendation)

        assert isinstance(report, list)
        assert any("SELL_PUT" in line for line in report)
        assert any("22.5" in line for line in report)
        assert any("75.0%" in line for line in report)  # Confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

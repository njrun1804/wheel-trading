"""Tests for Unity-specific assignment probability model."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from unity_wheel.analytics import AssignmentProbability, UnityAssignmentModel


class TestUnityAssignmentModel:
    """Test Unity assignment probability calculations."""

    @pytest.fixture
    def model(self):
        """Create assignment model instance."""
        return UnityAssignmentModel()

    def test_deep_otm_low_probability(self, model):
        """Deep OTM options should have very low assignment probability."""
        result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=30.0,  # 14% OTM
            days_to_expiry=30,
            volatility=0.50,
            near_earnings=False,
        )

        assert result.probability < 0.01  # Less than 1%
        assert result.confidence > 0.8
        assert "base_probability" in result.factors

    def test_deep_itm_high_probability(self, model):
        """Deep ITM options should have high assignment probability."""
        result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=30.0,  # 16.7% ITM
            days_to_expiry=7,
            volatility=0.50,
            near_earnings=False,
        )

        assert result.probability > 0.5  # More than 50%
        assert result.confidence > 0.8

    def test_earnings_multiplier(self, model):
        """Earnings should significantly increase assignment probability."""
        # Without earnings
        base_result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.0,  # Slightly ITM
            days_to_expiry=14,
            volatility=0.50,
            near_earnings=False,
        )

        # With earnings
        earnings_result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.0,  # Same setup
            days_to_expiry=14,
            volatility=0.50,
            near_earnings=True,
        )

        # Earnings should increase probability significantly
        assert earnings_result.probability > base_result.probability * 2
        assert "Elevated assignment risk due to upcoming earnings" in earnings_result.warnings
        assert earnings_result.factors["earnings_multiplier"] > 1.0

    def test_friday_weekend_gap_adjustment(self, model):
        """Friday should increase assignment probability (weekend gap risk)."""
        # Non-Friday
        base_result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.5,  # Slightly ITM
            days_to_expiry=7,
            volatility=0.50,
            near_earnings=False,
            is_friday=False,
        )

        # Friday
        friday_result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.5,  # Same setup
            days_to_expiry=7,
            volatility=0.50,
            near_earnings=False,
            is_friday=True,
        )

        # Friday should increase probability
        assert friday_result.probability > base_result.probability
        assert friday_result.factors["weekend_multiplier"] > 1.0

    def test_near_expiry_effect(self, model):
        """Near expiry should increase assignment probability."""
        # Far expiry
        far_result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.5,  # Slightly ITM
            days_to_expiry=30,
            volatility=0.50,
            near_earnings=False,
        )

        # Near expiry
        near_result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.5,  # Same setup
            days_to_expiry=3,
            volatility=0.50,
            near_earnings=False,
        )

        # Near expiry should have higher probability
        assert near_result.probability > far_result.probability
        assert near_result.factors["time_multiplier"] > far_result.factors["time_multiplier"]

    def test_dividend_adjustment(self, model):
        """Dividend should increase ITM assignment probability."""
        dividend_date = datetime.now() + timedelta(days=5)

        result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.0,  # ITM
            days_to_expiry=10,
            volatility=0.50,
            near_earnings=False,
            dividend_date=dividend_date,
        )

        assert result.factors.get("dividend_multiplier", 1.0) > 1.0
        assert any("Ex-dividend" in warning for warning in result.warnings)

    def test_assignment_curve(self, model):
        """Test assignment probability curve across strikes."""
        strikes = [30, 32, 34, 35, 36, 38, 40]
        spot = 35.0

        curve = model.get_assignment_curve(
            spot_price=spot,
            strikes=strikes,
            days_to_expiry=14,
            volatility=0.50,
            near_earnings=False,
        )

        # Verify curve shape
        assert len(curve) == len(strikes)

        # ITM strikes should have higher probability than OTM
        itm_strikes = [s for s in strikes if s < spot]
        otm_strikes = [s for s in strikes if s > spot]

        if itm_strikes and otm_strikes:
            max_itm_prob = max(curve[s].probability for s in itm_strikes)
            min_otm_prob = min(curve[s].probability for s in otm_strikes)
            assert max_itm_prob > min_otm_prob

    def test_suggest_safe_strike(self, model):
        """Test strike suggestion to avoid assignment."""
        available_strikes = [30, 32, 34, 35, 36, 38, 40]

        safe_strike = model.suggest_assignment_avoidance_strike(
            spot_price=35.0,
            available_strikes=available_strikes,
            days_to_expiry=14,
            volatility=0.50,
            max_acceptable_probability=0.10,
            near_earnings=False,
        )

        # Should suggest an OTM strike
        assert safe_strike is not None
        assert safe_strike < 35.0  # OTM for puts

        # Verify the suggested strike meets criteria
        result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=safe_strike,
            days_to_expiry=14,
            volatility=0.50,
            near_earnings=False,
        )
        assert result.probability <= 0.10

    def test_extreme_moneyness_cases(self, model):
        """Test extreme ITM/OTM cases."""
        # Extreme ITM
        deep_itm = model.probability_of_assignment(
            spot_price=40.0,
            strike_price=30.0,  # 33% ITM
            days_to_expiry=3,
            volatility=0.50,
            near_earnings=False,
        )
        assert deep_itm.probability > 0.8  # Should be very high
        assert "Deep ITM near expiry" in deep_itm.warnings

        # Extreme OTM
        deep_otm = model.probability_of_assignment(
            spot_price=30.0,
            strike_price=40.0,  # 25% OTM
            days_to_expiry=30,
            volatility=0.50,
            near_earnings=False,
        )
        assert deep_otm.probability < 0.01  # Should be very low

    def test_invalid_inputs(self, model):
        """Test handling of invalid inputs."""
        # Negative spot price
        result = model.probability_of_assignment(
            spot_price=-35.0,
            strike_price=34.0,
            days_to_expiry=14,
            volatility=0.50,
            near_earnings=False,
        )
        assert result.confidence < 0.2  # Very low confidence

        # Negative volatility
        result2 = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.0,
            days_to_expiry=14,
            volatility=-0.50,
            near_earnings=False,
        )
        assert result2.confidence < 0.9  # Reduced confidence

    def test_earnings_and_friday_combo(self, model):
        """Test combined effect of earnings and Friday."""
        result = model.probability_of_assignment(
            spot_price=35.0,
            strike_price=34.5,  # Slightly ITM
            days_to_expiry=7,
            volatility=0.60,
            near_earnings=True,
            is_friday=True,
        )

        # Should have multiple risk factors
        assert result.factors["earnings_multiplier"] > 1.0
        assert result.factors["weekend_multiplier"] > 1.0
        assert len(result.warnings) >= 2  # At least two warnings

    @pytest.mark.parametrize(
        "moneyness,expected_bucket",
        [
            (-0.10, "deep_otm"),  # 10% OTM
            (-0.03, "otm"),  # 3% OTM
            (0.005, "atm"),  # 0.5% ITM (ATM)
            (0.02, "slight_itm"),  # 2% ITM
            (0.05, "deep_itm"),  # 5% ITM
        ],
    )
    def test_moneyness_buckets(self, model, moneyness, expected_bucket):
        """Test correct bucketing of moneyness levels."""
        base_prob = model._get_base_probability(moneyness)
        expected_prob = model.HISTORICAL_RATES[expected_bucket]
        assert base_prob == expected_prob

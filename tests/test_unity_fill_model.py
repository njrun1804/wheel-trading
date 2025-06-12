"""Test Unity-specific fill model."""

import pytest

from unity_wheel.execution.unity_fill_model import FillEstimate, UnityFillModel


class TestUnityFillModel:
    """Test Unity fill price estimation."""

    @pytest.fixture
    def fill_model(self):
        """Create fill model instance."""
        return UnityFillModel()

    def test_basic_fill_estimation(self, fill_model):
        """Test basic fill price estimation."""
        # Test selling puts (opening)
        estimate, confidence = fill_model.estimate_fill_price(
            bid=2.00, ask=2.10, size=5, is_opening=True
        )

        assert isinstance(estimate, FillEstimate)
        assert 2.00 <= estimate.fill_price <= 2.10
        # With urgency=0.5, fill position is 0.1 + (0.4 * 0.5) = 0.3
        # So fill price = 2.00 + (0.10 * 0.3) = 2.03
        assert estimate.fill_price == 2.03
        assert estimate.commission == 3.25  # 5 * 0.65
        assert confidence > 0.8  # Adjusted for realistic confidence

    def test_closing_position_fill(self, fill_model):
        """Test buying back positions."""
        estimate, confidence = fill_model.estimate_fill_price(
            bid=1.00, ask=1.10, size=5, is_opening=False
        )

        assert 1.00 <= estimate.fill_price <= 1.10
        # With urgency=0.5, closing fill position is 0.9 - (0.3 * 0.5) = 0.75
        # So fill price = 1.00 + (0.10 * 0.75) = 1.075, rounded to 1.08
        assert estimate.fill_price == 1.08
        assert confidence > 0.8

    def test_size_impact(self, fill_model):
        """Test size impact on fill price."""
        # Small order - no impact
        small_est, _ = fill_model.estimate_fill_price(bid=2.00, ask=2.10, size=5, is_opening=True)

        # Large order - should have impact
        large_est, _ = fill_model.estimate_fill_price(bid=2.00, ask=2.10, size=25, is_opening=True)

        # Large orders when selling should get worse fills (lower price)
        assert large_est.fill_price < small_est.fill_price
        assert large_est.size_impact > 0

    def test_urgency_parameter(self, fill_model):
        """Test urgency affects fill positioning."""
        # Low urgency - better fill
        low_urgency, _ = fill_model.estimate_fill_price(
            bid=2.00, ask=2.10, size=5, is_opening=True, urgency=0.1
        )

        # High urgency - worse fill but faster execution
        high_urgency, _ = fill_model.estimate_fill_price(
            bid=2.00, ask=2.10, size=5, is_opening=True, urgency=0.9
        )

        # Higher urgency should fill closer to mid
        assert high_urgency.fill_price > low_urgency.fill_price

    def test_wide_spread_confidence(self, fill_model):
        """Test wide spreads reduce confidence."""
        # Tight spread
        tight_est, tight_conf = fill_model.estimate_fill_price(
            bid=2.00, ask=2.05, size=5, is_opening=True
        )

        # Wide spread
        wide_est, wide_conf = fill_model.estimate_fill_price(
            bid=2.00, ask=2.30, size=5, is_opening=True
        )

        assert wide_conf < tight_conf
        assert wide_conf < 0.9

    def test_round_trip_cost(self, fill_model):
        """Test round-trip cost estimation."""
        total_cost, confidence = fill_model.estimate_round_trip_cost(
            open_bid=2.00, open_ask=2.10, close_bid=1.00, close_ask=1.10, size=10
        )

        # Should include commissions for both legs
        assert total_cost >= 10 * 0.65 * 2  # At least commission
        # Combined confidence is min of both legs times 0.9
        assert confidence > 0.7  # Adjusted for realistic combined confidence

    def test_invalid_inputs(self, fill_model):
        """Test handling of invalid inputs."""
        # Invalid bid/ask
        estimate, confidence = fill_model.estimate_fill_price(
            bid=0, ask=2.10, size=5, is_opening=True
        )
        assert confidence == 0.0
        assert estimate.fill_price == 0

        # Invalid size
        estimate, confidence = fill_model.estimate_fill_price(
            bid=2.00, ask=2.10, size=0, is_opening=True
        )
        assert confidence == 0.0

    def test_extreme_prices(self, fill_model):
        """Test confidence adjustment for extreme prices."""
        # Very low price (likely far OTM)
        low_price_est, low_conf = fill_model.estimate_fill_price(
            bid=0.05, ask=0.10, size=5, is_opening=True
        )

        # Very high price (likely deep ITM)
        high_price_est, high_conf = fill_model.estimate_fill_price(
            bid=15.00, ask=15.10, size=5, is_opening=True
        )

        # Both should have reduced confidence
        assert low_conf < 0.9
        assert high_conf < 0.9

    def test_assignment_cost(self, fill_model):
        """Test assignment fee estimation."""
        cost = fill_model.estimate_assignment_cost(size=5)
        assert cost == 5.00  # Flat fee regardless of size

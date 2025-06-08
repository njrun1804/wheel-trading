"""Tests for wheel strategy implementation."""

from __future__ import annotations

import pytest

from src.models import Position, WheelPosition
from src.wheel import WheelStrategy


class TestWheelStrategy:
    """Test wheel strategy functionality."""

    @pytest.fixture
    def wheel(self):
        """Create wheel strategy instance."""
        return WheelStrategy()

    def test_find_optimal_put_strike(self, wheel):
        """Test finding optimal put strike."""
        strikes = [90, 95, 100, 105, 110]
        optimal = wheel.find_optimal_put_strike(
            current_price=100,
            available_strikes=strikes,
            volatility=0.25,
            days_to_expiry=45,
        )

        # Should pick a strike below current price
        assert optimal < 100
        assert optimal in strikes

    def test_find_optimal_put_strike_no_strikes(self, wheel):
        """Test handling empty strike list."""
        optimal = wheel.find_optimal_put_strike(
            current_price=100,
            available_strikes=[],
            volatility=0.25,
            days_to_expiry=45,
        )
        assert optimal is None

    def test_find_optimal_call_strike(self, wheel):
        """Test finding optimal call strike."""
        strikes = [95, 100, 105, 110, 115]
        optimal = wheel.find_optimal_call_strike(
            current_price=100,
            cost_basis=98,
            available_strikes=strikes,
            volatility=0.25,
            days_to_expiry=45,
        )

        # Should pick a strike above cost basis
        assert optimal >= 98
        assert optimal in strikes

    def test_find_optimal_call_strike_no_valid_strikes(self, wheel):
        """Test when no strikes are above cost basis."""
        strikes = [90, 95]  # All below cost basis
        optimal = wheel.find_optimal_call_strike(
            current_price=100,
            cost_basis=98,
            available_strikes=strikes,
            volatility=0.25,
            days_to_expiry=45,
        )
        assert optimal is None

    def test_calculate_position_size(self, wheel):
        """Test position sizing calculation."""
        # With 20% max allocation and $100k portfolio
        size = wheel.calculate_position_size(
            symbol="U",
            current_price=35,
            portfolio_value=100000,
        )

        # Max allocation: $20k / $3.5k per contract = 5.7, rounds to 5
        assert size == 5

        # Test with larger portfolio
        size = wheel.calculate_position_size(
            symbol="U",
            current_price=35,
            portfolio_value=1000000,
        )
        # Max allocation: $200k / $3.5k per contract = 57
        assert size == 57

    def test_should_roll_position_expiry(self, wheel):
        """Test rolling decision based on expiry."""
        position = Position(
            symbol="U",
            qty=-1,
            avg_price=2.50,
            option_type="put",
            strike=35,
        )

        # Should roll if close to expiry
        should_roll = wheel.should_roll_position(
            position=position,
            current_price=405,
            days_to_expiry=5,  # Less than 7 days
            current_delta=-0.3,
        )
        assert should_roll is True

    def test_should_roll_position_deep_itm_put(self, wheel):
        """Test rolling deep ITM put."""
        position = Position(
            symbol="U",
            qty=-1,
            avg_price=2.50,
            option_type="put",
            strike=35,
        )

        # Should roll if put is deep ITM
        should_roll = wheel.should_roll_position(
            position=position,
            current_price=28,  # Well below strike
            days_to_expiry=20,
            current_delta=-0.85,  # Deep ITM
        )
        assert should_roll is True

    def test_should_roll_position_deep_itm_call(self, wheel):
        """Test rolling deep ITM call."""
        position = Position(
            symbol="U",
            qty=-1,
            avg_price=2.50,
            option_type="call",
            strike=35,
        )

        # Should roll if call is deep ITM
        should_roll = wheel.should_roll_position(
            position=position,
            current_price=42,  # Well above strike
            days_to_expiry=20,
            current_delta=0.85,  # Deep ITM
        )
        assert should_roll is True

    def test_should_not_roll_position(self, wheel):
        """Test when position should not be rolled."""
        position = Position(
            symbol="U",
            qty=-1,
            avg_price=2.50,
            option_type="put",
            strike=35,
        )

        # Should not roll if conditions are fine
        should_roll = wheel.should_roll_position(
            position=position,
            current_price=38,
            days_to_expiry=30,
            current_delta=-0.3,  # Target delta
        )
        assert should_roll is False

    def test_track_position(self, wheel):
        """Test position tracking."""
        # First access creates new position
        pos = wheel.track_position("U")
        assert isinstance(pos, WheelPosition)
        assert pos.symbol == "U"
        assert pos.shares is None
        assert len(pos.cash_secured_puts) == 0

        # Second access returns same position
        pos2 = wheel.track_position("U")
        assert pos2 is pos

        # Different symbol creates different position
        pos3 = wheel.track_position("AAPL")
        assert pos3 is not pos
        assert pos3.symbol == "AAPL"

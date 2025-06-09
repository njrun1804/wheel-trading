"""Comprehensive tests for Position model including property-based testing."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pytest
from hypothesis import assume, given, strategies as st

from unity_wheel.models.position import Position, PositionType


class TestPositionBasic:
    """Basic unit tests for Position model."""

    def test_stock_position_creation(self) -> None:
        """Test creating a stock position."""
        pos = Position("U", 100)
        assert pos.symbol == "U"
        assert pos.quantity == 100
        assert pos.position_type == PositionType.STOCK
        assert pos.underlying == "U"
        assert pos.strike is None
        assert pos.expiration is None
        assert pos.is_long is True
        assert pos.is_short is False

    def test_short_stock_position(self) -> None:
        """Test creating a short stock position."""
        pos = Position("AAPL", -500)
        assert pos.symbol == "AAPL"
        assert pos.quantity == -500
        assert pos.position_type == PositionType.STOCK
        assert pos.is_long is False
        assert pos.is_short is True
        assert pos.abs_quantity == 500

    def test_call_option_position(self) -> None:
        """Test creating a call option position."""
        pos = Position("U241220C00080000", -1)
        assert pos.symbol == "U241220C00080000"
        assert pos.quantity == -1
        assert pos.position_type == PositionType.CALL
        assert pos.underlying == "U"
        assert pos.strike == 80.0
        assert pos.expiration == date(2024, 12, 20)
        assert pos.is_short is True

    def test_put_option_position(self) -> None:
        """Test creating a put option position."""
        pos = Position("AAPL250117P00150000", 10)
        assert pos.position_type == PositionType.PUT
        assert pos.underlying == "AAPL"
        assert pos.strike == 150.0
        assert pos.expiration == date(2025, 1, 17)
        assert pos.is_long is True

    def test_position_immutability(self) -> None:
        """Test that Position is immutable."""
        pos = Position("U", 100)
        with pytest.raises(AttributeError):
            pos.quantity = 200  # type: ignore
        with pytest.raises(AttributeError):
            pos.symbol = "AAPL"  # type: ignore

    def test_invalid_symbol_format(self) -> None:
        """Test invalid symbol formats raise errors."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Position("123ABC", 100)  # Invalid stock ticker
        
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Position("U241220X00080000", 1)  # Invalid option type (X)
        
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Position("", 100)  # Empty symbol

    def test_invalid_quantity(self) -> None:
        """Test invalid quantities raise errors."""
        with pytest.raises(ValueError, match="Quantity cannot be zero"):
            Position("U", 0)
        
        with pytest.raises(TypeError, match="Quantity must be an integer"):
            Position("U", 10.5)  # type: ignore

    def test_position_string_representation(self) -> None:
        """Test human-readable string representation."""
        # Stock position
        pos1 = Position("U", 100)
        assert str(pos1) == "Long 100 U"
        
        pos2 = Position("AAPL", -500)
        assert str(pos2) == "Short 500 AAPL"
        
        # Option positions
        pos3 = Position("U241220C00080000", -1)
        assert str(pos3) == "Short 1 U $80.00 Call exp 2024-12-20"
        
        pos4 = Position("AAPL250117P00150000", 10)
        assert str(pos4) == "Long 10 AAPL $150.00 Put exp 2025-01-17"

    def test_position_serialization(self) -> None:
        """Test converting Position to/from dictionary."""
        pos = Position("U241220C00080000", -5)
        
        # To dict
        data = pos.to_dict()
        assert data["symbol"] == "U241220C00080000"
        assert data["quantity"] == -5
        assert data["position_type"] == "call"
        assert data["underlying"] == "U"
        assert data["strike"] == 80.0
        assert data["expiration"] == "2024-12-20"
        
        # From dict
        pos2 = Position.from_dict(data)
        assert pos2.symbol == pos.symbol
        assert pos2.quantity == pos.quantity

    def test_edge_case_strikes(self) -> None:
        """Test positions with unusual strike prices."""
        # Very high strike
        pos1 = Position("SPY261231C99999999", 1)
        assert pos1.strike == 99999.999
        
        # Very low strike (penny strikes)
        pos2 = Position("U241220P00000500", 1)
        assert pos2.strike == 0.5
        
        # Zero strike (rare but possible)
        pos3 = Position("TEST250101C00000000", 1)
        assert pos3.strike == 0.0


class TestPositionPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        ticker=st.from_regex(r"[A-Z]{1,6}", fullmatch=True),
        quantity=st.integers(min_value=-999999, max_value=999999).filter(lambda x: x != 0),
    )
    def test_stock_position_properties(self, ticker: str, quantity: int) -> None:
        """Test stock positions with random valid inputs."""
        pos = Position(ticker, quantity)
        
        # Properties that must hold
        assert pos.symbol == ticker
        assert pos.quantity == quantity
        assert pos.position_type == PositionType.STOCK
        assert pos.underlying == ticker
        assert pos.strike is None
        assert pos.expiration is None
        assert pos.is_long == (quantity > 0)
        assert pos.is_short == (quantity < 0)
        assert pos.abs_quantity == abs(quantity)

    @given(
        ticker=st.from_regex(r"[A-Z]{1,6}", fullmatch=True),
        year=st.integers(min_value=24, max_value=30),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),  # Avoid month-end issues
        option_type=st.sampled_from(["C", "P"]),
        strike_cents=st.integers(min_value=1, max_value=99999999),
        quantity=st.integers(min_value=-999999, max_value=999999).filter(lambda x: x != 0),
    )
    def test_option_position_properties(
        self,
        ticker: str,
        year: int,
        month: int,
        day: int,
        option_type: str,
        strike_cents: int,
        quantity: int,
    ) -> None:
        """Test option positions with random valid inputs."""
        # Construct OCC symbol
        symbol = f"{ticker}{year:02d}{month:02d}{day:02d}{option_type}{strike_cents:08d}"
        
        pos = Position(symbol, quantity)
        
        # Properties that must hold
        assert pos.symbol == symbol
        assert pos.quantity == quantity
        assert pos.underlying == ticker
        assert pos.strike == strike_cents / 1000.0
        assert pos.expiration == date(2000 + year, month, day)
        
        if option_type == "C":
            assert pos.position_type == PositionType.CALL
        else:
            assert pos.position_type == PositionType.PUT

    @given(
        data=st.fixed_dictionaries(
            {
                "symbol": st.text(min_size=1),
                "quantity": st.integers(),
            }
        )
    )
    def test_position_from_dict_roundtrip(self, data: dict[str, Any]) -> None:
        """Test that valid positions can roundtrip through dict."""
        # Only test with valid position data
        try:
            pos = Position.from_dict(data)
            # If we can create it, we should be able to roundtrip
            data2 = pos.to_dict()
            pos2 = Position.from_dict(data2)
            assert pos.symbol == pos2.symbol
            assert pos.quantity == pos2.quantity
        except (ValueError, TypeError):
            # Invalid data is expected to fail
            pass

    @given(st.integers(min_value=0, max_value=10))
    def test_zero_and_near_zero_quantities(self, offset: int) -> None:
        """Test behavior around zero quantities."""
        if offset == 0:
            with pytest.raises(ValueError, match="Quantity cannot be zero"):
                Position("U", 0)
        else:
            # Should work fine for non-zero
            pos_positive = Position("U", offset)
            assert pos_positive.quantity == offset
            
            pos_negative = Position("U", -offset)
            assert pos_negative.quantity == -offset


class TestPositionEdgeCases:
    """Test edge cases and error conditions."""

    def test_maximum_values(self) -> None:
        """Test positions with maximum integer values."""
        import sys
        
        max_int = sys.maxsize
        pos = Position("U", max_int)
        assert pos.quantity == max_int
        assert pos.abs_quantity == max_int

    def test_option_date_edge_cases(self) -> None:
        """Test option positions with edge case dates."""
        # Leap year
        pos1 = Position("U240229C00100000", 1)
        assert pos1.expiration == date(2024, 2, 29)
        
        # End of year
        pos2 = Position("SPY241231P00400000", -1)
        assert pos2.expiration == date(2024, 12, 31)

    def test_case_sensitivity(self) -> None:
        """Test that symbols are case-insensitive."""
        pos1 = Position("u", 100)  # lowercase
        assert pos1.underlying == "U"  # Should be uppercase
        
        pos2 = Position("aapl", -50)
        assert pos2.underlying == "AAPL"

    def test_invalid_option_dates(self) -> None:
        """Test invalid dates in option symbols."""
        # Invalid month
        with pytest.raises(ValueError):
            Position("U241320C00080000", 1)
        
        # Invalid day
        with pytest.raises(ValueError):
            Position("U240232C00080000", 1)

    @pytest.mark.parametrize(
        "bad_symbol",
        [
            "U24122C00080000",      # Missing digit in date
            "U2412200C00080000",    # Extra digit in date  
            "U241220C0008000",      # Missing digit in strike
            "U241220C000800000",    # Extra digit in strike
            "TOOLONG241220C00080000",  # Ticker too long
            "241220C00080000",      # Missing ticker
            "U241220Q00080000",     # Invalid option type
        ],
    )
    def test_malformed_option_symbols(self, bad_symbol: str) -> None:
        """Test various malformed option symbols."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Position(bad_symbol, 1)
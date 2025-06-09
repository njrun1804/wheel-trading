"""Comprehensive tests for Account model including property-based testing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import assume, given, strategies as st

from unity_wheel.models.account import Account


class TestAccountBasic:
    """Basic unit tests for Account model."""

    def test_account_creation_basic(self) -> None:
        """Test creating a basic account."""
        now = datetime.now(timezone.utc)
        account = Account(
            cash_balance=50000.0,
            buying_power=100000.0,
            margin_used=10000.0,
            timestamp=now,
        )
        assert account.cash_balance == 50000.0
        assert account.buying_power == 100000.0
        assert account.margin_used == 10000.0
        assert account.timestamp == now

    def test_account_creation_cash_only(self) -> None:
        """Test creating a cash-only account (no margin)."""
        account = Account(
            cash_balance=25000.0,
            buying_power=25000.0,  # Same as cash
        )
        assert account.cash_balance == 25000.0
        assert account.buying_power == 25000.0
        assert account.margin_used == 0.0
        assert account.is_margin_account is False
        assert account.margin_available == 0.0
        assert account.margin_utilization == 0.0

    def test_account_creation_with_margin(self) -> None:
        """Test creating a margin account."""
        account = Account(
            cash_balance=50000.0,
            buying_power=100000.0,  # 2x margin
            margin_used=25000.0,
        )
        assert account.is_margin_account is True
        assert account.margin_available == 25000.0  # 50k total margin - 25k used
        assert account.margin_utilization == 0.5  # 25k used / 50k total

    def test_account_immutability(self) -> None:
        """Test that Account is immutable."""
        account = Account(cash_balance=50000.0, buying_power=100000.0)
        with pytest.raises(AttributeError):
            account.cash_balance = 60000.0  # type: ignore
        with pytest.raises(AttributeError):
            account.margin_used = 5000.0  # type: ignore

    def test_default_timestamp(self) -> None:
        """Test that default timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        account = Account(cash_balance=10000.0, buying_power=10000.0)
        after = datetime.now(timezone.utc)
        
        assert before <= account.timestamp <= after
        assert account.timestamp.tzinfo is not None

    def test_invalid_cash_balance(self) -> None:
        """Test that negative cash balance raises error."""
        with pytest.raises(ValueError, match="Cash balance cannot be negative"):
            Account(cash_balance=-1000.0, buying_power=10000.0)

    def test_invalid_buying_power(self) -> None:
        """Test that negative buying power raises error."""
        with pytest.raises(ValueError, match="Buying power cannot be negative"):
            Account(cash_balance=10000.0, buying_power=-1000.0)

    def test_invalid_margin_used(self) -> None:
        """Test that negative margin used raises error."""
        with pytest.raises(ValueError, match="Margin used cannot be negative"):
            Account(cash_balance=10000.0, buying_power=20000.0, margin_used=-1000.0)

    def test_buying_power_less_than_cash(self) -> None:
        """Test that buying power cannot be less than cash balance."""
        with pytest.raises(ValueError, match="Buying power .* cannot be less than cash balance"):
            Account(cash_balance=10000.0, buying_power=5000.0)

    def test_margin_used_exceeds_available(self) -> None:
        """Test that margin used cannot exceed total margin available."""
        with pytest.raises(ValueError, match="Margin used .* exceeds total margin available"):
            Account(
                cash_balance=10000.0,
                buying_power=20000.0,  # 10k margin available
                margin_used=15000.0,    # Exceeds 10k available
            )

    def test_timezone_required(self) -> None:
        """Test that timestamp must be timezone-aware."""
        naive_time = datetime.now()  # No timezone
        with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
            Account(cash_balance=10000.0, buying_power=10000.0, timestamp=naive_time)

    def test_net_liquidation_value(self) -> None:
        """Test net liquidation value calculation."""
        account = Account(
            cash_balance=50000.0,
            buying_power=100000.0,
            margin_used=20000.0,
        )
        # Simplified version just returns cash balance
        assert account.net_liquidation_value == 50000.0

    def test_has_sufficient_buying_power(self) -> None:
        """Test available buying power calculation."""
        account = Account(
            cash_balance=50000.0,
            buying_power=100000.0,
            margin_used=30000.0,
        )
        assert account.has_sufficient_buying_power == 70000.0  # 100k - 30k

    def test_validate_position_size(self) -> None:
        """Test position size validation."""
        account = Account(
            cash_balance=50000.0,
            buying_power=100000.0,
            margin_used=30000.0,
        )
        
        # Can take position requiring 50k
        assert account.validate_position_size(50000.0) is True
        
        # Can take position requiring exactly 70k (remaining buying power)
        assert account.validate_position_size(70000.0) is True
        
        # Cannot take position requiring 80k
        assert account.validate_position_size(80000.0) is False

    def test_account_string_representation(self) -> None:
        """Test human-readable string representation."""
        account = Account(
            cash_balance=50000.0,
            buying_power=100000.0,
            margin_used=25000.0,
        )
        expected = (
            "Account(cash=$50,000.00, buying_power=$100,000.00, "
            "margin_used=$25,000.00, utilization=50.0%)"
        )
        assert str(account) == expected

    def test_account_serialization(self) -> None:
        """Test converting Account to/from dictionary."""
        now = datetime.now(timezone.utc)
        account = Account(
            cash_balance=50000.0,
            buying_power=100000.0,
            margin_used=25000.0,
            timestamp=now,
        )
        
        # To dict
        data = account.to_dict()
        assert data["cash_balance"] == 50000.0
        assert data["buying_power"] == 100000.0
        assert data["margin_used"] == 25000.0
        assert data["timestamp"] == now.isoformat()
        assert data["margin_available"] == 25000.0
        assert data["margin_utilization"] == 0.5
        
        # From dict
        account2 = Account.from_dict(data)
        assert account2.cash_balance == account.cash_balance
        assert account2.buying_power == account.buying_power
        assert account2.margin_used == account.margin_used
        assert account2.timestamp == account.timestamp


class TestAccountPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        cash=st.floats(min_value=0.0, max_value=1e9, allow_nan=False),
        margin_multiplier=st.floats(min_value=1.0, max_value=4.0),
        margin_usage_pct=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_valid_account_properties(
        self,
        cash: float,
        margin_multiplier: float,
        margin_usage_pct: float,
    ) -> None:
        """Test accounts with random valid inputs."""
        buying_power = cash * margin_multiplier
        total_margin = buying_power - cash
        margin_used = total_margin * margin_usage_pct
        
        account = Account(
            cash_balance=cash,
            buying_power=buying_power,
            margin_used=margin_used,
        )
        
        # Properties that must hold
        assert account.cash_balance == cash
        assert account.buying_power == buying_power
        assert account.margin_used == margin_used
        assert account.margin_available == pytest.approx(total_margin - margin_used)
        
        if total_margin > 0:
            assert account.margin_utilization == pytest.approx(margin_usage_pct)
        else:
            assert account.margin_utilization == 0.0

    @given(
        cash=st.floats(min_value=0.01, max_value=1e6),
        buying_power=st.floats(min_value=0.01, max_value=1e6),
        margin_used=st.floats(min_value=0.0, max_value=1e6),
    )
    def test_account_validation_logic(
        self,
        cash: float,
        buying_power: float,
        margin_used: float,
    ) -> None:
        """Test that validation logic works correctly."""
        # Buying power must be >= cash
        if buying_power < cash:
            with pytest.raises(ValueError):
                Account(
                    cash_balance=cash,
                    buying_power=buying_power,
                    margin_used=0.0,
                )
            return
        
        # Margin used must not exceed available margin
        total_margin = buying_power - cash
        if margin_used > total_margin:
            with pytest.raises(ValueError):
                Account(
                    cash_balance=cash,
                    buying_power=buying_power,
                    margin_used=margin_used,
                )
            return
        
        # Should create successfully if all validations pass
        account = Account(
            cash_balance=cash,
            buying_power=buying_power,
            margin_used=margin_used,
        )
        assert account is not None

    @given(
        data=st.fixed_dictionaries(
            {
                "cash_balance": st.floats(min_value=0, max_value=1e6),
                "buying_power": st.floats(min_value=0, max_value=1e6),
                "margin_used": st.floats(min_value=0, max_value=1e6),
                "timestamp": st.datetimes(timezones=st.just(timezone.utc)),
            }
        )
    )
    def test_account_dict_roundtrip(self, data: dict[str, Any]) -> None:
        """Test that valid accounts can roundtrip through dict."""
        # Skip invalid combinations
        if data["buying_power"] < data["cash_balance"]:
            return
        if data["margin_used"] > (data["buying_power"] - data["cash_balance"]):
            return
        
        account = Account.from_dict(data)
        data2 = account.to_dict()
        account2 = Account.from_dict(data2)
        
        assert account.cash_balance == account2.cash_balance
        assert account.buying_power == account2.buying_power
        assert account.margin_used == account2.margin_used
        assert account.timestamp == account2.timestamp


class TestAccountEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_balances(self) -> None:
        """Test account with zero balances."""
        account = Account(
            cash_balance=0.0,
            buying_power=0.0,
            margin_used=0.0,
        )
        assert account.cash_balance == 0.0
        assert account.margin_available == 0.0
        assert account.margin_utilization == 0.0
        assert account.has_sufficient_buying_power == 0.0

    def test_maximum_margin_usage(self) -> None:
        """Test account at maximum margin usage."""
        account = Account(
            cash_balance=50000.0,
            buying_power=150000.0,  # 3x leverage
            margin_used=100000.0,   # All margin used
        )
        assert account.margin_available == 0.0
        assert account.margin_utilization == 1.0
        assert account.has_sufficient_buying_power == 50000.0  # Only cash left

    def test_typical_account_scenarios(self) -> None:
        """Test typical brokerage account scenarios."""
        # Retail cash account
        cash_account = Account(
            cash_balance=25000.0,
            buying_power=25000.0,
        )
        assert cash_account.is_margin_account is False
        
        # PDT margin account (4x intraday)
        pdt_account = Account(
            cash_balance=30000.0,
            buying_power=120000.0,  # 4x for day trading
            margin_used=50000.0,
        )
        assert pdt_account.is_margin_account is True
        assert pdt_account.margin_available == 40000.0
        
        # Portfolio margin account (higher leverage)
        pm_account = Account(
            cash_balance=150000.0,
            buying_power=1000000.0,  # ~6.7x leverage
            margin_used=400000.0,
        )
        assert pm_account.margin_utilization == pytest.approx(0.47, rel=0.01)

    def test_timestamp_formats(self) -> None:
        """Test various timestamp formats in from_dict."""
        base_data = {
            "cash_balance": 10000.0,
            "buying_power": 20000.0,
            "margin_used": 0.0,
        }
        
        # ISO format with timezone
        data1 = {**base_data, "timestamp": "2024-01-15T10:30:00+00:00"}
        account1 = Account.from_dict(data1)
        assert account1.timestamp.tzinfo is not None
        
        # Already a datetime object
        now = datetime.now(timezone.utc)
        data2 = {**base_data, "timestamp": now}
        account2 = Account.from_dict(data2)
        assert account2.timestamp == now

    @pytest.mark.parametrize(
        "cash,power,used,expected_util",
        [
            (100000, 100000, 0, 0.0),       # No margin account
            (100000, 200000, 0, 0.0),       # Margin not used
            (100000, 200000, 50000, 0.5),   # Half margin used
            (100000, 200000, 100000, 1.0),  # All margin used
            (50000, 150000, 75000, 0.75),   # 3/4 margin used
        ],
    )
    def test_margin_utilization_calculations(
        self,
        cash: float,
        power: float,
        used: float,
        expected_util: float,
    ) -> None:
        """Test margin utilization calculations."""
        account = Account(
            cash_balance=cash,
            buying_power=power,
            margin_used=used,
        )
        assert account.margin_utilization == pytest.approx(expected_util)
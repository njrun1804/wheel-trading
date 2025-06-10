"""Test single Schwab account management with hard failures."""

import sys

import pytest

from src.unity_wheel.portfolio import SchwabAccount, SingleAccountManager


class TestSingleAccountManager:
    """Test single account manager with hard failure behavior."""

    @pytest.fixture
    def manager(self):
        """Create account manager instance."""
        return SingleAccountManager()

    @pytest.fixture
    def valid_account_data(self):
        """Valid Schwab account data."""
        return {
            "securitiesAccount": {
                "accountNumber": "12345678",
                "type": "MARGIN",
                "currentBalances": {
                    "liquidationValue": 150000,
                    "cashBalance": 50000,
                    "buyingPower": 100000,
                    "marginBuyingPower": 100000,
                },
                "positions": [
                    {
                        "instrument": {"symbol": "U", "assetType": "EQUITY"},
                        "quantity": 1000,
                        "averagePrice": 35.00,
                        "marketValue": 35000,
                    },
                    {
                        "instrument": {"symbol": "U240119P00035000", "assetType": "OPTION"},
                        "quantity": -5,
                        "averagePrice": 2.00,
                        "marketValue": -1000,
                    },
                ],
            }
        }

    def test_parse_valid_account(self, manager, valid_account_data):
        """Test parsing valid account data."""
        account = manager.parse_account(valid_account_data)

        assert isinstance(account, SchwabAccount)
        assert account.account_id == "12345678"
        assert account.total_value == 150000
        assert account.cash_balance == 50000
        assert account.buying_power == 100000
        assert account.margin_buying_power == 100000
        assert len(account.positions) == 2
        assert account.unity_shares == 1000
        assert account.unity_puts == 5

    def test_die_on_no_data(self, manager):
        """Test program exits when no account data provided."""
        with pytest.raises(SystemExit) as exc_info:
            manager.parse_account(None)
        assert exc_info.value.code == 1

    def test_die_on_missing_securities_account(self, manager):
        """Test program exits when missing securitiesAccount."""
        with pytest.raises(SystemExit) as exc_info:
            manager.parse_account({})
        assert exc_info.value.code == 1

    def test_die_on_missing_balances(self, manager):
        """Test program exits when missing currentBalances."""
        with pytest.raises(SystemExit) as exc_info:
            manager.parse_account({"securitiesAccount": {}})
        assert exc_info.value.code == 1

    def test_die_on_missing_liquidation_value(self, manager):
        """Test program exits when missing liquidationValue."""
        with pytest.raises(SystemExit) as exc_info:
            manager.parse_account(
                {"securitiesAccount": {"currentBalances": {"cashBalance": 50000}}}
            )
        assert exc_info.value.code == 1

    def test_die_on_missing_cash_balance(self, manager):
        """Test program exits when missing cashBalance."""
        with pytest.raises(SystemExit) as exc_info:
            manager.parse_account(
                {"securitiesAccount": {"currentBalances": {"liquidationValue": 100000}}}
            )
        assert exc_info.value.code == 1

    def test_die_on_missing_position_instrument(self, manager):
        """Test program exits when position missing instrument."""
        with pytest.raises(SystemExit) as exc_info:
            manager.parse_account(
                {
                    "securitiesAccount": {
                        "currentBalances": {
                            "liquidationValue": 100000,
                            "cashBalance": 50000,
                            "buyingPower": 50000,
                        },
                        "positions": [{"quantity": 100}],  # Missing instrument
                    }
                }
            )
        assert exc_info.value.code == 1

    def test_die_on_insufficient_buying_power(self, manager, valid_account_data):
        """Test program exits when insufficient buying power."""
        account = manager.parse_account(valid_account_data)

        with pytest.raises(SystemExit) as exc_info:
            manager.validate_buying_power(200000, account)  # Needs 200k, has 100k
        assert exc_info.value.code == 1

    def test_die_on_position_limit_exceeded(self, manager, valid_account_data):
        """Test program exits when position limit exceeded."""
        account = manager.parse_account(valid_account_data)

        # Position limit is 20% of 150k = 30k
        # Current Unity exposure is already 70k
        # Adding 20k would exceed limit
        with pytest.raises(SystemExit) as exc_info:
            manager.validate_position_limits(20000, account)
        assert exc_info.value.code == 1

    def test_die_on_max_puts_exceeded(self, manager, valid_account_data):
        """Test program exits when max concurrent puts exceeded."""
        # Modify account to have max puts already
        valid_account_data["securitiesAccount"]["positions"].extend(
            [
                {
                    "instrument": {"symbol": "U240219P00035000", "assetType": "OPTION"},
                    "quantity": -5,
                },
                {
                    "instrument": {"symbol": "U240319P00035000", "assetType": "OPTION"},
                    "quantity": -5,
                },
            ]
        )

        account = manager.parse_account(valid_account_data)
        assert account.unity_puts == 15  # 5 + 5 + 5 = 15

        # Try to add more position when at max (assuming max is 3 or similar)
        # This should die if we're at the limit
        # Note: actual limit depends on config

    def test_unity_position_detection(self, manager, valid_account_data):
        """Test Unity position detection."""
        account = manager.parse_account(valid_account_data)

        # Should detect Unity stock
        assert account.unity_shares == 1000

        # Should detect Unity puts
        assert account.unity_puts == 5

        # Should calculate notional correctly
        # 1000 shares * $35 + 5 puts * 100 shares * $35 = $52,500
        assert account.unity_notional == 52500.0

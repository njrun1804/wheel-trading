import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError, ClientResponseError
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.unity_wheel.schwab import (
    PositionType,
    SchwabAccount,
    SchwabAuthError,
    SchwabClient,
    SchwabDataError,
    SchwabError,
    SchwabNetworkError,
    SchwabPosition,
)


@pytest.fixture
def mock_client():
    """Create a mock Schwab client."""
    client = SchwabClient(
        client_id="test_client_id",
        client_secret="test_client_secret",
        cache_dir=Path("/tmp/test_schwab_cache"),
    )
    client.session = MagicMock()
    client.access_token = "test_token"
    client.token_expiry = datetime.now() + timedelta(hours=1)
    return client


class TestSchwabPositionParsing:
    """Test position parsing and validation."""

    def test_parse_stock_position(self, mock_client):
        """Test parsing a stock position."""
        data = {
            "symbol": "AAPL",
            "quantity": "100",
            "assetType": "EQUITY",
            "marketValue": "15000.00",
            "averagePrice": "145.00",
            "unrealizedPnL": "500.00",
            "realizedPnL": "0.00",
        }

        position = mock_client._parse_position(data)

        assert position.symbol == "AAPL"
        assert position.quantity == Decimal("100")
        assert position.position_type == PositionType.STOCK
        assert position.market_value == Decimal("15000.00")
        assert position.cost_basis == Decimal("14500.00")  # 145 * 100
        assert position.validate()

    def test_parse_option_position(self, mock_client):
        """Test parsing an option position."""
        data = {
            "symbol": "AAPL  231215C00150000",
            "quantity": "-10",
            "assetType": "OPTION",
            "marketValue": "-5000.00",
            "averagePrice": "5.00",
            "unrealizedPnL": "1000.00",
        }

        position = mock_client._parse_position(data)

        assert position.symbol == "AAPL  231215C00150000"
        assert position.quantity == Decimal("-10")
        assert position.position_type == PositionType.OPTION
        assert position.is_short()
        assert position.underlying == "AAPL"
        assert position.strike == Decimal("150")
        assert position.option_type == "CALL"
        assert position.expiration == datetime(2023, 12, 15)
        assert position.validate()

    @given(
        underlying=st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=6),
        year=st.integers(min_value=20, max_value=30),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
        option_type=st.sampled_from(["C", "P"]),
        strike=st.integers(min_value=1, max_value=999999),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_occ_symbol_parsing(
        self, mock_client, underlying, year, month, day, option_type, strike
    ):
        """Property test for OCC symbol parsing."""
        # Construct OCC symbol
        symbol = f"{underlying:<6}{year:02d}{month:02d}{day:02d}{option_type}{strike:08d}"

        result = mock_client._parse_option_symbol(symbol)

        if result:
            assert result["underlying"] == underlying.strip()
            assert result["option_type"] == "CALL" if option_type == "C" else "PUT"
            assert result["strike"] == Decimal(strike) / 1000

    def test_invalid_position_detection(self, mock_client):
        """Test detection of invalid position data."""
        # Option without required fields
        position = SchwabPosition(
            symbol="AAPL231215C150",
            quantity=Decimal("-10"),
            position_type=PositionType.OPTION,
            market_value=Decimal("-5000"),
            cost_basis=Decimal("5000"),
            unrealized_pnl=Decimal("0"),
            # Missing underlying, strike, expiration
        )

        assert not position.validate()

    def test_position_validation_edge_cases(self):
        """Test position validation edge cases."""
        # Zero quantity with non-zero market value
        position = SchwabPosition(
            symbol="AAPL",
            quantity=Decimal("0"),
            position_type=PositionType.STOCK,
            market_value=Decimal("1000"),  # Should be 0
            cost_basis=Decimal("0"),
            unrealized_pnl=Decimal("0"),
        )

        assert not position.validate()

        # Negative strike price
        position = SchwabPosition(
            symbol="AAPL231215C150",
            quantity=Decimal("-10"),
            position_type=PositionType.OPTION,
            market_value=Decimal("-5000"),
            cost_basis=Decimal("5000"),
            unrealized_pnl=Decimal("0"),
            underlying="AAPL",
            strike=Decimal("-150"),  # Invalid
            expiration=datetime(2023, 12, 15),
            option_type="CALL",
        )

        assert not position.validate()


class TestSchwabAccountParsing:
    """Test account parsing and validation."""

    def test_parse_margin_account(self, mock_client):
        """Test parsing a margin account."""
        data = {
            "securitiesAccount": {
                "accountNumber": "123456789",
                "type": "MARGIN",
                "currentBalances": {
                    "liquidationValue": "100000.00",
                    "cashBalance": "25000.00",
                    "buyingPower": "200000.00",
                    "marginBalance": "75000.00",
                    "maintenanceRequirement": "30000.00",
                    "maintenanceCall": "0.00",
                },
            }
        }

        account = mock_client._parse_account(data)

        assert account.account_number == "123456789"
        assert account.account_type == "MARGIN"
        assert account.total_value == Decimal("100000.00")
        assert account.cash_balance == Decimal("25000.00")
        assert account.buying_power == Decimal("200000.00")
        assert account.margin_utilization == Decimal("0.5")  # (100k - 200k) / 100k
        assert account.validate()

    def test_account_validation(self):
        """Test account validation rules."""
        # Negative total value
        account = SchwabAccount(
            account_number="123",
            account_type="MARGIN",
            total_value=Decimal("-1000"),
            cash_balance=Decimal("0"),
            buying_power=Decimal("0"),
        )

        assert not account.validate()

        # Margin requirement exceeds total value
        account = SchwabAccount(
            account_number="123",
            account_type="MARGIN",
            total_value=Decimal("10000"),
            cash_balance=Decimal("1000"),
            buying_power=Decimal("5000"),
            margin_requirement=Decimal("15000"),  # Too high
        )

        assert not account.validate()


class TestCorporateActionDetection:
    """Test corporate action detection."""

    def test_detect_stock_split(self, mock_client):
        """Test detection of potential stock splits."""
        positions = [
            SchwabPosition(
                symbol="AAPL",
                quantity=Decimal("150"),  # Odd lot
                position_type=PositionType.STOCK,
                market_value=Decimal("22500"),
                cost_basis=Decimal("20000"),
                unrealized_pnl=Decimal("2500"),
            )
        ]

        actions = mock_client.detect_corporate_actions(positions)

        assert len(actions) == 1
        assert actions[0]["type"] == "POTENTIAL_SPLIT"
        assert actions[0]["symbol"] == "AAPL"
        assert actions[0]["quantity"] == Decimal("150")

    def test_detect_option_adjustment(self, mock_client):
        """Test detection of option adjustments."""
        positions = [
            SchwabPosition(
                symbol="AAPL231215C00152500",  # Non-standard strike
                quantity=Decimal("-10"),
                position_type=PositionType.OPTION,
                market_value=Decimal("-5000"),
                cost_basis=Decimal("5000"),
                unrealized_pnl=Decimal("0"),
                underlying="AAPL",
                strike=Decimal("152.50"),
                expiration=datetime(2023, 12, 15),
                option_type="CALL",
            ),
            SchwabPosition(
                symbol="AAPL231215C00167250",  # Another non-standard
                quantity=Decimal("-5"),
                position_type=PositionType.OPTION,
                market_value=Decimal("-1000"),
                cost_basis=Decimal("1000"),
                unrealized_pnl=Decimal("0"),
                underlying="AAPL",
                strike=Decimal("167.25"),
                expiration=datetime(2023, 12, 15),
                option_type="CALL",
            ),
        ]

        actions = mock_client.detect_corporate_actions(positions)

        # Should detect the 167.25 strike as non-standard (not divisible by 0.50)
        assert any(a["type"] == "POTENTIAL_ADJUSTMENT" for a in actions)
        assert any(a["strike"] == Decimal("167.25") for a in actions)


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_network_error_retry(self, mock_client):
        """Test retry on network errors."""
        # Mock session to fail twice then succeed
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.json = AsyncMock(return_value={"positions": []})

        mock_client.session.request = AsyncMock(
            side_effect=[
                ClientError("Network error"),
                ClientError("Network error"),
                response_mock,
            ]
        )

        # Should retry and eventually succeed
        result = await mock_client._make_request("GET", "test")
        assert result == {"positions": []}
        assert mock_client.session.request.call_count == 3

    @pytest.mark.asyncio
    async def test_auth_error_handling(self, mock_client):
        """Test handling of authentication errors."""
        response_mock = AsyncMock()
        response_mock.status = 401

        mock_client.session.request = AsyncMock(return_value=response_mock)

        with pytest.raises(SchwabAuthError):
            await mock_client._make_request("GET", "test")

        # Token should be cleared
        assert mock_client.access_token is None

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_client):
        """Test handling of rate limiting."""
        response_mock = AsyncMock()
        response_mock.status = 429
        response_mock.headers = {"Retry-After": "60"}

        mock_client.session.request = AsyncMock(return_value=response_mock)

        with pytest.raises(SchwabRateLimitError) as exc_info:
            await mock_client._make_request("GET", "test")

        assert "60s" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_to_cached_data(self, mock_client):
        """Test fallback to last known good data."""
        # Set up last known good data
        good_positions = [
            SchwabPosition(
                symbol="AAPL",
                quantity=Decimal("100"),
                position_type=PositionType.STOCK,
                market_value=Decimal("15000"),
                cost_basis=Decimal("14000"),
                unrealized_pnl=Decimal("1000"),
            )
        ]
        mock_client.last_known_good["positions"] = good_positions

        # Make API fail
        mock_client._make_request = AsyncMock(side_effect=SchwabNetworkError("Connection failed"))

        # Should return cached data
        positions = await mock_client.get_positions()
        assert positions == good_positions


class TestCaching:
    """Test caching behavior."""

    @pytest.mark.asyncio
    async def test_account_caching(self, mock_client):
        """Test account data is cached briefly."""
        account_data = {
            "securitiesAccount": {
                "accountNumber": "123456789",
                "type": "MARGIN",
                "currentBalances": {
                    "liquidationValue": "100000.00",
                    "cashBalance": "25000.00",
                    "buyingPower": "200000.00",
                },
            }
        }

        mock_client._make_request = AsyncMock(return_value=account_data)
        mock_client._get_accounts = AsyncMock(return_value=[{"accountNumber": "123456789"}])

        # First call should hit API
        account1 = await mock_client.get_account()
        assert mock_client._make_request.call_count == 1

        # Second call should use cache
        account2 = await mock_client.get_account()
        assert mock_client._make_request.call_count == 1
        assert account1 == account2

    @pytest.mark.asyncio
    async def test_positions_not_cached(self, mock_client):
        """Test positions are never cached."""
        positions_data = {
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": "100",
                    "assetType": "EQUITY",
                    "marketValue": "15000.00",
                    "averagePrice": "145.00",
                }
            ]
        }

        mock_client._make_request = AsyncMock(return_value=positions_data)
        mock_client._get_accounts = AsyncMock(return_value=[{"accountNumber": "123456789"}])

        # Each call should hit API
        await mock_client.get_positions()
        assert mock_client._make_request.call_count == 1

        await mock_client.get_positions()
        assert mock_client._make_request.call_count == 2


class TestDataValidation:
    """Test comprehensive data validation."""

    @pytest.mark.asyncio
    async def test_position_quantity_reconciliation(self, mock_client):
        """Test validation of position quantities."""
        # Duplicate positions (should trigger warning)
        positions_data = {
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": "100",
                    "assetType": "EQUITY",
                    "marketValue": "15000.00",
                    "averagePrice": "150.00",
                },
                {
                    "symbol": "AAPL",  # Duplicate
                    "quantity": "50",
                    "assetType": "EQUITY",
                    "marketValue": "7500.00",
                    "averagePrice": "150.00",
                },
            ]
        }

        mock_client._make_request = AsyncMock(return_value=positions_data)
        mock_client._get_accounts = AsyncMock(return_value=[{"accountNumber": "123456789"}])

        with patch("src.unity_wheel.schwab.client.logger") as mock_logger:
            positions = await mock_client.get_positions()

            # Should parse both but warn about duplicates
            assert len(positions) == 2
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_stale_data_detection(self, mock_client):
        """Test detection of stale data."""
        # Load old cached data
        old_position = SchwabPosition(
            symbol="AAPL",
            quantity=Decimal("100"),
            position_type=PositionType.STOCK,
            market_value=Decimal("15000"),
            cost_basis=Decimal("14000"),
            unrealized_pnl=Decimal("1000"),
            last_update=datetime.now() - timedelta(hours=25),  # Old data
        )

        # Check if data is stale
        assert (datetime.now() - old_position.last_update).total_seconds() > 86400


class TestIntegration:
    """Integration tests with mocked responses."""

    @pytest.mark.asyncio
    async def test_full_portfolio_fetch(self, mock_client):
        """Test fetching complete portfolio data."""
        # Mock comprehensive portfolio response
        positions_data = {
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": "100",
                    "assetType": "EQUITY",
                    "marketValue": "17500.00",
                    "averagePrice": "150.00",
                    "unrealizedPnL": "2500.00",
                },
                {
                    "symbol": "MSFT",
                    "quantity": "50",
                    "assetType": "EQUITY",
                    "marketValue": "17500.00",
                    "averagePrice": "300.00",
                    "unrealizedPnL": "2500.00",
                },
                {
                    "symbol": "AAPL  231215P00150000",
                    "quantity": "-10",
                    "assetType": "OPTION",
                    "marketValue": "-2000.00",
                    "averagePrice": "3.00",
                    "unrealizedPnL": "1000.00",
                },
                {
                    "symbol": "MSFT  231215C00350000",
                    "quantity": "-5",
                    "assetType": "OPTION",
                    "marketValue": "-1500.00",
                    "averagePrice": "4.00",
                    "unrealizedPnL": "500.00",
                },
            ]
        }

        account_data = {
            "securitiesAccount": {
                "accountNumber": "123456789",
                "type": "MARGIN",
                "currentBalances": {
                    "liquidationValue": "100000.00",
                    "cashBalance": "25000.00",
                    "buyingPower": "150000.00",
                    "marginBalance": "75000.00",
                    "maintenanceRequirement": "35000.00",
                },
            }
        }

        mock_client._make_request = AsyncMock(side_effect=[positions_data, account_data])
        mock_client._get_accounts = AsyncMock(return_value=[{"accountNumber": "123456789"}])

        # Fetch portfolio
        positions = await mock_client.get_positions()
        account = await mock_client.get_account()

        # Verify positions
        assert len(positions) == 4

        stock_positions = [p for p in positions if p.position_type == PositionType.STOCK]
        option_positions = [p for p in positions if p.position_type == PositionType.OPTION]

        assert len(stock_positions) == 2
        assert len(option_positions) == 2

        # All options should be short
        assert all(p.is_short() for p in option_positions)

        # Verify account
        assert account.total_value == Decimal("100000")
        assert account.margin_utilization < Decimal("0.4")  # Healthy margin

        # Check for corporate actions
        actions = mock_client.detect_corporate_actions(positions)
        assert len(actions) == 0  # No issues with this portfolio


@pytest.mark.asyncio
class TestAsyncContextManager:
    """Test async context manager behavior."""

    async def test_context_manager(self):
        """Test client works as async context manager."""
        client = SchwabClient(
            client_id="test",
            client_secret="secret",
        )

        # Mock authentication
        client._authenticate = AsyncMock()

        async with client as c:
            assert c.session is not None
            assert isinstance(c, SchwabClient)

        # Session should be closed
        assert client.session is None

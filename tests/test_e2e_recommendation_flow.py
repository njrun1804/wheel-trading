"""End-to-end integration test for the full recommendation flow.

Tests the complete flow from data ingestion through recommendation generation,
simulating real-world usage patterns and failure scenarios.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config.loader import ConfigurationLoader, get_config_loader
from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.auth.auth_client import AuthClient
from unity_wheel.data_providers.databento.types import OptionChain, OptionQuote, OptionType
from unity_wheel.models.account import Account
from unity_wheel.models.position import Position
from unity_wheel.monitoring.diagnostics import SystemDiagnostics
from unity_wheel.schwab.types import AccountData, PositionData, PositionType
from unity_wheel.storage.cache.general_cache import CacheManager
from unity_wheel.storage.storage import Storage


class TestEndToEndRecommendationFlow:
    """Test complete recommendation flow with various scenarios."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create mock configuration for testing."""
        config = {
            "portfolio": {
                "total_capital": 100000,
                "margin_multiplier": 2.0,  # 2x margin available
                "use_margin": True,
                "cash_reserve": 0.0,  # 0% cash reserve per user preference
            },
            "strategy": {
                "delta_target": 0.30,
                "days_to_expiry_target": 45,
                "days_to_expiry_range": [30, 60],
                "profit_target": 0.50,
                "stop_loss": None,  # No stop loss
                "roll_when_tested": True,
                "roll_days_before_expiry": 21,
            },
            "risk": {
                "max_position_size": 1.0,  # 100% of capital per position allowed
                "max_portfolio_delta": 2.0,  # Allow leveraged delta
                "var_limit": 0.50,  # 50% VaR acceptable
                "cvar_limit": 0.75,  # 75% CVaR acceptable
                "kelly_fraction": 0.5,
                "max_margin_usage": 0.95,  # Use up to 95% of available margin
            },
            "data": {
                "cache_dir": str(temp_cache_dir),
                "schwab_cache_ttl": 30,
                "market_data_cache_ttl": 300,
            },
        }

        loader = ConfigurationLoader(config)
        with patch("src.config.loader._config_loader", loader):
            yield loader

    @pytest.fixture
    def mock_schwab_data(self):
        """Create mock Schwab account and position data."""
        account_data = AccountData(
            account_number="12345678",
            total_cash=20000.0,
            buying_power=150000.0,  # Reflects margin availability
            maintenance_requirement=30000.0,
            option_level=3,
            is_margin_account=True,
        )

        # Existing wheel position - short put
        positions = [
            PositionData(
                symbol="U  241220P00040000",  # Unity $40 put, Dec 20 2024
                underlying="U",
                quantity=-5,  # Short 5 contracts
                option_type="PUT",
                strike_price=40.0,
                expiration_date="2024-12-20",
                days_to_expiration=45,
                mark_price=1.20,
                delta=-0.28,
                gamma=0.015,
                theta=-0.08,
                vega=0.12,
                implied_volatility=0.35,
            )
        ]

        return account_data, positions

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        return OptionChain(
            underlying="U",
            expiration=datetime(2025, 1, 17),  # Next month expiry
            spot_price=Decimal("45.50"),
            timestamp=datetime.now(timezone.utc),
            calls=[],  # Not needed for put selling
            puts=[
                OptionQuote(
                    instrument_id=1,
                    timestamp=datetime.now(timezone.utc),
                    bid_price=Decimal("1.10"),
                    ask_price=Decimal("1.15"),
                    bid_size=50,
                    ask_size=75,
                    strike=Decimal("41"),
                    delta=Decimal("-0.25"),
                    gamma=Decimal("0.012"),
                    theta=Decimal("-0.07"),
                    vega=Decimal("0.11"),
                    implied_volatility=Decimal("0.32"),
                ),
                OptionQuote(
                    instrument_id=2,
                    timestamp=datetime.now(timezone.utc),
                    bid_price=Decimal("1.35"),
                    ask_price=Decimal("1.40"),
                    bid_size=100,
                    ask_size=100,
                    strike=Decimal("42"),
                    delta=Decimal("-0.30"),
                    gamma=Decimal("0.014"),
                    theta=Decimal("-0.08"),
                    vega=Decimal("0.13"),
                    implied_volatility=Decimal("0.33"),
                ),
                OptionQuote(
                    instrument_id=3,
                    timestamp=datetime.now(timezone.utc),
                    bid_price=Decimal("1.65"),
                    ask_price=Decimal("1.70"),
                    bid_size=80,
                    ask_size=90,
                    strike=Decimal("43"),
                    delta=Decimal("-0.35"),
                    gamma=Decimal("0.015"),
                    theta=Decimal("-0.09"),
                    vega=Decimal("0.14"),
                    implied_volatility=Decimal("0.34"),
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_full_recommendation_flow(
        self, mock_config, mock_schwab_data, mock_market_data, temp_cache_dir
    ):
        """Test complete flow from data fetch to recommendation."""
        account_data, position_data = mock_schwab_data

        # Mock external dependencies
        with (
            patch("src.unity_wheel.schwab.client.SchwabClient") as MockSchwabClient,
            patch("src.unity_wheel.databento.client.DatabentoClient") as MockDatabentoClient,
            patch("src.unity_wheel.auth.client.AuthClient") as MockAuthClient,
        ):

            # Setup mock Schwab client
            mock_schwab = AsyncMock()
            mock_schwab.get_account.return_value = account_data
            mock_schwab.get_positions.return_value = position_data
            MockSchwabClient.return_value.__aenter__.return_value = mock_schwab

            # Setup mock Databento client
            mock_databento = AsyncMock()
            mock_databento.get_option_chain.return_value = mock_market_data
            MockDatabentoClient.return_value = mock_databento

            # Setup mock Auth client
            mock_auth = AsyncMock()
            mock_auth.get_token.return_value = "mock_token"
            MockAuthClient.return_value = mock_auth

            # Initialize components
            advisor = WheelAdvisor()

            # Convert Schwab data to internal models
            account = Account(
                total_value=Decimal(
                    str(account_data.total_cash + 40 * 100 * 5 * 1.20)
                ),  # Cash + position value
                cash_balance=Decimal(str(account_data.total_cash)),
                buying_power=Decimal(str(account_data.buying_power)),
                maintenance_requirement=Decimal(str(account_data.maintenance_requirement)),
            )

            positions = [
                Position(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    position_type=PositionType.OPTION,
                    option_type=OptionType.PUT,
                    strike=Decimal(str(pos.strike_price)),
                    expiration=datetime.strptime(pos.expiration_date, "%Y-%m-%d"),
                    underlying="U",
                    cost_basis=Decimal("6.00"),  # $1.20 * 5 contracts
                    current_price=Decimal(str(pos.mark_price)),
                    multiplier=100,
                )
                for pos in position_data
            ]

            # Get recommendation
            result = advisor.advise_position(
                account=account,
                positions=positions,
                market_data={"U": mock_market_data},
                config=mock_config.config,
            )

            # Verify recommendation
            assert result is not None
            assert result.confidence >= 0.7, "Confidence should be high with good data"

            # Should recommend rolling the position since spot moved up
            assert result.primary_action.action_type == "roll"
            assert result.primary_action.contracts == 5
            assert result.primary_action.new_strike == 42  # Target ~0.30 delta

            # Verify risk metrics
            assert result.risk_metrics.portfolio_var <= 0.50, "VaR should be within limit"
            assert result.risk_metrics.margin_usage <= 0.95, "Margin usage should be within limit"

            # Verify reasoning
            assert "spot price moved" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_api_failure_recovery(self, mock_config, temp_cache_dir):
        """Test graceful handling of API failures."""
        with (
            patch("src.unity_wheel.schwab.client.SchwabClient") as MockSchwabClient,
            patch("src.unity_wheel.databento.client.DatabentoClient") as MockDatabentoClient,
        ):

            # Mock Schwab API failure
            mock_schwab = AsyncMock()
            mock_schwab.get_account.side_effect = Exception("API timeout")
            MockSchwabClient.return_value.__aenter__.return_value = mock_schwab

            advisor = WheelAdvisor()

            # Should return degraded recommendation
            result = advisor.advise_position(
                account=Account(
                    total_value=Decimal("100000"),
                    cash_balance=Decimal("100000"),
                    buying_power=Decimal("100000"),
                ),
                positions=[],
                market_data={},
                config=mock_config.config,
            )

            assert result is not None
            assert result.confidence < 0.5, "Confidence should be low with missing data"
            assert result.primary_action.action_type == "hold"
            assert "data unavailable" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_margin_call_scenario(self, mock_config, temp_cache_dir):
        """Test recommendations during margin call."""
        account = Account(
            total_value=Decimal("50000"),
            cash_balance=Decimal("-5000"),  # Negative cash
            buying_power=Decimal("0"),
            maintenance_requirement=Decimal("45000"),  # Close to margin call
        )

        positions = [
            Position(
                symbol="U  241220P00045000",
                quantity=-10,  # Large short position
                position_type=PositionType.OPTION,
                option_type=OptionType.PUT,
                strike=Decimal("45"),
                expiration=datetime.now() + timedelta(days=30),
                underlying="U",
                cost_basis=Decimal("20.00"),
                current_price=Decimal("3.50"),  # Position moved against us
                multiplier=100,
            )
        ]

        advisor = WheelAdvisor()
        result = advisor.advise_position(
            account=account,
            positions=positions,
            market_data={},
            config=mock_config.config,
        )

        # Should recommend reducing position
        assert result.primary_action.action_type in ["close", "reduce"]
        assert "margin" in result.reasoning.lower()
        assert result.risk_metrics.margin_usage > 0.90

    @pytest.mark.asyncio
    async def test_max_position_sizing(self, mock_config, mock_market_data, temp_cache_dir):
        """Test position sizing with 100% capital allocation allowed."""
        account = Account(
            total_value=Decimal("100000"),
            cash_balance=Decimal("100000"),
            buying_power=Decimal("200000"),  # 2x margin
        )

        advisor = WheelAdvisor()
        result = advisor.advise_position(
            account=account,
            positions=[],  # No existing positions
            market_data={"U": mock_market_data},
            config=mock_config.config,
        )

        # With 100% allocation allowed, should use Kelly sizing up to margin limit
        assert result.primary_action.action_type == "open"

        # Calculate expected position size
        # With $42 strike at $1.35 bid, each contract uses $4200 margin
        # With $200k buying power, could do up to ~47 contracts
        # Kelly sizing at 50% should recommend ~23 contracts
        assert 15 <= result.primary_action.contracts <= 30

        # Should use significant portion of available capital
        position_value = result.primary_action.contracts * 42 * 100
        assert position_value >= 50000  # At least 50% of capital

    @pytest.mark.asyncio
    async def test_cache_behavior(
        self, mock_config, mock_schwab_data, mock_market_data, temp_cache_dir
    ):
        """Test caching behavior in the flow."""
        account_data, position_data = mock_schwab_data

        with (
            patch("src.unity_wheel.schwab.client.SchwabClient") as MockSchwabClient,
            patch("src.unity_wheel.databento.client.DatabentoClient") as MockDatabentoClient,
        ):

            # Track API calls
            mock_schwab = AsyncMock()
            mock_schwab.get_account.return_value = account_data
            mock_schwab.get_positions.return_value = position_data
            MockSchwabClient.return_value.__aenter__.return_value = mock_schwab

            mock_databento = AsyncMock()
            mock_databento.get_option_chain.return_value = mock_market_data
            MockDatabentoClient.return_value = mock_databento

            # First call - should hit APIs
            advisor1 = WheelAdvisor()
            result1 = advisor.advise_position(
                account=Account(total_value=Decimal("100000")),
                positions=[],
                market_data={"U": mock_market_data},
                config=mock_config.config,
            )

            # Second call within cache TTL - should use cache
            advisor2 = WheelAdvisor()
            result2 = advisor.advise_position(
                account=Account(total_value=Decimal("100000")),
                positions=[],
                market_data={"U": mock_market_data},
                config=mock_config.config,
            )

            # Results should be identical
            assert result1.primary_action.action_type == result2.primary_action.action_type
            assert result1.primary_action.strike == result2.primary_action.strike

    @pytest.mark.asyncio
    async def test_diagnostic_integration(self, mock_config, temp_cache_dir):
        """Test system diagnostics integration."""
        diagnostics = SystemDiagnostics()

        # Run full diagnostic suite
        results = await diagnostics.run_all_diagnostics()

        # Should complete without errors
        assert results["status"] != "error"
        assert "environment" in results
        assert "configuration" in results
        assert "performance" in results

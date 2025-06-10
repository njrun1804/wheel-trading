"""Tests for aggressive capital allocation strategy.

User preference: 100% capital allocation with margin usage up to broker limits.
No cash reserve requirements.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.unity_wheel.api.advisor import WheelAdvisor
from src.unity_wheel.models.account import Account
from src.unity_wheel.models.position import OptionType, Position, PositionType
from src.unity_wheel.risk.analytics import RiskAnalytics
from src.unity_wheel.risk.limits import TradingLimits
from src.unity_wheel.strategy.wheel import WheelStrategy


class TestAggressiveAllocation:
    """Test aggressive allocation strategies."""

    @pytest.fixture
    def aggressive_config(self):
        """Create aggressive configuration."""
        config = {
            "risk": {
                "max_position_size": 1.00,
                "max_margin_percent": 0.95,
                "max_drawdown_percent": 0.50,
                "limits": {
                    "max_var_95": 0.50,
                    "max_cvar_95": 0.75,
                    "max_kelly_fraction": 1.00,
                    "max_delta_exposure": 200.0,
                    "max_gamma_exposure": 50.0,
                    "max_vega_exposure": 5000.0,
                    "max_contracts_per_trade": 100,
                    "max_notional_percent": 2.00,
                },
            }
        }
        return config

    def test_full_capital_deployment(self, aggressive_config):
        """Test that strategy can deploy 100% of capital."""
        account = Account(
            total_value=Decimal("100000"),
            cash_balance=Decimal("100000"),
            buying_power=Decimal("200000"),  # 2x margin
        )

        strategy = WheelStrategy()
        limits = TradingLimits(aggressive_config["risk"]["limits"])

        # Calculate maximum position size
        max_size = strategy._calculate_position_size(
            account_value=account.total_value,
            buying_power=account.buying_power,
            strike=Decimal("100"),
            premium=Decimal("2.00"),
            config=aggressive_config,
        )

        # Should be able to use full buying power
        # Each contract requires $10,000 margin (100 shares * $100)
        # With $200k buying power, can do 20 contracts
        assert max_size >= 15  # Allow for some Kelly sizing reduction

        # Verify limits allow this
        position_value = max_size * 100 * 100  # contracts * multiplier * strike
        assert limits.check_position_size(position_value, account.total_value)

    def test_margin_utilization(self, aggressive_config):
        """Test margin usage up to 95% of available."""
        account = Account(
            total_value=Decimal("100000"),
            cash_balance=Decimal("20000"),
            buying_power=Decimal("150000"),
            maintenance_requirement=Decimal("30000"),
        )

        analytics = RiskAnalytics()
        limits = TradingLimits(aggressive_config["risk"]["limits"])

        # Create leveraged position
        positions = [
            Position(
                symbol="U  250117P00045000",
                quantity=-30,  # Large short position
                position_type=PositionType.OPTION,
                option_type=OptionType.PUT,
                strike=Decimal("45"),
                expiration=datetime.now() + timedelta(days=45),
                underlying="U",
                cost_basis=Decimal("45.00"),  # $1.50 * 30 contracts
                current_price=Decimal("1.50"),
                multiplier=100,
                delta=Decimal("-0.30"),
            )
        ]

        # Calculate margin usage
        margin_usage = analytics._calculate_margin_usage(
            positions=positions,
            account=account,
            spot_prices={"U": Decimal("46.00")},
        )

        # Should allow up to 95% margin usage
        assert limits.check_margin_usage(margin_usage)
        assert margin_usage <= 0.95

    def test_no_cash_reserve_requirement(self, aggressive_config):
        """Test that no cash reserve is required."""
        account = Account(
            total_value=Decimal("100000"),
            cash_balance=Decimal("0"),  # All capital deployed
            buying_power=Decimal("100000"),  # Margin available
        )

        advisor = WheelAdvisor()
        limits = TradingLimits(aggressive_config["risk"]["limits"])

        # Should still be able to make recommendations
        result = advisor._evaluate_new_position(
            account=account,
            target_delta=Decimal("-0.30"),
            available_strikes=[
                {"strike": Decimal("40"), "delta": Decimal("-0.25"), "premium": Decimal("1.00")},
                {"strike": Decimal("41"), "delta": Decimal("-0.30"), "premium": Decimal("1.20")},
                {"strike": Decimal("42"), "delta": Decimal("-0.35"), "premium": Decimal("1.40")},
            ],
            config=aggressive_config,
        )

        assert result is not None
        assert result["action"] == "open"  # Should recommend opening position

    def test_high_risk_tolerance(self, aggressive_config):
        """Test acceptance of high VaR and CVaR."""
        positions = []
        # Create a concentrated portfolio
        for i in range(5):
            positions.append(
                Position(
                    symbol=f"U  250117P000{40+i}000",
                    quantity=-20,  # Large positions
                    position_type=PositionType.OPTION,
                    option_type=OptionType.PUT,
                    strike=Decimal(str(40 + i)),
                    expiration=datetime.now() + timedelta(days=30),
                    underlying="U",
                    cost_basis=Decimal("30.00"),
                    current_price=Decimal("1.50"),
                    multiplier=100,
                    delta=Decimal("-0.40"),  # High delta
                    gamma=Decimal("0.02"),
                )
            )

        analytics = RiskAnalytics()
        limits = TradingLimits(aggressive_config["risk"]["limits"])

        metrics = analytics.calculate_portfolio_metrics(
            positions=positions,
            account_value=Decimal("100000"),
            spot_prices={"U": Decimal("42.00")},
        )

        # High VaR and CVaR should be acceptable
        assert metrics.var_95 <= 0.50  # Up to 50% VaR allowed
        assert metrics.cvar_95 <= 0.75  # Up to 75% CVaR allowed

        # Limits should pass
        assert limits.check_var_limit(metrics.var_95, Decimal("100000"))
        assert limits.check_risk_metrics(metrics, Decimal("100000"))

    def test_leveraged_delta_exposure(self, aggressive_config):
        """Test leveraged delta exposure beyond 100%."""
        positions = [
            Position(
                symbol="U  250117P00045000",
                quantity=-50,  # 50 short puts
                position_type=PositionType.OPTION,
                option_type=OptionType.PUT,
                strike=Decimal("45"),
                expiration=datetime.now() + timedelta(days=45),
                underlying="U",
                cost_basis=Decimal("75.00"),
                current_price=Decimal("1.50"),
                multiplier=100,
                delta=Decimal("-0.30"),  # Total delta = -1500
            )
        ]

        analytics = RiskAnalytics()
        limits = TradingLimits(aggressive_config["risk"]["limits"])

        total_delta = analytics._calculate_portfolio_delta(positions)

        # Should allow leveraged delta (up to 200%)
        assert abs(total_delta) <= 200.0 * 50  # 200 delta per contract * 50
        assert limits.check_delta_exposure(total_delta)

    def test_kelly_sizing_with_full_allocation(self, aggressive_config):
        """Test Kelly criterion with 100% allocation allowed."""
        strategy = WheelStrategy()

        # High win rate scenario
        win_rate = 0.85
        avg_win = Decimal("500")
        avg_loss = Decimal("1000")

        kelly_fraction = strategy._calculate_kelly_fraction(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            config=aggressive_config,
        )

        # With 85% win rate and 1:2 risk/reward, Kelly suggests large position
        # But we apply half-Kelly (0.5) for safety
        expected_kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        expected_fraction = min(expected_kelly * 0.5, 1.0)  # Cap at 100%

        assert abs(kelly_fraction - expected_fraction) < 0.1
        assert kelly_fraction <= 1.0  # Should cap at 100%

    def test_max_contracts_per_trade(self, aggressive_config):
        """Test no artificial limit on contracts per trade."""
        account = Account(
            total_value=Decimal("1000000"),  # $1M account
            cash_balance=Decimal("500000"),
            buying_power=Decimal("2000000"),  # 2x margin
        )

        strategy = WheelStrategy()
        limits = TradingLimits(aggressive_config["risk"]["limits"])

        # Calculate position size for low-priced underlying
        max_contracts = strategy._calculate_position_size(
            account_value=account.total_value,
            buying_power=account.buying_power,
            strike=Decimal("10"),  # $10 strike
            premium=Decimal("0.20"),
            config=aggressive_config,
        )

        # Should allow large number of contracts
        assert max_contracts > 50  # No artificial 10-contract limit
        assert limits.check_contracts_limit(max_contracts)

    def test_portfolio_concentration(self, aggressive_config):
        """Test that portfolio can be concentrated in single position."""
        account = Account(
            total_value=Decimal("100000"),
            cash_balance=Decimal("5000"),
            buying_power=Decimal("150000"),
        )

        # Single large position using most of capital
        positions = [
            Position(
                symbol="U  250117P00045000",
                quantity=-20,
                position_type=PositionType.OPTION,
                option_type=OptionType.PUT,
                strike=Decimal("45"),
                expiration=datetime.now() + timedelta(days=45),
                underlying="U",
                cost_basis=Decimal("30.00"),
                current_price=Decimal("1.50"),
                multiplier=100,
                delta=Decimal("-0.30"),
            )
        ]

        analytics = RiskAnalytics()

        # Position value = 20 * 100 * 45 = $90,000 (90% of portfolio)
        position_value = abs(positions[0].quantity) * positions[0].multiplier * positions[0].strike
        concentration = position_value / account.total_value

        assert concentration > 0.85  # High concentration

        # Should still be acceptable under aggressive limits
        metrics = analytics.calculate_portfolio_metrics(
            positions=positions,
            account_value=account.total_value,
            spot_prices={"U": Decimal("45.00")},
        )

        limits = TradingLimits(aggressive_config["risk"]["limits"])
        assert limits.check_risk_metrics(metrics, account.total_value)

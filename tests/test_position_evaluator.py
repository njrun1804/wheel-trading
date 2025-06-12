"""Test position evaluation and comparison framework."""

import pytest
from unity_wheel.models.position import Position
from unity_wheel.strategy.position_evaluator import PositionEvaluator, PositionValue, SwitchAnalysis


class TestPositionEvaluator:
    """Test position evaluation and switching logic."""

    def test_evaluate_position_short_put(self):
        """Test evaluating a short put position."""
        evaluator = PositionEvaluator()

        # Create a short put position
        position = Position(symbol="U250718P00030000", quantity=-1)

        # Evaluate the position
        result = evaluator.evaluate_position(
            position=position,
            current_price=35.0,
            risk_free_rate=0.05,
            volatility=0.45,
            days_to_expiry=45,
            bid=1.20,
            ask=1.30,
            contracts=1,
        )

        # Verify results
        assert isinstance(result, PositionValue)
        assert result.strike == 30.0
        assert result.current_value == 1.25  # Mid price
        assert result.time_value > 0  # OTM put has time value
        assert 0 < result.probability_itm < 0.5  # OTM probability
        assert result.expected_profit > 0  # Should be profitable
        assert result.confidence > 0.5

    def test_analyze_switch_beneficial(self):
        """Test analyzing a beneficial switch opportunity."""
        evaluator = PositionEvaluator()

        # Current position: Short $30 put with 15 DTE
        current_position = Position(symbol="U250625P00030000", quantity=-1)

        # Analyze switch to $32 put with 45 DTE
        analysis = evaluator.analyze_switch(
            current_position=current_position,
            current_bid=0.40,
            current_ask=0.50,
            current_dte=15,
            new_strike=32.0,
            new_expiry_days=45,
            new_bid=1.50,
            new_ask=1.60,
            underlying_price=35.0,
            volatility=0.45,
            risk_free_rate=0.05,
            contracts=1,
            min_benefit_threshold=25.0,
        )

        # Verify analysis
        assert isinstance(analysis, SwitchAnalysis)
        assert analysis.current_position.strike == 30.0
        assert analysis.new_position.strike == 32.0

        # Check costs
        assert analysis.close_commission == 0.65
        assert analysis.open_commission == 0.65
        assert analysis.total_switch_cost > 1.30  # Includes spread costs

        # This should be beneficial (collecting more premium)
        assert analysis.new_expected_value > analysis.current_expected_value
        assert analysis.breakeven_days != float("inf")
        assert analysis.breakeven_days > 0

    def test_analyze_switch_not_beneficial(self):
        """Test analyzing a non-beneficial switch."""
        evaluator = PositionEvaluator()

        # Current position: Short $32 put with 40 DTE (good position)
        current_position = Position(symbol="U250801P00032000", quantity=-1)

        # Analyze switch to slightly different strike
        analysis = evaluator.analyze_switch(
            current_position=current_position,
            current_bid=1.40,
            current_ask=1.50,
            current_dte=40,
            new_strike=31.0,
            new_expiry_days=42,
            new_bid=1.20,
            new_ask=1.30,
            underlying_price=35.0,
            volatility=0.45,
            risk_free_rate=0.05,
            contracts=1,
        )

        # Should not recommend switch (minimal benefit)
        assert not analysis.should_switch
        assert "below minimum threshold" in analysis.rationale

    def test_find_best_switch_opportunity(self):
        """Test finding best switch from multiple options."""
        evaluator = PositionEvaluator()

        # Current position
        current_position = Position(symbol="U250625P00030000", quantity=-1)

        # Available strikes: (strike, dte, bid, ask)
        available_strikes = [
            (29.0, 30, 0.80, 0.90),  # Slightly lower, less time
            (31.0, 45, 1.40, 1.50),  # Higher strike, more time
            (32.0, 45, 1.80, 1.90),  # Even higher strike
            (30.0, 60, 1.60, 1.70),  # Same strike, more time
        ]

        best = evaluator.find_best_switch_opportunity(
            current_position=current_position,
            current_bid=0.40,
            current_ask=0.50,
            current_dte=15,
            available_strikes=available_strikes,
            underlying_price=35.0,
            volatility=0.45,
            risk_free_rate=0.05,
            contracts=1,
        )

        # Should find a beneficial switch
        assert best is not None
        assert best.should_switch
        assert best.new_position.strike in [31.0, 32.0]  # Higher strikes
        assert best.switch_benefit > 0

    def test_position_value_calculations(self):
        """Test detailed position value calculations."""
        evaluator = PositionEvaluator()

        # Short put near the money
        position = Position(symbol="U250718P00035000", quantity=-1)

        result = evaluator.evaluate_position(
            position=position,
            current_price=35.0,
            risk_free_rate=0.05,
            volatility=0.50,
            days_to_expiry = config.trading.target_dte,
            bid=2.00,
            ask=2.10,
            contracts=1,
        )

        # ATM put should have significant value
        assert result.current_value > 1.5
        assert result.probability_itm > 0.4  # Near 50% for ATM
        assert result.delta < -0.3  # Negative delta for put
        assert result.theta < 0  # Time decay

    def test_switch_with_high_volatility(self):
        """Test switching decisions during high volatility."""
        evaluator = PositionEvaluator()

        current_position = Position(symbol="U250625P00030000", quantity=-1)

        # High volatility scenario
        analysis = evaluator.analyze_switch(
            current_position=current_position,
            current_bid=1.50,
            current_ask=1.60,
            current_dte=20,
            new_strike=28.0,  # Lower strike for safety
            new_expiry_days=35,
            new_bid=1.80,
            new_ask=1.90,
            underlying_price=32.0,  # Price dropped
            volatility=0.80,  # High volatility
            risk_free_rate=0.05,
            contracts=1,
        )

        # In high vol, might want to roll down and out
        assert analysis is not None
        assert analysis.new_position.strike < analysis.current_position.strike


@pytest.mark.parametrize(
    "contracts,expected_benefit_multiplier",
    [
        (1, 1.0),
        (5, 5.0),
        (10, 10.0),
    ],
)
def test_contract_scaling(contracts, expected_benefit_multiplier):
    """Test that switch benefits scale with contract size."""
    evaluator = PositionEvaluator()

    current_position = Position(symbol="U250625P00030000", quantity=-contracts)

    analysis = evaluator.analyze_switch(
        current_position=current_position,
        current_bid=0.40,
        current_ask=0.50,
        current_dte=15,
        new_strike=32.0,
        new_expiry_days=45,
        new_bid=1.50,
        new_ask=1.60,
        underlying_price=35.0,
        volatility=0.45,
        risk_free_rate=0.05,
        contracts=contracts,
    )

    # Benefit should scale with contracts
    base_benefit = 100  # Approximate base benefit
    assert abs(analysis.switch_benefit) > base_benefit * expected_benefit_multiplier * 0.5

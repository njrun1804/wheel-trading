"""Position evaluation and comparison framework with switching cost analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..math.options import black_scholes_price_validated, probability_itm_validated
from ..models.position import Position, PositionType
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PositionValue:
    """Expected value metrics for a position."""

    symbol: str
    strike: float
    expiry_days: int
    position_type: PositionType

    # Value components
    current_value: float  # Current mid price
    intrinsic_value: float  # Max(S-K, 0) for calls, Max(K-S, 0) for puts
    time_value: float  # Current value - intrinsic

    # Expected returns
    expected_profit: float  # Expected profit to expiry
    daily_expected_return: float  # Profit / DTE
    annualized_return: float  # Daily return * 365

    # Risk metrics
    probability_itm: float  # Probability of finishing ITM
    expected_loss_if_assigned: float  # Loss if assigned
    max_risk: float  # Maximum potential loss

    # Greeks
    delta: float
    theta: float
    gamma: float
    vega: float

    # Confidence
    confidence: float  # Overall confidence in calculations


@dataclass
class SwitchAnalysis:
    """Analysis of switching from one position to another."""

    current_position: PositionValue
    new_position: PositionValue

    # Switching costs
    close_commission: float  # Commission to close current
    open_commission: float  # Commission to open new
    close_spread_cost: float  # Half bid-ask spread to close
    open_spread_cost: float  # Half bid-ask spread to open
    total_switch_cost: float  # Total cost to switch

    # Value comparison
    current_expected_value: float  # Expected value keeping current
    new_expected_value: float  # Expected value after switch
    switch_benefit: float  # New value - current value - switch cost

    # Normalized metrics
    switch_benefit_pct: float  # Switch benefit as % of capital
    daily_return_improvement: float  # Improvement in daily return
    risk_adjusted_benefit: float  # Benefit adjusted for risk change
    breakeven_days: float  # Days to recoup switching cost

    # Decision
    should_switch: bool
    confidence: float
    rationale: str


class PositionEvaluator:
    """Evaluates and compares option positions with switching analysis."""

    def __init__(
        self,
        commission_per_contract: float = 0.65,
        spread_cost_multiplier: float = 0.5,  # Assume we pay half the spread
    ):
        """
        Initialize position evaluator.

        Parameters
        ----------
        commission_per_contract : float
            Commission per option contract
        spread_cost_multiplier : float
            Fraction of bid-ask spread paid (0.5 = half spread)
        """
        self.commission = commission_per_contract
        self.spread_multiplier = spread_cost_multiplier

    def evaluate_position(
        self,
        position: Position,
        current_price: float,
        risk_free_rate: float,
        volatility: float,
        days_to_expiry: int,
        bid: float,
        ask: float,
        contracts: int = 1,
    ) -> PositionValue:
        """
        Evaluate expected value of a position.

        Parameters
        ----------
        position : Position
            Option position to evaluate
        current_price : float
            Current underlying price
        risk_free_rate : float
            Risk-free rate
        volatility : float
            Implied volatility
        days_to_expiry : int
            Days to expiration
        bid : float
            Current bid price
        ask : float
            Current ask price
        contracts : int
            Number of contracts

        Returns
        -------
        PositionValue
            Comprehensive position evaluation
        """
        # Calculate time to expiry in years
        T = days_to_expiry / 365.0

        # Current value (mid price)
        current_value = (bid + ask) / 2.0

        # Calculate intrinsic value
        if position.position_type == PositionType.PUT:
            intrinsic_value = max(position.strike - current_price, 0)
        else:  # CALL
            intrinsic_value = max(current_price - position.strike, 0)

        time_value = current_value - intrinsic_value

        # Calculate probability ITM (using available function)
        prob_result = probability_itm_validated(
            S=current_price,
            K=position.strike,
            T=T,
            r=risk_free_rate,
            sigma=volatility,
            option_type="put" if position.position_type == PositionType.PUT else "call",
        )

        # Calculate theoretical value
        theoretical = black_scholes_price_validated(
            S=current_price,
            K=position.strike,
            T=T,
            r=risk_free_rate,
            sigma=volatility,
            option_type="put" if position.position_type == PositionType.PUT else "call",
        )

        # Probability of finishing ITM
        probability_itm = prob_result.value if prob_result.confidence > 0 else 0.5

        # Expected profit/loss calculations
        if position.is_short:  # Selling options
            # We collect premium, maximum profit is keeping it all
            max_profit = current_value * 100 * contracts

            # Expected loss if assigned
            if position.position_type == PositionType.PUT:
                # If put is assigned, we buy at strike, sell at market
                loss_if_assigned = max(position.strike - current_price, 0) * 100 * contracts
            else:  # CALL
                # If call is assigned, we sell at strike, buy at market
                loss_if_assigned = max(current_price - position.strike, 0) * 100 * contracts

            # Expected profit = premium kept * P(OTM) - loss * P(ITM)
            expected_profit = (
                max_profit * (1 - probability_itm) - loss_if_assigned * probability_itm
            )

            # Max risk for short put is strike price (stock goes to 0)
            if position.position_type == PositionType.PUT:
                max_risk = position.strike * 100 * contracts
            else:  # Short call has unlimited risk, cap at 2x strike
                max_risk = position.strike * 200 * contracts

        else:  # Long options
            # We pay premium, maximum loss is premium paid
            max_loss = current_value * 100 * contracts

            # Expected profit if ITM
            if position.position_type == PositionType.PUT:
                profit_if_itm = max(position.strike - current_price, 0) * 100 * contracts
            else:  # CALL
                profit_if_itm = max(current_price - position.strike, 0) * 100 * contracts

            # Expected profit = profit * P(ITM) - premium
            expected_profit = profit_if_itm * probability_itm - max_loss

            # Max risk is premium paid
            max_risk = max_loss

        # Calculate returns
        daily_return = expected_profit / days_to_expiry if days_to_expiry > 0 else 0
        annualized_return = daily_return * 365

        # Overall confidence based on calculations
        confidence = (
            min(prob_result.confidence, theoretical.confidence)
            if prob_result.confidence > 0
            else 0.5
        )

        return PositionValue(
            symbol=position.symbol,
            strike=position.strike,
            expiry_days=days_to_expiry,
            position_type=position.position_type,
            current_value=current_value,
            intrinsic_value=intrinsic_value,
            time_value=time_value,
            expected_profit=expected_profit,
            daily_expected_return=daily_return,
            annualized_return=annualized_return,
            probability_itm=probability_itm,
            expected_loss_if_assigned=loss_if_assigned if position.is_short else 0,
            max_risk=max_risk,
            # Approximate Greeks
            delta=(
                -probability_itm if position.position_type == PositionType.PUT else probability_itm
            ),
            theta=-time_value / days_to_expiry if days_to_expiry > 0 else 0,  # Simplified
            gamma=0.0,  # Would need separate calculation
            vega=0.0,  # Would need separate calculation
            confidence=confidence,
        )

    def analyze_switch(
        self,
        current_position: Position,
        current_bid: float,
        current_ask: float,
        current_dte: int,
        new_strike: float,
        new_expiry_days: int,
        new_bid: float,
        new_ask: float,
        underlying_price: float,
        volatility: float,
        risk_free_rate: float,
        contracts: int = 1,
        min_benefit_threshold: float = 50.0,  # Minimum $50 benefit to switch
    ) -> SwitchAnalysis:
        """
        Analyze whether to switch from current position to new position.

        Parameters
        ----------
        current_position : Position
            Current option position
        current_bid : float
            Current position bid price
        current_ask : float
            Current position ask price
        current_dte : int
            Current position days to expiry
        new_strike : float
            New position strike price
        new_expiry_days : int
            New position days to expiry
        new_bid : float
            New position bid price
        new_ask : float
            New position ask price
        underlying_price : float
            Current underlying price
        volatility : float
            Implied volatility
        risk_free_rate : float
            Risk-free rate
        contracts : int
            Number of contracts
        min_benefit_threshold : float
            Minimum benefit required to switch

        Returns
        -------
        SwitchAnalysis
            Complete switching analysis with recommendation
        """
        # Evaluate current position
        current_eval = self.evaluate_position(
            position=current_position,
            current_price=underlying_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            days_to_expiry=current_dte,
            bid=current_bid,
            ask=current_ask,
            contracts=contracts,
        )

        # Create hypothetical new position
        # Generate proper OCC symbol with a future date
        from datetime import datetime, timedelta

        new_expiry_date = datetime.now() + timedelta(days=new_expiry_days)
        yy = new_expiry_date.strftime("%y")
        mm = new_expiry_date.strftime("%m")
        dd = new_expiry_date.strftime("%d")
        option_type = "P" if current_position.position_type == PositionType.PUT else "C"

        new_position = Position(
            symbol=f"{current_position.underlying}{yy}{mm}{dd}{option_type}{int(new_strike * 1000):08d}",
            quantity=-contracts if current_position.is_short else contracts,
        )

        # Evaluate new position
        new_eval = self.evaluate_position(
            position=new_position,
            current_price=underlying_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            days_to_expiry=new_expiry_days,
            bid=new_bid,
            ask=new_ask,
            contracts=contracts,
        )

        # Calculate switching costs
        close_commission = self.commission * contracts
        open_commission = self.commission * contracts

        # Spread costs (we pay to cross the spread)
        current_spread = current_ask - current_bid
        new_spread = new_ask - new_bid

        close_spread_cost = current_spread * self.spread_multiplier * 100 * contracts
        open_spread_cost = new_spread * self.spread_multiplier * 100 * contracts

        total_switch_cost = (
            close_commission + open_commission + close_spread_cost + open_spread_cost
        )

        # Expected values
        current_expected = current_eval.expected_profit
        new_expected = new_eval.expected_profit

        # Switch benefit
        switch_benefit = new_expected - current_expected - total_switch_cost

        # Calculate normalized metrics
        capital_required = new_eval.max_risk
        switch_benefit_pct = (
            (switch_benefit / capital_required * 100) if capital_required > 0 else 0
        )

        # Daily return improvement
        daily_improvement = new_eval.daily_expected_return - current_eval.daily_expected_return

        # Days required to recover switching cost
        breakeven_days = (
            total_switch_cost / daily_improvement if daily_improvement > 0 else float("inf")
        )

        # Risk-adjusted benefit (penalize if new position is riskier)
        risk_ratio = new_eval.probability_itm / (
            current_eval.probability_itm + 0.01
        )  # Avoid div by zero
        risk_adjusted_benefit = switch_benefit / risk_ratio

        # Decision logic
        should_switch = False
        rationale = ""

        if switch_benefit < min_benefit_threshold:
            rationale = f"Switch benefit ${switch_benefit:.2f} below minimum threshold ${min_benefit_threshold:.2f}"
        elif switch_benefit_pct < 0.5:  # Less than 0.5% improvement
            rationale = f"Switch benefit {switch_benefit_pct:.2f}% too small relative to capital"
        elif daily_improvement < 0:
            rationale = "New position has lower daily return"
        elif new_eval.probability_itm > current_eval.probability_itm * 1.5:
            rationale = "New position has significantly higher assignment risk"
        else:
            should_switch = True
            rationale = (
                f"Switch provides ${switch_benefit:.2f} benefit "
                f"({switch_benefit_pct:.1f}% of capital) with "
                f"${daily_improvement:.2f}/day improvement"
            )

        # Calculate confidence
        confidence = (
            min(current_eval.confidence, new_eval.confidence) * 0.9
        )  # Reduce for complexity

        return SwitchAnalysis(
            current_position=current_eval,
            new_position=new_eval,
            close_commission=close_commission,
            open_commission=open_commission,
            close_spread_cost=close_spread_cost,
            open_spread_cost=open_spread_cost,
            total_switch_cost=total_switch_cost,
            current_expected_value=current_expected,
            new_expected_value=new_expected,
            switch_benefit=switch_benefit,
            switch_benefit_pct=switch_benefit_pct,
            daily_return_improvement=daily_improvement,
            risk_adjusted_benefit=risk_adjusted_benefit,
            breakeven_days=breakeven_days,
            should_switch=should_switch,
            confidence=confidence,
            rationale=rationale,
        )

    def find_best_switch_opportunity(
        self,
        current_position: Position,
        current_bid: float,
        current_ask: float,
        current_dte: int,
        available_strikes: List[Tuple[float, int, float, float]],  # (strike, dte, bid, ask)
        underlying_price: float,
        volatility: float,
        risk_free_rate: float,
        contracts: int = 1,
    ) -> Optional[SwitchAnalysis]:
        """
        Find the best switching opportunity from available strikes.

        Parameters
        ----------
        current_position : Position
            Current position
        current_bid : float
            Current position bid
        current_ask : float
            Current position ask
        current_dte : int
            Current position DTE
        available_strikes : List[Tuple[float, int, float, float]]
            List of (strike, dte, bid, ask) tuples
        underlying_price : float
            Current underlying price
        volatility : float
            Implied volatility
        risk_free_rate : float
            Risk-free rate
        contracts : int
            Number of contracts

        Returns
        -------
        Optional[SwitchAnalysis]
            Best switching opportunity if found, None otherwise
        """
        best_analysis = None
        best_benefit = -float("inf")

        for strike, dte, bid, ask in available_strikes:
            # Skip if too close to current position
            if abs(strike - current_position.strike) < 0.50 and abs(dte - current_dte) < 7:
                continue

            # Analyze this switch opportunity
            try:
                analysis = self.analyze_switch(
                    current_position=current_position,
                    current_bid=current_bid,
                    current_ask=current_ask,
                    current_dte=current_dte,
                    new_strike=strike,
                    new_expiry_days=dte,
                    new_bid=bid,
                    new_ask=ask,
                    underlying_price=underlying_price,
                    volatility=volatility,
                    risk_free_rate=risk_free_rate,
                    contracts=contracts,
                )

                # Track best opportunity
                if analysis.switch_benefit > best_benefit:
                    best_benefit = analysis.switch_benefit
                    best_analysis = analysis

            except Exception as e:
                logger.warning(
                    f"Failed to analyze switch to {strike}/{dte}",
                    extra={"error": str(e), "strike": strike, "dte": dte},
                )
                continue

        return best_analysis if best_analysis and best_analysis.should_switch else None

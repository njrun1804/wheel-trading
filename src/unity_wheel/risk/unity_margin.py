"""Unity-specific margin calculations with account type differentiation.

This module provides accurate margin calculations for Unity (U) options trading,
accounting for the stock's high volatility and different account types.
"""
from __future__ import annotations


import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..config.loader import get_config

from ..utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))


@dataclass
class MarginResult:
    """Result of margin calculation."""

    margin_required: float
    margin_type: str  # 'cash', 'standard', 'unity_adjusted'
    calculation_method: str
    account_type: str
    confidence: float = 0.99
    details: Dict[str, float] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class UnityMarginCalculator:
    """Calculate margin requirements for Unity options with volatility adjustments."""

    # Unity volatility adjustment factor
    UNITY_VOLATILITY_MULTIPLIER = 1.5

    # Standard margin requirements
    STANDARD_MARGIN_PERCENT = 0.20  # 20% of underlying
    MINIMUM_MARGIN_PERCENT = 0.10  # 10% minimum
    MINIMUM_PER_SHARE = 2.50  # $2.50 per share minimum

    def __init__(self):
        """Initialize with configuration."""
        self.config = get_config()

    def calculate_unity_margin(
        self,
        contracts: int,
        strike: float,
        current_price: float,
        premium_received: float,
        account_type: str = "margin",
        option_type: str = "put",
    ) -> MarginResult:
        """
        Calculate margin requirement for Unity options.

        Args:
            contracts: Number of option contracts
            strike: Strike price
            current_price: Current Unity stock price
            premium_received: Premium received per contract (in dollars)
            account_type: 'ira', 'cash', or 'margin'
            option_type: 'put' or 'call'

        Returns:
            MarginResult with calculated margin and details
        """
        # Validate inputs
        if contracts <= 0:
            return MarginResult(
                margin_required=0,
                margin_type="invalid",
                calculation_method="invalid_input",
                account_type=account_type,
                confidence=0.0,
            )

        # Calculate notional value
        notional_value = contracts * 100 * strike
        total_premium = contracts * premium_received

        # IRA accounts require full cash securing
        if account_type.lower() == "ira":
            margin = notional_value  # 100% cash secured

            logger.info(
                "ira_margin_calculated",
                extra={
                    "contracts": contracts,
                    "strike": strike,
                    "margin": margin,
                    "account_type": account_type,
                },
            )

            return MarginResult(
                margin_required=margin,
                margin_type="cash",
                calculation_method="ira_full_cash",
                account_type=account_type,
                details={"notional_value": notional_value, "cash_secured_percent": 1.0},
            )

        # Cash account - also requires full securing for puts
        if account_type.lower() == "cash" and option_type.lower() == "put":
            margin = notional_value  # 100% cash secured

            return MarginResult(
                margin_required=margin,
                margin_type="cash",
                calculation_method="cash_secured_put",
                account_type=account_type,
                details={"notional_value": notional_value, "cash_secured_percent": 1.0},
            )

        # Margin account - calculate standard margin requirement
        # Standard margin formula for short puts:
        # Greater of:
        # 1. 20% of underlying - OTM amount + premium
        # 2. 10% of strike price + premium
        # 3. $2.50 per share + premium

        # Calculate OTM amount (0 if ITM)
        otm_amount = (
            max(0, current_price - strike)
            if option_type.lower() == "put"
            else max(0, strike - current_price)
        )
        otm_per_contract = otm_amount * 100

        # Method 1: 20% of underlying - OTM + premium
        method1_base = (self.STANDARD_MARGIN_PERCENT * current_price * 100) - otm_per_contract
        method1 = max(0, method1_base - premium_received)  # Premium reduces margin

        # Method 2: 10% of strike + premium
        method2_base = self.MINIMUM_MARGIN_PERCENT * strike * 100
        method2 = max(0, method2_base - premium_received)

        # Method 3: $2.50 per share + premium
        method3_base = self.MINIMUM_PER_SHARE * 100
        method3 = max(0, method3_base - premium_received)

        # Standard margin is the greater of all methods
        standard_margin_per_contract = max(method1, method2, method3)
        standard_margin_total = contracts * standard_margin_per_contract

        # Apply Unity volatility adjustment
        unity_adjusted_margin = standard_margin_total * self.UNITY_VOLATILITY_MULTIPLIER

        # Determine which method was binding
        if standard_margin_per_contract == method1:
            method_used = "20%_underlying_minus_otm"
        elif standard_margin_per_contract == method2:
            method_used = "10%_strike"
        else:
            method_used = "minimum_per_share"

        logger.info(
            "unity_margin_calculated",
            extra={
                "contracts": contracts,
                "strike": strike,
                "current_price": current_price,
                "premium_received": premium_received,
                "standard_margin": standard_margin_total,
                "unity_adjusted": unity_adjusted_margin,
                "method_used": method_used,
                "volatility_multiplier": self.UNITY_VOLATILITY_MULTIPLIER,
            },
        )

        return MarginResult(
            margin_required=unity_adjusted_margin,
            margin_type="unity_adjusted",
            calculation_method=method_used,
            account_type=account_type,
            details={
                "standard_margin": standard_margin_total,
                "unity_multiplier": self.UNITY_VOLATILITY_MULTIPLIER,
                "method1_margin": method1 * contracts,
                "method2_margin": method2 * contracts,
                "method3_margin": method3 * contracts,
                "otm_amount": otm_amount,
                "premium_offset": total_premium,
            },
        )

    def calculate_portfolio_margin(
        self,
        contracts: int,
        strike: float,
        current_price: float,
        premium_received: float,
        implied_volatility: float,
        account_type: str = "portfolio",
    ) -> MarginResult:
        """
        Calculate portfolio margin for sophisticated accounts.

        Portfolio margin uses risk-based calculations rather than
        fixed percentages, but we'll apply Unity adjustment.

        Args:
            contracts: Number of contracts
            strike: Strike price
            current_price: Current stock price
            premium_received: Premium per contract
            implied_volatility: Implied volatility (decimal)
            account_type: Account type (should be 'portfolio')

        Returns:
            MarginResult with portfolio margin calculation
        """
        # Simplified portfolio margin calculation
        # Real brokers use complex stress testing

        # Base calculation: 15% move stress test
        stress_move_percent = 0.15

        # Adjust stress move based on Unity's volatility
        unity_stress_move = stress_move_percent * (1 + implied_volatility)

        # Calculate potential loss on stress move
        stressed_price = current_price * (1 - unity_stress_move)

        # Loss if assigned at stressed price
        if stressed_price < strike:
            loss_per_contract = (strike - stressed_price) * 100
        else:
            loss_per_contract = 0

        # Margin is potential loss minus premium
        margin_per_contract = max(0, loss_per_contract - premium_received)
        total_margin = contracts * margin_per_contract

        # Apply Unity adjustment even to portfolio margin
        unity_adjusted = total_margin * self.UNITY_VOLATILITY_MULTIPLIER

        return MarginResult(
            margin_required=unity_adjusted,
            margin_type="portfolio_unity_adjusted",
            calculation_method="stress_test_15pct",
            account_type=account_type,
            details={
                "stress_move_percent": unity_stress_move,
                "stressed_price": stressed_price,
                "potential_loss": loss_per_contract * contracts,
                "standard_portfolio_margin": total_margin,
                "unity_multiplier": self.UNITY_VOLATILITY_MULTIPLIER,
            },
        )

    def get_margin_by_account_type(
        self,
        contracts: int,
        strike: float,
        current_price: float,
        premium_received: float,
        account_type: str,
        implied_volatility: Optional[float] = None,
        option_type: str = "put",
    ) -> MarginResult:
        """
        Calculate margin based on account type with proper routing.

        Args:
            contracts: Number of contracts
            strike: Strike price
            current_price: Current stock price
            premium_received: Premium per contract
            account_type: 'ira', 'cash', 'margin', or 'portfolio'
            implied_volatility: IV for portfolio margin calculations
            option_type: 'put' or 'call'

        Returns:
            MarginResult with appropriate calculation
        """
        account_type_lower = account_type.lower()

        # Route to appropriate calculation
        if account_type_lower == "portfolio" and implied_volatility is not None:
            return self.calculate_portfolio_margin(
                contracts=contracts,
                strike=strike,
                current_price=current_price,
                premium_received=premium_received,
                implied_volatility=implied_volatility,
                account_type=account_type,
            )
        else:
            return self.calculate_unity_margin(
                contracts=contracts,
                strike=strike,
                current_price=current_price,
                premium_received=premium_received,
                account_type=account_type,
                option_type=option_type,
            )


def calculate_unity_margin_requirement(
    contracts: int,
    strike: float,
    current_price: float,
    premium_received: float,
    account_type: str = "margin",
    option_type: str = "put",
) -> Tuple[float, Dict[str, float]]:
    """
    Convenience function for quick margin calculations.

    Returns:
        (margin_required, details_dict)
    """
    calculator = UnityMarginCalculator()
    result = calculator.calculate_unity_margin(
        contracts=contracts,
        strike=strike,
        current_price=current_price,
        premium_received=premium_received,
        account_type=account_type,
        option_type=option_type,
    )
    return result.margin_required, result.details
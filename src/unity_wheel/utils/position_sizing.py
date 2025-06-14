"""Dynamic position sizing utilities for the wheel trading system.

This module provides intelligent position sizing based on:
- Account value and buying power
- Kelly criterion and confidence
- Risk limits and volatility
- Option premium and margin requirements
"""
from __future__ import annotations


import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config.loader import get_config

from ..risk.unity_margin import UnityMarginCalculator
from .logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    contracts: int
    notional_value: float
    margin_required: float
    position_pct: float
    warnings: list[str]
    sizing_method: str
    confidence: float = 0.95  # Default high confidence

    @property
    def is_valid(self) -> bool:
        """Check if position size is valid."""
        return self.contracts > 0 and len(self.warnings) == 0


class DynamicPositionSizer:
    """Calculate dynamic position sizes based on multiple factors."""

    def __init__(self):
        """Initialize with configuration."""
        self.config = get_config()
        self.margin_calculator = UnityMarginCalculator()
        self._load_limits()

    def _load_limits(self) -> None:
        """Load risk limits from configuration."""
        risk_config = self.config.risk

        # Position limits
        self.max_position_pct = risk_config.max_position_size
        self.max_margin_pct = risk_config.max_margin_percent

        # Contract limits (for safety)
        self.min_contracts = 1
        self.max_contracts_absolute = risk_config.limits.max_contracts_per_trade

        # Notional limits
        self.max_notional_pct = risk_config.limits.max_notional_percent

        logger.info(
            "position_sizer_initialized",
            extra={
                "max_position_pct": self.max_position_pct,
                "max_margin_pct": self.max_margin_pct,
                "max_contracts": self.max_contracts_absolute,
            },
        )

    def calculate_position_size(
        self,
        portfolio_value: float,
        buying_power: float,
        strike_price: float,
        option_premium: float,
        kelly_fraction: float,
        volatility_factor: float = 1.0,
        confidence: float = 1.0,
        existing_exposure: float = 0.0,
        account_type: str = "margin",
        current_price: Optional[float] = None,
    ) -> PositionSizeResult:
        """Calculate optimal position size with risk constraints.

        CODEX GUIDANCE:
        - Always validate inputs with die() for required data
        - Return PositionSizeResult with confidence score
        - Log all decisions with structured logging
        - Check against MAX_POSITION_SIZE constant (20%)
        - Enforce Unity-specific MAX_CONCURRENT_PUTS (3)

        Parameters
        ----------
        portfolio_value : float
            Total portfolio value (must be positive)
        buying_power : float
            Available buying power (cash + margin available)
        strike_price : float
            Strike price of the option (must be positive)
        option_premium : float
            Premium per contract in dollars (must be positive)
        kelly_fraction : float
            Kelly criterion fraction (0-1, typically 0.25-0.5)
        volatility_factor : float, optional
            Volatility adjustment factor (0.1-3.0, default 1.0)
        confidence : float, optional
            Confidence in the trade (0-1, default 1.0)
        existing_exposure : float, optional
            Current portfolio exposure in dollars (default 0)
        account_type : str, optional
            Account type: 'ira', 'cash', 'margin', 'portfolio' (default 'margin')
        current_price : float, optional
            Current stock price (defaults to strike if not provided)

        Returns
        -------
        PositionSizeResult
            Contains:
            - contracts: Number of contracts to trade
            - position_value: Total position value in dollars
            - margin_required: Margin requirement
            - confidence: Confidence in the sizing (0-1)
            - warnings: List of any warnings
            - constraints_applied: Dict of which limits were hit

        Examples
        --------
        >>> sizer = DynamicPositionSizer(config)
        >>> result = sizer.calculate_position_size(
        ...     portfolio_value = config.trading.portfolio_value,
        ...     buying_power=200000,
        ...     strike_price=45.0,
        ...     option_premium=1.50,
        ...     kelly_fraction=0.25
        ... )
        >>> if result.confidence > 0.7:
        ...     print(f"Trade {result.contracts} contracts")

        See Also
        --------
        - patterns/position_sizing_examples.py for more examples
        - tests/test_position_sizing.py for property tests
        - RiskAnalytics.calculate_kelly_criterion() for Kelly calculation

        Notes
        -----
        The position sizing algorithm applies multiple constraints:
        1. Kelly criterion with safety factor
        2. Maximum position size (20% of portfolio)
        3. Unity-specific limit (3 concurrent puts)
        4. Account type restrictions (IRA: cash-secured only)
        5. Margin availability checks
        """
        warnings = []

        # 1. Kelly-based position size
        kelly_position_value = portfolio_value * kelly_fraction * volatility_factor * confidence

        # 2. Calculate notional value per contract
        notional_per_contract = strike_price * 100  # 100 shares per contract

        # 3. Calculate margin requirement using Unity-specific calculator
        # Use current price if provided, otherwise assume ATM
        if current_price is None:
            current_price = strike_price

        margin_result = self.margin_calculator.calculate_unity_margin(
            contracts=1,  # Calculate per contract first
            strike=strike_price,
            current_price=current_price,
            premium_received=option_premium,
            account_type=account_type,
            option_type="put",  # Wheel strategy uses puts
        )
        margin_per_contract = margin_result.margin_required

        # 4. Calculate max contracts based on different constraints

        # a) Kelly-based contracts
        kelly_contracts = int(kelly_position_value / notional_per_contract)

        # b) Margin-based contracts
        max_margin_usage = buying_power * self.max_margin_pct
        margin_contracts = int(max_margin_usage / margin_per_contract)

        # c) Position limit contracts
        max_position_value = portfolio_value * self.max_position_pct
        remaining_capacity = max_position_value - existing_exposure
        position_contracts = int(remaining_capacity / notional_per_contract)

        # d) Notional limit contracts
        max_notional = portfolio_value * self.max_notional_pct
        notional_contracts = int(max_notional / notional_per_contract)

        # 5. Take minimum of all constraints
        contracts = min(
            kelly_contracts,
            margin_contracts,
            position_contracts,
            notional_contracts,
            self.max_contracts_absolute,
        )

        # 6. Apply minimum contract requirement
        if contracts < self.min_contracts:
            contracts = 0
            warnings.append(f"Position too small: {contracts} < {self.min_contracts} minimum")

        # 7. Add warnings for binding constraints
        sizing_method = "kelly"

        if contracts == margin_contracts and contracts < kelly_contracts:
            warnings.append(f"Limited by margin: {margin_contracts} contracts")
            sizing_method = "margin"

        if contracts == position_contracts and contracts < kelly_contracts:
            warnings.append(f"Limited by position size: {position_contracts} contracts")
            sizing_method = "position_limit"

        if contracts == notional_contracts and contracts < kelly_contracts:
            warnings.append(f"Limited by notional: {notional_contracts} contracts")
            sizing_method = "notional_limit"

        if contracts == self.max_contracts_absolute and contracts < kelly_contracts:
            warnings.append(f"Limited by absolute max: {self.max_contracts_absolute} contracts")
            sizing_method = "absolute_limit"

        # 8. Final calculations
        final_notional = contracts * notional_per_contract
        final_margin = contracts * margin_per_contract
        position_pct = final_notional / portfolio_value if portfolio_value > 0 else 0

        # 9. Safety checks
        if position_pct > self.max_position_pct:
            warnings.append(f"Position exceeds max % of portfolio: {position_pct:.1%}")

        if final_margin > buying_power:
            warnings.append(
                f"Insufficient buying power: ${final_margin:,.0f} > ${buying_power:,.0f}"
            )
            contracts = 0  # Can't execute

        # Log decision
        logger.info(
            "position_size_calculated",
            extra={
                "contracts": contracts,
                "kelly_contracts": kelly_contracts,
                "margin_contracts": margin_contracts,
                "position_contracts": position_contracts,
                "notional_contracts": notional_contracts,
                "sizing_method": sizing_method,
                "position_pct": position_pct,
                "warnings": warnings,
            },
        )

        # Calculate confidence based on constraints and warnings
        confidence = 0.95  # Start with high confidence

        # Reduce confidence for each constraint that limited position
        if sizing_method != "kelly":
            confidence *= 0.90  # Position was constrained

        # Reduce confidence for warnings
        if warnings:
            confidence *= max(0.70, 1.0 - len(warnings) * 0.05)

        # Reduce confidence for extreme Kelly fractions
        if kelly_fraction < 0.10 or kelly_fraction > 0.40:
            confidence *= 0.85

        return PositionSizeResult(
            contracts=contracts,
            notional_value=final_notional,
            margin_required=final_margin,
            position_pct=position_pct,
            warnings=warnings,
            sizing_method=sizing_method,
            confidence=confidence,
        )

    def adjust_for_small_account(
        self,
        portfolio_value: float,
        min_trade_value: float = 5000,
    ) -> Tuple[int, str]:
        """
        Special handling for small accounts.

        Args:
            portfolio_value: Total portfolio value
            min_trade_value: Minimum sensible trade value

        Returns:
            (min_contracts, warning_message)
        """
        if portfolio_value < min_trade_value:
            return (
                0,
                f"Account too small: ${portfolio_value:,.0f} < ${min_trade_value:,.0f} minimum",
            )

        if portfolio_value < 25000:
            # Small account: limit to 1-2 contracts max
            return 1, "Small account: limited to 1 contract positions"

        if portfolio_value < 50000:
            # Medium account: allow up to 5 contracts
            return 5, "Medium account: limited to 5 contracts maximum"

        # Larger accounts: no special restrictions
        return self.max_contracts_absolute, ""

    def contracts_for_trade(
        self,
        portfolio_value: float,
        buying_power: float,
        strike_price: float,
        option_premium: float,
        **kwargs,
    ) -> int:
        """Determine contracts using defaults and risk limits."""

        result = self.calculate_position_size(
            portfolio_value=portfolio_value,
            buying_power=buying_power,
            strike_price=strike_price,
            option_premium=option_premium,
            kelly_fraction=self.config.risk.kelly_fraction,
            **kwargs,
        )
        return result.contracts

    def calculate_from_recommendation(
        self,
        portfolio_value: float,
        buying_power: float,
        recommendation: Dict,
        account_type: str = "margin",
    ) -> PositionSizeResult:
        """
        Calculate position size from a recommendation dict.

        Convenience method that extracts necessary fields.
        """
        # Extract required fields with defaults
        strike = recommendation.get("strike", 0)
        premium = recommendation.get("premium", 0)
        kelly = recommendation.get("kelly_fraction", 0.25)
        vol_factor = recommendation.get("volatility_factor", 1.0)
        confidence = recommendation.get("confidence", 0.8)
        current_price = recommendation.get("current_price", strike)

        # Check for small accounts
        max_for_account, warning = self.adjust_for_small_account(portfolio_value)

        result = self.calculate_position_size(
            portfolio_value=portfolio_value,
            buying_power=buying_power,
            strike_price=strike,
            option_premium=premium * 100,  # Convert to dollars
            kelly_fraction=kelly,
            volatility_factor=vol_factor,
            confidence=confidence,
            account_type=account_type,
            current_price=current_price,
        )

        # Apply small account limits
        if result.contracts > max_for_account:
            result.contracts = max_for_account
            result.warnings.append(warning)
            result.sizing_method = "account_size_limit"

            # Recalculate values
            result.notional_value = result.contracts * strike * 100
            # Use Unity margin calculator for accurate margin
            margin_recalc = self.margin_calculator.calculate_unity_margin(
                contracts=result.contracts,
                strike=strike,
                current_price=strike,  # Assume ATM if not provided
                premium_received=premium * 100,
                account_type="margin",  # Default to margin for recommendation
                option_type="put",
            )
            result.margin_required = margin_recalc.margin_required
            result.position_pct = result.notional_value / portfolio_value

        return result


def calculate_dynamic_contracts(
    portfolio_value: float,
    buying_power: float,
    strike_price: float,
    option_premium: float,
    kelly_fraction: float = 0.25,
    account_type: str = "margin",
    current_price: Optional[float] = None,
    **kwargs,
) -> int:
    """
    Simple interface for dynamic contract calculation.

    Returns just the number of contracts.
    """
    sizer = DynamicPositionSizer()
    result = sizer.calculate_position_size(
        portfolio_value=portfolio_value,
        buying_power=buying_power,
        strike_price=strike_price,
        option_premium=option_premium,
        kelly_fraction=kelly_fraction,
        account_type=account_type,
        current_price=current_price,
        **kwargs,
    )
    return result.contracts
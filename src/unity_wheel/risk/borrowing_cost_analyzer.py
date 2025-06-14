"""Borrowing cost analyzer for capital allocation decisions.

Helps determine whether to use borrowed funds or pay down debt
when considering positions.
"""
from __future__ import annotations


import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from ..math import CalculationResult

import numpy as np

from src.config.loader import get_config
from ..utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))


@dataclass
class BorrowingSource:
    """Represents a source of borrowed funds."""

    name: str
    balance: float
    annual_rate: float  # As decimal (0.07 for 7%)
    minimum_payment: float = 0.0
    is_revolving: bool = True  # Can re-borrow if paid down

    @property
    def daily_rate(self) -> float:
        """Daily interest rate."""
        return self.annual_rate / 365

    @property
    def monthly_rate(self) -> float:
        """Monthly interest rate."""
        return self.annual_rate / 12

    def daily_cost(self, amount: Optional[float] = None) -> float:
        """Calculate daily borrowing cost."""
        principal = amount if amount is not None else self.balance
        return principal * self.daily_rate

    def monthly_cost(self, amount: Optional[float] = None) -> float:
        """Calculate monthly borrowing cost."""
        principal = amount if amount is not None else self.balance
        return principal * self.monthly_rate

    def cost_for_period(self, days: int, amount: Optional[float] = None) -> float:
        """Calculate borrowing cost for a specific period."""
        principal = amount if amount is not None else self.balance
        return principal * self.daily_rate * days


@dataclass
class CapitalAllocationResult:
    """Result of capital allocation analysis."""

    action: str  # 'invest', 'paydown_debt', 'mixed'
    invest_amount: float
    paydown_amount: float
    source_to_use: Optional[str] = None  # Which borrowing source if investing
    hurdle_rate: float = 0.0  # Minimum return needed
    expected_return: float = 0.0
    borrowing_cost: float = 0.0
    net_benefit: float = 0.0
    confidence: float = 0.95
    reasoning: str = ""
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BorrowingCostAnalyzer:
    """Analyze borrowing costs and optimal capital allocation."""

    # Default borrowing sources
    AMEX_LOAN_RATE = 0.07  # 7% APR
    SCHWAB_MARGIN_RATE = 0.10  # 10% APR

    # No artificial adjustments - pure math only
    CONFIDENCE_MULTIPLIER = 1.0  # No safety factor
    TAX_ADJUSTMENT = 1.0  # Tax-free environment

    def __init__(self, rate_fetcher: Optional[Callable[[str], float]] = None, auto_update: bool = False):
        """Initialize with configuration.

        Parameters
        ----------
        rate_fetcher : Callable[[str], float] or None
            Optional callback to fetch real-time borrowing rates.
        auto_update : bool
            If ``True`` the analyzer fetches fresh rates before each
            allocation analysis.
        """
        self.config = get_config()
        self.sources: Dict[str, BorrowingSource] = {}
        self.rate_fetcher = rate_fetcher
        self.auto_update = auto_update
        self._setup_default_sources()

    def _setup_default_sources(self):
        """Set up default borrowing sources."""
        # Amex loan
        self.add_source(
            BorrowingSource(
                name="amex_loan",
                balance=45000,
                annual_rate=self.AMEX_LOAN_RATE,
                minimum_payment=1000,  # Assumed
                is_revolving=False,
            )
        )

        # Schwab margin
        self.add_source(
            BorrowingSource(
                name="schwab_margin",
                balance=0,  # Available but not used yet
                annual_rate=self.SCHWAB_MARGIN_RATE,
                minimum_payment=0,
                is_revolving=True,
            )
        )

    def add_source(self, source: BorrowingSource):
        """Add a borrowing source."""
        self.sources[source.name] = source
        logger.info(
            "borrowing_source_added",
            extra={
                "name": source.name,
                "rate": f"{source.annual_rate:.1%}",
                "balance": source.balance,
            },
        )

    def update_rates(self) -> Dict[str, float]:
        """Update borrowing rates using the configured fetcher."""
        if not self.rate_fetcher:
            return {}

        updates: Dict[str, float] = {}
        for name, source in self.sources.items():
            try:
                new_rate = self.rate_fetcher(name)
                if new_rate is not None and new_rate > 0:
                    source.annual_rate = new_rate
                    updates[name] = new_rate
            except (ValueError, KeyError, AttributeError) as exc:  # pragma: no cover - defensive
                logger.warning(
                    "rate_update_failed",
                    extra={"source": name, "error": str(exc)},
                )

        if updates:
            logger.info("rates_updated", extra={"updates": updates})

        return updates

    def calculate_hurdle_rate(
        self, borrowing_source: str, holding_period_days: int = 45, include_tax: bool = True
    ) -> CalculationResult:
        """
        Calculate the minimum return needed to justify borrowing.

        Args:
            borrowing_source: Name of borrowing source
            holding_period_days: Expected holding period
            include_tax: Whether to account for taxes on gains

        Returns:
            Hurdle rate as annualized percentage
        """
        if borrowing_source not in self.sources:
            raise ValueError(f"Unknown borrowing source: {borrowing_source}")

        source = self.sources[borrowing_source]

        # Base hurdle rate is the borrowing cost
        base_rate = source.annual_rate

        logger.info(
            "hurdle_rate_calculated",
            extra={
                "source": borrowing_source,
                "hurdle_rate": f"{base_rate:.1%}",
                "holding_days": holding_period_days,
            },
        )

        hurdle = base_rate * self.CONFIDENCE_MULTIPLIER

        if include_tax and self.TAX_ADJUSTMENT != 0:
            hurdle /= self.TAX_ADJUSTMENT

        return CalculationResult(hurdle, 0.95, [])

    def analyze_position_allocation(
        self,
        position_size: float,
        expected_annual_return: float,
        holding_period_days: int = 45,
        available_cash: float = 0,
        confidence: float = 0.8,
    ) -> CapitalAllocationResult:
        """
        Analyze whether to use borrowed funds or pay down debt.

        Args:
            position_size: Size of position to consider
            expected_annual_return: Expected annualized return (as decimal)
            holding_period_days: Expected holding period
            available_cash: Cash available without borrowing
            confidence: Confidence in expected return (0-1)

        Returns:
            CapitalAllocationResult with recommendation
        """
        # Optionally refresh borrowing rates
        if self.auto_update:
            self.update_rates()

        # Adjust expected return by confidence
        adjusted_return = expected_annual_return * confidence

        # Find cheapest borrowing source
        cheapest_source = min(self.sources.values(), key=lambda s: s.annual_rate)

        # Calculate hurdle rate for cheapest source
        hurdle_rate = self.calculate_hurdle_rate(cheapest_source.name, holding_period_days)

        # Calculate expected profit from position
        period_return = adjusted_return * (holding_period_days / 365)
        expected_profit = position_size * period_return

        # Calculate borrowing cost if we need to borrow
        need_to_borrow = max(0, position_size - available_cash)
        borrowing_cost = 0
        source_to_use = None

        if need_to_borrow > 0:
            # Find best source to borrow from
            source_to_use = self._select_borrowing_source(need_to_borrow)
            if source_to_use:
                borrowing_cost = source_to_use.cost_for_period(holding_period_days, need_to_borrow)

        # Net benefit calculation
        net_benefit = expected_profit - borrowing_cost

        # Decision logic
        if adjusted_return < hurdle_rate:
            # Return too low - pay down debt instead
            action = "paydown_debt"
            invest_amount = 0
            paydown_amount = position_size
            reasoning = (
                f"Expected return ({adjusted_return:.1%}) below hurdle rate "
                f"({hurdle_rate:.1%}). Better to pay down {cheapest_source.name}."
            )

        elif need_to_borrow > 0 and net_benefit < 0:
            # Borrowing cost exceeds profit
            action = "paydown_debt"
            invest_amount = 0
            paydown_amount = position_size
            reasoning = (
                f"Borrowing cost (${borrowing_cost:.0f}) exceeds expected "
                f"profit (${expected_profit:.0f}). Pay down debt instead."
            )

        elif need_to_borrow > 0:
            # Profitable even with borrowing
            action = "invest"
            invest_amount = position_size
            paydown_amount = 0
            reasoning = (
                f"Expected profit (${expected_profit:.0f}) exceeds borrowing "
                f"cost (${borrowing_cost:.0f}) by ${net_benefit:.0f}."
            )

        else:
            # Can invest with cash on hand
            action = "invest"
            invest_amount = position_size
            paydown_amount = 0
            reasoning = "Sufficient cash available. No borrowing needed."

        # Build result
        result = CapitalAllocationResult(
            action=action,
            invest_amount=invest_amount,
            paydown_amount=paydown_amount,
            source_to_use=source_to_use.name if source_to_use else None,
            hurdle_rate=hurdle_rate,
            expected_return=adjusted_return,
            borrowing_cost=borrowing_cost,
            net_benefit=net_benefit,
            confidence=confidence,
            reasoning=reasoning,
            details={
                "expected_profit": expected_profit,
                "need_to_borrow": need_to_borrow,
                "holding_days": holding_period_days,
                "cheapest_source": cheapest_source.name,
                "all_sources": {
                    name: {
                        "rate": f"{source.annual_rate:.1%}",
                        "balance": source.balance,
                        "daily_cost": source.daily_cost(),
                    }
                    for name, source in self.sources.items()
                },
            },
        )

        logger.info(
            "capital_allocation_analyzed",
            extra={
                "action": action,
                "position_size": position_size,
                "expected_return": f"{expected_annual_return:.1%}",
                "hurdle_rate": f"{hurdle_rate:.1%}",
                "net_benefit": net_benefit,
                "confidence": confidence,
            },
        )

        return result

    def _select_borrowing_source(self, amount: float) -> Optional[BorrowingSource]:
        """Select optimal borrowing source for given amount."""
        # Sort by rate (cheapest first)
        sorted_sources = sorted(self.sources.values(), key=lambda s: s.annual_rate)

        for source in sorted_sources:
            # Check if source is available
            # For revolving credit, always available
            # For term loans, only if we haven't borrowed yet (balance > 0 means existing loan)
            if source.is_revolving:
                return source

        # If no revolving sources, return cheapest regardless
        # This assumes we can refinance or take additional loans
        return sorted_sources[0] if sorted_sources else None

    def calculate_paydown_benefit(
        self, paydown_amount: float, source_name: str, time_horizon_days: int = 365
    ) -> Dict[str, float]:
        """
        Calculate the benefit of paying down debt.

        Args:
            paydown_amount: Amount to pay down
            source_name: Which debt to pay down
            time_horizon_days: Time horizon for benefit calculation

        Returns:
            Dict with interest saved and effective return
        """
        if source_name not in self.sources:
            raise ValueError(f"Unknown source: {source_name}")

        source = self.sources[source_name]

        # Interest saved over time horizon
        interest_saved = source.cost_for_period(time_horizon_days, paydown_amount)

        # Effective annualized return from paying down debt
        effective_return = source.annual_rate

        # Tax-free environment
        after_tax_benefit = interest_saved

        return {
            "interest_saved": interest_saved,
            "effective_return": effective_return,
            "after_tax_benefit": after_tax_benefit,
            "daily_savings": source.daily_cost(paydown_amount),
            "monthly_savings": source.monthly_cost(paydown_amount),
        }

    def optimize_capital_deployment(
        self,
        available_capital: float,
        opportunities: List[Dict[str, float]],
        max_leverage: float = 1.5,
    ) -> Dict[str, CapitalAllocationResult]:
        """
        Optimize capital deployment across multiple opportunities.

        Args:
            available_capital: Total capital available
            opportunities: List of investment opportunities with
                         'size', 'expected_return', 'confidence'
            max_leverage: Maximum leverage ratio (1.5 = can borrow 50% more)

        Returns:
            Dict mapping opportunity index to allocation decision
        """
        # Calculate max deployable capital
        max_capital = available_capital * max_leverage

        # Sort opportunities by risk-adjusted return
        sorted_opps = sorted(
            enumerate(opportunities),
            key=lambda x: x[1]["expected_return"] * x[1].get("confidence", 1.0),
            reverse=True,
        )

        allocations = {}
        remaining_capital = available_capital
        total_borrowed = 0

        for idx, opp in sorted_opps:
            # Check if we can/should take this opportunity
            result = self.analyze_position_allocation(
                position_size=opp["size"],
                expected_annual_return=opp["expected_return"],
                holding_period_days=opp.get("holding_days", 45),
                available_cash=remaining_capital,
                confidence=opp.get("confidence", 0.8),
            )

            if result.action == "invest" and (
                remaining_capital > 0 or total_borrowed < (max_capital - available_capital)
            ):
                allocations[f"opportunity_{idx}"] = result
                remaining_capital -= min(result.invest_amount, remaining_capital)
                if result.source_to_use:
                    total_borrowed += max(0, result.invest_amount - remaining_capital)
            else:
                # Skip this opportunity
                result.action = "skip"
                result.reasoning = "Better opportunities available or leverage limit reached"
                allocations[f"opportunity_{idx}"] = result

        return allocations

    def get_current_borrowing_summary(self) -> Dict[str, Dict]:
        """Get summary of current borrowing costs."""
        summary = {}
        total_daily_cost = 0
        total_monthly_cost = 0
        total_balance = 0

        for name, source in self.sources.items():
            daily_cost = source.daily_cost()
            monthly_cost = source.monthly_cost()

            summary[name] = {
                "balance": source.balance,
                "annual_rate": f"{source.annual_rate:.1%}",
                "daily_cost": daily_cost,
                "monthly_cost": monthly_cost,
                "annual_cost": source.balance * source.annual_rate,
                "is_revolving": source.is_revolving,
            }

            total_daily_cost += daily_cost
            total_monthly_cost += monthly_cost
            total_balance += source.balance

        # Add totals
        summary["totals"] = {
            "total_debt": total_balance,
            "daily_cost": total_daily_cost,
            "monthly_cost": total_monthly_cost,
            "annual_cost": total_daily_cost * 365,
            "blended_rate": (
                f"{(total_daily_cost * 365 / total_balance):.1%}" if total_balance > 0 else "0.0%"
            ),
        }

        return summary


def analyze_borrowing_decision(
    position_size: float, expected_return: float, confidence: float = 0.8, available_cash: float = 0
) -> CapitalAllocationResult:
    """
    Convenience function for quick borrowing analysis.

    Args:
        position_size: Size of position
        expected_return: Expected annualized return
        confidence: Confidence in return
        available_cash: Cash available without borrowing

    Returns:
        CapitalAllocationResult with recommendation
    """
    analyzer = BorrowingCostAnalyzer()
    return analyzer.analyze_position_allocation(
        position_size=position_size,
        expected_annual_return=expected_return,
        confidence=confidence,
        available_cash=available_cash,
    )
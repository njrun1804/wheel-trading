"""Pure mathematical borrowing cost analyzer without safety factors.

This module provides exact financial calculations for capital allocation
decisions in a tax-free environment.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from src.config.loader import get_config

from ..utils.logging import StructuredLogger

logger = StructuredLogger(logging.getLogger(__name__))


@dataclass
class LoanTerms:
    """Exact loan terms and calculations."""

    name: str
    principal: float
    annual_rate: float
    is_revolving: bool = True
    compound_frequency: int = 365  # Daily compounding

    @property
    def daily_rate(self) -> float:
        """Exact daily rate for compounding."""
        return self.annual_rate / self.compound_frequency

    def compound_interest(self, days: int, amount: Optional[float] = None) -> float:
        """Calculate compound interest for given days."""
        p = amount if amount is not None else self.principal
        # Compound interest formula: A = P(1 + r/n)^(nt)
        # Where n = compound frequency, t = time in years
        return p * ((1 + self.daily_rate) ** days - 1)

    def effective_annual_rate(self) -> float:
        """Calculate effective annual rate with compounding."""
        # EAR = (1 + r/n)^n - 1
        return (1 + self.daily_rate) ** self.compound_frequency - 1

    def present_value(self, future_amount: float, days: int) -> float:
        """Calculate present value of future amount."""
        return future_amount / ((1 + self.daily_rate) ** days)


@dataclass
class InvestmentAnalysis:
    """Complete investment analysis with pure mathematics."""

    action: str  # 'invest', 'paydown', 'indifferent'
    npv: float  # Net Present Value
    irr: Optional[float]  # Internal Rate of Return
    break_even_return: float  # Exact return needed to break even
    opportunity_cost: float  # Cost of not paying down debt

    # Cash flows
    initial_outflow: float
    expected_inflow: float
    borrowing_cost: float

    # Detailed metrics
    effective_borrowing_rate: float
    days_to_break_even: Optional[int]
    return_multiple: float  # Expected return / borrowing cost

    # Sensitivity analysis
    sensitivity: Dict[str, float] = None

    def __post_init__(self):
        if self.sensitivity is None:
            self.sensitivity = {}


class PureBorrowingAnalyzer:
    """Pure mathematical analysis of borrowing decisions."""

    def __init__(self):
        """Initialize with loan terms."""
        self.config = get_config()
        self.loans: Dict[str, LoanTerms] = {}
        self._setup_loans()

    def _setup_loans(self):
        """Set up loan terms."""
        # Amex personal loan
        self.loans["amex"] = LoanTerms(
            name="amex", principal=45000, annual_rate=0.07, is_revolving=False
        )

        # Schwab margin
        self.loans["schwab"] = LoanTerms(
            name="schwab", principal=0, annual_rate=0.10, is_revolving=True  # No current balance
        )

    def calculate_npv(self, cash_flows: List[Tuple[int, float]], discount_rate: float) -> float:
        """
        Calculate Net Present Value of cash flows.

        Args:
            cash_flows: List of (day, amount) tuples
            discount_rate: Annual discount rate

        Returns:
            NPV of cash flows
        """
        daily_discount = discount_rate / 365
        npv = 0.0

        for day, amount in cash_flows:
            npv += amount / ((1 + daily_discount) ** day)

        return npv

    def calculate_irr(self, cash_flows: List[Tuple[int, float]]) -> Optional[float]:
        """
        Calculate Internal Rate of Return.

        Args:
            cash_flows: List of (day, amount) tuples

        Returns:
            Annual IRR or None if no solution
        """

        def npv_at_rate(annual_rate):
            return self.calculate_npv(cash_flows, annual_rate)

        try:
            # Find IRR using Brent's method
            # Search between -50% and 500% annual return
            daily_irr = brentq(npv_at_rate, -0.5, 5.0, maxiter=100)
            return daily_irr
        except (ValueError, RuntimeError) as exc:
            logger.warning("IRR calculation failed: %s", exc, exc_info=exc)
            return None

    def analyze_investment(
        self,
        investment_amount: float,
        expected_return: float,  # Total return, not annualized
        holding_days: int,
        available_cash: float = 0,
        loan_source: str = "schwab",
        skip_sensitivity: bool = False,
    ) -> InvestmentAnalysis:
        """
        Analyze investment opportunity with pure mathematics.

        Args:
            investment_amount: Amount needed for investment
            expected_return: Expected total return (decimal)
            holding_days: Investment holding period
            available_cash: Cash available without borrowing
            loan_source: Which loan to use if borrowing

        Returns:
            Complete investment analysis
        """
        loan = self.loans[loan_source]

        # Calculate borrowing need
        borrow_amount = max(0, investment_amount - available_cash)

        # Expected cash inflow
        expected_inflow = investment_amount * (1 + expected_return)

        # Calculate exact borrowing cost with compounding
        if borrow_amount > 0:
            borrowing_cost = loan.compound_interest(holding_days, borrow_amount)
            effective_rate = loan.effective_annual_rate()
        else:
            borrowing_cost = 0
            effective_rate = 0

        # Build cash flow timeline
        cash_flows = [
            (0, -investment_amount),  # Initial investment
            (holding_days, expected_inflow),  # Return of principal + profit
        ]

        if borrow_amount > 0:
            # Repay loan with interest
            cash_flows.append((holding_days, -(borrow_amount + borrowing_cost)))

        # Calculate NPV using loan rate as discount rate
        npv = self.calculate_npv(cash_flows, loan.annual_rate)

        # Calculate IRR
        irr = self.calculate_irr(cash_flows)

        # Break-even analysis
        if borrow_amount > 0:
            # What return exactly breaks even with borrowing cost?
            break_even_return = borrowing_cost / investment_amount
        else:
            break_even_return = 0

        # Opportunity cost of not paying down existing debt
        opportunity_cost = self._calculate_opportunity_cost(investment_amount, holding_days)

        # Days to break even (when cumulative return exceeds borrowing cost)
        days_to_break_even = None
        if borrow_amount > 0 and expected_return > 0:
            daily_return_rate = expected_return / holding_days
            daily_borrow_rate = loan.daily_rate

            # Solve for days when returns = borrowing cost
            if daily_return_rate > daily_borrow_rate:
                days_to_break_even = int(
                    np.log(1 + break_even_return) / np.log(1 + daily_return_rate)
                )

        # Return multiple
        return_multiple = (
            (expected_inflow - investment_amount) / borrowing_cost
            if borrowing_cost > 0
            else float("inf")
        )

        # Sensitivity analysis (skip if recursive call)
        if skip_sensitivity:
            sensitivity = {}
        else:
            sensitivity = self._sensitivity_analysis(
                investment_amount, expected_return, holding_days, borrow_amount, loan
            )

        # Decision logic - pure math, no safety factors
        if npv > 0:
            action = "invest"
        elif abs(npv) < 0.01 * investment_amount:  # Within 1% of breakeven
            action = "indifferent"
        else:
            action = "paydown"

        return InvestmentAnalysis(
            action=action,
            npv=npv,
            irr=irr,
            break_even_return=break_even_return,
            opportunity_cost=opportunity_cost,
            initial_outflow=investment_amount,
            expected_inflow=expected_inflow,
            borrowing_cost=borrowing_cost,
            effective_borrowing_rate=effective_rate,
            days_to_break_even=days_to_break_even,
            return_multiple=return_multiple,
            sensitivity=sensitivity,
        )

    def _calculate_opportunity_cost(self, amount: float, days: int) -> float:
        """Calculate opportunity cost of not paying down cheapest debt."""
        # Find loan with highest rate (highest opportunity cost)
        if not self.loans:
            return 0

        highest_rate_loan = max(
            self.loans.values(), key=lambda l: l.annual_rate if l.principal > 0 else 0
        )

        if highest_rate_loan.principal > 0:
            # Interest saved by paying down this loan
            return highest_rate_loan.compound_interest(
                days, min(amount, highest_rate_loan.principal)
            )
        return 0

    def _sensitivity_analysis(
        self,
        investment_amount: float,
        expected_return: float,
        holding_days: int,
        borrow_amount: float,
        loan: LoanTerms,
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on key parameters."""
        sensitivity = {}

        # Base case NPV
        base_npv = self.analyze_investment(
            investment_amount,
            expected_return,
            holding_days,
            investment_amount - borrow_amount,
            loan.name,
            skip_sensitivity=True,
        ).npv

        # 10% worse return
        worse_return = expected_return * 0.9
        worse_npv = self.analyze_investment(
            investment_amount,
            worse_return,
            holding_days,
            investment_amount - borrow_amount,
            loan.name,
            skip_sensitivity=True,
        ).npv
        sensitivity["return_10pct_worse"] = worse_npv - base_npv

        # 20% longer holding period
        longer_days = int(holding_days * 1.2)
        longer_npv = self.analyze_investment(
            investment_amount,
            expected_return,
            longer_days,
            investment_amount - borrow_amount,
            loan.name,
            skip_sensitivity=True,
        ).npv
        sensitivity["holding_20pct_longer"] = longer_npv - base_npv

        # 1% higher borrowing rate
        if borrow_amount > 0:
            original_rate = loan.annual_rate
            loan.annual_rate = original_rate + 0.01
            higher_rate_npv = self.analyze_investment(
                investment_amount,
                expected_return,
                holding_days,
                investment_amount - borrow_amount,
                loan.name,
                skip_sensitivity=True,
            ).npv
            loan.annual_rate = original_rate  # Reset
            sensitivity["rate_1pct_higher"] = higher_rate_npv - base_npv

        return sensitivity

    def compare_opportunities(
        self, opportunities: List[Dict[str, float]], available_capital: float
    ) -> Dict[str, InvestmentAnalysis]:
        """
        Compare multiple investment opportunities.

        Args:
            opportunities: List of dicts with 'amount', 'return', 'days'
            available_capital: Total capital available

        Returns:
            Analysis for each opportunity, ranked by NPV
        """
        results = {}

        for i, opp in enumerate(opportunities):
            analysis = self.analyze_investment(
                investment_amount=opp["amount"],
                expected_return=opp["return"],
                holding_days=opp["days"],
                available_cash=min(available_capital, opp["amount"]),
            )
            results[f"opportunity_{i}"] = analysis

        # Sort by NPV
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1].npv, reverse=True))

        return sorted_results

    def calculate_portfolio_irr(self, positions: List[Dict[str, any]]) -> float:
        """
        Calculate portfolio-wide IRR for multiple positions.

        Args:
            positions: List of position dicts with cash flows

        Returns:
            Portfolio IRR
        """
        # Aggregate all cash flows
        all_flows = []

        for pos in positions:
            if "cash_flows" in pos:
                all_flows.extend(pos["cash_flows"])

        # Sort by day
        all_flows.sort(key=lambda x: x[0])

        # Calculate portfolio IRR
        return self.calculate_irr(all_flows) or 0.0


def analyze_pure_borrowing(
    investment_amount: float,
    expected_return_pct: float,  # As percentage
    holding_days: int,
    available_cash: float = 0,
) -> InvestmentAnalysis:
    """
    Convenience function for pure borrowing analysis.

    Args:
        investment_amount: Amount needed
        expected_return_pct: Expected return as percentage (e.g., 15 for 15%)
        holding_days: Holding period
        available_cash: Cash available

    Returns:
        Investment analysis
    """
    analyzer = PureBorrowingAnalyzer()
    return analyzer.analyze_investment(
        investment_amount=investment_amount,
        expected_return=expected_return_pct / 100,  # Convert to decimal
        holding_days=holding_days,
        available_cash=available_cash,
    )

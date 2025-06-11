from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, List, Optional

import numpy as np

from .advanced_financial_modeling import AdvancedFinancialModeling
from .borrowing_cost_analyzer import BorrowingCostAnalyzer


@dataclass
class PortfolioLeg:
    """Represents a potential portfolio leg (stock or option)."""

    name: str
    capital: float  # Capital required for the leg
    expected_return: float  # Annualized expected return
    volatility: float  # Annualized volatility
    time_horizon: int = 45


class PortfolioPermutationOptimizer:
    """Enumerate debt and position permutations to maximize Sharpe ratio."""

    def __init__(
        self,
        financial_modeler: Optional[AdvancedFinancialModeling] = None,
        borrowing_analyzer: Optional[BorrowingCostAnalyzer] = None,
    ) -> None:
        self.financial_modeler = financial_modeler or AdvancedFinancialModeling()
        self.borrowing_analyzer = borrowing_analyzer or BorrowingCostAnalyzer()

    def optimize(
        self,
        legs: List[PortfolioLeg],
        cash: float,
        paydown_options: List[float],
        margin_options: List[float],
        max_legs: Optional[int] = None,
        risk_free_rate: float = 0.05,
    ) -> Dict:
        """Evaluate permutations and return the highest Sharpe ratio combination."""
        max_legs = max_legs or len(legs)
        amex_balance = self.borrowing_analyzer.sources["amex_loan"].balance

        best: Dict = {"sharpe": float("-inf")}
        for paydown, margin in product(paydown_options, margin_options):
            if paydown > cash or paydown > amex_balance:
                continue
            cash_after_paydown = cash - paydown
            remaining_debt = amex_balance - paydown + margin
            for r in range(1, min(max_legs, len(legs)) + 1):
                for combo in combinations(legs, r):
                    capital_required = sum(l.capital for l in combo)
                    if capital_required > cash_after_paydown + margin:
                        continue
                    weights = np.array([l.capital for l in combo], dtype=float)
                    weights = weights / capital_required
                    expected = float(sum(w * l.expected_return for w, l in zip(weights, combo)))
                    vol = float(
                        np.sqrt(sum((w * l.volatility) ** 2 for w, l in zip(weights, combo)))
                    )
                    horizon = max(l.time_horizon for l in combo)
                    mc = self.financial_modeler.monte_carlo_simulation(
                        expected_return=expected,
                        volatility=vol,
                        time_horizon=horizon,
                        position_size=capital_required,
                        borrowed_amount=margin,
                        n_simulations=2000,
                        random_seed=42,
                    )
                    metrics, _ = self.financial_modeler.calculate_risk_adjusted_metrics(
                        returns=np.array(mc.returns),
                        borrowed_capital=remaining_debt,
                        total_capital=capital_required,
                        risk_free_rate=risk_free_rate,
                    )
                    if metrics.net_sharpe > best.get("sharpe", float("-inf")):
                        best = {
                            "sharpe": metrics.net_sharpe,
                            "paydown": paydown,
                            "margin": margin,
                            "legs": [l.name for l in combo],
                            "metrics": metrics,
                        }
        best.setdefault("legs", [])
        return best

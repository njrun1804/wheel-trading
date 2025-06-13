"""
from __future__ import annotations

Wheel Strategy Portfolio Optimization Engine

Implements sophisticated optimization algorithms for wheel strategy
with Unity stock at scale ($200k+ portfolios).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ...config.unified_config import get_config
config = get_config()


try:
    from ..models.greeks import Greeks
    from ..models.position import Position
    from ..risk.analytics import RiskAnalytics
except ImportError:
    # Fallback for standalone execution
    Position = None
    Greeks = None
    RiskAnalytics = None

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods for different scenarios"""

    HEURISTIC = "heuristic"
    INTELLIGENT_BUCKETING = "intelligent_bucketing"
    FULL_OPTIMIZATION = "full_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARETO_FRONTIER = "pareto_frontier"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""

    max_position_size: float = 0.25  # Max 25% in any single position
    min_cash_reserve: float = 0.10  # Min 10% cash
    max_options_allocation: float = 0.50  # Max 50% in options
    min_confidence_score: float = 0.30  # Min confidence for trades
    max_drawdown: float = 0.15  # Max 15% drawdown
    target_return: float = 0.15  # Target 15% annual return
    risk_tolerance: float = 0.12  # Max 12% annual volatility


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""

    positions: list[dict[str, Any]]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    max_drawdown: float
    confidence_score: float
    optimization_time: float
    method_used: OptimizationMethod
    metadata: dict[str, Any] = field(default_factory=dict)


class PositionSpace:
    """Defines the space of possible positions"""

    def __init__(self, capital: float, stock_price: float):
        self.capital = capital
        self.stock_price = stock_price

    def generate_stock_positions(self) -> list[int]:
        """Generate possible stock position sizes"""
        max_shares = int(self.capital * 0.8 / self.stock_price)  # Max 80% in stock

        # Adaptive granularity
        positions = []

        # Fine for small positions (0-1000 shares)
        positions.extend(range(0, min(1000, max_shares), 100))

        # Medium for mid positions (1000-5000 shares)
        if max_shares > 1000:
            positions.extend(range(1000, min(5000, max_shares), 500))

        # Coarse for large positions (5000+ shares)
        if max_shares > 5000:
            positions.extend(range(5000, max_shares + 1, 1000))

        return sorted(list(set(positions)))

    def generate_option_strikes(self, option_type: str = "put") -> list[float]:
        """Generate option strike prices"""
        if option_type == "put":
            # Put strikes: 75-95% of stock price
            strikes = np.linspace(self.stock_price * 0.75, self.stock_price * 0.95, 15)
        else:  # call
            # Call strikes: 105-125% of stock price
            strikes = np.linspace(self.stock_price * 1.05, self.stock_price * 1.25, 15)

        return strikes.tolist()

    def generate_expirations(self) -> list[int]:
        """Generate expiration dates"""
        return [7, 14, 21, 30, 45, 60, 90]  # Days to expiration


class HeuristicOptimizer:
    """Fast heuristic-based optimization for real-time decisions"""

    def __init__(self, constraints: OptimizationConstraints):
        self.constraints = constraints

    def optimize(
        self, capital: float, current_positions: list, market_data: dict
    ) -> OptimizationResult:
        """Quick heuristic optimization"""
        start_time = time.time()

        stock_price = market_data.get("stock_price", 20.0)
        implied_vol = market_data.get("implied_vol", 0.25)

        # Simple heuristic rules
        positions = []

        # 1. Stock position: 30-70% based on market conditions
        if implied_vol > 0.30:  # High IV, favor options
            stock_allocation = 0.30
        elif implied_vol < 0.20:  # Low IV, favor stock
            stock_allocation = 0.70
        else:  # Normal IV
            stock_allocation = 0.50

        stock_value = capital * stock_allocation
        shares = int(stock_value / stock_price / 100) * 100  # Round to lots

        positions.append(
            {
                "type": "stock",
                "symbol": "U",
                "quantity": shares,
                "allocation": shares * stock_price / capital,
            }
        )

        # 2. Put options: 20-30% allocation
        remaining_capital = capital - (shares * stock_price)
        put_allocation = min(0.30, remaining_capital / capital)

        if put_allocation > 0.05:  # Only if significant allocation
            put_strike = stock_price * 0.85  # 15% OTM
            put_premium = stock_price * 0.02  # Approximate premium
            put_contracts = int(put_allocation * capital / (put_premium * 100))

            positions.append(
                {
                    "type": "put",
                    "symbol": "U",
                    "strike": put_strike,
                    "expiration": 30,
                    "quantity": put_contracts,
                    "allocation": put_contracts * put_premium * 100 / capital,
                }
            )

        # 3. Covered calls if holding stock
        if shares > 0:
            call_contracts = shares // 100
            call_strike = stock_price * 1.10  # 10% OTM

            positions.append(
                {
                    "type": "call",
                    "symbol": "U",
                    "strike": call_strike,
                    "expiration": 21,
                    "quantity": -call_contracts,  # Short calls
                    "allocation": 0.05,  # Estimated premium
                }
            )

        # Calculate performance metrics (simplified)
        expected_return = 0.12 + (implied_vol - 0.25) * 0.2  # Adjust for IV
        expected_risk = 0.15 * implied_vol / 0.25
        sharpe_ratio = expected_return / expected_risk

        optimization_time = time.time() - start_time

        return OptimizationResult(
            positions=positions,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.12,
            confidence_score=0.75,
            optimization_time=optimization_time,
            method_used=OptimizationMethod.HEURISTIC,
        )


class IntelligentBucketingOptimizer:
    """Optimization using intelligent bucketing for medium complexity"""

    def __init__(self, constraints: OptimizationConstraints):
        self.constraints = constraints

    def optimize(
        self, capital: float, current_positions: list, market_data: dict
    ) -> OptimizationResult:
        """Optimize using intelligent bucketing"""
        start_time = time.time()

        stock_price = market_data.get("stock_price", 20.0)
        space = PositionSpace(capital, stock_price)

        # Generate candidate portfolios
        candidates = self._generate_candidates(capital, stock_price, space, market_data)

        # Evaluate candidates in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            evaluations = list(
                executor.map(lambda c: self._evaluate_portfolio(c, market_data), candidates)
            )

        # Select best portfolio based on Sharpe ratio
        best_idx = np.argmax([e["sharpe_ratio"] for e in evaluations])
        best_portfolio = candidates[best_idx]
        best_eval = evaluations[best_idx]

        optimization_time = time.time() - start_time

        return OptimizationResult(
            positions=best_portfolio,
            expected_return=best_eval["expected_return"],
            expected_risk=best_eval["expected_risk"],
            sharpe_ratio=best_eval["sharpe_ratio"],
            max_drawdown=best_eval["max_drawdown"],
            confidence_score=best_eval["confidence_score"],
            optimization_time=optimization_time,
            method_used=OptimizationMethod.INTELLIGENT_BUCKETING,
        )

    def _generate_candidates(
        self, capital: float, stock_price: float, space: PositionSpace, market_data: dict
    ) -> list[list[dict]]:
        """Generate candidate portfolios using intelligent bucketing"""
        candidates = []

        stock_positions = space.generate_stock_positions()[:10]  # Top 10 buckets
        put_strikes = space.generate_option_strikes("put")[:5]  # Top 5 strikes
        call_strikes = space.generate_option_strikes("call")[:5]
        expirations = [21, 30, 45]  # Focus on liquid expirations

        # Generate combinations
        for shares in stock_positions:
            stock_value = shares * stock_price
            remaining_capital = capital - stock_value

            if remaining_capital < capital * 0.1:  # Need min cash
                continue

            for put_strike in put_strikes:
                for exp in expirations:
                    # Create portfolio
                    portfolio = []

                    # Add stock position
                    if shares > 0:
                        portfolio.append(
                            {
                                "type": "stock",
                                "symbol": "U",
                                "quantity": shares,
                                "price": stock_price,
                            }
                        )

                    # Add put position
                    put_premium = self._estimate_option_premium(
                        stock_price, put_strike, exp, market_data, "put"
                    )
                    max_put_contracts = int(remaining_capital * 0.4 / (put_premium * 100))

                    if max_put_contracts > 0:
                        portfolio.append(
                            {
                                "type": "put",
                                "symbol": "U",
                                "strike": put_strike,
                                "expiration": exp,
                                "quantity": max_put_contracts,
                                "premium": put_premium,
                            }
                        )

                    # Add call position if holding stock
                    if shares > 0:
                        call_contracts = shares // 100
                        call_strike = call_strikes[len(call_strikes) // 2]  # ATM-ish
                        call_premium = self._estimate_option_premium(
                            stock_price, call_strike, exp, market_data, "call"
                        )

                        portfolio.append(
                            {
                                "type": "call",
                                "symbol": "U",
                                "strike": call_strike,
                                "expiration": exp,
                                "quantity": -call_contracts,  # Short
                                "premium": call_premium,
                            }
                        )

                    candidates.append(portfolio)

        return candidates[:100]  # Limit to top 100 candidates

    def _estimate_option_premium(
        self, stock_price: float, strike: float, days: int, market_data: dict, option_type: str
    ) -> float:
        """Estimate option premium using simplified Black-Scholes"""
        iv = market_data.get("implied_vol", 0.25)
        rate = market_data.get("risk_free_rate", 0.05)

        # Simplified premium estimation
        time_to_exp = days / 365.0
        moneyness = strike / stock_price

        if option_type == "put":
            # Put premium approximation
            intrinsic = max(strike - stock_price, 0)
            time_value = stock_price * iv * np.sqrt(time_to_exp) * 0.4
            premium = intrinsic + time_value
        else:  # call
            # Call premium approximation
            intrinsic = max(stock_price - strike, 0)
            time_value = stock_price * iv * np.sqrt(time_to_exp) * 0.4
            premium = intrinsic + time_value

        return max(premium, 0.01)  # Minimum premium

    def _evaluate_portfolio(self, portfolio: list[dict], market_data: dict) -> dict:
        """Evaluate a portfolio's risk/return characteristics"""

        # Calculate portfolio metrics
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_value = 0

        for position in portfolio:
            if position["type"] == "stock":
                delta = position["quantity"]
                gamma = 0
                theta = 0
                value = position["quantity"] * position["price"]
            else:  # option
                # Simplified Greeks
                delta = position["quantity"] * 50  # Approximate delta
                gamma = position["quantity"] * 10
                theta = position["quantity"] * -5
                value = position["quantity"] * position["premium"] * 100

            total_delta += delta
            total_gamma += gamma
            total_theta += theta
            total_value += abs(value)

        # Risk metrics
        portfolio_delta = abs(total_delta)
        portfolio_volatility = np.sqrt(portfolio_delta * 0.25**2)  # Simplified

        # Return estimation
        theta_income = total_theta * 365 / total_value if total_value > 0 else 0
        expected_return = theta_income + 0.08  # Base stock return

        # Risk-adjusted metrics
        sharpe_ratio = expected_return / max(portfolio_volatility, 0.01)
        max_drawdown = portfolio_volatility * 2  # Approximate
        confidence_score = min(0.9, 0.3 + sharpe_ratio * 0.1)

        return {
            "expected_return": expected_return,
            "expected_risk": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "confidence_score": confidence_score,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta": total_theta,
        }


class PortfolioOptimizer:
    """Main portfolio optimization engine"""

    def __init__(self, constraints: OptimizationConstraints = None):
        self.constraints = constraints or OptimizationConstraints()
        self.heuristic = HeuristicOptimizer(self.constraints)
        self.bucketing = IntelligentBucketingOptimizer(self.constraints)

    def optimize(
        self,
        capital: float,
        current_positions: list[Position],
        market_data: dict,
        method: OptimizationMethod = None,
    ) -> OptimizationResult:
        """Run portfolio optimization"""

        # Auto-select method if not specified
        if method is None:
            method = self._select_optimization_method(capital, len(current_positions))

        logger.info(f"Running {method.value} optimization for ${capital:,.0f} portfolio")

        if method == OptimizationMethod.HEURISTIC:
            return self.heuristic.optimize(capital, current_positions, market_data)
        elif method == OptimizationMethod.INTELLIGENT_BUCKETING:
            return self.bucketing.optimize(capital, current_positions, market_data)
        else:
            # Fallback to heuristic for unsupported methods
            logger.warning(f"Method {method.value} not implemented, using heuristic")
            return self.heuristic.optimize(capital, current_positions, market_data)

    def _select_optimization_method(self, capital: float, num_positions: int) -> OptimizationMethod:
        """Auto-select optimization method based on portfolio characteristics"""

        if capital < 50_000:
            return OptimizationMethod.HEURISTIC
        elif capital < 500_000:
            return OptimizationMethod.INTELLIGENT_BUCKETING
        else:
            return OptimizationMethod.FULL_OPTIMIZATION

    def get_optimization_recommendations(self, capital: float) -> dict[str, Any]:
        """Get recommendations for optimization approach"""

        method = self._select_optimization_method(capital, 0)

        return {
            "recommended_method": method.value,
            "expected_time_seconds": {
                OptimizationMethod.HEURISTIC: 0.1,
                OptimizationMethod.INTELLIGENT_BUCKETING: 5.0,
                OptimizationMethod.FULL_OPTIMIZATION: 60.0,
            }[method],
            "expected_accuracy": {
                OptimizationMethod.HEURISTIC: 0.85,
                OptimizationMethod.INTELLIGENT_BUCKETING: 0.95,
                OptimizationMethod.FULL_OPTIMIZATION: 0.99,
            }[method],
            "constraints": {
                "max_position_size": self.constraints.max_position_size,
                "min_cash_reserve": self.constraints.min_cash_reserve,
                "target_return": self.constraints.target_return,
                "risk_tolerance": self.constraints.risk_tolerance,
            },
        }


def run_optimization_demo() -> None:
    """Demo of the optimization engine"""

    constraints = OptimizationConstraints(
        max_position_size = config.trading.max_position_size, target_return=0.15, risk_tolerance=0.12
    )

    optimizer = PortfolioOptimizer(constraints)

    # Test scenarios
    scenarios = [
        {"capital": 50_000, "method": OptimizationMethod.HEURISTIC},
        {"capital": 200_000, "method": OptimizationMethod.INTELLIGENT_BUCKETING},
        {"capital": 1_000_000, "method": OptimizationMethod.FULL_OPTIMIZATION},
    ]

    market_data = {"stock_price": 20.0, "implied_vol": 0.25, "risk_free_rate": 0.05}

    print("=== PORTFOLIO OPTIMIZATION DEMO ===\n")

    for scenario in scenarios:
        capital = scenario["capital"]
        method = scenario["method"]

        print(f"Scenario: ${capital:,.0f} portfolio")

        # Get recommendations
        recommendations = optimizer.get_optimization_recommendations(capital)
        print(f"  Recommended method: {recommendations['recommended_method']}")
        print(f"  Expected time: {recommendations['expected_time_seconds']:.1f}s")
        print(f"  Expected accuracy: {recommendations['expected_accuracy']:.1%}")

        # Run optimization
        result = optimizer.optimize(capital, [], market_data, method)

        print("  Result:")
        print(f"    Expected return: {result.expected_return:.1%}")
        print(f"    Expected risk: {result.expected_risk:.1%}")
        print(f"    Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"    Confidence: {result.confidence_score:.1%}")
        print(f"    Optimization time: {result.optimization_time:.3f}s")
        print(f"    Positions: {len(result.positions)}")
        print()


if __name__ == "__main__":
    run_optimization_demo()
"""
This module provides sophisticated financial analysis for Unity wheel trading
with borrowed capital considerations.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from ..utils.logging import StructuredLogger
from ..utils.random_utils import set_seed
from .borrowing_cost_analyzer import BorrowingCostAnalyzer
from .pure_borrowing_analyzer import PureBorrowingAnalyzer

logger = StructuredLogger(logging.getLogger(__name__))


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    mean_return: float
    std_return: float
    percentiles: dict[int, float]  # 5th, 25th, 50th, 75th, 95th
    probability_profit: float
    probability_loss: float
    expected_shortfall: float  # CVaR
    max_drawdown: float
    paths: np.ndarray | None = None  # Sample paths for visualization
    confidence_interval: tuple[float, float] = None  # 95% CI


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics with borrowing costs."""

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    omega_ratio: float

    # With borrowing costs
    adjusted_sharpe: float
    adjusted_sortino: float
    net_sharpe: float  # After borrowing costs

    # Leverage metrics
    leverage_ratio: float
    debt_to_equity: float
    interest_coverage: float


@dataclass
class OptimalCapitalStructure:
    """Optimal debt/equity mix analysis."""

    optimal_leverage: float
    optimal_debt_ratio: float
    expected_return: float
    risk_level: float
    sharpe_ratio: float

    # Sensitivity
    leverage_curve: list[tuple[float, float, float]]  # (leverage, return, risk)
    efficient_frontier: list[tuple[float, float]]  # (risk, return) points


@dataclass
class MultiPeriodOptimization:
    """Multi-period borrowing and trading optimization."""

    periods: list[dict[str, float]]  # Period details
    total_return: float
    total_borrowing_cost: float
    net_return: float
    optimal_borrowing_schedule: list[float]
    optimal_position_schedule: list[float]
    reinvestment_strategy: str


class AdvancedFinancialModeling:
    """Advanced financial modeling for Unity wheel trading."""

    def __init__(self, borrowing_analyzer: BorrowingCostAnalyzer | None = None):
        """Initialize with borrowing analyzer."""
        # TODO: Pass config through constructor
        self.config = self._get_default_config()
        self.borrowing_analyzer = borrowing_analyzer or BorrowingCostAnalyzer()
        self.pure_analyzer = PureBorrowingAnalyzer()

    def _get_default_config(self):
        """Get default configuration."""

        class DefaultConfig:
            class unity:
                ticker = "U"

            class risk:
                max_position_size = 0.20

        return DefaultConfig()

    def monte_carlo_simulation(
        self,
        expected_return: float,
        volatility: float,
        time_horizon: int,  # days
        position_size: float,
        borrowed_amount: float = 0,
        n_simulations: int = 10000,
        include_path_dependency: bool = False,
        random_seed: int | None = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for Unity wheel returns.

        Args:
            expected_return: Expected annualized return
            volatility: Annualized volatility
            time_horizon: Time horizon in days
            position_size: Total position size
            borrowed_amount: Amount borrowed
            n_simulations: Number of simulation paths
            include_path_dependency: Model path-dependent features

        Returns:
            MonteCarloResult with statistics
        """
        set_seed(random_seed)

        # Convert to daily parameters
        daily_return = expected_return / 252
        daily_vol = volatility / np.sqrt(252)
        n_days = time_horizon

        # Calculate borrowing cost if applicable
        daily_borrow_cost = 0
        if borrowed_amount > 0:
            # Use Schwab margin rate
            loan = self.borrowing_analyzer.loans.get("schwab_margin")
            if loan:
                daily_borrow_cost = loan.daily_rate * borrowed_amount

        # Initialize arrays
        returns = np.zeros(n_simulations)
        paths = np.zeros((n_simulations, n_days + 1)) if include_path_dependency else None

        # Run simulations
        for i in range(n_simulations):
            if include_path_dependency:
                # Geometric Brownian Motion with discrete monitoring
                prices = np.zeros(n_days + 1)
                prices[0] = position_size

                # Daily returns with potential early exit
                for t in range(1, n_days + 1):
                    # Unity-specific: Higher kurtosis (fat tails)
                    shock = np.random.standard_t(df=5) * np.sqrt(
                        0.8
                    ) + np.random.normal() * np.sqrt(0.2)
                    daily_ret = daily_return + daily_vol * shock

                    prices[t] = prices[t - 1] * (1 + daily_ret)

                    # Early exit if severe drawdown (Unity characteristic)
                    if prices[t] < position_size * 0.7:  # 30% loss
                        prices[t:] = prices[t]
                        break

                paths[i] = prices
                final_value = prices[-1]

            else:
                # Simple terminal value simulation
                # Use student-t distribution for fatter tails (Unity characteristic)
                shock = np.random.standard_t(df=5) * np.sqrt(n_days)
                log_return = (daily_return - 0.5 * daily_vol**2) * n_days + daily_vol * shock
                final_value = position_size * np.exp(log_return)

            # Calculate return after borrowing costs
            gross_profit = final_value - position_size
            total_borrow_cost = daily_borrow_cost * n_days
            net_profit = gross_profit - total_borrow_cost
            returns[i] = net_profit / position_size

        # Calculate statistics
        percentiles = {
            5: np.percentile(returns, 5),
            25: np.percentile(returns, 25),
            50: np.percentile(returns, 50),
            75: np.percentile(returns, 75),
            95: np.percentile(returns, 95),
        }

        # Risk metrics
        negative_returns = returns[returns < 0]
        expected_shortfall = np.mean(negative_returns) if len(negative_returns) > 0 else 0

        # Maximum drawdown (if paths available)
        max_dd = 0
        if paths is not None:
            for path in paths:
                running_max = np.maximum.accumulate(path)
                drawdowns = (path - running_max) / running_max
                max_dd = min(max_dd, np.min(drawdowns))

        return MonteCarloResult(
            mean_return=np.mean(returns),
            std_return=np.std(returns),
            percentiles=percentiles,
            probability_profit=np.mean(returns > 0),
            probability_loss=np.mean(returns < 0),
            expected_shortfall=expected_shortfall,
            max_drawdown=max_dd,
            paths=paths[:100] if paths is not None else None,  # Keep sample for viz
            confidence_interval=(
                np.mean(returns) - 1.96 * np.std(returns) / np.sqrt(n_simulations),
                np.mean(returns) + 1.96 * np.std(returns) / np.sqrt(n_simulations),
            ),
        )

    def calculate_risk_adjusted_metrics(
        self,
        returns: np.ndarray,
        borrowed_capital: float,
        total_capital: float,
        risk_free_rate: float = 0.05,
        benchmark_returns: np.ndarray | None = None,
    ) -> tuple[RiskAdjustedMetrics, float]:
        """
        Calculate comprehensive risk-adjusted metrics.

        Args:
            returns: Array of period returns
            borrowed_capital: Amount borrowed
            total_capital: Total capital deployed
            risk_free_rate: Risk-free rate (annualized)
            benchmark_returns: Benchmark returns for relative metrics

        Returns:
            RiskAdjustedMetrics with all ratios
        """
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Downside deviation (for Sortino)
        downside_returns = returns[returns < risk_free_rate / 252]  # Daily risk-free
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return

        # Calculate borrowing cost impact
        leverage_ratio = total_capital / (total_capital - borrowed_capital)
        daily_borrow_rate = self.borrowing_analyzer.loans["schwab_margin"].daily_rate
        daily_borrow_cost = daily_borrow_rate * borrowed_capital / total_capital

        # Adjusted returns (after borrowing costs)
        adjusted_returns = returns - daily_borrow_cost
        adjusted_mean = np.mean(adjusted_returns)
        adjusted_std = np.std(adjusted_returns)

        # Sharpe Ratio (standard and adjusted)
        sharpe_ratio = (mean_return - risk_free_rate / 252) / std_return if std_return > 0 else 0
        adjusted_sharpe = (
            (adjusted_mean - risk_free_rate / 252) / adjusted_std if adjusted_std > 0 else 0
        )
        net_sharpe = adjusted_sharpe  # After all costs

        # Sortino Ratio
        sortino_ratio = (
            (mean_return - risk_free_rate / 252) / downside_std if downside_std > 0 else 0
        )
        adjusted_sortino = (
            (adjusted_mean - risk_free_rate / 252) / downside_std if downside_std > 0 else 0
        )

        # Calmar Ratio (return / max drawdown)
        running_wealth = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(running_wealth)
        drawdowns = (running_wealth - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.01
        calmar_ratio = mean_return * 252 / max_drawdown if max_drawdown > 0 else 0

        # Information Ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns)
            information_ratio = (
                np.mean(active_returns) / tracking_error if tracking_error > 0 else 0
            )
        else:
            information_ratio = 0

        # Treynor Ratio (uses beta instead of total risk)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            treynor_ratio = (mean_return - risk_free_rate / 252) / beta if beta > 0 else 0
        else:
            beta = 1.0
            treynor_ratio = sharpe_ratio

        # Omega Ratio (probability-weighted ratio of gains vs losses)
        threshold = risk_free_rate / 252
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else float("inf")

        # Leverage metrics
        debt_to_equity = borrowed_capital / (total_capital - borrowed_capital)
        interest_coverage = (
            mean_return * total_capital / (daily_borrow_cost * total_capital)
            if daily_borrow_cost > 0
            else float("inf")
        )

        metrics = RiskAdjustedMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            omega_ratio=omega_ratio,
            adjusted_sharpe=adjusted_sharpe,
            adjusted_sortino=adjusted_sortino,
            net_sharpe=net_sharpe,
            leverage_ratio=leverage_ratio,
            debt_to_equity=debt_to_equity,
            interest_coverage=interest_coverage,
        )

        confidence = 1.0 if len(returns) >= 252 else 0.8

        return metrics, confidence

    def optimize_capital_structure(
        self,
        expected_return: float,
        volatility: float,
        max_leverage: float = 2.0,
        risk_tolerance: float = 0.5,
        n_points: int = 50,
    ) -> OptimalCapitalStructure:
        """
        Find optimal debt/equity mix for Unity wheel trading.

        Args:
            expected_return: Expected return on assets
            volatility: Asset volatility
            max_leverage: Maximum leverage allowed
            risk_tolerance: Risk tolerance (0=min risk, 1=max return)
            n_points: Number of points for analysis

        Returns:
            OptimalCapitalStructure with optimal leverage
        """
        leverage_points = np.linspace(1.0, max_leverage, n_points)
        results = []

        # Get borrowing rate
        borrow_rate = self.borrowing_analyzer.loans["schwab_margin"].annual_rate

        for leverage in leverage_points:
            # Leveraged return = L * (R_asset - R_debt) + R_debt
            # Where L = leverage ratio
            leveraged_return = leverage * (expected_return - borrow_rate) + borrow_rate

            # Leveraged volatility = L * volatility
            leveraged_vol = leverage * volatility

            # Sharpe ratio
            sharpe = (leveraged_return - borrow_rate) / leveraged_vol if leveraged_vol > 0 else 0

            results.append((leverage, leveraged_return, leveraged_vol, sharpe))

        # Convert to arrays
        leverages = np.array([r[0] for r in results])
        returns = np.array([r[1] for r in results])
        vols = np.array([r[2] for r in results])
        sharpes = np.array([r[3] for r in results])

        # Find optimal based on risk tolerance
        # Utility = return - (1-risk_tolerance) * variance
        utilities = returns - (1 - risk_tolerance) * vols**2
        optimal_idx = np.argmax(utilities)

        # Build efficient frontier (return/risk pairs)
        frontier = [(vols[i], returns[i]) for i in range(len(vols))]
        frontier.sort(key=lambda x: x[0])  # Sort by risk

        # Leverage curve for visualization
        leverage_curve = [(leverages[i], returns[i], vols[i]) for i in range(len(leverages))]

        return OptimalCapitalStructure(
            optimal_leverage=leverages[optimal_idx],
            optimal_debt_ratio=(leverages[optimal_idx] - 1) / leverages[optimal_idx],
            expected_return=returns[optimal_idx],
            risk_level=vols[optimal_idx],
            sharpe_ratio=sharpes[optimal_idx],
            leverage_curve=leverage_curve,
            efficient_frontier=frontier,
        )

    def multi_period_optimization(
        self,
        periods: list[
            dict[str, float]
        ],  # Each period: {'days': X, 'expected_return': Y, 'volatility': Z}
        initial_capital: float,
        max_leverage: float = 1.5,
        reinvest_profits: bool = True,
    ) -> MultiPeriodOptimization:
        """
        Optimize borrowing and position sizing over multiple periods.

        Args:
            periods: List of period specifications
            initial_capital: Starting capital
            max_leverage: Maximum leverage allowed
            reinvest_profits: Whether to reinvest profits

        Returns:
            MultiPeriodOptimization with optimal schedule
        """
        n_periods = len(periods)

        # Decision variables: borrowing amount for each period
        def objective(borrowings):
            """Maximize total return over all periods."""
            capital = initial_capital
            total_return = 0
            total_borrow_cost = 0

            for i, (borrow, period) in enumerate(zip(borrowings, periods, strict=False)):
                # Position size = capital + borrowing
                position = capital + borrow

                # Expected return for period
                period_return = period["expected_return"] * period["days"] / 365
                gross_profit = position * period_return

                # Borrowing cost
                if borrow > 0:
                    loan = self.borrowing_analyzer.loans["schwab_margin"]
                    borrow_cost = loan.compound_interest(period["days"], borrow)
                else:
                    borrow_cost = 0

                # Net profit
                net_profit = gross_profit - borrow_cost
                total_return += net_profit
                total_borrow_cost += borrow_cost

                # Update capital for next period
                if reinvest_profits:
                    capital += net_profit

            return -total_return  # Negative for minimization

        # Constraints
        constraints = []

        # Leverage constraint for each period
        for i in range(n_periods):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, idx=i: max_leverage * initial_capital - x[idx],
                }
            )

        # Non-negative borrowing
        bounds = [(0, max_leverage * initial_capital) for _ in range(n_periods)]

        # Initial guess: moderate leverage
        x0 = [(max_leverage - 1) * initial_capital * 0.5 for _ in range(n_periods)]

        # Optimize
        result = optimize.minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        # Extract results
        optimal_borrowings = result.x
        optimal_positions = [initial_capital + b for b in optimal_borrowings]

        # Calculate detailed results
        capital = initial_capital
        period_details = []
        total_return = 0
        total_borrow_cost = 0

        for i, (borrow, period) in enumerate(zip(optimal_borrowings, periods, strict=False)):
            position = capital + borrow
            period_return = period["expected_return"] * period["days"] / 365
            gross_profit = position * period_return

            if borrow > 0:
                loan = self.borrowing_analyzer.loans["schwab_margin"]
                borrow_cost = loan.compound_interest(period["days"], borrow)
            else:
                borrow_cost = 0

            net_profit = gross_profit - borrow_cost

            period_details.append(
                {
                    "period": i + 1,
                    "days": period["days"],
                    "capital": capital,
                    "borrowed": borrow,
                    "position": position,
                    "gross_return": gross_profit,
                    "borrow_cost": borrow_cost,
                    "net_return": net_profit,
                }
            )

            total_return += gross_profit
            total_borrow_cost += borrow_cost

            if reinvest_profits:
                capital += net_profit

        return MultiPeriodOptimization(
            periods=period_details,
            total_return=total_return,
            total_borrowing_cost=total_borrow_cost,
            net_return=total_return - total_borrow_cost,
            optimal_borrowing_schedule=list(optimal_borrowings),
            optimal_position_schedule=optimal_positions,
            reinvestment_strategy="compound" if reinvest_profits else "withdraw",
        )

    def correlation_analysis(
        self,
        unity_returns: np.ndarray,
        interest_rates: np.ndarray,
        other_factors: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, float], float]:
        """
        Analyze correlation between Unity returns and interest rates.

        Args:
            unity_returns: Historical Unity returns
            interest_rates: Historical interest rates
            other_factors: Other factors to correlate

        Returns:
            Dictionary of correlation metrics
        """
        # Basic correlation
        pearson_corr = np.corrcoef(unity_returns, interest_rates)[0, 1]

        # Rank correlation (more robust)
        spearman_corr = stats.spearmanr(unity_returns, interest_rates)[0]

        # Rolling correlation (detect regime changes)
        window = min(60, len(unity_returns) // 4)  # 60-day or 25% of data
        rolling_corr = []

        for i in range(window, len(unity_returns)):
            window_corr = np.corrcoef(
                unity_returns[i - window : i], interest_rates[i - window : i]
            )[0, 1]
            rolling_corr.append(window_corr)

        # Correlation stability
        corr_std = np.std(rolling_corr) if rolling_corr else 0

        results = {
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "correlation_stability": 1 - corr_std,  # Higher is more stable
            "current_correlation": rolling_corr[-1] if rolling_corr else pearson_corr,
            "min_correlation": min(rolling_corr) if rolling_corr else pearson_corr,
            "max_correlation": max(rolling_corr) if rolling_corr else pearson_corr,
        }

        # Additional factors
        if other_factors:
            for name, data in other_factors.items():
                if len(data) == len(unity_returns):
                    results[f"{name}_correlation"] = np.corrcoef(unity_returns, data)[0, 1]

        # Interest rate sensitivity (beta to rates)
        if len(interest_rates) > 1:
            # Simple regression
            rate_changes = np.diff(interest_rates)
            return_changes = np.diff(unity_returns)

            if len(rate_changes) > 0:
                rate_beta = np.cov(return_changes, rate_changes)[0, 1] / np.var(rate_changes)
                results["interest_rate_beta"] = rate_beta

        return results

    def calculate_var_with_leverage(
        self,
        position_size: float,
        borrowed_amount: float,
        returns_distribution: np.ndarray,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
    ) -> dict[str, float]:
        """
        Calculate Value at Risk with borrowed capital.

        Args:
            position_size: Total position size
            borrowed_amount: Amount borrowed
            returns_distribution: Historical or simulated returns
            confidence_level: VaR confidence level
            time_horizon: Time horizon in days

        Returns:
            Dictionary with VaR metrics
        """
        # Adjust returns for leverage
        leverage_ratio = position_size / (position_size - borrowed_amount)
        leveraged_returns = returns_distribution * leverage_ratio

        # Account for borrowing costs
        loan = self.borrowing_analyzer.loans["schwab_margin"]
        daily_borrow_cost = loan.daily_rate * borrowed_amount / position_size
        net_returns = leveraged_returns - daily_borrow_cost * time_horizon

        # Calculate VaR (negative of percentile)
        var_percentile = (1 - confidence_level) * 100
        var_basic = -np.percentile(returns_distribution, var_percentile) * position_size
        var_leveraged = -np.percentile(net_returns, var_percentile) * position_size

        # Conditional VaR (Expected Shortfall)
        threshold_basic = np.percentile(returns_distribution, var_percentile)
        threshold_leveraged = np.percentile(net_returns, var_percentile)

        cvar_basic = (
            -np.mean(returns_distribution[returns_distribution <= threshold_basic]) * position_size
        )
        cvar_leveraged = -np.mean(net_returns[net_returns <= threshold_leveraged]) * position_size

        # Marginal VaR (impact of $1 more borrowing)
        marginal_position = position_size + 1
        marginal_leverage = marginal_position / (marginal_position - borrowed_amount - 1)
        marginal_returns = (
            returns_distribution * marginal_leverage
            - loan.daily_rate * (borrowed_amount + 1) / marginal_position
        )
        marginal_var = -np.percentile(marginal_returns, var_percentile) * marginal_position
        marginal_var_impact = marginal_var - var_leveraged

        # Worst case (with model risk)
        model_risk_multiplier = 1.2  # 20% model risk buffer
        worst_case_var = var_leveraged * model_risk_multiplier

        results = {
            "var_basic": var_basic,
            "var_leveraged": var_leveraged,
            "cvar_basic": cvar_basic,
            "cvar_leveraged": cvar_leveraged,
            "leverage_impact": var_leveraged - var_basic,
            "marginal_var": marginal_var_impact,
            "worst_case_var": worst_case_var,
            "var_to_equity": var_leveraged / (position_size - borrowed_amount),
            "probability_margin_call": np.mean(
                net_returns < -0.30
            ),  # 30% loss triggers margin call
        }

        conf = 1.0 if len(returns_distribution) >= 250 else 0.8

        return results, conf

    def optimize_portfolio_leverage(
        self,
        portfolio: list[dict[str, float]],
        max_leverage: float = 2.0,
        n_points: int = 20,
        n_simulations: int = 1000,
        risk_tolerance: float = 0.5,
        random_seed: int | None = None,
    ) -> OptimalCapitalStructure:
        """Find optimal leverage for an entire portfolio using Monte Carlo."""
        if not portfolio:
            raise ValueError("Portfolio cannot be empty")

        weights = np.array([p.get("weight", 1.0) for p in portfolio], dtype=float)
        weights /= weights.sum()

        expected = float(
            sum(w * p["expected_return"] for w, p in zip(weights, portfolio, strict=False))
        )
        volatility = float(
            np.sqrt(
                sum((w * p["volatility"]) ** 2 for w, p in zip(weights, portfolio, strict=False))
            )
        )
        horizon = max(int(p.get("time_horizon", 252)) for p in portfolio)

        leverage_points = np.linspace(1.0, max_leverage, n_points)
        curve = []

        for lev in leverage_points:
            mc = self.monte_carlo_simulation(
                expected_return=expected,
                volatility=volatility,
                time_horizon=horizon,
                position_size=lev,
                borrowed_amount=max(0.0, lev - 1.0),
                n_simulations=n_simulations,
                include_path_dependency=False,
                random_seed=random_seed,
            )
            ret = mc.mean_return * lev
            risk = mc.std_return * lev
            curve.append((lev, ret, risk))

        leverages = np.array([c[0] for c in curve])
        returns = np.array([c[1] for c in curve])
        risks = np.array([c[2] for c in curve])

        utilities = returns - (1 - risk_tolerance) * risks**2
        optimal_idx = int(np.argmax(utilities))

        frontier = [(risks[i], returns[i]) for i in range(len(risks))]
        frontier.sort(key=lambda x: x[0])

        leverage_curve = [(leverages[i], returns[i], risks[i]) for i in range(len(leverages))]

        return OptimalCapitalStructure(
            optimal_leverage=leverages[optimal_idx],
            optimal_debt_ratio=(leverages[optimal_idx] - 1) / leverages[optimal_idx],
            expected_return=returns[optimal_idx],
            risk_level=risks[optimal_idx],
            sharpe_ratio=returns[optimal_idx] / risks[optimal_idx] if risks[optimal_idx] > 0 else 0,
            leverage_curve=leverage_curve,
            efficient_frontier=frontier,
        )

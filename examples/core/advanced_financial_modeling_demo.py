"""Demonstration of advanced financial modeling features."""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unity_wheel.risk.advanced_financial_modeling import AdvancedFinancialModeling
from unity_wheel.risk.borrowing_cost_analyzer import BorrowingCostAnalyzer


def demo_monte_carlo():
    """Demonstrate Monte Carlo simulation."""
    print("Monte Carlo Simulation for Unity Wheel")
    print("=" * 60)

    modeler = AdvancedFinancialModeling()

    # Unity wheel parameters
    expected_return = 0.20  # 20% annualized
    volatility = 0.60  # 60% Unity volatility
    position_size = 35000
    borrowed_amount = 25000  # Borrowing $25k
    time_horizon = 45  # 45 days

    print(f"Position: ${position_size:,}")
    print(f"Borrowed: ${borrowed_amount:,}")
    print(f"Expected Return: {expected_return:.0%} annualized")
    print(f"Volatility: {volatility:.0%}")
    print(f"Time: {time_horizon} days")
    print()

    # Run simulation
    mc_result = modeler.monte_carlo_simulation(
        expected_return=expected_return,
        volatility=volatility,
        time_horizon=time_horizon,
        position_size=position_size,
        borrowed_amount=borrowed_amount,
        n_simulations=10000,
        include_path_dependency=True,
    )

    print("Results (10,000 simulations):")
    print(f"  Mean Return: {mc_result.mean_return:.2%}")
    print(f"  Std Deviation: {mc_result.std_return:.2%}")
    print(f"  Probability of Profit: {mc_result.probability_profit:.1%}")
    print(f"  Probability of Loss: {mc_result.probability_loss:.1%}")
    print()

    print("Percentiles:")
    for pct, value in mc_result.percentiles.items():
        print(f"  {pct}th percentile: {value:.2%}")
    print()

    print(f"Expected Shortfall (CVaR): {mc_result.expected_shortfall:.2%}")
    print(f"Maximum Drawdown: {mc_result.max_drawdown:.2%}")
    print(
        f"95% Confidence Interval: [{mc_result.confidence_interval[0]:.2%}, {mc_result.confidence_interval[1]:.2%}]"
    )


def demo_risk_adjusted_metrics():
    """Demonstrate risk-adjusted metrics calculation."""
    print("\n" + "=" * 60)
    print("Risk-Adjusted Metrics with Borrowing")
    print("=" * 60)

    modeler = AdvancedFinancialModeling()

    # Generate sample returns (Unity-like with fat tails)
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=252) * 0.03 + 0.0008  # Daily returns

    # Calculate metrics
    metrics = modeler.calculate_risk_adjusted_metrics(
        returns=returns,
        borrowed_capital=25000,
        total_capital=35000,
        risk_free_rate=0.05,
        benchmark_returns=np.random.normal(0.0004, 0.01, 252),  # SPY-like
    )

    print("Standard Metrics:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")
    print(f"  Information Ratio: {metrics.information_ratio:.3f}")
    print(f"  Treynor Ratio: {metrics.treynor_ratio:.3f}")
    print(f"  Omega Ratio: {metrics.omega_ratio:.3f}")
    print()

    print("With Borrowing Costs:")
    print(f"  Adjusted Sharpe: {metrics.adjusted_sharpe:.3f}")
    print(f"  Adjusted Sortino: {metrics.adjusted_sortino:.3f}")
    print(f"  Net Sharpe (all costs): {metrics.net_sharpe:.3f}")
    print()

    print("Leverage Metrics:")
    print(f"  Leverage Ratio: {metrics.leverage_ratio:.2f}x")
    print(f"  Debt-to-Equity: {metrics.debt_to_equity:.2f}")
    print(f"  Interest Coverage: {metrics.interest_coverage:.1f}x")


def demo_optimal_capital_structure():
    """Demonstrate optimal leverage calculation."""
    print("\n" + "=" * 60)
    print("Optimal Capital Structure Analysis")
    print("=" * 60)

    modeler = AdvancedFinancialModeling()

    # Unity parameters
    expected_return = 0.25  # 25% expected return
    volatility = 0.60  # 60% volatility

    # Find optimal leverage
    optimal = modeler.optimize_capital_structure(
        expected_return=expected_return,
        volatility=volatility,
        max_leverage=2.0,
        risk_tolerance=0.6,  # Moderate-aggressive
        n_points=20,
    )

    print(f"Expected Asset Return: {expected_return:.0%}")
    print(f"Asset Volatility: {volatility:.0%}")
    print(f"Borrowing Cost: 10% (Schwab margin)")
    print()

    print("Optimal Structure:")
    print(f"  Leverage: {optimal.optimal_leverage:.2f}x")
    print(f"  Debt Ratio: {optimal.optimal_debt_ratio:.1%}")
    print(f"  Expected Return: {optimal.expected_return:.1%}")
    print(f"  Risk Level: {optimal.risk_level:.1%}")
    print(f"  Sharpe Ratio: {optimal.sharpe_ratio:.3f}")
    print()

    print("Leverage Analysis (sample points):")
    for i in [0, 4, 9, 14, 19]:  # Sample 5 points
        lev, ret, risk = optimal.leverage_curve[i]
        print(f"  {lev:.2f}x leverage: {ret:.1%} return, {risk:.1%} volatility")


def demo_multi_period_optimization():
    """Demonstrate multi-period optimization."""
    print("\n" + "=" * 60)
    print("Multi-Period Optimization")
    print("=" * 60)

    modeler = AdvancedFinancialModeling()

    # Define multiple Unity wheel periods
    periods = [
        {"days": 45, "expected_return": 0.20, "volatility": 0.50},  # Normal
        {"days": 30, "expected_return": 0.30, "volatility": 0.70},  # High vol
        {"days": 60, "expected_return": 0.15, "volatility": 0.40},  # Low vol
        {"days": 45, "expected_return": 0.25, "volatility": 0.60},  # Medium
    ]

    initial_capital = 50000

    # Optimize
    result = modeler.multi_period_optimization(
        periods=periods, initial_capital=initial_capital, max_leverage=1.5, reinvest_profits=True
    )

    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Strategy: {result.reinvestment_strategy}")
    print()

    print("Optimal Borrowing Schedule:")
    for i, (period, borrow) in enumerate(zip(periods, result.optimal_borrowing_schedule)):
        print(
            f"  Period {i+1} ({period['days']}d, {period['expected_return']:.0%} return): Borrow ${borrow:,.0f}"
        )
    print()

    print("Results:")
    print(f"  Total Return: ${result.total_return:,.2f}")
    print(f"  Total Borrowing Cost: ${result.total_borrowing_cost:,.2f}")
    print(f"  Net Return: ${result.net_return:,.2f}")
    print(f"  Net Return %: {result.net_return/initial_capital:.1%}")


def demo_correlation_analysis():
    """Demonstrate correlation analysis."""
    print("\n" + "=" * 60)
    print("Unity Returns vs Interest Rates Correlation")
    print("=" * 60)

    modeler = AdvancedFinancialModeling()

    # Generate synthetic data (real would come from historical)
    np.random.seed(42)
    n_days = 500

    # Interest rates (slow moving)
    rates = np.cumsum(np.random.normal(0, 0.0001, n_days)) + 0.05

    # Unity returns (somewhat correlated with rates)
    base_returns = np.random.standard_t(df=5, size=n_days) * 0.03
    rate_impact = -np.diff(np.concatenate([[rates[0]], rates])) * 50  # Negative correlation
    unity_returns = base_returns + rate_impact + 0.001

    # Other factors
    vix = np.random.gamma(2, 10, n_days)  # Volatility index
    spy_returns = np.random.normal(0.0004, 0.01, n_days)  # Market returns

    # Analyze
    correlation = modeler.correlation_analysis(
        unity_returns=unity_returns,
        interest_rates=rates,
        other_factors={"vix": vix, "spy_returns": spy_returns},
    )

    print("Correlation Analysis:")
    for metric, value in correlation.items():
        if "correlation" in metric:
            print(f"  {metric}: {value:.3f}")
        elif "beta" in metric:
            print(f"  {metric}: {value:.3f}")


def demo_var_with_leverage():
    """Demonstrate VaR calculation with leverage."""
    print("\n" + "=" * 60)
    print("Value at Risk with Borrowed Capital")
    print("=" * 60)

    modeler = AdvancedFinancialModeling()

    # Generate returns distribution
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=1000) * 0.03 + 0.001

    position_size = 100000
    borrowed_amount = 60000

    # Calculate VaR
    var_metrics, conf = modeler.calculate_var_with_leverage(
        position_size=position_size,
        borrowed_amount=borrowed_amount,
        returns_distribution=returns,
        confidence_level=0.95,
        time_horizon=1,
    )

    print(f"Position: ${position_size:,}")
    print(f"Borrowed: ${borrowed_amount:,}")
    print(f"Equity: ${position_size - borrowed_amount:,}")
    print()

    print("Value at Risk (95% confidence, 1 day):")
    print(f"  VaR (unleveraged): ${var_metrics['var_basic']:,.0f}")
    print(f"  VaR (leveraged): ${var_metrics['var_leveraged']:,.0f}")
    print(f"  Leverage Impact: ${var_metrics['leverage_impact']:,.0f}")
    print()

    print("Conditional VaR (Expected Shortfall):")
    print(f"  CVaR (unleveraged): ${var_metrics['cvar_basic']:,.0f}")
    print(f"  CVaR (leveraged): ${var_metrics['cvar_leveraged']:,.0f}")
    print()

    print("Risk Ratios:")
    print(f"  VaR to Equity: {var_metrics['var_to_equity']:.1%}")
    print(f"  Marginal VaR ($1 more borrowing): ${var_metrics['marginal_var']:.2f}")
    print(f"  Probability of Margin Call: {var_metrics['probability_margin_call']:.1%}")
    print(f"  Worst Case VaR (with model risk): ${var_metrics['worst_case_var']:,.0f}")


def main():
    """Run all demonstrations."""
    print("Advanced Financial Modeling for Unity Wheel Trading")
    print("=" * 60)
    print()

    demo_monte_carlo()
    demo_risk_adjusted_metrics()
    demo_optimal_capital_structure()
    demo_multi_period_optimization()
    demo_correlation_analysis()
    demo_var_with_leverage()

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("- Monte Carlo shows fat-tailed risk for Unity")
    print("- Borrowing reduces Sharpe ratio but can increase returns")
    print("- Optimal leverage around 1.3-1.5x for Unity's volatility")
    print("- Multi-period optimization suggests varying leverage by volatility")
    print("- Unity shows negative correlation with interest rates")
    print("- Leveraged VaR significantly higher than unleveraged")


if __name__ == "__main__":
    main()

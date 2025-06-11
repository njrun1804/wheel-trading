from src.unity_wheel.risk.portfolio_permutation_optimizer import (
    PortfolioLeg,
    PortfolioPermutationOptimizer,
)


def test_permutation_optimizer_basic():
    optimizer = PortfolioPermutationOptimizer()
    legs = [
        PortfolioLeg(name="short_put", capital=10000, expected_return=0.25, volatility=0.40),
        PortfolioLeg(name="long_call", capital=10000, expected_return=0.10, volatility=0.30),
    ]

    result = optimizer.optimize(
        legs=legs,
        cash=20000,
        paydown_options=[0, 10000],
        margin_options=[0, 10000],
        max_legs=2,
    )

    assert result["sharpe"] != float("-inf")
    assert len(result["legs"]) > 0

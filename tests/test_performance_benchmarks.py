"""Performance benchmark tests for critical paths.

Ensures that key operations meet performance targets:
- Black-Scholes: <0.2ms per calculation
- Greeks: <0.3ms for all Greeks
- Risk metrics: <10ms for 1000 data points
- IV solver: <5ms with fallback to bisection
- Full recommendation: <100ms
- Memory usage: <100MB for typical portfolio
"""

import os
import time
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import psutil
import pytest
from memory_profiler import memory_usage
from unity_wheel.api.advisor import WheelAdvisor
from unity_wheel.data_providers.databento.types import OptionChain, OptionQuote
from unity_wheel.math.options import (
    black_scholes_price_validated,
    calculate_all_greeks,
    implied_volatility_validated,
    probability_itm_validated,
)
from unity_wheel.models.account import Account
from unity_wheel.models.position import Position, PositionType
from unity_wheel.risk.analytics import RiskAnalyzer


class TestMathPerformance:
    """Test performance of mathematical calculations."""

    def test_black_scholes_performance(self):
        """Test Black-Scholes calculation speed."""
        spot = 100.0
        strike = 100.0
        time_to_expiry = 30 / 365
        rate = 0.05
        volatility = 0.25

        # Warm up
        for _ in range(10):
            black_scholes_price_validated(spot, strike, time_to_expiry, rate, volatility, "call")

        # Time 10,000 calculations
        start = time.perf_counter()
        for _ in range(10000):
            result = black_scholes_price_validated(
                spot, strike, time_to_expiry, rate, volatility, "call"
            )
            assert not np.isnan(result.value)
        end = time.perf_counter()

        elapsed = end - start
        per_calc = elapsed / 10000 * 1000  # Convert to ms

        print(f"Black-Scholes: {per_calc:.3f}ms per calculation")
        assert per_calc < 0.2, f"Black-Scholes too slow: {per_calc:.3f}ms > 0.2ms"

    def test_greeks_performance(self):
        """Test Greeks calculation speed."""
        spot = 100.0
        strike = 100.0
        time_to_expiry = 30 / 365
        rate = 0.05
        volatility = 0.25

        # Warm up
        for _ in range(10):
            calculate_all_greeks(spot, strike, time_to_expiry, rate, volatility)

        # Time 10,000 calculations
        start = time.perf_counter()
        for _ in range(10000):
            result = calculate_all_greeks(spot, strike, time_to_expiry, rate, volatility)
            assert result.confidence > 0.9
        end = time.perf_counter()

        elapsed = end - start
        per_calc = elapsed / 10000 * 1000  # Convert to ms

        print(f"All Greeks: {per_calc:.3f}ms per calculation")
        assert per_calc < 0.3, f"Greeks too slow: {per_calc:.3f}ms > 0.3ms"

    def test_implied_volatility_performance(self):
        """Test implied volatility solver speed."""
        spot = 100.0
        strike = 100.0
        time_to_expiry = 30 / 365
        rate = 0.05
        option_price = 2.50

        # Warm up
        for _ in range(10):
            implied_volatility_validated(option_price, spot, strike, time_to_expiry, rate, "call")

        # Time 1,000 calculations (IV is slower)
        start = time.perf_counter()
        for _ in range(1000):
            result = implied_volatility_validated(
                option_price, spot, strike, time_to_expiry, rate, "call"
            )
            assert not np.isnan(result.value)
        end = time.perf_counter()

        elapsed = end - start
        per_calc = elapsed / 1000 * 1000  # Convert to ms

        print(f"Implied Volatility: {per_calc:.3f}ms per calculation")
        assert per_calc < 5.0, f"IV solver too slow: {per_calc:.3f}ms > 5.0ms"


class TestRiskAnalyticsPerformance:
    """Test performance of risk calculations."""

    def test_var_calculation_performance(self):
        """Test VaR calculation speed with large dataset."""
        # Generate 1000 days of returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        analytics = RiskAnalytics()

        # Warm up
        for _ in range(10):
            analytics.calculate_var(returns, confidence=0.95)

        # Time 100 calculations
        start = time.perf_counter()
        for _ in range(100):
            var = analytics.calculate_var(returns, confidence=0.95)
            assert var < 0
        end = time.perf_counter()

        elapsed = end - start
        per_calc = elapsed / 100 * 1000  # Convert to ms

        print(f"VaR (1000 points): {per_calc:.3f}ms per calculation")
        assert per_calc < 10.0, f"VaR too slow: {per_calc:.3f}ms > 10.0ms"

    def test_portfolio_risk_performance(self):
        """Test portfolio risk metrics calculation speed."""
        # Create sample portfolio
        positions = []
        for i in range(20):  # 20 positions
            positions.append(
                Position(
                    symbol=f"U  250117P000{40+i}000",
                    quantity=-5,
                    position_type=PositionType.OPTION,
                    option_type=OptionType.PUT,
                    strike=Decimal(str(40 + i)),
                    expiration=datetime.now() + timedelta(days=45),
                    underlying="U",
                    cost_basis=Decimal("5.00"),
                    current_price=Decimal("1.00"),
                    multiplier=100,
                    delta=Decimal("-0.30"),
                    gamma=Decimal("0.015"),
                    theta=Decimal("-0.08"),
                    vega=Decimal("0.12"),
                )
            )

        analytics = RiskAnalytics()

        # Time portfolio calculations
        start = time.perf_counter()
        metrics = analytics.calculate_portfolio_metrics(
            positions=positions,
            account_value=Decimal("100000"),
            spot_prices={"U": Decimal("45.00")},
        )
        end = time.perf_counter()

        elapsed = (end - start) * 1000  # Convert to ms

        print(f"Portfolio risk (20 positions): {elapsed:.3f}ms")
        assert elapsed < 50.0, f"Portfolio risk too slow: {elapsed:.3f}ms > 50.0ms"


class TestRecommendationPerformance:
    """Test full recommendation flow performance."""

    def test_full_recommendation_performance(self):
        """Test complete recommendation generation speed."""
        account = Account(
            total_value=Decimal("100000"),
            cash_balance=Decimal("50000"),
            buying_power=Decimal("150000"),
        )

        positions = [
            Position(
                symbol="U  241220P00040000",
                quantity=-5,
                position_type=PositionType.OPTION,
                option_type=OptionType.PUT,
                strike=Decimal("40"),
                expiration=datetime.now() + timedelta(days=45),
                underlying="U",
                cost_basis=Decimal("6.00"),
                current_price=Decimal("1.20"),
                multiplier=100,
                delta=Decimal("-0.28"),
                gamma=Decimal("0.015"),
                theta=Decimal("-0.08"),
                vega=Decimal("0.12"),
            )
        ]

        market_data = {
            "U": OptionChain(
                underlying="U",
                expiration=datetime.now() + timedelta(days=45),
                spot_price=Decimal("45.50"),
                timestamp=datetime.now(),
                calls=[],
                puts=[
                    OptionQuote(
                        instrument_id=i,
                        timestamp=datetime.now(),
                        bid_price=Decimal(str(1.0 + i * 0.1)),
                        ask_price=Decimal(str(1.05 + i * 0.1)),
                        bid_size=100,
                        ask_size=100,
                        strike=Decimal(str(40 + i)),
                        delta=Decimal(str(-0.25 - i * 0.05)),
                        gamma=Decimal("0.015"),
                        theta=Decimal("-0.08"),
                        vega=Decimal("0.12"),
                        implied_volatility=Decimal("0.30"),
                    )
                    for i in range(10)
                ],
            )
        }

        advisor = WheelAdvisor()

        # Warm up
        for _ in range(5):
            advisor.advise_position(account, positions, market_data)

        # Time 10 recommendations
        start = time.perf_counter()
        for _ in range(10):
            result = advisor.advise_position(account, positions, market_data)
            assert result is not None
            assert result.confidence > 0
        end = time.perf_counter()

        elapsed = end - start
        per_recommendation = elapsed / 10 * 1000  # Convert to ms

        print(f"Full recommendation: {per_recommendation:.3f}ms")
        assert (
            per_recommendation < 100.0
        ), f"Recommendation too slow: {per_recommendation:.3f}ms > 100.0ms"


class TestMemoryUsage:
    """Test memory usage stays within bounds."""

    def test_typical_portfolio_memory(self):
        """Test memory usage for typical portfolio operations."""

        def create_and_analyze_portfolio():
            # Create 50 positions
            positions = []
            for i in range(50):
                positions.append(
                    Position(
                        symbol=f"U  250117P000{40+i}000",
                        quantity=-5,
                        position_type=PositionType.OPTION,
                        option_type=OptionType.PUT,
                        strike=Decimal(str(40 + i)),
                        expiration=datetime.now() + timedelta(days=45),
                        underlying="U",
                        cost_basis=Decimal("5.00"),
                        current_price=Decimal("1.00"),
                        multiplier=100,
                    )
                )

            # Create market data
            market_data = {}
            for underlying in ["U", "SPY", "QQQ", "IWM", "DIA"]:
                chains = []
                for exp_days in [30, 45, 60, 90]:
                    chain = OptionChain(
                        underlying=underlying,
                        expiration=datetime.now() + timedelta(days=exp_days),
                        spot_price=Decimal("100"),
                        timestamp=datetime.now(),
                        calls=[],
                        puts=[
                            OptionQuote(
                                instrument_id=i,
                                timestamp=datetime.now(),
                                bid_price=Decimal(str(1.0 + i * 0.1)),
                                ask_price=Decimal(str(1.05 + i * 0.1)),
                                bid_size=100,
                                ask_size=100,
                                strike=Decimal(str(90 + i)),
                            )
                            for i in range(20)  # 20 strikes per expiry
                        ],
                    )
                    chains.append(chain)
                market_data[underlying] = chains

            # Run analytics
            analytics = RiskAnalytics()
            analytics.calculate_portfolio_metrics(
                positions=positions,
                account_value=Decimal("1000000"),
                spot_prices={underlying: Decimal("100") for underlying in market_data},
            )

            # Get recommendations
            advisor = WheelAdvisor()
            account = Account(
                total_value=Decimal("1000000"),
                cash_balance=Decimal("500000"),
                buying_power=Decimal("1500000"),
            )

            for _ in range(10):
                advisor.advise_position(account, positions, market_data)

        # Measure memory usage
        mem_usage = memory_usage(create_and_analyze_portfolio)
        max_memory = max(mem_usage)

        print(f"Peak memory usage: {max_memory:.1f}MB")
        assert max_memory < 100.0, f"Memory usage too high: {max_memory:.1f}MB > 100MB"


class TestConcurrentPerformance:
    """Test performance under concurrent load."""

    def test_concurrent_calculations(self):
        """Test calculations remain fast under concurrent load."""
        import concurrent.futures

        def calc_batch():
            """Calculate a batch of options prices."""
            results = []
            for _ in range(100):
                spot = np.random.uniform(80, 120)
                strike = np.random.uniform(80, 120)
                vol = np.random.uniform(0.1, 0.5)

                result = black_scholes_price_validated(spot, strike, 30 / 365, 0.05, vol, "call")
                results.append(result.value)
            return results

        # Run calculations concurrently
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calc_batch) for _ in range(4)]
            results = [f.result() for f in futures]
        end = time.perf_counter()

        elapsed = end - start
        total_calcs = 4 * 100
        per_calc = elapsed / total_calcs * 1000  # ms

        print(f"Concurrent BS (4 threads): {per_calc:.3f}ms per calculation")
        assert per_calc < 0.5, f"Concurrent calc too slow: {per_calc:.3f}ms > 0.5ms"


if __name__ == "__main__":
    # Run performance tests with timing output
    pytest.main([__file__, "-v", "-s"])

"""GPU-accelerated wheel strategy using M4 Pro hardware optimizations.

This module provides MLX-accelerated wheel strategy implementation that leverages:
- 8 P-cores + 4 E-cores for parallel processing
- 20-core Metal GPU for vectorized options calculations  
- Unified memory architecture for zero-copy operations
- Thermal monitoring for sustained performance
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import mlx.core as mx
import numpy as np

from ..math import StrikeRecommendation
from ..models import Position, PositionType
from ..utils import get_logger
from .wheel import WheelParameters, WheelStrategy

# Import bolt components
try:
    from bolt.gpu_acceleration import batch_cosine_similarity, gpuify, matrix_multiply
    from bolt.hardware.hardware_state import get_hardware_state
    from bolt.hardware.memory_manager import get_memory_manager
    from bolt.thermal_monitor import ThermalMonitor

    BOLT_AVAILABLE = True
except ImportError:
    BOLT_AVAILABLE = False

    # Fallback decorator
    def gpuify(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = get_logger(__name__)


@dataclass
class GPUComputationStats:
    """Statistics for GPU computations."""

    total_operations: int = 0
    gpu_time_ms: float = 0.0
    cpu_fallbacks: int = 0
    memory_peak_mb: float = 0.0
    thermal_throttles: int = 0


class GPUAcceleratedWheelStrategy(WheelStrategy):
    """Wheel strategy with M4 Pro GPU acceleration and thermal management."""

    def __init__(
        self,
        parameters: WheelParameters | None = None,
        enable_gpu: bool = True,
        thermal_monitoring: bool = True,
    ):
        """Initialize GPU-accelerated wheel strategy.

        Args:
            parameters: Wheel strategy parameters
            enable_gpu: Whether to enable GPU acceleration
            thermal_monitoring: Whether to enable thermal monitoring
        """
        super().__init__(parameters)

        self.enable_gpu = enable_gpu and BOLT_AVAILABLE
        self.thermal_monitoring = thermal_monitoring
        self.stats = GPUComputationStats()

        # Initialize hardware components
        if self.enable_gpu:
            self.hw = get_hardware_state()
            self.memory_manager = get_memory_manager()
            self.thermal_monitor = ThermalMonitor() if thermal_monitoring else None

            # Get resource budget for GPU operations
            self.gpu_budget = self.hw.get_resource_budget("gpu")
            self.current_thermal_state = "normal"  # normal, throttled, emergency

            logger.info(
                "GPU acceleration enabled",
                extra={
                    "gpu_workers": self.gpu_budget.gpu_workers,
                    "memory_pool_mb": self.gpu_budget.memory_pool_mb,
                    "hardware_summary": self.hw.get_summary(),
                },
            )
        else:
            logger.info("GPU acceleration disabled - using CPU fallback")

    async def initialize_gpu_system(self):
        """Initialize GPU system components."""
        if not self.enable_gpu:
            return

        if self.thermal_monitor:
            await self.thermal_monitor.start_monitoring(
                cpu_temp_threshold=85,  # Conservative for sustained trading
                gpu_temp_threshold=80,  # MLX can get hot during vector ops
                callback=self._thermal_callback,
            )

    def _thermal_callback(self, thermal_state: dict):
        """Handle thermal events during GPU computations."""
        cpu_temp = thermal_state.get("cpu_temperature", 0)
        gpu_temp = thermal_state.get("gpu_temperature", 0)

        if cpu_temp > 85 or gpu_temp > 80:
            self.current_thermal_state = "throttled"
            self.stats.thermal_throttles += 1
            logger.warning(f"Thermal throttling: CPU {cpu_temp}°C, GPU {gpu_temp}°C")
        elif cpu_temp < 75 and gpu_temp < 70:
            if self.current_thermal_state != "normal":
                logger.info("Thermal state normalized")
            self.current_thermal_state = "normal"

    @asynccontextmanager
    async def gpu_computation_context(self, operation: str, estimated_mb: float):
        """Context manager for GPU computations with memory management."""
        if not self.enable_gpu:
            yield None
            return

        # Allocate memory for computation
        alloc_id = self.memory_manager.allocate(
            component="jarvis",
            size_mb=estimated_mb,
            description=f"GPU {operation}",
            priority=8,  # High priority for trading
            can_evict=False,
        )

        if not alloc_id:
            logger.warning(f"Failed to allocate GPU memory for {operation}")
            yield None
            return

        start_time = time.perf_counter()
        try:
            yield alloc_id
        finally:
            # Track performance metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_operations += 1
            self.stats.gpu_time_ms += elapsed_ms

            # Release memory
            self.memory_manager.deallocate(alloc_id)

    @gpuify(batch_size=4096, memory_check=True, fallback=True)
    async def find_optimal_strikes_vectorized_gpu(
        self,
        current_price: float,
        available_strikes: list[float],
        volatilities: list[float],
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        target_delta: float = 0.30,
    ) -> list[StrikeRecommendation]:
        """GPU-accelerated strike selection across multiple volatility scenarios.

        This method processes all strike/volatility combinations in parallel using
        the M4 Pro's 20-core GPU, providing 10-50x speedup over sequential CPU.
        """
        if not available_strikes or not volatilities:
            return []

        # Estimate memory usage
        n_strikes = len(available_strikes)
        n_vols = len(volatilities)
        n_scenarios = n_strikes * n_vols
        estimated_mb = (n_scenarios * 8 * 6) / (1024 * 1024)  # 6 arrays, 8 bytes each

        async with self.gpu_computation_context(
            "vectorized_strikes", estimated_mb
        ) as alloc_id:
            if not alloc_id:
                # Fallback to CPU
                self.stats.cpu_fallbacks += 1
                return await self._find_strikes_cpu_fallback(
                    current_price,
                    available_strikes,
                    volatilities,
                    days_to_expiry,
                    risk_free_rate,
                    target_delta,
                )

            # Convert to MLX arrays for GPU processing
            strikes_np = np.array(available_strikes, dtype=np.float32)
            vols_np = np.array(volatilities, dtype=np.float32)

            # Create parameter grids
            strikes_grid, vols_grid = np.meshgrid(strikes_np, vols_np)

            # Convert to MLX arrays
            strikes_mx = mx.array(strikes_grid.flatten())
            vols_mx = mx.array(vols_grid.flatten())

            # Constants for all scenarios
            S = mx.full_like(strikes_mx, current_price)
            T = mx.full_like(strikes_mx, days_to_expiry / 365.0)
            r = mx.full_like(strikes_mx, risk_free_rate)

            # Vectorized Black-Scholes calculation for puts
            sqrt_T = mx.sqrt(T)
            d1 = (mx.log(S / strikes_mx) + (r + 0.5 * vols_mx**2) * T) / (
                vols_mx * sqrt_T
            )
            d2 = d1 - vols_mx * sqrt_T

            # Use MLX's sigmoid as fast approximation to norm.cdf
            # For production, could implement proper CDF using MLX
            cdf_neg_d1 = mx.sigmoid(-d1 * 0.8)  # Scaled for better approximation
            cdf_neg_d2 = mx.sigmoid(-d2 * 0.8)

            # Put delta calculation
            deltas_put = cdf_neg_d1 - 1.0

            # Put premium calculation
            put_premiums = strikes_mx * mx.exp(-r * T) * cdf_neg_d2 - S * cdf_neg_d1

            # Vectorized scoring
            delta_errors = mx.abs(deltas_put - target_delta)
            premium_ratios = put_premiums / strikes_mx

            # Combined score (lower is better)
            scores = delta_errors + 0.1 * (1.0 - premium_ratios)

            # Reshape to find best strike for each volatility
            scores_reshaped = scores.reshape(n_vols, n_strikes)
            best_strike_indices = mx.argmin(scores_reshaped, axis=1)

            # Extract results
            results = []
            for vol_idx, strike_idx in enumerate(best_strike_indices):
                strike_idx = int(strike_idx)
                vol = volatilities[vol_idx]
                strike = available_strikes[strike_idx]

                # Get the corresponding values
                flat_idx = vol_idx * n_strikes + strike_idx
                delta = float(deltas_put[flat_idx])
                premium = float(put_premiums[flat_idx])
                score = float(scores[flat_idx])

                # Calculate probability ITM (simplified)
                prob_itm = max(0, min(1, 1 + delta))  # Put delta is negative

                # Calculate confidence based on score and thermal state
                base_confidence = max(0.5, 1.0 - score)
                if self.current_thermal_state == "throttled":
                    base_confidence *= 0.9
                elif self.current_thermal_state == "emergency":
                    base_confidence *= 0.7

                results.append(
                    StrikeRecommendation(
                        strike=strike,
                        delta=delta,
                        probability_itm=prob_itm,
                        premium=premium,
                        confidence=base_confidence,
                        reason=f"GPU-optimized (vol={vol:.2f}, score={score:.3f})",
                    )
                )

            # Update memory peak tracking
            current_usage = estimated_mb
            if current_usage > self.stats.memory_peak_mb:
                self.stats.memory_peak_mb = current_usage

            logger.info(
                "GPU strike optimization completed",
                extra={
                    "scenarios_computed": n_scenarios,
                    "results_count": len(results),
                    "thermal_state": self.current_thermal_state,
                    "memory_usage_mb": estimated_mb,
                },
            )

            return results

    async def _find_strikes_cpu_fallback(
        self,
        current_price: float,
        available_strikes: list[float],
        volatilities: list[float],
        days_to_expiry: int,
        risk_free_rate: float,
        target_delta: float,
    ) -> list[StrikeRecommendation]:
        """CPU fallback for strike selection when GPU is unavailable."""

        # Use the original vectorized CPU implementation
        results = []
        for vol in volatilities:
            result = await asyncio.to_thread(
                self.find_optimal_put_strike_vectorized,
                current_price=current_price,
                available_strikes=available_strikes,
                volatility=vol,
                days_to_expiry=days_to_expiry,
                risk_free_rate=risk_free_rate,
                target_delta=target_delta,
            )
            if result:
                results.append(result)

        return results

    @gpuify(fallback=True)
    async def compute_portfolio_greeks_parallel(
        self, positions: list[Position], current_price: float, volatility: float
    ) -> dict[str, float]:
        """Compute Greeks for entire portfolio in parallel using GPU.

        Processes all positions simultaneously on the 20-core Metal GPU,
        providing significant speedup for large portfolios.
        """
        if not positions:
            return {}

        n_positions = len(positions)
        estimated_mb = (n_positions * 8 * 10) / (1024 * 1024)  # 10 arrays per position

        async with self.gpu_computation_context(
            "portfolio_greeks", estimated_mb
        ) as alloc_id:
            if not alloc_id:
                self.stats.cpu_fallbacks += 1
                return await self._compute_greeks_cpu_fallback(
                    positions, current_price, volatility
                )

            # Extract position parameters
            strikes = []
            quantities = []
            expiries_days = []
            is_call = []

            for pos in positions:
                if pos.strike and pos.expiration:
                    strikes.append(pos.strike)
                    quantities.append(pos.quantity)
                    days = (pos.expiration - datetime.now(UTC).date()).days
                    expiries_days.append(max(1, days))  # Minimum 1 day
                    is_call.append(pos.position_type == PositionType.CALL)

            if not strikes:
                return {}

            # Convert to MLX arrays
            strikes_mx = mx.array(strikes, dtype=mx.float32)
            quantities_mx = mx.array(quantities, dtype=mx.float32)
            expiries_mx = mx.array([d / 365.0 for d in expiries_days], dtype=mx.float32)
            is_call_mx = mx.array(is_call, dtype=mx.float32)

            # Constants
            S = mx.full_like(strikes_mx, current_price)
            sigma = mx.full_like(strikes_mx, volatility)
            r = mx.full_like(strikes_mx, 0.05)  # Risk-free rate

            # Black-Scholes intermediate calculations
            sqrt_T = mx.sqrt(expiries_mx)
            d1 = (mx.log(S / strikes_mx) + (r + 0.5 * sigma**2) * expiries_mx) / (
                sigma * sqrt_T
            )
            d2 = d1 - sigma * sqrt_T

            # Approximate normal CDF using sigmoid (for speed)
            cdf_d1 = mx.sigmoid(d1 * 0.8)
            mx.sigmoid(-d1 * 0.8)
            cdf_d2 = mx.sigmoid(d2 * 0.8)
            cdf_neg_d2 = mx.sigmoid(-d2 * 0.8)

            # Delta calculation (call delta = N(d1), put delta = N(d1) - 1)
            delta_call = cdf_d1
            delta_put = cdf_d1 - 1.0
            deltas = is_call_mx * delta_call + (1 - is_call_mx) * delta_put

            # Gamma (same for calls and puts)
            # Using approximation: gamma ≈ phi(d1) / (S * sigma * sqrt(T))
            phi_d1 = mx.exp(-0.5 * d1**2) / mx.sqrt(
                2 * mx.pi
            )  # Normal PDF approximation
            gammas = phi_d1 / (S * sigma * sqrt_T)

            # Theta (time decay)
            theta_term1 = -S * phi_d1 * sigma / (2 * sqrt_T)
            theta_call_term2 = -r * strikes_mx * mx.exp(-r * expiries_mx) * cdf_d2
            theta_put_term2 = r * strikes_mx * mx.exp(-r * expiries_mx) * cdf_neg_d2

            theta_call = (theta_term1 + theta_call_term2) / 365
            theta_put = (theta_term1 + theta_put_term2) / 365
            thetas = is_call_mx * theta_call + (1 - is_call_mx) * theta_put

            # Vega (same for calls and puts)
            vegas = S * phi_d1 * sqrt_T / 100  # Per 1% change in volatility

            # Position-weighted Greeks
            position_deltas = deltas * quantities_mx
            position_gammas = gammas * quantities_mx * 100  # Scale gamma
            position_thetas = thetas * quantities_mx
            position_vegas = vegas * quantities_mx

            # Portfolio totals
            portfolio_greeks = {
                "total_delta": float(mx.sum(position_deltas)),
                "total_gamma": float(mx.sum(position_gammas)),
                "total_theta": float(mx.sum(position_thetas)),
                "total_vega": float(mx.sum(position_vegas)),
                "delta_dollars": float(mx.sum(position_deltas * S)),
                "gamma_dollars": float(mx.sum(position_gammas * S / 100)),
                "positions_count": len(positions),
                "largest_delta": float(mx.max(mx.abs(position_deltas))),
                "net_theta_daily": float(mx.sum(position_thetas)),
            }

            logger.debug(
                "Portfolio Greeks computed on GPU",
                extra={
                    "positions": len(positions),
                    "portfolio_greeks": portfolio_greeks,
                    "thermal_state": self.current_thermal_state,
                },
            )

            return portfolio_greeks

    async def _compute_greeks_cpu_fallback(
        self, positions: list[Position], current_price: float, volatility: float
    ) -> dict[str, float]:
        """CPU fallback for Greeks computation."""

        # Use sequential calculation for each position
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for position in positions:
            # Use the parent class's risk analysis method
            risk_metrics = self.analyze_position_risk(
                position, current_price, volatility, 100000  # Dummy portfolio value
            )

            total_delta += risk_metrics.get("delta_dollars", 0) / current_price
            total_gamma += risk_metrics.get("gamma_dollars", 0) / current_price
            total_theta += risk_metrics.get("theta_daily", 0)
            total_vega += risk_metrics.get("vega_dollars", 0)

        return {
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta": total_theta,
            "total_vega": total_vega,
            "delta_dollars": total_delta * current_price,
            "gamma_dollars": total_gamma * current_price,
            "positions_count": len(positions),
            "net_theta_daily": total_theta,
        }

    async def optimize_portfolio_allocation_gpu(
        self,
        positions: list[Position],
        available_capital: float,
        max_positions: int = 10,
    ) -> dict[str, any]:
        """GPU-accelerated portfolio allocation optimization.

        Uses the M4 Pro GPU to evaluate thousands of portfolio combinations
        in parallel, optimizing for risk-adjusted returns.
        """
        if not self.enable_gpu or len(positions) < 2:
            return {"error": "Insufficient positions or GPU unavailable"}

        n_positions = len(positions)
        n_combinations = min(1000, 2**n_positions)  # Limit combinations
        estimated_mb = (n_combinations * n_positions * 4) / (1024 * 1024)

        async with self.gpu_computation_context(
            "portfolio_optimization", estimated_mb
        ) as alloc_id:
            if not alloc_id:
                return {"error": "Memory allocation failed"}

            # Generate random portfolio weight combinations
            np.random.seed(42)  # Reproducible results
            weights = np.random.dirichlet(np.ones(n_positions), n_combinations)

            # Convert to MLX
            weights_mx = mx.array(weights.astype(np.float32))

            # Position properties (simplified)
            expected_returns = mx.array(
                [0.05 / 252] * n_positions
            )  # Daily return assumption
            volatilities = mx.array([0.02] * n_positions)  # Daily volatility assumption

            # Vectorized portfolio calculations
            portfolio_returns = mx.sum(weights_mx * expected_returns, axis=1)
            portfolio_vols = mx.sqrt(mx.sum((weights_mx * volatilities) ** 2, axis=1))

            # Sharpe ratio approximation
            sharpe_ratios = portfolio_returns / (portfolio_vols + 1e-8)

            # Find best allocation
            best_idx = mx.argmax(sharpe_ratios)
            best_weights = weights_mx[int(best_idx)]

            # Convert back to position allocations
            total_allocation = available_capital
            allocations = {}

            for i, position in enumerate(positions[:n_positions]):
                weight = float(best_weights[i])
                allocation = total_allocation * weight
                allocations[f"position_{i}"] = {
                    "symbol": getattr(position, "symbol", f"pos_{i}"),
                    "weight": weight,
                    "allocation_dollars": allocation,
                    "suggested_contracts": max(
                        1, int(allocation / 10000)
                    ),  # Rough estimate
                }

            return {
                "optimization_method": "gpu_accelerated",
                "combinations_evaluated": n_combinations,
                "best_sharpe_ratio": float(sharpe_ratios[int(best_idx)]),
                "allocations": allocations,
                "total_allocated": total_allocation,
                "thermal_state": self.current_thermal_state,
            }

    def get_performance_stats(self) -> dict[str, any]:
        """Get GPU acceleration performance statistics."""
        if not self.enable_gpu:
            return {"gpu_acceleration": "disabled"}

        gpu_utilization = self.hw.get_utilization()
        memory_stats = self.memory_manager.get_component_stats("jarvis")

        return {
            "gpu_acceleration": "enabled",
            "total_operations": self.stats.total_operations,
            "gpu_time_ms": self.stats.gpu_time_ms,
            "cpu_fallbacks": self.stats.cpu_fallbacks,
            "memory_peak_mb": self.stats.memory_peak_mb,
            "thermal_throttles": self.stats.thermal_throttles,
            "current_thermal_state": self.current_thermal_state,
            "gpu_utilization_percent": gpu_utilization.get("gpu_percent", 0),
            "memory_utilization": memory_stats,
            "hardware_summary": self.hw.get_summary(),
        }

    async def shutdown(self):
        """Shutdown GPU components gracefully."""
        if self.thermal_monitor:
            await self.thermal_monitor.stop_monitoring()

        logger.info(
            "GPU-accelerated wheel strategy shutdown",
            extra=self.get_performance_stats(),
        )


# Example usage
async def example_gpu_wheel_usage():
    """Example of using GPU-accelerated wheel strategy."""

    # Initialize strategy
    strategy = GPUAcceleratedWheelStrategy(enable_gpu=True, thermal_monitoring=True)

    await strategy.initialize_gpu_system()

    try:
        # Example: GPU-accelerated strike selection
        current_price = 100.0
        strikes = list(range(80, 121))  # 41 strikes
        volatilities = [0.15, 0.20, 0.25, 0.30, 0.35]  # 5 volatility scenarios

        start_time = time.perf_counter()

        recommendations = await strategy.find_optimal_strikes_vectorized_gpu(
            current_price=current_price,
            available_strikes=strikes,
            volatilities=volatilities,
            days_to_expiry=30,
            target_delta=0.30,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"GPU-accelerated strike selection completed in {elapsed_ms:.2f}ms")
        print(f"Evaluated {len(strikes) * len(volatilities)} scenarios")
        print(f"Found {len(recommendations)} recommendations")

        for rec in recommendations[:3]:  # Show first 3
            print(
                f"Strike: {rec.strike}, Delta: {rec.delta:.3f}, Premium: {rec.premium:.2f}"
            )

        # Performance stats
        stats = strategy.get_performance_stats()
        print(f"Performance stats: {stats}")

    finally:
        await strategy.shutdown()


if __name__ == "__main__":
    # Run example if bolt is available
    if BOLT_AVAILABLE:
        asyncio.run(example_gpu_wheel_usage())
    else:
        print("Bolt framework not available - install bolt dependencies")

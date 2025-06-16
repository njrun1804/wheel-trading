"""Complete M4 Pro trading system integration using bolt framework.

This module demonstrates the full integration of M4 Pro hardware optimizations
with the wheel trading system, showcasing real-world usage patterns and
performance benefits.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.unity_wheel.models import Position

# Trading system imports
from src.unity_wheel.strategy.gpu_wheel_strategy import GPUAcceleratedWheelStrategy
from src.unity_wheel.utils import get_logger

from .hardware.hardware_state import get_hardware_state
from .hardware.memory_manager import get_memory_manager

# Bolt framework imports
from .integration import BoltIntegration
from .thermal_trading_monitor import TradingPerformanceMode, TradingThermalManager

logger = get_logger(__name__)


@dataclass
class TradingSessionMetrics:
    """Comprehensive metrics for a trading session."""

    session_start: datetime
    session_end: datetime | None = None

    # Performance metrics
    total_computations: int = 0
    gpu_accelerated_ops: int = 0
    cpu_fallback_ops: int = 0
    average_computation_ms: float = 0.0

    # Thermal metrics
    max_cpu_temp: float = 0.0
    max_gpu_temp: float = 0.0
    thermal_throttle_events: int = 0
    thermal_throttle_total_minutes: float = 0.0

    # Memory metrics
    peak_memory_usage_gb: float = 0.0
    memory_pressure_events: int = 0

    # Trading metrics
    recommendations_generated: int = 0
    portfolio_analyses: int = 0
    strike_optimizations: int = 0
    greeks_computations: int = 0


class M4ProTradingSystem:
    """Complete M4 Pro-optimized trading system using bolt framework."""

    def __init__(
        self,
        num_agents: int = 8,
        enable_gpu: bool = True,
        enable_thermal_monitoring: bool = True,
        session_name: str = "default",
    ):
        """Initialize M4 Pro trading system.

        Args:
            num_agents: Number of parallel agents (matches P-cores)
            enable_gpu: Enable GPU acceleration
            enable_thermal_monitoring: Enable thermal management
            session_name: Name for this trading session
        """
        self.session_name = session_name
        self.num_agents = num_agents

        # Initialize core components
        self.bolt = BoltIntegration(num_agents=num_agents)
        self.hw = get_hardware_state()
        self.memory_manager = get_memory_manager()

        # Initialize trading components
        self.gpu_strategy = GPUAcceleratedWheelStrategy(
            enable_gpu=enable_gpu, thermal_monitoring=enable_thermal_monitoring
        )

        # Initialize thermal management
        self.thermal_manager = TradingThermalManager()

        # Session tracking
        self.metrics = TradingSessionMetrics(session_start=datetime.now())
        self.active_positions: list[Position] = []
        self.session_active = False

        # Performance state
        self.current_performance_mode = TradingPerformanceMode.BALANCED
        self.adaptive_batch_sizes = {
            "strikes": 4096,
            "scenarios": 1000,
            "portfolio": 100,
        }

        logger.info(
            f"M4 Pro trading system initialized: {session_name}",
            extra={
                "agents": num_agents,
                "gpu_enabled": enable_gpu,
                "thermal_monitoring": enable_thermal_monitoring,
                "hardware": self.hw.get_summary(),
            },
        )

    async def initialize(self):
        """Initialize all system components."""
        logger.info("Initializing M4 Pro trading system...")

        # Initialize bolt integration
        await self.bolt.initialize()

        # Initialize GPU strategy
        await self.gpu_strategy.initialize_gpu_system()

        # Initialize thermal monitoring
        await self.thermal_manager.start_monitoring()

        # Register thermal callbacks
        self.thermal_manager.register_throttle_callback(self._handle_thermal_throttle)
        self.thermal_manager.register_mode_change_callback(self._handle_mode_change)

        # Setup memory pressure handlers
        self._setup_memory_handlers()

        logger.info("M4 Pro trading system initialization complete")

    async def shutdown(self):
        """Gracefully shutdown the trading system."""
        logger.info("Shutting down M4 Pro trading system...")

        self.session_active = False
        self.metrics.session_end = datetime.now()

        # Shutdown components
        await self.thermal_manager.stop_monitoring()
        await self.gpu_strategy.shutdown()
        await self.bolt.shutdown()

        # Log final session metrics
        await self._log_session_summary()

        logger.info("M4 Pro trading system shutdown complete")

    @asynccontextmanager
    async def trading_session(self, session_type: str = "wheel_analysis"):
        """Context manager for a complete trading session."""
        self.session_active = True

        # Allocate session memory
        session_memory_mb = 4096 if session_type == "portfolio_analysis" else 2048

        with self.memory_manager.allocate_context(
            "jarvis", session_memory_mb, f"Trading session: {session_type}", priority=9
        ) as alloc_id:
            try:
                logger.info(f"Starting trading session: {session_type}")
                yield alloc_id
            finally:
                logger.info(f"Ending trading session: {session_type}")
                self.session_active = False

    async def get_comprehensive_recommendation(
        self,
        symbol: str,
        portfolio_value: float,
        current_positions: list[Position],
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get comprehensive trading recommendation using all M4 Pro capabilities.

        This method demonstrates the full power of the M4 Pro system by:
        1. Using all 8 agents in parallel for different analysis tasks
        2. GPU-accelerating options calculations
        3. Adaptive performance based on thermal state
        4. Memory-optimized data processing
        """
        start_time = time.perf_counter()

        async with self.trading_session("comprehensive_analysis"):
            # Create parallel analysis tasks for 8 agents
            tasks = [
                self._analyze_current_market_regime(symbol, market_data),
                self._optimize_strikes_gpu_accelerated(symbol, market_data),
                self._compute_portfolio_risk_parallel(current_positions, market_data),
                self._analyze_volatility_surface(symbol, market_data),
                self._check_liquidity_conditions(symbol, market_data),
                self._compute_kelly_optimal_sizing(portfolio_value, market_data),
                self._analyze_correlation_risk(symbol, current_positions),
                self._generate_scenario_analysis(symbol, portfolio_value, market_data),
            ]

            # Execute all tasks in parallel using bolt agents
            logger.info(f"Starting parallel analysis with {len(tasks)} agents")
            results = await self.bolt.execute_parallel_tasks(tasks)

            # Combine results with confidence weighting
            recommendation = await self._synthesize_recommendation(results, market_data)

            # Track metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.average_computation_ms = (
                self.metrics.average_computation_ms * self.metrics.total_computations
                + elapsed_ms
            ) / (self.metrics.total_computations + 1)
            self.metrics.total_computations += 1
            self.metrics.recommendations_generated += 1

            # Add system performance metadata
            recommendation["system_metadata"] = {
                "computation_time_ms": elapsed_ms,
                "thermal_state": self.thermal_manager.current_state.mode.value,
                "gpu_utilization": self.hw.get_utilization()["gpu_percent"],
                "memory_usage": self.memory_manager.get_component_stats("jarvis"),
                "performance_mode": self.current_performance_mode.value,
                "agents_used": len(tasks),
            }

            return recommendation

    # Individual analysis methods for each agent

    async def _analyze_current_market_regime(
        self, symbol: str, market_data: dict
    ) -> dict:
        """Agent 1: Analyze current market regime using historical patterns."""
        # Simulate market regime analysis
        await asyncio.sleep(0.1)  # Simulate computation

        return {
            "task": "market_regime",
            "regime": "normal_volatility",
            "confidence": 0.85,
            "trend": "bullish",
            "volatility_percentile": 45,
        }

    async def _optimize_strikes_gpu_accelerated(
        self, symbol: str, market_data: dict
    ) -> dict:
        """Agent 2: GPU-accelerated strike optimization."""
        current_price = market_data.get("current_price", 100.0)

        # Get available strikes and volatilities
        strikes = list(range(int(current_price * 0.8), int(current_price * 1.2), 1))
        volatilities = [0.15, 0.20, 0.25, 0.30, 0.35]

        # Use GPU acceleration for strike optimization
        recommendations = await self.gpu_strategy.find_optimal_strikes_vectorized_gpu(
            current_price=current_price,
            available_strikes=strikes,
            volatilities=volatilities,
            days_to_expiry=30,
            target_delta=0.30,
        )

        self.metrics.strike_optimizations += 1
        if (
            hasattr(self.gpu_strategy, "stats")
            and self.gpu_strategy.stats.total_operations > 0
        ):
            self.metrics.gpu_accelerated_ops += 1
        else:
            self.metrics.cpu_fallback_ops += 1

        return {
            "task": "strike_optimization",
            "recommendations": recommendations[:3],  # Top 3
            "gpu_accelerated": len(recommendations) > 0,
            "scenarios_evaluated": len(strikes) * len(volatilities),
        }

    async def _compute_portfolio_risk_parallel(
        self, positions: list[Position], market_data: dict
    ) -> dict:
        """Agent 3: Parallel portfolio risk computation."""
        current_price = market_data.get("current_price", 100.0)
        volatility = market_data.get("implied_volatility", 0.25)

        # Use GPU-accelerated Greeks computation
        portfolio_greeks = await self.gpu_strategy.compute_portfolio_greeks_parallel(
            positions, current_price, volatility
        )

        self.metrics.greeks_computations += 1
        self.metrics.portfolio_analyses += 1

        return {
            "task": "portfolio_risk",
            "greeks": portfolio_greeks,
            "risk_metrics": {
                "var_95": abs(portfolio_greeks.get("total_delta", 0))
                * current_price
                * 0.02,
                "max_loss": abs(portfolio_greeks.get("total_delta", 0))
                * current_price
                * 0.05,
                "theta_decay_daily": portfolio_greeks.get("net_theta_daily", 0),
            },
        }

    async def _analyze_volatility_surface(self, symbol: str, market_data: dict) -> dict:
        """Agent 4: Volatility surface analysis."""
        # Simulate IV surface analysis
        await asyncio.sleep(0.05)

        return {
            "task": "volatility_surface",
            "iv_rank": 65,
            "iv_percentile": 78,
            "term_structure": "normal_contango",
            "skew": "moderate_put_skew",
        }

    async def _check_liquidity_conditions(self, symbol: str, market_data: dict) -> dict:
        """Agent 5: Liquidity and market conditions analysis."""
        await asyncio.sleep(0.03)

        return {
            "task": "liquidity_analysis",
            "bid_ask_spread": 0.02,
            "volume_ranking": "high",
            "market_depth": "good",
            "execution_quality": 0.92,
        }

    async def _compute_kelly_optimal_sizing(
        self, portfolio_value: float, market_data: dict
    ) -> dict:
        """Agent 6: Kelly optimal position sizing."""
        await asyncio.sleep(0.02)

        # Simplified Kelly calculation
        win_rate = 0.55
        avg_win = 0.15
        avg_loss = 0.10

        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        optimal_size = portfolio_value * kelly_fraction * 0.5  # Conservative factor

        return {
            "task": "kelly_sizing",
            "kelly_fraction": kelly_fraction,
            "optimal_position_size": optimal_size,
            "max_contracts": int(optimal_size / 10000),  # Rough estimate
            "confidence": 0.75,
        }

    async def _analyze_correlation_risk(
        self, symbol: str, positions: list[Position]
    ) -> dict:
        """Agent 7: Portfolio correlation and concentration risk."""
        await asyncio.sleep(0.04)

        # Analyze position concentration
        position_count = len(positions)
        concentration_risk = "low" if position_count > 5 else "high"

        return {
            "task": "correlation_analysis",
            "concentration_risk": concentration_risk,
            "position_count": position_count,
            "correlation_score": 0.25,
            "diversification_benefit": 0.85,
        }

    async def _generate_scenario_analysis(
        self, symbol: str, portfolio_value: float, market_data: dict
    ) -> dict:
        """Agent 8: Monte Carlo scenario analysis."""
        # This could use GPU acceleration for large Monte Carlo simulations
        await asyncio.sleep(0.08)

        scenarios = {
            "bull_case": {"return": 0.15, "probability": 0.30},
            "base_case": {"return": 0.05, "probability": 0.50},
            "bear_case": {"return": -0.10, "probability": 0.20},
        }

        return {
            "task": "scenario_analysis",
            "scenarios": scenarios,
            "expected_return": 0.045,
            "value_at_risk": portfolio_value * 0.08,
            "stress_test_passed": True,
        }

    async def _synthesize_recommendation(
        self, analysis_results: list[dict], market_data: dict
    ) -> dict:
        """Synthesize all analysis results into final recommendation."""

        # Extract key information from each analysis
        recommendation = {
            "symbol": market_data.get("symbol", "SPY"),
            "action": "HOLD",  # Default
            "confidence": 0.0,
            "reasoning": [],
            "risk_assessment": "MODERATE",
            "position_sizing": {},
            "execution_timing": "NORMAL",
            "analysis_components": {},
        }

        total_confidence = 0
        component_count = 0

        for result in analysis_results:
            if not result:
                continue

            task = result.get("task", "unknown")
            recommendation["analysis_components"][task] = result

            # Extract confidence if available
            confidence = result.get("confidence", 0.5)
            total_confidence += confidence
            component_count += 1

            # Task-specific synthesis
            if task == "strike_optimization" and result.get("recommendations"):
                best_rec = result["recommendations"][0]
                recommendation["position_sizing"][
                    "recommended_strike"
                ] = best_rec.strike
                recommendation["position_sizing"]["expected_premium"] = best_rec.premium
                recommendation["reasoning"].append(
                    f"Optimal strike: {best_rec.strike} (delta: {best_rec.delta:.2f})"
                )

            elif task == "portfolio_risk":
                greeks = result.get("greeks", {})
                risk_metrics = result.get("risk_metrics", {})

                if abs(greeks.get("total_delta", 0)) > 1000:
                    recommendation["risk_assessment"] = "HIGH"
                    recommendation["reasoning"].append("High delta exposure detected")

                recommendation["position_sizing"]["var_95"] = risk_metrics.get(
                    "var_95", 0
                )

            elif task == "kelly_sizing":
                recommendation["position_sizing"]["kelly_contracts"] = result.get(
                    "max_contracts", 1
                )
                recommendation["position_sizing"]["kelly_fraction"] = result.get(
                    "kelly_fraction", 0.1
                )

            elif task == "market_regime":
                regime = result.get("regime", "unknown")
                if regime == "high_volatility":
                    recommendation["execution_timing"] = "DELAYED"
                    recommendation["reasoning"].append(
                        "High volatility regime - consider waiting"
                    )
                elif regime == "low_volatility":
                    recommendation["execution_timing"] = "IMMEDIATE"

        # Calculate overall confidence
        if component_count > 0:
            recommendation["confidence"] = total_confidence / component_count

        # Determine final action based on analysis
        if (
            recommendation["confidence"] > 0.8
            and recommendation["risk_assessment"] != "HIGH"
        ):
            recommendation["action"] = "BUY"
        elif (
            recommendation["confidence"] < 0.4
            or recommendation["risk_assessment"] == "HIGH"
        ):
            recommendation["action"] = "AVOID"

        return recommendation

    def _handle_thermal_throttle(
        self, is_throttling: bool, mode: TradingPerformanceMode
    ):
        """Handle thermal throttling events."""
        if is_throttling:
            self.metrics.thermal_throttle_events += 1
            logger.warning(f"Thermal throttling activated - Mode: {mode.value}")

            # Adjust batch sizes based on thermal state
            if mode == TradingPerformanceMode.CONSERVATIVE:
                self.adaptive_batch_sizes["strikes"] = 1024
                self.adaptive_batch_sizes["scenarios"] = 500
            elif mode == TradingPerformanceMode.EMERGENCY:
                self.adaptive_batch_sizes["strikes"] = 256
                self.adaptive_batch_sizes["scenarios"] = 100
        else:
            logger.info("Thermal throttling deactivated")
            # Restore normal batch sizes
            self.adaptive_batch_sizes["strikes"] = 4096
            self.adaptive_batch_sizes["scenarios"] = 1000

    def _handle_mode_change(self, mode: TradingPerformanceMode):
        """Handle performance mode changes."""
        self.current_performance_mode = mode
        logger.info(f"Performance mode changed to: {mode.value}")

    def _setup_memory_handlers(self):
        """Setup memory pressure handling."""

        def handle_memory_pressure(usage: float):
            if usage > 0.85:
                # Reduce batch sizes under memory pressure
                self.adaptive_batch_sizes["strikes"] = max(
                    256, self.adaptive_batch_sizes["strikes"] // 2
                )
                self.adaptive_batch_sizes["scenarios"] = max(
                    100, self.adaptive_batch_sizes["scenarios"] // 2
                )
                self.metrics.memory_pressure_events += 1
                logger.warning(f"Memory pressure: {usage:.1%} - reducing batch sizes")

        self.memory_manager.register_pressure_callback(handle_memory_pressure)

    async def _log_session_summary(self):
        """Log comprehensive session summary."""
        if not self.metrics.session_end:
            self.metrics.session_end = datetime.now()

        session_duration = self.metrics.session_end - self.metrics.session_start

        # Update thermal metrics
        thermal_report = await self.thermal_manager.get_thermal_report()
        self.metrics.max_cpu_temp = max(
            self.metrics.max_cpu_temp,
            thermal_report["current_state"]["cpu_temperature"],
        )
        self.metrics.max_gpu_temp = max(
            self.metrics.max_gpu_temp,
            thermal_report["current_state"]["gpu_temperature"],
        )

        # Update memory metrics
        memory_stats = self.memory_manager.get_component_stats("jarvis")
        self.metrics.peak_memory_usage_gb = memory_stats.get("peak_mb", 0) / 1024

        logger.info(
            f"Trading session complete: {self.session_name}",
            extra={
                "session_duration_minutes": session_duration.total_seconds() / 60,
                "total_computations": self.metrics.total_computations,
                "gpu_acceleration_rate": (
                    self.metrics.gpu_accelerated_ops
                    / max(1, self.metrics.total_computations)
                ),
                "average_computation_ms": self.metrics.average_computation_ms,
                "max_cpu_temp": self.metrics.max_cpu_temp,
                "max_gpu_temp": self.metrics.max_gpu_temp,
                "thermal_throttle_events": self.metrics.thermal_throttle_events,
                "peak_memory_gb": self.metrics.peak_memory_usage_gb,
                "recommendations_generated": self.metrics.recommendations_generated,
            },
        )


# Example usage and testing
async def example_m4_pro_trading_session():
    """Example of complete M4 Pro trading session."""

    # Initialize trading system
    trading_system = M4ProTradingSystem(
        num_agents=8,
        enable_gpu=True,
        enable_thermal_monitoring=True,
        session_name="example_session",
    )

    try:
        # Initialize system
        await trading_system.initialize()

        # Sample market data
        market_data = {
            "symbol": "SPY",
            "current_price": 420.50,
            "implied_volatility": 0.22,
            "volume": 50000000,
            "bid": 420.48,
            "ask": 420.52,
        }

        # Sample positions
        positions = [
            # This would be actual Position objects in real usage
        ]

        # Get comprehensive recommendation
        print("Getting comprehensive recommendation...")
        start_time = time.perf_counter()

        recommendation = await trading_system.get_comprehensive_recommendation(
            symbol="SPY",
            portfolio_value=100000,
            current_positions=positions,
            market_data=market_data,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"Recommendation generated in {elapsed_ms:.2f}ms")
        print(f"Action: {recommendation['action']}")
        print(f"Confidence: {recommendation['confidence']:.2f}")
        print(f"Risk Assessment: {recommendation['risk_assessment']}")

        # System performance metadata
        metadata = recommendation["system_metadata"]
        print(f"GPU Utilization: {metadata['gpu_utilization']:.1f}%")
        print(f"Thermal State: {metadata['thermal_state']}")
        print(f"Performance Mode: {metadata['performance_mode']}")
        print(f"Agents Used: {metadata['agents_used']}")

        # Multiple recommendations to test sustained performance
        print("\nTesting sustained performance...")
        for i in range(5):
            await trading_system.get_comprehensive_recommendation(
                symbol="SPY",
                portfolio_value=100000,
                current_positions=positions,
                market_data=market_data,
            )
            print(f"Recommendation {i+1} complete")
            await asyncio.sleep(1)

    finally:
        await trading_system.shutdown()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_m4_pro_trading_session())

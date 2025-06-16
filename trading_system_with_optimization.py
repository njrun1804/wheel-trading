#!/usr/bin/env python3
"""
Trading System with Embedded Optimization
Complete trading system with built-in monitoring and optimization
"""

import asyncio
import contextlib
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

# Import unified system manager
from unified_system_manager import UnifiedSystemManager

# Import trading components (assuming they exist)
try:
    from src.unity_wheel.api.advisor import WheelAdvisor
    from src.unity_wheel.storage.storage import Storage
    from src.unity_wheel.strategy.wheel import WheelStrategy

    HAS_TRADING_COMPONENTS = True
except ImportError as e:
    HAS_TRADING_COMPONENTS = False
    print(f"⚠️  Trading components not found - running in monitoring-only mode: {e}")


@dataclass
class TradingSystemConfig:
    """Configuration for trading system"""

    enable_optimization: bool = True
    enable_monitoring: bool = True
    enable_gpu_acceleration: bool = True
    max_memory_usage_gb: float = 8.0
    max_cpu_usage_percent: float = 80.0
    monitoring_interval: int = 30
    optimization_interval: int = 300


class TradingSystemWithOptimization:
    """Main trading system with embedded optimization"""

    def __init__(self, config: TradingSystemConfig | None = None):
        self.config = config or TradingSystemConfig()
        self.running = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/trading_system.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("TradingSystem")

        # Initialize system manager
        self.system_manager = (
            UnifiedSystemManager() if self.config.enable_optimization else None
        )

        # Initialize trading components
        self.trading_components = {}
        if HAS_TRADING_COMPONENTS:
            self._initialize_trading_components()

        # Setup system callbacks
        if self.system_manager:
            self.system_manager.add_callback(self._system_event_callback)

        self.logger.info("Trading system initialized")

    def _initialize_trading_components(self):
        """Initialize trading system components"""
        try:
            self.trading_components = {
                "advisor": WheelAdvisor(),
                "strategy": WheelStrategy(),
                "storage": Storage(),
            }
            self.logger.info("Trading components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize trading components: {e}")
            self.trading_components = {}

    def _system_event_callback(self, event_type: str, data: dict[str, Any]):
        """Handle system events"""
        if event_type == "memory":
            memory_gb = data["available_mb"] / 1024
            if memory_gb < 2.0:
                self.logger.warning(
                    f"Low memory detected: {memory_gb:.1f}GB - Reducing trading operations"
                )
                self._reduce_trading_operations()
            elif memory_gb < 1.0:
                self.logger.critical(
                    f"Critical memory: {memory_gb:.1f}GB - Pausing trading"
                )
                self._pause_trading_operations()

        elif event_type == "cpu":
            if data.get("cpu_percent", 0) > 90:
                self.logger.warning("High CPU usage - Optimizing trading processes")
                self._optimize_trading_processes()

    def _reduce_trading_operations(self):
        """Reduce trading operations to save resources"""
        self.logger.info("Reducing trading operations for resource conservation")
        # Implementation would reduce polling frequency, batch operations, etc.

    def _pause_trading_operations(self):
        """Pause trading operations during critical resource shortage"""
        self.logger.warning("Pausing trading operations due to resource constraints")
        # Implementation would pause non-critical trading operations

    def _optimize_trading_processes(self):
        """Optimize trading processes for better performance"""
        self.logger.info("Optimizing trading processes")
        # Implementation would optimize database queries, reduce polling, etc.

    async def start_trading(self):
        """Start trading operations"""
        if not HAS_TRADING_COMPONENTS:
            self.logger.warning(
                "Trading components not available - skipping trading operations"
            )
            return

        self.logger.info("Starting trading operations...")

        # Example trading loop
        while self.running:
            try:
                # Get market data and make trading decisions
                # This is where your actual trading logic would go

                # Simulate trading operation
                await asyncio.sleep(60)  # Wait 1 minute between operations

                # Check system health
                if self.system_manager:
                    status = self.system_manager.get_status()
                    memory_gb = status["metrics"]["memory_available_gb"]
                    cpu_percent = status["metrics"]["cpu_percent"]

                    self.logger.info(
                        f"Trading cycle complete - Memory: {memory_gb:.1f}GB, CPU: {cpu_percent:.1f}%"
                    )

            except Exception as e:
                self.logger.error(f"Trading error: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    def start(self):
        """Start the complete system"""
        if self.running:
            return

        self.running = True
        self.logger.info("Starting trading system with embedded optimization...")

        # Start system optimization if enabled
        if self.system_manager and self.config.enable_optimization:
            self.system_manager.start()
            self.logger.info("System optimization started")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start trading operations
        if HAS_TRADING_COMPONENTS:
            # Run trading in async loop
            with contextlib.suppress(KeyboardInterrupt):
                asyncio.run(self.start_trading())
        else:
            # Run in monitoring-only mode
            self.logger.info("Running in monitoring-only mode")
            try:
                while self.running:
                    if self.system_manager:
                        status = self.system_manager.get_status()
                        print(
                            f"\r📊 System Status - "
                            f"CPU: {status['metrics']['cpu_percent']:.1f}% | "
                            f"RAM: {status['metrics']['memory_available_gb']:.1f}GB | "
                            f"Load: {status['metrics']['load_average'][0]:.2f} | "
                            f"Processes: {status['metrics']['process_count']}",
                            end="",
                        )

                    time.sleep(10)
            except KeyboardInterrupt:
                pass

    def stop(self):
        """Stop the complete system"""
        self.running = False
        self.logger.info("Stopping trading system...")

        # Stop system optimization
        if self.system_manager:
            self.system_manager.stop()
            self.logger.info("System optimization stopped")

        self.logger.info("Trading system stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def get_status(self) -> dict[str, Any]:
        """Get complete system status"""
        status = {
            "trading_system": {
                "running": self.running,
                "has_trading_components": HAS_TRADING_COMPONENTS,
                "config": {
                    "optimization_enabled": self.config.enable_optimization,
                    "monitoring_enabled": self.config.enable_monitoring,
                    "gpu_acceleration": self.config.enable_gpu_acceleration,
                },
            }
        }

        # Add system manager status if available
        if self.system_manager:
            status["system_optimization"] = self.system_manager.get_status()

        return status


# Simple CLI interface
def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Trading System with Embedded Optimization"
    )
    parser.add_argument(
        "--no-optimization", action="store_true", help="Disable system optimization"
    )
    parser.add_argument(
        "--no-monitoring", action="store_true", help="Disable system monitoring"
    )
    parser.add_argument(
        "--max-memory", type=float, default=8.0, help="Max memory usage in GB"
    )
    parser.add_argument(
        "--max-cpu", type=float, default=80.0, help="Max CPU usage percentage"
    )
    parser.add_argument("--status", action="store_true", help="Show status and exit")

    args = parser.parse_args()

    # Create configuration
    config = TradingSystemConfig(
        enable_optimization=not args.no_optimization,
        enable_monitoring=not args.no_monitoring,
        max_memory_usage_gb=args.max_memory,
        max_cpu_usage_percent=args.max_cpu,
    )

    # Create and start system
    system = TradingSystemWithOptimization(config)

    if args.status:
        # Just show status
        status = system.get_status()
        print(json.dumps(status, indent=2))
        return

    try:
        print("🚀 Starting Trading System with Embedded Optimization")
        print("   All monitoring and optimization embedded in main process")
        print("   No external daemons or services required")
        print("   Press Ctrl+C to stop")
        print()

        system.start()

    except KeyboardInterrupt:
        print("\n🛑 Stopping Trading System...")
        system.stop()
        print("✅ Stopped successfully")


if __name__ == "__main__":
    main()

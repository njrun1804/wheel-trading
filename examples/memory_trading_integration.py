#!/usr/bin/env python3
"""
Trading System Integration with Memory Management

This example shows how to integrate the memory management system
with existing trading components for optimal resource utilization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptionsData:
    """Mock options data structure"""

    symbol: str
    expiry: datetime
    strikes: list[float]
    prices: dict[float, dict[str, float]]  # strike -> {call/put: price}

    def estimate_memory_mb(self) -> float:
        """Estimate memory usage for this options data"""
        # Rough estimate: each option has ~10 fields * 8 bytes
        num_options = len(self.strikes) * 2  # calls and puts
        return (num_options * 10 * 8) / (1024 * 1024)


class MemoryAwareOptionsProcessor:
    """Options processor that uses memory management system"""

    def __init__(self):
        # Import memory management components
        from src.unity_wheel.memory import (
            TaskPriority,
            allocate_for_trading,
            get_memory_manager,
            schedule_trading_task,
        )

        self.memory_manager = get_memory_manager()
        self.allocate_for_trading = allocate_for_trading
        self.schedule_trading_task = schedule_trading_task
        self.TaskPriority = TaskPriority

        # Set up pressure monitoring
        self.memory_manager.register_pressure_callback(self._handle_memory_pressure)

        # Processing configuration
        self.batch_size = 100  # Default batch size
        self.max_batch_size = 500
        self.min_batch_size = 10

    def _handle_memory_pressure(self, pressure_level: float):
        """Handle memory pressure by reducing batch sizes"""
        if pressure_level > 0.85:
            self.batch_size = max(self.min_batch_size, int(self.batch_size * 0.5))
            logger.warning(
                f"Memory pressure {pressure_level:.1%} - reduced batch size to {self.batch_size}"
            )
        elif pressure_level < 0.6:
            self.batch_size = min(self.max_batch_size, int(self.batch_size * 1.2))
            logger.info(
                f"Memory pressure low {pressure_level:.1%} - increased batch size to {self.batch_size}"
            )

    def process_options_chain(self, symbol: str, expiry: datetime) -> dict:
        """Process options chain with memory management"""
        # Estimate memory requirements
        estimated_mb = self._estimate_processing_memory(symbol, expiry)

        logger.info(
            f"Processing options chain for {symbol} {expiry.strftime('%Y-%m-%d')}"
        )
        logger.info(f"Estimated memory requirement: {estimated_mb:.1f}MB")

        try:
            with self.allocate_for_trading(
                estimated_mb,
                f"Options chain {symbol} {expiry.strftime('%Y-%m-%d')}",
                priority=7,
                tags=["options", symbol.lower(), "chain_processing"],
            ) as alloc_id:
                # Simulate loading options data
                options_data = self._load_options_data(symbol, expiry)

                # Process with memory-aware batching
                results = self._process_with_batching(options_data, alloc_id)

                logger.info(f"Successfully processed {len(results)} options")
                return results

        except MemoryError as e:
            logger.error(f"Memory allocation failed for {symbol}: {e}")
            # Fallback to smaller processing
            return self._process_reduced(symbol, expiry)

    def _estimate_processing_memory(self, symbol: str, expiry: datetime) -> float:
        """Estimate memory needed for processing"""
        # Base memory for options data
        base_mb = 50

        # Additional memory based on symbol popularity
        popularity_multiplier = {
            "SPY": 3.0,
            "QQQ": 2.5,
            "IWM": 2.0,
            "AAPL": 2.5,
            "MSFT": 2.0,
            "GOOGL": 2.0,
        }.get(symbol, 1.0)

        # Additional memory for near-term expiries (more strikes)
        days_to_expiry = (expiry - datetime.now()).days
        expiry_multiplier = (
            2.0 if days_to_expiry <= 7 else 1.5 if days_to_expiry <= 30 else 1.0
        )

        return base_mb * popularity_multiplier * expiry_multiplier

    def _load_options_data(self, symbol: str, expiry: datetime) -> OptionsData:
        """Mock loading of options data"""
        # Generate mock strikes
        base_price = {"SPY": 450, "QQQ": 350, "IWM": 200}.get(symbol, 100)
        strikes = [base_price + i * 5 for i in range(-20, 21)]

        # Generate mock prices
        prices = {}
        for strike in strikes:
            prices[strike] = {
                "call": max(0.01, base_price - strike + np.random.normal(0, 5)),
                "put": max(0.01, strike - base_price + np.random.normal(0, 5)),
            }

        return OptionsData(symbol, expiry, strikes, prices)

    def _process_with_batching(
        self, options_data: OptionsData, alloc_id: str
    ) -> list[dict]:
        """Process options data in memory-aware batches"""
        results = []
        strikes = options_data.strikes

        # Process in batches
        for i in range(0, len(strikes), self.batch_size):
            batch_strikes = strikes[i : i + self.batch_size]

            # Track memory allocation access
            self.memory_manager.access_allocation(alloc_id)

            # Process batch
            batch_results = self._process_strike_batch(options_data, batch_strikes)
            results.extend(batch_results)

            logger.debug(
                f"Processed batch {i//self.batch_size + 1}: {len(batch_strikes)} strikes"
            )

        return results

    def _process_strike_batch(
        self, options_data: OptionsData, strikes: list[float]
    ) -> list[dict]:
        """Process a batch of strikes"""
        results = []

        for strike in strikes:
            if strike in options_data.prices:
                call_price = options_data.prices[strike]["call"]
                put_price = options_data.prices[strike]["put"]

                # Mock Greeks calculation
                delta_call = max(0, min(1, (call_price / strike) * 0.5))
                delta_put = delta_call - 1

                results.append(
                    {
                        "symbol": options_data.symbol,
                        "expiry": options_data.expiry,
                        "strike": strike,
                        "call_price": call_price,
                        "put_price": put_price,
                        "call_delta": delta_call,
                        "put_delta": delta_put,
                        "processed_at": datetime.now(),
                    }
                )

        return results

    def _process_reduced(self, symbol: str, expiry: datetime) -> dict:
        """Fallback processing with minimal memory"""
        logger.warning(f"Using reduced processing for {symbol}")

        try:
            with self.allocate_for_trading(
                10,  # Minimal 10MB
                f"Reduced {symbol} processing",
                priority=5,
                tags=["options", "reduced", symbol.lower()],
            ):
                # Simplified processing
                return {
                    "symbol": symbol,
                    "expiry": expiry,
                    "status": "reduced_processing",
                    "message": "Processed with memory constraints",
                }

        except MemoryError:
            logger.error(f"Even reduced processing failed for {symbol}")
            return {"error": "Memory allocation failed"}


class MemoryAwareMLTraining:
    """ML training that adapts to memory constraints"""

    def __init__(self):
        from src.unity_wheel.memory import (
            TaskPriority,
            allocate_for_ml,
            allocate_tensor_memory,
            get_resource_scheduler,
            schedule_ml_task,
        )

        self.allocate_for_ml = allocate_for_ml
        self.allocate_tensor_memory = allocate_tensor_memory
        self.schedule_ml_task = schedule_ml_task
        self.TaskPriority = TaskPriority
        self.scheduler = get_resource_scheduler()

    def train_model_async(self, model_config: dict) -> str:
        """Schedule ML model training asynchronously"""
        estimated_memory = self._estimate_training_memory(model_config)

        task_id = self.schedule_ml_task(
            name=f"Train {model_config['name']}",
            func=self._train_model,
            memory_mb=estimated_memory,
            priority=self.TaskPriority.HIGH
            if model_config.get("urgent")
            else self.TaskPriority.NORMAL,
            model_config=model_config,
        )

        logger.info(
            f"Scheduled training for {model_config['name']} (Task ID: {task_id})"
        )
        return task_id

    def _estimate_training_memory(self, config: dict) -> float:
        """Estimate memory needed for model training"""

        # Scale by model size
        model_size_mb = config.get("model_size_mb", 100)
        batch_size = config.get("batch_size", 32)

        # Memory = model + batches + gradients + optimizer state
        estimated = model_size_mb * (1 + batch_size * 0.1 + 1 + 0.5)

        return min(estimated, 2000)  # Cap at 2GB

    def _train_model(self, model_config: dict) -> dict:
        """Train ML model with memory management"""
        model_name = model_config["name"]

        try:
            with self.allocate_for_ml(
                self._estimate_training_memory(model_config),
                f"Training {model_name}",
                priority=8,
                tags=["training", "ml_model", model_name.lower()],
            ) as alloc_id:
                logger.info(f"Starting training for {model_name}")

                # Simulate training steps with tensor allocations
                results = self._simulate_training(model_config, alloc_id)

                logger.info(f"Training completed for {model_name}")
                return results

        except MemoryError as e:
            logger.error(f"Training failed for {model_name}: {e}")
            return {"error": str(e), "model": model_name}

    def _simulate_training(self, config: dict, alloc_id: str) -> dict:
        """Simulate model training with memory allocations"""
        import time

        epochs = config.get("epochs", 5)
        batch_size = config.get("batch_size", 32)

        results = {
            "model": config["name"],
            "epochs": epochs,
            "batch_size": batch_size,
            "training_loss": [],
            "validation_loss": [],
        }

        for epoch in range(epochs):
            # Simulate epoch training
            with self.allocate_tensor_memory(
                (batch_size, 512), np.float32, f"Epoch {epoch+1} batch data"
            ):
                # Simulate training step
                time.sleep(0.1)  # Simulate computation

                # Mock loss values
                train_loss = 1.0 - (epoch / epochs) * 0.8 + np.random.normal(0, 0.05)
                val_loss = train_loss + np.random.normal(0, 0.02)

                results["training_loss"].append(max(0.1, train_loss))
                results["validation_loss"].append(max(0.1, val_loss))

                logger.debug(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.3f}")

        return results


def main():
    """Demonstrate memory-aware trading system integration"""
    print("=" * 80)
    print("MEMORY-AWARE TRADING SYSTEM INTEGRATION DEMO")
    print("=" * 80)

    # Initialize components
    options_processor = MemoryAwareOptionsProcessor()
    ml_trainer = MemoryAwareMLTraining()

    print("\n1. PROCESSING OPTIONS CHAINS")
    print("-" * 50)

    # Process multiple options chains
    symbols = ["SPY", "QQQ", "AAPL"]
    expiries = [
        datetime.now() + timedelta(days=7),
        datetime.now() + timedelta(days=30),
        datetime.now() + timedelta(days=60),
    ]

    options_results = []
    for symbol in symbols:
        for expiry in expiries[:2]:  # Process first 2 expiries
            try:
                result = options_processor.process_options_chain(symbol, expiry)
                options_results.append(result)
                print(f"✓ Processed {symbol} {expiry.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"✗ Failed {symbol} {expiry.strftime('%Y-%m-%d')}: {e}")

    print(f"\n✓ Processed {len(options_results)} options chains")

    print("\n2. SCHEDULING ML TRAINING TASKS")
    print("-" * 50)

    # Schedule ML training tasks
    model_configs = [
        {
            "name": "OptionsPredictor",
            "model_size_mb": 150,
            "batch_size": 64,
            "epochs": 3,
            "urgent": True,
        },
        {
            "name": "RiskModel",
            "model_size_mb": 200,
            "batch_size": 32,
            "epochs": 5,
            "urgent": False,
        },
        {
            "name": "SentimentAnalyzer",
            "model_size_mb": 300,
            "batch_size": 16,
            "epochs": 4,
            "urgent": False,
        },
    ]

    training_tasks = []
    for config in model_configs:
        try:
            task_id = ml_trainer.train_model_async(config)
            training_tasks.append((task_id, config["name"]))
            print(f"✓ Scheduled training: {config['name']}")
        except Exception as e:
            print(f"✗ Failed to schedule {config['name']}: {e}")

    print(f"\n✓ Scheduled {len(training_tasks)} training tasks")

    print("\n3. MONITORING TASK EXECUTION")
    print("-" * 50)

    # Monitor training tasks
    completed_tasks = 0
    start_time = time.time()

    while completed_tasks < len(training_tasks) and time.time() - start_time < 30:
        for task_id, model_name in training_tasks:
            task = ml_trainer.scheduler.get_task_status(task_id)
            if (
                task
                and task.state.value == "completed"
                and task_id not in [t[0] for t in training_tasks[:completed_tasks]]
            ):
                print(f"✓ Training completed: {model_name}")
                if task.result and "training_loss" in task.result:
                    final_loss = task.result["training_loss"][-1]
                    print(f"  Final training loss: {final_loss:.3f}")
                completed_tasks += 1

        time.sleep(1)

    print(f"\n✓ Completed {completed_tasks}/{len(training_tasks)} training tasks")

    print("\n4. MEMORY USAGE SUMMARY")
    print("-" * 50)

    # Get final memory report
    from src.unity_wheel.memory import memory_usage_report

    report = memory_usage_report()

    print("System Memory Status:")
    sys_info = report["system"]
    print(f"  System Usage: {sys_info['system_usage_percent']:.1f}%")
    print(f"  Allocated: {sys_info['allocated_mb']:.1f}MB")
    print(f"  Pressure Level: {sys_info['pressure_level']:.1%}")

    print("\nComponent Usage:")
    for component, stats in report["components"].items():
        print(f"  {component.replace('_', ' ').title()}:")
        print(
            f"    Usage: {stats['usage_percent']:.1f}% "
            f"({stats['allocated_mb']:.1f}/{stats['budget_mb']:.1f}MB)"
        )
        print(f"    Allocations: {stats['allocation_count']}")
        print(f"    Peak: {stats['peak_mb']:.1f}MB")

    print("\n5. CLEANUP")
    print("-" * 50)

    # Cleanup
    from src.unity_wheel.memory import get_memory_manager, get_resource_scheduler

    scheduler = get_resource_scheduler()
    manager = get_memory_manager()

    # Get final statistics
    sched_stats = scheduler.get_statistics()
    print("Final scheduler stats:")
    print(f"  Tasks completed: {sched_stats['tasks']['completed']}")
    print(f"  Completion rate: {sched_stats['performance']['completion_rate']:.1%}")
    print(
        f"  Avg execution time: {sched_stats['performance']['average_execution_time']:.2f}s"
    )

    # Shutdown
    scheduler.stop()
    manager.shutdown()

    print("✓ System shutdown complete")

    print("\n" + "=" * 80)
    print("INTEGRATION DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    try:
        import time

        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()

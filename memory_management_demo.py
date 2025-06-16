#!/usr/bin/env python3
"""
Unity Wheel Memory Management System - Comprehensive Demo

This script demonstrates all features of the unified memory management system
including allocation strategies, pressure monitoring, cleanup, pools, and scheduling.
"""

import logging
import time

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("UNITY WHEEL MEMORY MANAGEMENT SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 80)

    # Import memory management components
    from src.unity_wheel.memory import (
        TaskPriority,
        allocate_adaptive_memory,
        allocate_for_database,
        allocate_for_ml,
        allocate_for_trading,
        allocate_tensor_memory,
        create_ml_tensor_pool,
        get_memory_manager,
        get_resource_scheduler,
        memory_usage_report,
        schedule_ml_task,
        schedule_trading_task,
    )

    print("\n1. INITIALIZING MEMORY MANAGEMENT SYSTEM")
    print("-" * 50)

    # Get memory manager (automatically starts monitoring)
    manager = get_memory_manager()
    scheduler = get_resource_scheduler()

    # Register pressure callbacks
    def pressure_callback(pressure_level):
        logger.warning(f"Memory pressure detected: {pressure_level:.1%}")

    def emergency_callback():
        logger.critical("Emergency memory situation - reducing operations")

    manager.register_pressure_callback(pressure_callback)
    manager.register_emergency_callback(emergency_callback)

    print("✓ Memory manager initialized")
    print("✓ Resource scheduler started")
    print("✓ Pressure monitoring active")

    time.sleep(2)

    print("\n2. BASIC MEMORY ALLOCATION EXAMPLES")
    print("-" * 50)

    # Example 1: Direct allocation
    print("Example 1: Direct memory allocation")
    alloc_id = manager.allocate(
        component="trading_data",
        size_mb=50,
        description="SPY options chain",
        priority=7,
        tags=["options", "spy", "real_time"],
    )

    if alloc_id:
        print(f"✓ Allocated 50MB for trading data (ID: {alloc_id})")
        time.sleep(1)
        manager.deallocate(alloc_id)
        print("✓ Deallocated successfully")

    time.sleep(1)

    # Example 2: Context manager allocation
    print("\nExample 2: Context manager allocation")
    try:
        with allocate_for_trading(100, "Options processing", priority=8) as alloc_id:
            print(f"✓ Allocated 100MB for trading (ID: {alloc_id})")
            time.sleep(1)
            # Automatic deallocation on exit
        print("✓ Automatic deallocation completed")
    except MemoryError as e:
        print(f"✗ Allocation failed: {e}")

    time.sleep(1)

    print("\n3. COMPONENT-SPECIFIC ALLOCATIONS")
    print("-" * 50)

    allocation_examples = [
        (
            "Trading Data",
            lambda: allocate_for_trading(75, "Market data processing", priority=7),
        ),
        ("ML Models", lambda: allocate_for_ml(200, "BERT embeddings", priority=8)),
        ("Database", lambda: allocate_for_database(150, "Query results", priority=6)),
        (
            "Tensor Memory",
            lambda: allocate_tensor_memory((500, 768), np.float32, "Word embeddings"),
        ),
    ]

    for name, allocator in allocation_examples:
        try:
            with allocator() as result:
                if isinstance(result, tuple):
                    alloc_id = result
                else:
                    alloc_id = result
                print(f"✓ {name}: Allocated successfully")
                time.sleep(0.5)
        except Exception as e:
            print(f"✗ {name}: Failed - {e}")

    print("\n4. ADAPTIVE MEMORY ALLOCATION")
    print("-" * 50)

    print("Attempting adaptive allocation (starts at 500MB, reduces until successful)")
    try:
        with allocate_adaptive_memory(500, "Large dataset processing") as (
            alloc_id,
            actual_size,
        ):
            print(f"✓ Successfully allocated {actual_size:.1f}MB (requested 500MB)")
            time.sleep(1)
    except MemoryError as e:
        print(f"✗ Even adaptive allocation failed: {e}")

    print("\n5. MEMORY POOLS DEMONSTRATION")
    print("-" * 50)

    # Create specialized pools
    print("Creating specialized memory pools...")
    tensor_pool = create_ml_tensor_pool(500)  # 500MB tensor pool

    # Allocate tensors
    tensor_allocations = []
    for i in range(3):
        shape = (100 + i * 50, 512)
        alloc_id = tensor_pool.allocate_tensor(
            shape=shape, dtype=np.float32, description=f"Tensor batch {i+1}", priority=6
        )
        if alloc_id:
            tensor_allocations.append(alloc_id)
            tensor = tensor_pool.get_tensor(alloc_id)
            print(f"✓ Allocated tensor {shape} - {tensor.nbytes / (1024**2):.1f}MB")

    time.sleep(1)

    # Clean up tensors
    for alloc_id in tensor_allocations:
        tensor_pool.deallocate(alloc_id)
    print(f"✓ Cleaned up {len(tensor_allocations)} tensor allocations")

    print("\n6. RESOURCE SCHEDULING EXAMPLES")
    print("-" * 50)

    # Define some example tasks
    def trading_task(symbol: str, duration: float = 2.0):
        """Simulate trading data processing"""
        logger.info(f"Processing trading data for {symbol}")
        time.sleep(duration)
        return f"Processed {symbol} data"

    def ml_task(model_name: str, duration: float = 3.0):
        """Simulate ML model training"""
        logger.info(f"Training model {model_name}")
        time.sleep(duration)
        return f"Trained {model_name}"

    def database_task(query: str, duration: float = 1.5):
        """Simulate database query"""
        logger.info(f"Executing query: {query[:30]}...")
        time.sleep(duration)
        return "Query results"

    # Schedule tasks with different priorities
    task_ids = []

    # High priority trading tasks
    for symbol in ["SPY", "AAPL", "MSFT"]:
        task_id = schedule_trading_task(
            name=f"Process {symbol}",
            func=trading_task,
            memory_mb=50,
            priority=TaskPriority.HIGH,
            symbol=symbol,
            duration=1.0,
        )
        task_ids.append(task_id)
        print(f"✓ Scheduled trading task: {symbol}")

    # ML tasks
    for model in ["BERT", "GPT", "ResNet"]:
        task_id = schedule_ml_task(
            name=f"Train {model}",
            func=ml_task,
            memory_mb=300,
            priority=TaskPriority.NORMAL,
            model_name=model,
            duration=2.0,
        )
        task_ids.append(task_id)
        print(f"✓ Scheduled ML task: {model}")

    print(f"\n✓ Scheduled {len(task_ids)} tasks total")

    # Monitor task execution
    print("\nMonitoring task execution...")
    completed_tasks = 0
    start_time = time.time()

    while completed_tasks < len(task_ids) and time.time() - start_time < 30:
        for task_id in task_ids:
            task = scheduler.get_task_status(task_id)
            if task and task.state.value == "completed" and task_id not in []:
                completed_tasks += 1
                print(
                    f"✓ Task completed: {task.name} ({task.completed_at - task.started_at:.1f}s)"
                )

        time.sleep(1)

    print(f"\n✓ Completed {completed_tasks}/{len(task_ids)} tasks")

    print("\n7. MEMORY PRESSURE SIMULATION")
    print("-" * 50)

    print("Simulating memory pressure by allocating large blocks...")

    try:
        # Allocate progressively larger blocks to trigger pressure
        for size in [200, 300, 400, 500, 600]:
            try:
                with allocate_for_trading(
                    size, f"Pressure test {size}MB", priority=5
                ) as alloc_id:
                    print(f"✓ Allocated {size}MB - monitoring pressure...")

                    # Check current pressure
                    pressure_level = manager.pressure_monitor.get_pressure_level()
                    classification = (
                        manager.pressure_monitor.get_pressure_classification()
                    )
                    print(
                        f"  Memory pressure: {pressure_level:.1%} ({classification.value})"
                    )

                    if pressure_level > 0.8:
                        print("  ⚠️  High pressure detected!")

                    time.sleep(1)

            except MemoryError:
                print(f"✗ Could not allocate {size}MB - system limit reached")
                break

    except Exception as e:
        print(f"Pressure simulation error: {e}")

    print("\n8. COMPREHENSIVE SYSTEM REPORT")
    print("-" * 50)

    # Get comprehensive system report
    report = memory_usage_report()

    print("System Memory Status:")
    sys_info = report["system"]
    print(f"  Total System: {sys_info['total_system_gb']:.1f}GB")
    print(f"  Usable: {sys_info['usable_gb']:.1f}GB")
    print(f"  Currently Allocated: {sys_info['allocated_mb']:.1f}MB")
    print(f"  System Usage: {sys_info['system_usage_percent']:.1f}%")
    print(f"  Pressure Level: {sys_info['pressure_level']:.1%}")
    print(f"  Emergency Mode: {'Yes' if sys_info.get('emergency_mode') else 'No'}")

    print("\nComponent Memory Usage:")
    for component, stats in report["components"].items():
        print(f"  {component.title()}:")
        print(
            f"    Allocated: {stats['allocated_mb']:.1f}MB / {stats['budget_mb']:.1f}MB"
        )
        print(f"    Usage: {stats['usage_percent']:.1f}%")
        print(f"    Peak: {stats['peak_mb']:.1f}MB")
        print(f"    Allocations: {stats['allocation_count']}")
        print(f"    Evictions: {stats['eviction_count']}")

    # Scheduler statistics
    print("\nResource Scheduler Statistics:")
    sched_stats = scheduler.get_statistics()
    tasks = sched_stats["tasks"]
    perf = sched_stats["performance"]

    print(f"  Tasks Submitted: {tasks['submitted']}")
    print(f"  Tasks Completed: {tasks['completed']}")
    print(f"  Tasks Failed: {tasks['failed']}")
    print(f"  Tasks Running: {tasks['running']}")
    print(f"  Tasks Pending: {tasks['pending']}")
    print(f"  Completion Rate: {perf['completion_rate']:.1%}")
    print(f"  Average Wait Time: {perf['average_wait_time']:.2f}s")
    print(f"  Average Execution Time: {perf['average_execution_time']:.2f}s")

    # Pressure monitoring statistics
    if hasattr(manager, "pressure_monitor"):
        print("\nPressure Monitor Statistics:")
        pressure_stats = manager.pressure_monitor.get_stats()
        current = pressure_stats["current"]
        stats = pressure_stats["stats"]

        print(f"  Current Level: {current['level']}")
        print(f"  Current Pressure: {current['pressure']:.1%}")
        print(f"  Available Memory: {current['available_gb']:.1f}GB")
        print(f"  Total Readings: {stats['readings_count']}")
        print(f"  Pressure Events: {stats['pressure_events']}")
        print(f"  Critical Events: {stats['critical_events']}")
        print(f"  Emergency Events: {stats['emergency_events']}")
        print(f"  Max Pressure Seen: {stats['max_pressure_seen']:.1%}")

    # Cleanup system statistics
    if hasattr(manager, "cleanup_system"):
        print("\nCleanup System Statistics:")
        cleanup_stats = manager.cleanup_system.get_stats()
        runs = cleanup_stats["runs"]
        performance = cleanup_stats["performance"]

        print(f"  Total Cleanup Runs: {runs['total']}")
        print(f"  Light Cleanups: {runs['light']}")
        print(f"  Moderate Cleanups: {runs['moderate']}")
        print(f"  Aggressive Cleanups: {runs['aggressive']}")
        print(f"  Emergency Cleanups: {runs['emergency']}")
        print(f"  Memory Freed: {performance['memory_freed_gb']:.2f}GB")
        print(f"  Objects Collected: {performance['objects_collected']}")
        print(f"  Average Cleanup Time: {performance['average_cleanup_time']:.2f}s")

    print("\n9. CLEANUP AND SHUTDOWN")
    print("-" * 50)

    print("Triggering manual cleanup...")
    manager.trigger_cleanup(aggressive=False)

    print("Stopping scheduler...")
    scheduler.stop()

    print("Shutting down memory manager...")
    manager.shutdown()

    print("✓ All systems shut down cleanly")

    print("\n" + "=" * 80)
    print("MEMORY MANAGEMENT DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        try:
            from src.unity_wheel.memory import (
                get_memory_manager,
                get_resource_scheduler,
            )

            manager = get_memory_manager()
            scheduler = get_resource_scheduler()
            scheduler.stop()
            manager.shutdown()
        except:
            pass
        print("Demo cleanup completed")

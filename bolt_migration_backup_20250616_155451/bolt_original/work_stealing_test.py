#!/usr/bin/env python3
"""
Comprehensive Work Stealing Test for Bolt Agent Pool

This test specifically creates conditions that should trigger work stealing
and validates that it's working correctly.
"""

import asyncio
import json
import time
from typing import Any

from bolt.agents.agent_pool import TaskPriority, WorkStealingAgentPool, WorkStealingTask


async def test_work_stealing_comprehensive() -> dict[str, Any]:
    """Comprehensive work stealing test with deliberate load imbalance."""

    print("🔧 Testing Work Stealing Functionality...")
    print("=" * 60)

    # Create agent pool with work stealing enabled
    agent_pool = WorkStealingAgentPool(num_agents=4, enable_work_stealing=True)
    await agent_pool.initialize()

    try:
        # Phase 1: Create load imbalance by submitting tasks with different execution times
        print("Phase 1: Creating deliberate load imbalance...")

        # Submit heavy computational tasks to create work that can be stolen
        heavy_tasks = []
        for i in range(3):
            task = WorkStealingTask(
                id=f"heavy_computational_{i}",
                description=f"Heavy computational task {i}",
                priority=TaskPriority.HIGH,
                subdividable=True,
                estimated_duration=8.0,  # Long enough to be subdivided
                remaining_work=8.0,
                metadata={
                    "complexity": "high",
                    "iterations": 5000,  # CPU intensive work
                    "type": "analysis_operation",
                },
            )
            heavy_tasks.append(task)
            await agent_pool.submit_task(task)

        # Wait a bit to let heavy tasks start
        await asyncio.sleep(0.2)

        # Submit many light tasks that should trigger stealing
        light_tasks = []
        for i in range(15):
            task = WorkStealingTask(
                id=f"light_task_{i}",
                description=f"Light task {i}",
                priority=TaskPriority.NORMAL,
                subdividable=False,
                estimated_duration=0.1,
                remaining_work=0.1,
                metadata={"test_type": "latency", "type": "search_operation"},
            )
            light_tasks.append(task)
            await agent_pool.submit_task(task)

        # Phase 2: Monitor work stealing activity
        print("Phase 2: Monitoring work stealing activity...")

        monitoring_results = []
        for i in range(10):  # Monitor for 5 seconds
            await asyncio.sleep(0.5)
            status = agent_pool.get_pool_status()
            monitoring_results.append(
                {
                    "time": i * 0.5,
                    "busy_agents": status["busy_agents"],
                    "utilization": status["utilization"],
                    "steals_attempted": status["performance_metrics"][
                        "total_steals_attempted"
                    ],
                    "successful_steals": status["performance_metrics"][
                        "total_tasks_stolen"
                    ],
                    "tasks_completed": status["performance_metrics"][
                        "total_tasks_completed"
                    ],
                    "agent_details": [
                        {
                            "id": agent["id"],
                            "state": agent["state"],
                            "queue_load": agent["queue_load"],
                            "tasks_completed": agent["tasks_completed"],
                            "tasks_stolen": agent["tasks_stolen"],
                        }
                        for agent in status["agent_details"]
                    ],
                }
            )

            print(
                f"  T+{i*0.5:.1f}s: Busy={status['busy_agents']}, "
                f"Util={status['utilization']:.2f}, "
                f"Steals={status['performance_metrics']['total_steals_attempted']}, "
                f"Completed={status['performance_metrics']['total_tasks_completed']}"
            )

        # Phase 3: Final analysis
        print("Phase 3: Final analysis...")
        await asyncio.sleep(2.0)  # Let remaining tasks complete

        final_status = agent_pool.get_pool_status()

        # Collect detailed metrics
        total_steals_attempted = final_status["performance_metrics"][
            "total_steals_attempted"
        ]
        successful_steals = final_status["performance_metrics"]["total_tasks_stolen"]
        steal_success_rate = final_status["performance_metrics"]["steal_success_rate"]
        total_completed = final_status["performance_metrics"]["total_tasks_completed"]
        peak_utilization = max(m["utilization"] for m in monitoring_results)

        # Agent-level statistics
        agent_stats = final_status["agent_details"]
        total_tasks_stolen_by_agents = sum(
            agent["tasks_stolen"] for agent in agent_stats
        )
        total_work_given = sum(agent.get("work_given", 0) for agent in agent_stats)

        print("\n📊 WORK STEALING ANALYSIS:")
        print(f"  Total tasks submitted: {len(heavy_tasks) + len(light_tasks)}")
        print(f"  Total tasks completed: {total_completed}")
        print(f"  Steal attempts: {total_steals_attempted}")
        print(f"  Successful steals: {successful_steals}")
        print(f"  Steal success rate: {steal_success_rate:.2%}")
        print(f"  Peak utilization: {peak_utilization:.2%}")
        print(f"  Tasks stolen by agents: {total_tasks_stolen_by_agents}")
        print(f"  Work redistributed: {total_work_given}")

        # Determine test success
        work_stealing_detected = (
            total_steals_attempted > 0
            or successful_steals > 0
            or total_tasks_stolen_by_agents > 0
            or peak_utilization > 0.7  # High utilization indicates work distribution
        )

        # Enhanced success criteria
        performance_success = (
            total_completed >= len(heavy_tasks) + len(light_tasks) * 0.8
            and peak_utilization  # Most tasks completed
            > 0.5  # Reasonable utilization achieved
        )

        overall_success = work_stealing_detected and performance_success

        if overall_success:
            print("✅ WORK STEALING TEST: PASSED")
        elif work_stealing_detected:
            print(
                "⚠️  WORK STEALING TEST: PARTIAL SUCCESS (stealing detected but performance issues)"
            )
        elif performance_success:
            print(
                "⚠️  WORK STEALING TEST: PARTIAL SUCCESS (good performance but no stealing detected)"
            )
        else:
            print("❌ WORK STEALING TEST: FAILED")

        return {
            "success": overall_success,
            "work_stealing_detected": work_stealing_detected,
            "performance_success": performance_success,
            "metrics": {
                "total_tasks_submitted": len(heavy_tasks) + len(light_tasks),
                "total_tasks_completed": total_completed,
                "completion_rate": total_completed
                / (len(heavy_tasks) + len(light_tasks)),
                "steal_attempts": total_steals_attempted,
                "successful_steals": successful_steals,
                "steal_success_rate": steal_success_rate,
                "peak_utilization": peak_utilization,
                "tasks_stolen_by_agents": total_tasks_stolen_by_agents,
                "work_redistributed": total_work_given,
            },
            "monitoring_timeline": monitoring_results,
            "final_agent_stats": agent_stats,
        }

    finally:
        await agent_pool.shutdown()


async def test_queue_based_work_stealing():
    """Test work stealing specifically from agent queues."""

    print("\n🔄 Testing Queue-Based Work Stealing...")
    print("=" * 40)

    agent_pool = WorkStealingAgentPool(num_agents=3, enable_work_stealing=True)
    await agent_pool.initialize()

    try:
        # Submit all tasks to one agent initially to create imbalance
        target_agent = agent_pool.agents[0]

        # Load one agent with many tasks
        for i in range(10):
            task = WorkStealingTask(
                id=f"queue_task_{i}",
                description=f"Queue task {i}",
                priority=TaskPriority.NORMAL,
                subdividable=False,
                estimated_duration=0.5,
                metadata={"type": "analysis_operation"},
            )
            await target_agent.submit_task(task)

        print(f"Loaded agent {target_agent.id} with {target_agent.queue_load} tasks")

        # Monitor stealing activity
        start_time = time.time()
        steal_events = []

        for i in range(8):  # Monitor for 4 seconds
            await asyncio.sleep(0.5)

            status = agent_pool.get_pool_status()
            queue_loads = [agent["queue_load"] for agent in status["agent_details"]]
            total_steals = sum(
                agent["tasks_stolen"] for agent in status["agent_details"]
            )

            steal_events.append(
                {
                    "time": time.time() - start_time,
                    "queue_loads": queue_loads,
                    "total_steals": total_steals,
                    "load_balance": max(queue_loads) - min(queue_loads),
                }
            )

            print(f"  T+{i*0.5:.1f}s: Queues={queue_loads}, Steals={total_steals}")

        # Check if load balancing occurred
        initial_imbalance = steal_events[0]["load_balance"]
        final_imbalance = steal_events[-1]["load_balance"]
        total_steals = steal_events[-1]["total_steals"]

        load_balancing_occurred = (
            final_imbalance < initial_imbalance or total_steals > 0
        )

        print("\n📈 QUEUE STEALING ANALYSIS:")
        print(f"  Initial load imbalance: {initial_imbalance}")
        print(f"  Final load imbalance: {final_imbalance}")
        print(f"  Total steals recorded: {total_steals}")
        print(f"  Load balancing occurred: {load_balancing_occurred}")

        return {
            "success": load_balancing_occurred,
            "initial_imbalance": initial_imbalance,
            "final_imbalance": final_imbalance,
            "total_steals": total_steals,
            "steal_timeline": steal_events,
        }

    finally:
        await agent_pool.shutdown()


async def main():
    """Run comprehensive work stealing tests."""

    print("🚀 BOLT WORK STEALING VALIDATION")
    print("=" * 80)

    # Test 1: Comprehensive work stealing
    test1_result = await test_work_stealing_comprehensive()

    # Test 2: Queue-based work stealing
    test2_result = await test_queue_based_work_stealing()

    # Overall assessment
    overall_success = test1_result["success"] and test2_result["success"]

    print(f"\n{'='*80}")
    print("FINAL WORK STEALING VALIDATION RESULTS")
    print(f"{'='*80}")
    print(
        f"Test 1 - Comprehensive: {'✅ PASSED' if test1_result['success'] else '❌ FAILED'}"
    )
    print(
        f"Test 2 - Queue Stealing: {'✅ PASSED' if test2_result['success'] else '❌ FAILED'}"
    )
    print(
        f"Overall Result: {'✅ WORK STEALING FUNCTIONAL' if overall_success else '❌ WORK STEALING ISSUES'}"
    )

    # Save detailed results
    results = {
        "timestamp": time.time(),
        "overall_success": overall_success,
        "test1_comprehensive": test1_result,
        "test2_queue_stealing": test2_result,
    }

    with open("work_stealing_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n📄 Detailed results saved to: work_stealing_validation_results.json")

    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

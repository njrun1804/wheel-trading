#!/usr/bin/env python3
"""
Bob stress test - High concurrency validation.
Tests the 12-agent system under heavy load to validate M4 optimizations.
"""

import asyncio
import time
from bob_simple_test import Simple12AgentCoordinator, SimpleTask, AgentRole


async def run_stress_test():
    """Run high-concurrency stress test."""
    print("🔥 Bob M4 Pro STRESS TEST")
    print("=" * 50)
    print("Testing 12-agent system under high concurrent load")
    print("=" * 50)
    
    coordinator = Simple12AgentCoordinator()
    
    # Create a large number of concurrent tasks
    stress_tasks = []
    
    # 20 simple tasks
    for i in range(20):
        stress_tasks.append(SimpleTask(
            task_id=f"stress_simple_{i}",
            description=f"Simple analysis task {i}",
            complexity="simple",
            preferred_roles=[AgentRole.ANALYZER, AgentRole.RESEARCHER],
            estimated_duration=0.3
        ))
    
    # 15 moderate tasks  
    for i in range(15):
        stress_tasks.append(SimpleTask(
            task_id=f"stress_moderate_{i}",
            description=f"Moderate optimization task {i}",
            complexity="moderate", 
            preferred_roles=[AgentRole.OPTIMIZER, AgentRole.VALIDATOR],
            estimated_duration=0.8
        ))
    
    # 10 complex tasks
    for i in range(10):
        stress_tasks.append(SimpleTask(
            task_id=f"stress_complex_{i}",
            description=f"Complex architecture task {i}",
            complexity="complex",
            preferred_roles=[AgentRole.ARCHITECT, AgentRole.INTEGRATOR],
            estimated_duration=1.5
        ))
    
    # 5 enterprise tasks
    for i in range(5):
        stress_tasks.append(SimpleTask(
            task_id=f"stress_enterprise_{i}",
            description=f"Enterprise analysis task {i}",
            complexity="enterprise",
            preferred_roles=[AgentRole.ANALYZER, AgentRole.SYNTHESIZER],
            estimated_duration=2.0
        ))
    
    print(f"🎯 Stress test parameters:")
    print(f"  📊 Total tasks: {len(stress_tasks)}")
    print(f"  📋 Simple: 20, Moderate: 15, Complex: 10, Enterprise: 5")
    print(f"  🤖 Agents available: 12 (8 P-core + 4 E-core)")
    print(f"  ⚡ Expected load factor: {len(stress_tasks)/12:.1f}x oversubscription")
    print()
    
    # Execute stress test
    print("🚀 Launching stress test...")
    start_time = time.time()
    
    results = await coordinator.execute_parallel_tasks(stress_tasks)
    
    total_time = time.time() - start_time
    
    # Analyze stress test results
    successful_tasks = sum(1 for r in results if r.get("success", False))
    total_task_time = sum(r.get("duration", 0) for r in results)
    parallel_efficiency = total_task_time / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("🔥 STRESS TEST RESULTS")
    print("=" * 60)
    
    print(f"✅ Tasks Completed: {successful_tasks}/{len(stress_tasks)} ({successful_tasks/len(stress_tasks):.1%})")
    print(f"⏱️  Wall Clock Time: {total_time:.2f}s")
    print(f"🕐 Total Task Time: {total_task_time:.2f}s")
    print(f"🚀 Parallel Speedup: {parallel_efficiency:.1f}x")
    print(f"📊 Throughput: {len(stress_tasks)/total_time:.1f} tasks/second")
    
    # Get detailed performance report
    report = coordinator.get_performance_report()
    metrics = report["system_metrics"]
    
    print(f"\n📈 RESOURCE UTILIZATION:")
    print(f"  P-core utilization: {metrics['p_core_utilization']:.1%}")
    print(f"  E-core utilization: {metrics['e_core_utilization']:.1%}")
    print(f"  Average task duration: {metrics['avg_task_duration']:.2f}s")
    
    # Agent workload distribution
    print(f"\n🤖 AGENT WORKLOAD DISTRIBUTION:")
    agent_stats = report["agent_stats"]
    
    p_core_agents = [(role, stats) for role, stats in agent_stats.items() 
                     if stats["core_type"] == "P-core" and stats["tasks_completed"] > 0]
    e_core_agents = [(role, stats) for role, stats in agent_stats.items() 
                     if stats["core_type"] == "E-core" and stats["tasks_completed"] > 0]
    
    print(f"  P-core agents:")
    for role, stats in sorted(p_core_agents, key=lambda x: x[1]["tasks_completed"], reverse=True):
        efficiency = stats["tasks_completed"] / stats["total_duration"] if stats["total_duration"] > 0 else 0
        print(f"    {role:>12}: {stats['tasks_completed']:>2} tasks, {stats['avg_duration']:.2f}s avg, {efficiency:.1f} tasks/s")
    
    print(f"  E-core agents:")
    for role, stats in sorted(e_core_agents, key=lambda x: x[1]["tasks_completed"], reverse=True):
        efficiency = stats["tasks_completed"] / stats["total_duration"] if stats["total_duration"] > 0 else 0
        print(f"    {role:>12}: {stats['tasks_completed']:>2} tasks, {stats['avg_duration']:.2f}s avg, {efficiency:.1f} tasks/s")
    
    # Performance assessment
    print(f"\n💡 STRESS TEST ANALYSIS:")
    
    # Throughput analysis
    if len(stress_tasks)/total_time > 10:
        print("  ✅ Excellent throughput - System handles high concurrent load well")
    elif len(stress_tasks)/total_time > 5:
        print("  ✅ Good throughput - System scales well under load")
    else:
        print("  ⚠️  Moderate throughput - System may be reaching limits")
    
    # Load balancing analysis
    max_tasks = max(stats["tasks_completed"] for stats in agent_stats.values())
    min_tasks = min(stats["tasks_completed"] for stats in agent_stats.values() if stats["tasks_completed"] > 0)
    load_balance_ratio = min_tasks / max_tasks if max_tasks > 0 else 0
    
    if load_balance_ratio > 0.7:
        print("  ✅ Excellent load balancing across agents")
    elif load_balance_ratio > 0.5:
        print("  ✅ Good load balancing - Minor optimization opportunities")
    else:
        print("  ⚠️  Uneven load distribution - Task routing could be improved")
    
    # Efficiency analysis
    if parallel_efficiency > 8:
        print("  ✅ Excellent parallel efficiency - M4 optimizations highly effective")
    elif parallel_efficiency > 5:
        print("  ✅ Good parallel efficiency - Multi-agent coordination working well")
    else:
        print("  ⚠️  Moderate efficiency - Consider optimization tuning")
    
    # Overall stress test assessment
    stress_score = (
        (successful_tasks / len(stress_tasks)) * 0.3 +      # Success rate
        min(parallel_efficiency / 10, 1.0) * 0.3 +         # Efficiency
        min((len(stress_tasks)/total_time) / 10, 1.0) * 0.2 + # Throughput
        load_balance_ratio * 0.2                            # Load balancing
    )
    
    print(f"\n🎯 OVERALL STRESS TEST ASSESSMENT:")
    if stress_score >= 0.8:
        print("  🚀 OUTSTANDING: System excels under high concurrent load")
    elif stress_score >= 0.7:
        print("  ✅ EXCELLENT: System handles stress test very well")
    elif stress_score >= 0.6:
        print("  ✅ GOOD: System performs well under load with minor issues")
    elif stress_score >= 0.5:
        print("  ⚠️  MODERATE: System functional but shows stress under load")
    else:
        print("  ❌ NEEDS IMPROVEMENT: System struggles under high concurrent load")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if metrics['p_core_utilization'] < 0.8:
        print("  🔧 Consider improving P-core task routing for compute-intensive work")
    
    if load_balance_ratio < 0.6:
        print("  🔧 Implement better load balancing to distribute work more evenly")
    
    if parallel_efficiency < 8:
        print("  🔧 Optimize coordination overhead to improve parallel efficiency")
    
    if len(stress_tasks)/total_time < 8:
        print("  🔧 Consider increasing concurrency limits or optimizing task execution")
    
    print(f"\n📊 KEY METRICS SUMMARY:")
    print(f"  Success Rate: {successful_tasks/len(stress_tasks):.1%}")
    print(f"  Throughput: {len(stress_tasks)/total_time:.1f} tasks/sec")
    print(f"  Parallel Speedup: {parallel_efficiency:.1f}x")
    print(f"  Load Balance Ratio: {load_balance_ratio:.2f}")
    print(f"  Stress Score: {stress_score:.2f}/1.0")
    
    return {
        "success_rate": successful_tasks / len(stress_tasks),
        "throughput": len(stress_tasks) / total_time,
        "parallel_efficiency": parallel_efficiency,
        "load_balance_ratio": load_balance_ratio,
        "stress_score": stress_score,
        "total_time": total_time,
        "total_tasks": len(stress_tasks)
    }


async def main():
    """Main stress test function."""
    try:
        results = await run_stress_test()
        print(f"\n🏁 Stress test completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = asyncio.run(main())
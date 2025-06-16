#!/usr/bin/env python3
"""
Advanced BOB features demonstration.

This example showcases:
- Custom agent behaviors
- Task pipelines
- Hardware optimization
- Parallel execution patterns
- Advanced error recovery
- Performance profiling
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from bolt.core.integration import BoltIntegration
from bolt.agents.types import Task, TaskPriority, AgentBehavior
from bolt.hardware.performance_monitor import PerformanceMonitor
from bolt.error_handling import BoltException, ErrorSeverity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom task types for financial analysis
class FinancialTaskType(Enum):
    RISK_ANALYSIS = "risk_analysis"
    PERFORMANCE_CALC = "performance_calculation"
    OPTIMIZATION = "optimization"
    DATA_VALIDATION = "data_validation"
    REPORT_GENERATION = "report_generation"


@dataclass
class FinancialTask(Task):
    """Extended task for financial operations."""
    task_type: FinancialTaskType
    symbols: List[str]
    time_sensitive: bool = False
    regulatory_required: bool = False


class FinancialAnalysisAgent(AgentBehavior):
    """Custom agent behavior for financial analysis tasks."""
    
    def __init__(self):
        super().__init__()
        self.specialization = "financial_analysis"
        self.priority_boost = {
            FinancialTaskType.RISK_ANALYSIS: 2,
            FinancialTaskType.REGULATORY: 3,
        }
    
    async def preprocess_task(self, task: Task) -> Task:
        """Preprocess task with financial context."""
        if isinstance(task, FinancialTask):
            # Boost priority for risk and regulatory tasks
            if task.task_type == FinancialTaskType.RISK_ANALYSIS:
                task.priority = TaskPriority.HIGH
            elif task.regulatory_required:
                task.priority = TaskPriority.CRITICAL
            
            # Add market data context
            task.data = task.data or {}
            task.data['market_hours'] = self._is_market_hours()
            task.data['volatility_regime'] = await self._get_volatility_regime()
        
        return task
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process financial task with specialized logic."""
        if isinstance(task, FinancialTask):
            if task.task_type == FinancialTaskType.RISK_ANALYSIS:
                return await self._analyze_risk(task)
            elif task.task_type == FinancialTaskType.OPTIMIZATION:
                return await self._optimize_strategy(task)
        
        # Fallback to default processing
        return await super().process_task(task)
    
    async def _analyze_risk(self, task: FinancialTask) -> Dict[str, Any]:
        """Specialized risk analysis."""
        # Simulate complex risk calculation
        await asyncio.sleep(0.1)
        return {
            "risk_metrics": {
                "var_95": 1000.0,
                "cvar_95": 1500.0,
                "max_drawdown": 0.15
            }
        }
    
    def _is_market_hours(self) -> bool:
        """Check if market is open."""
        # Simplified check
        from datetime import datetime
        now = datetime.now()
        return 9 <= now.hour < 16 and now.weekday() < 5
    
    async def _get_volatility_regime(self) -> str:
        """Determine current volatility regime."""
        # Simulate API call
        await asyncio.sleep(0.05)
        return "normal"  # low, normal, high


async def custom_agent_example():
    """Demonstrate custom agent behaviors."""
    # Create BOB with custom agent configuration
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Register custom agent behavior
        financial_agent = FinancialAnalysisAgent()
        await bob.register_agent_behavior("financial", financial_agent)
        
        # Create financial tasks
        tasks = [
            FinancialTask(
                id="risk_001",
                description="Calculate portfolio risk metrics",
                task_type=FinancialTaskType.RISK_ANALYSIS,
                symbols=["U"],
                priority=TaskPriority.MEDIUM,
                time_sensitive=True
            ),
            FinancialTask(
                id="opt_001",
                description="Optimize wheel strategy parameters",
                task_type=FinancialTaskType.OPTIMIZATION,
                symbols=["U"],
                priority=TaskPriority.LOW
            ),
            FinancialTask(
                id="reg_001",
                description="Generate regulatory risk report",
                task_type=FinancialTaskType.REPORT_GENERATION,
                symbols=["U"],
                regulatory_required=True,
                priority=TaskPriority.MEDIUM  # Will be upgraded to CRITICAL
            )
        ]
        
        # Execute with custom behavior
        logger.info("Executing financial tasks with custom agent behavior...")
        
        # BOB will automatically route to specialized agents
        results = await bob.execute_tasks(tasks)
        
        # Display results
        logger.info("\n=== Custom Agent Results ===")
        for result in results:
            logger.info(f"Task {result['task_id']}: {result['status']}")
            if result['priority_upgraded']:
                logger.info(f"  Priority upgraded: {result['original_priority']} → {result['final_priority']}")
            if result.get('specialized_processing'):
                logger.info(f"  Processed by: {result['agent_type']}")
        
        return results
        
    finally:
        await bob.shutdown()


async def task_pipeline_example():
    """Demonstrate multi-stage task pipelines."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Define pipeline stages
        pipeline_stages = [
            {
                "name": "Data Collection",
                "query": "Collect historical options data for Unity stock (last 6 months)",
                "output_key": "historical_data"
            },
            {
                "name": "Data Validation",
                "query": "Validate options data for completeness and accuracy",
                "input_key": "historical_data",
                "output_key": "validated_data"
            },
            {
                "name": "Performance Analysis",
                "query": "Analyze wheel strategy performance using validated data",
                "input_key": "validated_data",
                "output_key": "performance_metrics"
            },
            {
                "name": "Optimization",
                "query": "Optimize strategy parameters based on performance analysis",
                "input_key": "performance_metrics",
                "output_key": "optimized_params"
            },
            {
                "name": "Report Generation",
                "query": "Generate comprehensive report with recommendations",
                "input_keys": ["performance_metrics", "optimized_params"],
                "output_key": "final_report"
            }
        ]
        
        # Create and execute pipeline
        logger.info("Executing multi-stage analysis pipeline...")
        
        pipeline_context = {}
        stage_results = []
        
        for i, stage in enumerate(pipeline_stages):
            logger.info(f"\n[Stage {i+1}/{len(pipeline_stages)}] {stage['name']}")
            
            # Build context from previous stages
            stage_context = {}
            if 'input_key' in stage:
                stage_context[stage['input_key']] = pipeline_context.get(stage['input_key'])
            elif 'input_keys' in stage:
                for key in stage['input_keys']:
                    stage_context[key] = pipeline_context.get(key)
            
            # Execute stage
            start_time = time.time()
            result = await bob.execute_query(stage['query'], context=stage_context)
            duration = time.time() - start_time
            
            # Store output for next stages
            if 'output_key' in stage:
                pipeline_context[stage['output_key']] = result.get_result('data')
            
            # Track stage results
            stage_results.append({
                'stage': stage['name'],
                'duration': duration,
                'success': result['success'],
                'tasks_completed': len([r for r in result['results'] if r['success']])
            })
            
            logger.info(f"  Completed in {duration:.2f}s")
            logger.info(f"  Tasks: {stage_results[-1]['tasks_completed']} completed")
        
        # Pipeline summary
        logger.info("\n=== Pipeline Summary ===")
        total_duration = sum(r['duration'] for r in stage_results)
        logger.info(f"Total duration: {total_duration:.2f}s")
        logger.info(f"Stages completed: {len([r for r in stage_results if r['success']])}/{len(stage_results)}")
        
        # Display final report preview
        final_report = pipeline_context.get('final_report', {})
        if final_report:
            logger.info("\nFinal Report Preview:")
            logger.info(f"  Recommended Delta Target: {final_report.get('delta_target', 'N/A')}")
            logger.info(f"  Recommended DTE: {final_report.get('dte_target', 'N/A')}")
            logger.info(f"  Expected Annual Return: {final_report.get('expected_return', 'N/A')}")
        
        return stage_results
        
    finally:
        await bob.shutdown()


async def hardware_optimization_example():
    """Demonstrate hardware-specific optimizations."""
    # Configure for maximum M4 Pro performance
    from bolt.hardware.cpu_optimizer import CPUOptimizer
    from bolt.gpu_acceleration import GPUAccelerator
    
    # Setup CPU optimization
    cpu_optimizer = CPUOptimizer()
    cpu_optimizer.configure({
        "performance_cores": list(range(8)),  # P-cores 0-7
        "efficiency_cores": list(range(8, 12)),  # E-cores 8-11
        "affinity_mode": "performance",
        "prefetch_distance": 256
    })
    
    # Setup GPU acceleration
    gpu_config = {
        "backend": "metal",
        "compute_units": 20,
        "memory_fraction": 0.8
    }
    
    # Create optimized BOB instance
    bob = BoltIntegration(
        num_agents=8,
        hardware_config={
            "cpu_optimizer": cpu_optimizer,
            "gpu_config": gpu_config,
            "memory_pools": {
                "agent_workspace": 512 * 1024 * 1024,  # 512MB per agent
                "cache_size": 2 * 1024 * 1024 * 1024   # 2GB cache
            }
        }
    )
    
    try:
        await bob.initialize()
        logger.info("Hardware-optimized BOB initialized")
        
        # Run compute-intensive query
        query = """
        Perform Monte Carlo simulation for wheel strategy:
        - 10,000 scenarios
        - 1 year time horizon
        - Include path-dependent features
        - Calculate VaR and CVaR
        - Generate return distribution
        """
        
        # Monitor hardware utilization
        monitor = PerformanceMonitor()
        monitor.start()
        
        logger.info("Starting compute-intensive Monte Carlo simulation...")
        start_time = time.time()
        
        result = await bob.execute_query(
            query,
            context={
                "use_gpu": True,
                "batch_size": 1000,  # Process 1000 scenarios at a time
                "precision": "float16",  # Use half precision for speed
                "parallel_paths": True
            }
        )
        
        duration = time.time() - start_time
        metrics = monitor.get_metrics()
        
        # Display results
        logger.info(f"\n=== Hardware Performance ===")
        logger.info(f"Simulation completed in {duration:.2f}s")
        logger.info(f"Scenarios per second: {10000/duration:.0f}")
        logger.info(f"CPU utilization: {metrics['avg_cpu_percent']:.1f}%")
        logger.info(f"GPU utilization: {metrics['avg_gpu_percent']:.1f}%")
        logger.info(f"Memory used: {metrics['peak_memory_gb']:.1f}GB")
        logger.info(f"GPU memory: {metrics['gpu_memory_used_gb']:.1f}GB")
        
        # Simulation results
        sim_results = result.get_result('simulation_results', {})
        logger.info(f"\n=== Simulation Results ===")
        logger.info(f"Expected Return: {sim_results.get('expected_return', 0):.2%}")
        logger.info(f"95% VaR: ${sim_results.get('var_95', 0):,.2f}")
        logger.info(f"95% CVaR: ${sim_results.get('cvar_95', 0):,.2f}")
        logger.info(f"Win Rate: {sim_results.get('win_rate', 0):.1%}")
        
        return result
        
    finally:
        monitor.stop()
        await bob.shutdown()


async def parallel_execution_patterns():
    """Demonstrate advanced parallel execution patterns."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Pattern 1: Map-Reduce for multiple symbols
        symbols = ["SPY", "QQQ", "IWM", "DIA", "U"]
        
        logger.info(f"Pattern 1: Map-Reduce analysis for {len(symbols)} symbols")
        
        # Map phase - analyze each symbol in parallel
        map_tasks = []
        for symbol in symbols:
            task = bob.execute_query(
                f"Analyze wheel strategy opportunities for {symbol}",
                context={"symbol": symbol, "quick_analysis": True}
            )
            map_tasks.append(task)
        
        # Execute all analyses in parallel
        map_results = await asyncio.gather(*map_tasks, return_exceptions=True)
        
        # Reduce phase - combine results
        reduce_query = f"""
        Compare wheel strategy opportunities across symbols and rank by:
        1. Expected return
        2. Risk-adjusted return (Sharpe)
        3. Liquidity score
        Provide top 3 recommendations
        """
        
        reduce_result = await bob.execute_query(
            reduce_query,
            context={"symbol_analyses": map_results}
        )
        
        recommendations = reduce_result.get_result('recommendations', [])
        logger.info("\nTop Symbol Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            logger.info(f"{i}. {rec['symbol']}: {rec['reason']}")
        
        # Pattern 2: Pipeline with parallel stages
        logger.info("\n\nPattern 2: Parallel pipeline execution")
        
        # Define parallel stage groups
        stage_groups = [
            # Group 1: Data collection (can run in parallel)
            [
                "Fetch current options chain",
                "Get historical volatility data",
                "Retrieve market sentiment indicators"
            ],
            # Group 2: Analysis (depends on Group 1)
            [
                "Calculate optimal strikes",
                "Analyze volatility surface",
                "Assess market regime"
            ],
            # Group 3: Decision (depends on Group 2)
            [
                "Generate trade signals"
            ]
        ]
        
        group_results = []
        for group_idx, group in enumerate(stage_groups):
            logger.info(f"\nExecuting parallel group {group_idx + 1}")
            
            # Execute all queries in group in parallel
            group_tasks = [bob.execute_query(query) for query in group]
            results = await asyncio.gather(*group_tasks)
            group_results.append(results)
            
            # Show completion
            for query, result in zip(group, results):
                success = all(r['success'] for r in result['results'])
                logger.info(f"  {'✓' if success else '✗'} {query}")
        
        # Pattern 3: Speculative execution with fallback
        logger.info("\n\nPattern 3: Speculative execution with fallback")
        
        primary_query = "Use advanced ML model to predict option prices"
        fallback_query = "Use Black-Scholes model for option pricing"
        quick_query = "Use simple approximation for option pricing"
        
        # Start all three in parallel
        logger.info("Starting speculative execution...")
        tasks = {
            'primary': asyncio.create_task(bob.execute_query(primary_query)),
            'fallback': asyncio.create_task(bob.execute_query(fallback_query)),
            'quick': asyncio.create_task(bob.execute_query(quick_query))
        }
        
        # Wait for first successful result
        result = None
        method_used = None
        
        for method in ['quick', 'primary', 'fallback']:
            try:
                if method == 'quick':
                    # Quick should finish first
                    result = await asyncio.wait_for(tasks['quick'], timeout=1.0)
                    method_used = 'quick'
                    break
                elif method == 'primary':
                    # Try primary with timeout
                    result = await asyncio.wait_for(tasks['primary'], timeout=3.0)
                    method_used = 'primary'
                    break
            except asyncio.TimeoutError:
                logger.info(f"  {method} method timed out, trying next...")
                continue
        
        # Use fallback if nothing else worked
        if not result:
            result = await tasks['fallback']
            method_used = 'fallback'
        
        # Cancel remaining tasks
        for task in tasks.values():
            if not task.done():
                task.cancel()
        
        logger.info(f"  Used {method_used} method for pricing")
        logger.info(f"  Result: {result.get_result('status', 'completed')}")
        
        return group_results
        
    finally:
        await bob.shutdown()


async def advanced_error_recovery():
    """Demonstrate advanced error recovery patterns."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Configure advanced error handling
        bob.configure_error_handling({
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 30,
                "half_open_requests": 1
            },
            "retry_policy": {
                "max_attempts": 3,
                "backoff_type": "exponential",
                "backoff_base": 2,
                "jitter": True
            },
            "fallback_strategies": {
                "graceful_degradation": True,
                "cached_results": True,
                "alternative_methods": True
            }
        })
        
        # Test 1: Retry with exponential backoff
        logger.info("Test 1: Retry with exponential backoff")
        
        # Simulate intermittent failure
        query = "Fetch real-time option prices (may fail intermittently)"
        
        try:
            result = await bob.execute_query(
                query,
                context={"simulate_failures": True, "failure_rate": 0.6}
            )
            logger.info(f"  Success after {result.get_metadata('retry_count', 0)} retries")
        except BoltException as e:
            logger.error(f"  Failed after all retries: {e.message}")
        
        # Test 2: Circuit breaker pattern
        logger.info("\nTest 2: Circuit breaker pattern")
        
        # Simulate service degradation
        for i in range(5):
            try:
                result = await bob.execute_query(
                    "Call unreliable external service",
                    context={"simulate_failures": i < 3}  # First 3 calls fail
                )
                logger.info(f"  Call {i+1}: Success")
            except BoltException as e:
                logger.warning(f"  Call {i+1}: {e.message}")
                if "circuit breaker open" in str(e).lower():
                    logger.info("    Circuit breaker activated - failing fast")
        
        # Test 3: Graceful degradation
        logger.info("\nTest 3: Graceful degradation")
        
        complex_query = """
        Perform comprehensive analysis:
        1. Real-time market data (may fail)
        2. Complex ML predictions (may timeout)
        3. Historical analysis (reliable)
        4. Basic calculations (always works)
        """
        
        result = await bob.execute_query(
            complex_query,
            context={
                "allow_partial_results": True,
                "min_required_components": 2
            }
        )
        
        # Check degradation
        degradation = result.get_metadata('degradation', {})
        logger.info(f"  Full functionality: {not degradation.get('degraded', True)}")
        if degradation.get('degraded'):
            logger.info(f"  Completed components: {degradation.get('completed', [])}")
            logger.info(f"  Failed components: {degradation.get('failed', [])}")
            logger.info(f"  Quality score: {degradation.get('quality_score', 0):.1%}")
        
        # Test 4: Intelligent fallback
        logger.info("\nTest 4: Intelligent fallback strategies")
        
        fallback_query = "Calculate option Greeks using best available method"
        
        result = await bob.execute_query(
            fallback_query,
            context={
                "methods": [
                    {"name": "gpu_accelerated", "timeout": 1.0},
                    {"name": "parallel_cpu", "timeout": 3.0},
                    {"name": "single_threaded", "timeout": 10.0}
                ],
                "prefer_speed": True
            }
        )
        
        method_used = result.get_metadata('method_used')
        logger.info(f"  Method used: {method_used}")
        logger.info(f"  Fallback chain: {result.get_metadata('fallback_chain', [])}")
        
        return True
        
    finally:
        await bob.shutdown()


async def performance_profiling_example():
    """Demonstrate performance profiling capabilities."""
    from bolt.profiling import Profiler, ProfileScope
    
    bob = BoltIntegration(num_agents=8)
    profiler = Profiler({
        "cpu_profiling": True,
        "memory_profiling": True,
        "gpu_profiling": True,
        "network_profiling": True,
        "disk_io_profiling": True
    })
    
    try:
        await bob.initialize()
        
        # Start profiling
        profiler.start()
        
        # Complex workload for profiling
        workload_query = """
        Execute comprehensive trading system analysis:
        1. Load 1 year of options data
        2. Calculate Greeks for all positions
        3. Run 1000 Monte Carlo simulations
        4. Generate performance attribution
        5. Create risk decomposition report
        """
        
        # Profile different aspects
        with profiler.profile_scope("data_loading"):
            # Data loading phase
            pass
        
        with profiler.profile_scope("computation"):
            # Heavy computation
            result = await bob.execute_query(workload_query)
        
        with profiler.profile_scope("report_generation"):
            # Report generation
            pass
        
        # Stop profiling and get results
        profile_data = profiler.stop()
        
        # Display profiling results
        logger.info("\n=== Performance Profile ===")
        
        # Overall metrics
        logger.info(f"Total execution time: {profile_data['total_time']:.2f}s")
        logger.info(f"CPU time: {profile_data['cpu_time']:.2f}s")
        logger.info(f"Wall time: {profile_data['wall_time']:.2f}s")
        logger.info(f"CPU efficiency: {profile_data['cpu_efficiency']:.1%}")
        
        # Breakdown by scope
        logger.info("\nTime by scope:")
        for scope, metrics in profile_data['scopes'].items():
            logger.info(f"  {scope}: {metrics['duration']:.2f}s ({metrics['percentage']:.1%})")
        
        # Resource usage
        logger.info("\nResource usage:")
        logger.info(f"  Peak memory: {profile_data['peak_memory_gb']:.2f}GB")
        logger.info(f"  Memory allocations: {profile_data['allocation_count']:,}")
        logger.info(f"  GPU kernel launches: {profile_data['gpu_kernel_count']:,}")
        logger.info(f"  Network calls: {profile_data['network_call_count']}")
        logger.info(f"  Disk I/O: {profile_data['disk_io_mb']:.1f}MB")
        
        # Bottleneck analysis
        bottlenecks = profile_data.get('bottlenecks', [])
        if bottlenecks:
            logger.info("\nIdentified bottlenecks:")
            for bottleneck in bottlenecks:
                logger.info(f"  - {bottleneck['description']}")
                logger.info(f"    Impact: {bottleneck['impact']:.1%} of total time")
                logger.info(f"    Suggestion: {bottleneck['suggestion']}")
        
        # Export detailed profile
        profiler.export_flamegraph("profile_flame.svg")
        profiler.export_chrome_trace("profile_trace.json")
        logger.info("\nDetailed profiles exported to profile_flame.svg and profile_trace.json")
        
        return profile_data
        
    finally:
        await bob.shutdown()


async def main():
    """Run all advanced examples."""
    examples = [
        ("Custom Agent Behaviors", custom_agent_example),
        ("Task Pipelines", task_pipeline_example),
        ("Hardware Optimization", hardware_optimization_example),
        ("Parallel Execution Patterns", parallel_execution_patterns),
        ("Advanced Error Recovery", advanced_error_recovery),
        ("Performance Profiling", performance_profiling_example)
    ]
    
    for name, example_func in examples:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*70}")
        
        try:
            await example_func()
            logger.info(f"\n✓ {name} completed successfully")
        except Exception as e:
            logger.error(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Pause between examples
        await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())
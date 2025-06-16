#!/usr/bin/env python3
"""
Basic BOB usage examples for wheel trading system.

This demonstrates fundamental BOB operations including:
- System initialization
- Query execution
- Result processing
- Error handling
"""

import asyncio
import logging
from typing import Dict, Any

from bolt.core.integration import BoltIntegration
from bolt.error_handling import BoltException, ErrorSeverity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_query_example():
    """Execute a basic query with BOB."""
    # Initialize BOB with default settings
    bob = BoltIntegration(num_agents=8)
    
    try:
        # Initialize system (sub-second startup)
        logger.info("Initializing BOB system...")
        await bob.initialize()
        logger.info("BOB initialized successfully")
        
        # Execute a simple query
        query = "find all TODO comments in the trading module"
        logger.info(f"Executing query: {query}")
        
        result = await bob.execute_query(query)
        
        # Process results
        logger.info(f"Query completed in {result['total_duration']:.2f}s")
        logger.info(f"Tasks executed: {len(result['results'])}")
        
        # Display task results
        for task_result in result['results']:
            if task_result['success']:
                logger.info(f"✓ {task_result['task']}")
                if 'count' in task_result.get('result', {}):
                    logger.info(f"  Found {task_result['result']['count']} TODOs")
            else:
                logger.error(f"✗ {task_result['task']}: {task_result.get('error', 'Unknown error')}")
        
        return result
        
    except BoltException as e:
        logger.error(f"BOB error: {e.message}")
        if e.recovery_hints:
            logger.info("Recovery suggestions:")
            for hint in e.recovery_hints:
                logger.info(f"  - {hint}")
        raise
        
    finally:
        # Always cleanup
        logger.info("Shutting down BOB...")
        await bob.shutdown()


async def analysis_only_example():
    """Analyze a query without execution."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Complex query for analysis
        query = "refactor the entire risk management module for better performance"
        
        # Analyze without execution
        logger.info(f"Analyzing query: {query}")
        analysis = await bob.analyze_query(query)
        
        # Display analysis results
        logger.info("\n=== Query Analysis ===")
        logger.info(f"Estimated duration: {analysis['estimated_duration']:.1f}s")
        logger.info(f"Relevant files: {len(analysis['relevant_files'])}")
        logger.info(f"Planned tasks: {len(analysis['tasks'])}")
        logger.info(f"Required resources: CPU={analysis['resource_requirements']['cpu_cores']}, "
                   f"Memory={analysis['resource_requirements']['memory_gb']:.1f}GB")
        
        # Show planned tasks
        logger.info("\n=== Planned Tasks ===")
        for i, task in enumerate(analysis['tasks'], 1):
            logger.info(f"{i}. {task['description']} "
                       f"(Priority: {task['priority']}, "
                       f"Est. time: {task.get('estimated_duration', 'N/A')}s)")
        
        return analysis
        
    finally:
        await bob.shutdown()


async def context_aware_query():
    """Execute a query with additional context."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Query with trading context
        query = "optimize the wheel strategy implementation"
        
        # Provide context for better results
        context = {
            "focus_modules": ["unity_wheel.strategy", "unity_wheel.risk"],
            "optimization_goals": ["performance", "memory_usage", "readability"],
            "constraints": {
                "maintain_api_compatibility": True,
                "max_memory_increase_percent": 10
            },
            "current_metrics": {
                "avg_execution_time_ms": 150,
                "memory_usage_mb": 256
            }
        }
        
        logger.info(f"Executing context-aware query: {query}")
        result = await bob.execute_query(query, context=context)
        
        # Process optimization results
        logger.info("\n=== Optimization Results ===")
        for task_result in result['results']:
            if task_result['task'] == 'performance_analysis':
                metrics = task_result.get('result', {})
                logger.info(f"Performance improvements found: {metrics.get('improvement_count', 0)}")
                logger.info(f"Estimated speedup: {metrics.get('speedup_percent', 0):.1f}%")
            elif task_result['task'] == 'memory_optimization':
                metrics = task_result.get('result', {})
                logger.info(f"Memory optimizations: {metrics.get('optimization_count', 0)}")
                logger.info(f"Estimated savings: {metrics.get('memory_saved_mb', 0):.1f}MB")
        
        return result
        
    finally:
        await bob.shutdown()


async def error_handling_example():
    """Demonstrate error handling and recovery."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Query that might fail
        query = "analyze performance of non-existent module"
        
        try:
            result = await bob.execute_query(query, timeout=30.0)
            logger.info("Query succeeded unexpectedly")
            
        except BoltException as e:
            logger.warning(f"Expected error occurred: {e.message}")
            
            # Check severity and handle appropriately
            if e.severity == ErrorSeverity.LOW:
                logger.info("Low severity - continuing with partial results")
                # Access partial results if available
                if hasattr(e, 'partial_results'):
                    logger.info(f"Partial results available: {len(e.partial_results)} tasks completed")
                    
            elif e.severity == ErrorSeverity.MEDIUM:
                logger.warning("Medium severity - attempting recovery")
                # Try alternative approach
                alternative_query = "analyze performance of unity_wheel module instead"
                result = await bob.execute_query(alternative_query)
                logger.info("Recovery successful with alternative query")
                
            elif e.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                logger.error("High/Critical severity - cannot recover")
                raise
        
    finally:
        await bob.shutdown()


async def performance_monitoring_example():
    """Monitor performance during query execution."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Enable performance monitoring
        async with bob.performance_monitor() as monitor:
            # Execute complex query
            query = "analyze and optimize all database queries in the trading system"
            
            logger.info("Starting monitored query execution...")
            result = await bob.execute_query(query)
            
            # Get performance metrics
            metrics = monitor.get_metrics()
            
            logger.info("\n=== Performance Metrics ===")
            logger.info(f"Peak CPU usage: {metrics['peak_cpu_percent']:.1f}%")
            logger.info(f"Peak memory usage: {metrics['peak_memory_gb']:.1f}GB")
            logger.info(f"Average GPU utilization: {metrics['avg_gpu_percent']:.1f}%")
            logger.info(f"Total tasks processed: {metrics['total_tasks']}")
            logger.info(f"Average task duration: {metrics['avg_task_duration_ms']:.1f}ms")
            logger.info(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
            
            # Check for performance issues
            if metrics['peak_memory_gb'] > 16:
                logger.warning("High memory usage detected - consider reducing agent count")
            
            if metrics['avg_task_duration_ms'] > 1000:
                logger.warning("Slow task execution - consider query optimization")
        
        return result
        
    finally:
        await bob.shutdown()


async def batch_query_example():
    """Execute multiple related queries efficiently."""
    bob = BoltIntegration(num_agents=8)
    
    try:
        await bob.initialize()
        
        # Related queries to batch
        queries = [
            "find all deprecated functions in the codebase",
            "identify unused imports across all modules",
            "locate duplicate code patterns",
            "check for missing type hints in public APIs"
        ]
        
        logger.info(f"Executing batch of {len(queries)} queries...")
        
        # Execute queries in parallel
        tasks = [bob.execute_query(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process batch results
        logger.info("\n=== Batch Results ===")
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"✗ {query}: {str(result)}")
            else:
                success_count = sum(1 for r in result['results'] if r['success'])
                logger.info(f"✓ {query}: {success_count}/{len(result['results'])} tasks succeeded")
        
        return results
        
    finally:
        await bob.shutdown()


async def main():
    """Run all examples."""
    examples = [
        ("Basic Query", basic_query_example),
        ("Analysis Only", analysis_only_example),
        ("Context-Aware Query", context_aware_query),
        ("Error Handling", error_handling_example),
        ("Performance Monitoring", performance_monitoring_example),
        ("Batch Queries", batch_query_example)
    ]
    
    for name, example_func in examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running example: {name}")
        logger.info(f"{'='*60}")
        
        try:
            await example_func()
            logger.info(f"Example '{name}' completed successfully")
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}")
        
        # Brief pause between examples
        await asyncio.sleep(2)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(main())
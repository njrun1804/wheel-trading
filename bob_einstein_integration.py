#!/usr/bin/env python3
"""
BOB-Einstein Integration Example

Demonstrates how BOB's context gathering system integrates with Einstein's
search capabilities for ultra-fast comprehensive code understanding.
"""

import asyncio
import logging
import time
from pathlib import Path

from bob.context.gatherer import ContextGatherer, ContextRequest
from einstein.unified_index import UnifiedIndex
from einstein.high_performance_search import HighPerformanceSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BOBEinsteinIntegration:
    """
    Integration layer between BOB context gathering and Einstein search.
    
    Provides a unified interface for complex code understanding tasks
    with <200ms response time.
    """
    
    def __init__(self):
        # Initialize components
        self.gatherer = ContextGatherer(max_parallel_queries=12)
        self.einstein = UnifiedIndex()
        self.high_perf_search = HighPerformanceSearch()
        
        logger.info("BOB-Einstein integration initialized")
    
    async def analyze_code_pattern(self, pattern: str, context_depth: int = 3):
        """
        Analyze a code pattern across the codebase.
        
        Combines:
        - Semantic search for similar patterns
        - Structural analysis of implementations
        - Dependency tracking
        - Performance metrics
        """
        logger.info(f"Analyzing code pattern: {pattern}")
        start_time = time.time()
        
        # Gather comprehensive context
        request = ContextRequest(
            query=pattern,
            context_type="comprehensive",
            max_files=50,
            max_depth=context_depth,
            timeout_ms=200,
        )
        
        context = await self.gatherer.gather_context(request)
        
        # Analyze results
        analysis = {
            "pattern": pattern,
            "processing_time_ms": context.processing_time_ms,
            "implementations": [],
            "usage_patterns": [],
            "performance_insights": [],
            "recommendations": [],
        }
        
        # Extract implementations
        for match in context.semantic_matches[:10]:
            analysis["implementations"].append({
                "file": match["file_path"],
                "line": match["line_number"],
                "code": match["content"],
                "similarity": match["score"],
            })
        
        # Analyze usage patterns
        for symbol in context.key_symbols[:5]:
            deps = next((d for d in context.dependencies if d["symbol"] == symbol), None)
            if deps:
                analysis["usage_patterns"].append({
                    "symbol": symbol,
                    "used_by": len(deps.get("imported_by", [])),
                    "uses": len(deps.get("imports", [])),
                    "complexity": context.complexity_metrics.get(symbol, {}).get("avg", 0),
                })
        
        # Generate insights
        if context.analytical_insights:
            analysis["performance_insights"] = [
                {
                    "type": insight.get("type", "general"),
                    "message": insight.get("message", ""),
                    "severity": insight.get("severity", "info"),
                }
                for insight in context.analytical_insights[:5]
            ]
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(context)
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Pattern analysis completed in {elapsed:.1f}ms")
        
        return analysis
    
    async def find_optimization_opportunities(self, component: str):
        """
        Find optimization opportunities in a component.
        
        Uses Einstein's analytical capabilities to identify:
        - Performance bottlenecks
        - Code complexity issues
        - Redundant implementations
        - Refactoring opportunities
        """
        logger.info(f"Finding optimization opportunities for: {component}")
        
        # Search for component
        request = ContextRequest(
            query=f"{component} performance complexity optimization",
            context_type="analytical",
            include_semantic=True,
            include_analytical=True,
            include_dependencies=True,
            timeout_ms=300,
        )
        
        context = await self.gatherer.gather_context(request)
        
        opportunities = {
            "component": component,
            "bottlenecks": [],
            "complexity_issues": [],
            "redundancies": [],
            "refactoring_suggestions": [],
        }
        
        # Identify bottlenecks
        for file in context.related_files[:10]:
            metrics = await self._get_file_metrics(file)
            if metrics and metrics.get("execution_time_ms", 0) > 100:
                opportunities["bottlenecks"].append({
                    "file": file,
                    "execution_time_ms": metrics["execution_time_ms"],
                    "hot_spots": metrics.get("hot_spots", []),
                })
        
        # Find complexity issues
        for symbol, metrics in context.complexity_metrics.items():
            if isinstance(metrics, dict) and metrics.get("max", 0) > 10:
                opportunities["complexity_issues"].append({
                    "symbol": symbol,
                    "complexity": metrics["max"],
                    "recommendation": "Consider breaking down into smaller functions",
                })
        
        # Detect redundancies
        similar_impls = await self._find_similar_implementations(component)
        if len(similar_impls) > 1:
            opportunities["redundancies"] = similar_impls
        
        # Generate refactoring suggestions
        opportunities["refactoring_suggestions"] = await self._generate_refactoring_suggestions(
            context, opportunities
        )
        
        return opportunities
    
    async def trace_execution_path(self, entry_point: str, max_depth: int = 5):
        """
        Trace execution path from an entry point.
        
        Builds a complete execution graph showing:
        - Call hierarchy
        - Data flow
        - Side effects
        - Performance characteristics
        """
        logger.info(f"Tracing execution path from: {entry_point}")
        
        # Start with entry point
        request = ContextRequest(
            query=f"function {entry_point} calls dependencies",
            context_type="structural",
            include_structural=True,
            include_dependencies=True,
            max_depth=max_depth,
            timeout_ms=500,
        )
        
        context = await self.gatherer.gather_context(request)
        
        # Build execution graph
        execution_graph = {
            "entry_point": entry_point,
            "nodes": {},
            "edges": [],
            "performance_data": {},
        }
        
        # Process dependencies recursively
        visited = set()
        queue = [(entry_point, 0)]
        
        while queue and len(visited) < 100:  # Limit graph size
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current)
            
            # Find dependencies for current node
            deps = next(
                (d for d in context.dependencies if d["symbol"] == current),
                {"calls": [], "imports": []}
            )
            
            # Add node
            execution_graph["nodes"][current] = {
                "type": "function",
                "depth": depth,
                "calls": deps.get("calls", []),
                "imports": deps.get("imports", []),
            }
            
            # Add edges and queue children
            for called in deps.get("calls", []):
                execution_graph["edges"].append({
                    "from": current,
                    "to": called,
                    "type": "calls",
                })
                queue.append((called, depth + 1))
        
        # Add performance data
        for node in execution_graph["nodes"]:
            perf_data = await self._get_performance_data(node)
            if perf_data:
                execution_graph["performance_data"][node] = perf_data
        
        return execution_graph
    
    def _generate_recommendations(self, context) -> list[str]:
        """Generate recommendations based on context analysis."""
        recommendations = []
        
        # Check for high complexity
        if context.complexity_metrics:
            max_complexity = max(
                m.get("max", 0) for m in context.complexity_metrics.values()
                if isinstance(m, dict)
            )
            if max_complexity > 15:
                recommendations.append(
                    f"High complexity detected ({max_complexity}). "
                    "Consider refactoring complex functions."
                )
        
        # Check for too many dependencies
        if context.dependencies:
            max_deps = max(
                len(d.get("imports", [])) + len(d.get("calls", []))
                for d in context.dependencies
            )
            if max_deps > 20:
                recommendations.append(
                    f"High coupling detected ({max_deps} dependencies). "
                    "Consider reducing dependencies."
                )
        
        # Check for scattered implementation
        if len(context.related_files) > 10:
            recommendations.append(
                f"Implementation scattered across {len(context.related_files)} files. "
                "Consider consolidating related functionality."
            )
        
        return recommendations
    
    async def _get_file_metrics(self, file_path: str) -> dict:
        """Get performance metrics for a file."""
        # This would integrate with actual metrics collection
        # For now, return mock data
        return {
            "execution_time_ms": 50,
            "memory_usage_mb": 10,
            "hot_spots": ["line 42", "line 156"],
        }
    
    async def _find_similar_implementations(self, component: str) -> list[dict]:
        """Find similar implementations of a component."""
        # This would use semantic search to find duplicates
        # For now, return mock data
        return []
    
    async def _generate_refactoring_suggestions(self, context, opportunities) -> list[str]:
        """Generate specific refactoring suggestions."""
        suggestions = []
        
        if opportunities["bottlenecks"]:
            suggestions.append(
                "Profile and optimize performance bottlenecks in: " +
                ", ".join(b["file"] for b in opportunities["bottlenecks"][:3])
            )
        
        if opportunities["complexity_issues"]:
            suggestions.append(
                "Refactor complex functions: " +
                ", ".join(c["symbol"] for c in opportunities["complexity_issues"][:3])
            )
        
        return suggestions
    
    async def _get_performance_data(self, symbol: str) -> dict:
        """Get performance data for a symbol."""
        # This would integrate with tracing/profiling data
        # For now, return mock data
        return {
            "avg_execution_time_ms": 10,
            "call_count": 100,
            "memory_allocated_mb": 5,
        }


async def demonstrate_integration():
    """Demonstrate BOB-Einstein integration capabilities."""
    integration = BOBEinsteinIntegration()
    
    # 1. Analyze code pattern
    logger.info("\n" + "="*60)
    logger.info("1. Analyzing wheel trading pattern")
    logger.info("="*60)
    
    analysis = await integration.analyze_code_pattern(
        "wheel trading strategy calculate position size",
        context_depth=3
    )
    
    logger.info(f"Found {len(analysis['implementations'])} implementations")
    logger.info(f"Processing time: {analysis['processing_time_ms']:.1f}ms")
    
    for impl in analysis["implementations"][:3]:
        logger.info(f"  - {impl['file']}:{impl['line']} (similarity: {impl['similarity']:.2f})")
    
    if analysis["recommendations"]:
        logger.info("Recommendations:")
        for rec in analysis["recommendations"]:
            logger.info(f"  - {rec}")
    
    # 2. Find optimization opportunities
    logger.info("\n" + "="*60)
    logger.info("2. Finding optimization opportunities")
    logger.info("="*60)
    
    opportunities = await integration.find_optimization_opportunities("risk_management")
    
    if opportunities["complexity_issues"]:
        logger.info("Complexity issues found:")
        for issue in opportunities["complexity_issues"][:3]:
            logger.info(f"  - {issue['symbol']}: complexity={issue['complexity']}")
    
    if opportunities["refactoring_suggestions"]:
        logger.info("Refactoring suggestions:")
        for suggestion in opportunities["refactoring_suggestions"]:
            logger.info(f"  - {suggestion}")
    
    # 3. Trace execution path
    logger.info("\n" + "="*60)
    logger.info("3. Tracing execution path")
    logger.info("="*60)
    
    execution_graph = await integration.trace_execution_path("run_wheel_strategy", max_depth=3)
    
    logger.info(f"Execution graph: {len(execution_graph['nodes'])} nodes, {len(execution_graph['edges'])} edges")
    
    # Show call hierarchy
    for node, data in list(execution_graph["nodes"].items())[:5]:
        logger.info(f"  {' ' * data['depth'] * 2}{node} -> {', '.join(data['calls'][:3])}")


async def benchmark_integration():
    """Benchmark integration performance."""
    integration = BOBEinsteinIntegration()
    
    logger.info("\n" + "="*60)
    logger.info("Benchmarking BOB-Einstein Integration")
    logger.info("="*60)
    
    # Complex queries that exercise all capabilities
    queries = [
        ("analyze numpy usage patterns", "pattern analysis"),
        ("optimize data processing pipeline", "optimization search"),
        ("trace API request handling", "execution tracing"),
        ("find duplicate risk calculations", "similarity search"),
        ("analyze Einstein search performance", "meta-analysis"),
    ]
    
    total_time = 0
    for query, query_type in queries:
        start = time.time()
        
        if query_type == "pattern analysis":
            await integration.analyze_code_pattern(query)
        elif query_type == "optimization search":
            await integration.find_optimization_opportunities(query.split()[1])
        elif query_type == "execution tracing":
            await integration.trace_execution_path(query.split()[-1])
        else:
            # Generic context gathering
            request = ContextRequest(query=query, timeout_ms=300)
            await integration.gatherer.gather_context(request)
        
        elapsed = (time.time() - start) * 1000
        total_time += elapsed
        
        logger.info(f"{query_type}: {elapsed:.1f}ms - {query}")
    
    avg_time = total_time / len(queries)
    logger.info(f"\nBenchmark complete:")
    logger.info(f"  Average response time: {avg_time:.1f}ms")
    logger.info(f"  Total time: {total_time:.1f}ms")
    
    if avg_time < 200:
        logger.info("✅ Performance target achieved! (<200ms)")
    else:
        logger.warning(f"⚠️  Performance target missed: {avg_time:.1f}ms")


async def main():
    """Run integration demonstration and benchmarks."""
    try:
        await demonstrate_integration()
        await benchmark_integration()
        
        logger.info("\n✅ BOB-Einstein integration demonstration complete!")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
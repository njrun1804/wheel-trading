#!/usr/bin/env python3
"""
Integrated High-Performance Einstein Search System

Combines all optimizations for consistent sub-50ms search responses:
1. Performance-optimized search engine
2. Real-time performance monitoring
3. Memory optimization and management
4. Load testing and validation
5. Comprehensive metrics and alerting
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from einstein.performance_optimized_search import (
    PerformanceOptimizedSearch,
    OptimizedSearchMetrics,
    get_performance_optimized_search,
)
from einstein.search_memory_optimizer import get_search_memory_optimizer
from einstein.search_performance_monitor import get_search_performance_monitor
from einstein.high_performance_search import SearchResult

logger = logging.getLogger(__name__)


class IntegratedHighPerformanceSearch:
    """
    Integrated high-performance search system with comprehensive optimization.
    
    Achieves consistent sub-50ms search responses through:
    - Performance-optimized search engine
    - Real-time monitoring and alerting
    - Memory optimization and GC management
    - Load balancing and concurrency control
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        
        # Core components
        self.search_engine: Optional[PerformanceOptimizedSearch] = None
        self.performance_monitor = get_search_performance_monitor(project_root)
        self.memory_optimizer = get_search_memory_optimizer()
        
        # Integration state
        self.initialized = False
        self.running = False
        
        # Performance targets
        self.target_response_time_ms = 40  # Target 40ms for 50ms headroom
        self.target_success_rate_percent = 95  # 95% of queries under 50ms
        
    async def initialize(self):
        """Initialize all components of the integrated search system."""
        if self.initialized:
            logger.warning("Integrated search system already initialized")
            return
        
        logger.info("🚀 Initializing Integrated High-Performance Search System...")
        start_time = time.time()
        
        try:
            # Initialize search engine
            logger.info("   Initializing performance-optimized search engine...")
            self.search_engine = await get_performance_optimized_search(self.project_root)
            
            # Start performance monitoring
            logger.info("   Starting performance monitoring...")
            await self.performance_monitor.start_monitoring()
            
            # Start memory optimization
            logger.info("   Starting memory optimization...")
            await self.memory_optimizer.start_optimization()
            
            self.initialized = True
            self.running = True
            
            init_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Integrated search system initialized in {init_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Failed to initialize integrated search system: {e}")
            await self.cleanup()
            raise
    
    async def search(
        self, 
        query: str, 
        max_results: int = 50,
        timeout_ms: Optional[int] = None
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Execute optimized search with comprehensive monitoring.
        
        Returns:
            Tuple of (search_results, comprehensive_metrics)
        """
        if not self.initialized or not self.search_engine:
            raise RuntimeError("Integrated search system not initialized")
        
        search_start_time = time.time()
        
        try:
            # Execute search with performance optimization
            results, search_metrics = await self.search_engine.search(
                query=query,
                max_results=max_results,
                timeout_ms=timeout_ms or self.target_response_time_ms
            )
            
            # Record performance metrics
            self.performance_monitor.record_search_operation(
                response_time_ms=search_metrics.total_time_ms,
                cache_hit=search_metrics.cache_hit,
                error=False,
                query_type=search_metrics.strategy_used
            )
            
            # Prepare comprehensive metrics
            comprehensive_metrics = {
                "search_metrics": {
                    "query_hash": search_metrics.query_hash,
                    "total_time_ms": search_metrics.total_time_ms,
                    "cache_hit": search_metrics.cache_hit,
                    "result_count": search_metrics.result_count,
                    "strategy_used": search_metrics.strategy_used,
                    "target_met": search_metrics.total_time_ms < 50,
                },
                "system_performance": self.performance_monitor.get_current_performance(),
                "memory_stats": self.memory_optimizer.get_optimization_stats(),
                "timestamp": time.time(),
            }
            
            return results, comprehensive_metrics
            
        except Exception as e:
            # Record error
            error_time = (time.time() - search_start_time) * 1000
            self.performance_monitor.record_search_operation(
                response_time_ms=error_time,
                cache_hit=False,
                error=True,
                query_type="error"
            )
            
            logger.error(f"Search failed for query '{query}': {e}")
            
            # Return empty results with error info
            error_metrics = {
                "search_metrics": {
                    "query_hash": "error",
                    "total_time_ms": error_time,
                    "cache_hit": False,
                    "result_count": 0,
                    "strategy_used": "error",
                    "target_met": False,
                    "error": str(e),
                },
                "system_performance": self.performance_monitor.get_current_performance(),
                "memory_stats": self.memory_optimizer.get_optimization_stats(),
                "timestamp": time.time(),
            }
            
            return [], error_metrics
    
    async def batch_search(
        self, 
        queries: List[str], 
        max_results_per_query: int = 20
    ) -> List[Tuple[List[SearchResult], Dict[str, Any]]]:
        """Execute batch search with load balancing."""
        logger.info(f"Executing batch search for {len(queries)} queries")
        
        # Execute all queries in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(8)  # Limit concurrent searches
        
        async def search_with_semaphore(query):
            async with semaphore:
                return await self.search(query, max_results_per_query)
        
        start_time = time.time()
        results = await asyncio.gather(
            *[search_with_semaphore(query) for query in queries],
            return_exceptions=True
        )
        
        total_time = time.time() - start_time
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch search failed for query {i}: {result}")
                processed_results.append(([], {"error": str(result)}))
            else:
                processed_results.append(result)
        
        logger.info(f"Batch search completed in {total_time:.2f}s ({len(queries)/total_time:.1f} QPS)")
        
        return processed_results
    
    async def run_performance_validation(self) -> Dict[str, Any]:
        """Run comprehensive performance validation."""
        logger.info("🧪 Running performance validation...")
        
        validation_queries = [
            "class WheelStrategy",
            "def calculate_delta", 
            "import pandas",
            "TODO: implement",
            "FIXME: bug fix",
            "async def process",
            "Exception handling",
            "logger.error",
            "test_performance",
            "calculate options",
            "risk analysis",
            "performance optimization",
            "database connection",
            "machine learning",
            "algorithmic trading",
        ]
        
        # Run single query tests
        logger.info("   Testing individual query performance...")
        single_query_results = []
        
        for query in validation_queries[:10]:  # Test subset for individual queries
            results, metrics = await self.search(query)
            single_query_results.append({
                "query": query,
                "response_time_ms": metrics["search_metrics"]["total_time_ms"],
                "result_count": metrics["search_metrics"]["result_count"],
                "cache_hit": metrics["search_metrics"]["cache_hit"],
                "target_met": metrics["search_metrics"]["target_met"],
            })
        
        # Run batch test
        logger.info("   Testing batch query performance...")
        batch_start = time.time()
        batch_results = await self.batch_search(validation_queries)
        batch_time = time.time() - batch_start
        
        # Analyze batch results
        batch_response_times = []
        batch_target_met = 0
        batch_cache_hits = 0
        
        for results, metrics in batch_results:
            if "search_metrics" in metrics:
                response_time = metrics["search_metrics"]["total_time_ms"]
                batch_response_times.append(response_time)
                if metrics["search_metrics"]["target_met"]:
                    batch_target_met += 1
                if metrics["search_metrics"]["cache_hit"]:
                    batch_cache_hits += 1
        
        # Load test
        logger.info("   Running load test...")
        load_test_results = await self._run_load_test()
        
        # Compile validation report
        import numpy as np
        
        single_response_times = [r["response_time_ms"] for r in single_query_results]
        single_target_met = sum(1 for r in single_query_results if r["target_met"])
        
        validation_report = {
            "validation_timestamp": time.time(),
            "single_query_performance": {
                "queries_tested": len(single_query_results),
                "avg_response_time_ms": np.mean(single_response_times) if single_response_times else 0,
                "p50_response_time_ms": np.percentile(single_response_times, 50) if single_response_times else 0,
                "p99_response_time_ms": np.percentile(single_response_times, 99) if single_response_times else 0,
                "target_success_rate": (single_target_met / len(single_query_results) * 100) if single_query_results else 0,
                "details": single_query_results,
            },
            "batch_performance": {
                "queries_tested": len(validation_queries),
                "total_time_seconds": batch_time,
                "queries_per_second": len(validation_queries) / batch_time if batch_time > 0 else 0,
                "avg_response_time_ms": np.mean(batch_response_times) if batch_response_times else 0,
                "p99_response_time_ms": np.percentile(batch_response_times, 99) if batch_response_times else 0,
                "target_success_rate": (batch_target_met / len(batch_results) * 100) if batch_results else 0,
                "cache_hit_rate": (batch_cache_hits / len(batch_results) * 100) if batch_results else 0,
            },
            "load_test": load_test_results,
            "system_status": self.get_system_status(),
            "performance_grade": self._calculate_performance_grade(single_response_times, batch_response_times),
        }
        
        logger.info("✅ Performance validation complete")
        return validation_report
    
    async def _run_load_test(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Run load test for specified duration."""
        logger.info(f"   Running {duration_seconds}s load test...")
        
        load_queries = ["test query", "class", "def", "import", "TODO"] * 20
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        completed_searches = 0
        response_times = []
        errors = 0
        
        while time.time() < end_time:
            try:
                query = load_queries[completed_searches % len(load_queries)]
                results, metrics = await self.search(query, max_results=10)
                
                response_times.append(metrics["search_metrics"]["total_time_ms"])
                completed_searches += 1
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
            except Exception as e:
                errors += 1
                logger.debug(f"Load test search failed: {e}")
        
        actual_duration = time.time() - start_time
        
        import numpy as np
        
        return {
            "duration_seconds": actual_duration,
            "completed_searches": completed_searches,
            "queries_per_second": completed_searches / actual_duration,
            "errors": errors,
            "error_rate_percent": (errors / completed_searches * 100) if completed_searches > 0 else 0,
            "avg_response_time_ms": np.mean(response_times) if response_times else 0,
            "p95_response_time_ms": np.percentile(response_times, 95) if response_times else 0,
            "p99_response_time_ms": np.percentile(response_times, 99) if response_times else 0,
        }
    
    def _calculate_performance_grade(self, single_times: List[float], batch_times: List[float]) -> str:
        """Calculate overall performance grade."""
        import numpy as np
        
        if not single_times and not batch_times:
            return "F"
        
        all_times = single_times + batch_times
        if not all_times:
            return "F"
        
        p99_time = np.percentile(all_times, 99)
        avg_time = np.mean(all_times)
        success_rate = sum(1 for t in all_times if t < 50) / len(all_times) * 100
        
        # Grading criteria
        if p99_time < 30 and avg_time < 20 and success_rate >= 98:
            return "A+"
        elif p99_time < 40 and avg_time < 25 and success_rate >= 95:
            return "A"
        elif p99_time < 50 and avg_time < 30 and success_rate >= 90:
            return "B"
        elif p99_time < 75 and avg_time < 40 and success_rate >= 80:
            return "C"
        elif p99_time < 100 and success_rate >= 60:
            return "D"
        else:
            return "F"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        performance_report = self.performance_monitor.get_performance_report()
        memory_stats = self.memory_optimizer.get_optimization_stats()
        search_stats = self.search_engine.get_performance_stats() if self.search_engine else {}
        
        return {
            "system_running": self.running,
            "components_initialized": self.initialized,
            "performance_monitoring": performance_report,
            "memory_optimization": memory_stats,
            "search_engine": search_stats,
            "health_summary": {
                "performance_grade": performance_report.get("performance_status", {}).get("performance_grade", "Unknown"),
                "memory_pressure": memory_stats.get("current_memory", {}).get("level", "unknown"),
                "active_alerts": performance_report.get("alerts", {}).get("active_count", 0),
                "target_compliance": performance_report.get("performance_status", {}).get("target_met", False),
            }
        }
    
    async def cleanup(self):
        """Cleanup all components."""
        if not self.running:
            return
        
        logger.info("🧹 Cleaning up Integrated High-Performance Search System...")
        
        try:
            # Stop monitoring
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()
            
            # Stop memory optimization
            if self.memory_optimizer:
                await self.memory_optimizer.stop_optimization()
            
            # Cleanup search engine
            if self.search_engine:
                await self.search_engine.cleanup()
            
            self.running = False
            logger.info("✅ Integrated search system cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global instance
_integrated_search_system = None


async def get_integrated_search_system(project_root: Optional[Path] = None) -> IntegratedHighPerformanceSearch:
    """Get global integrated search system instance."""
    global _integrated_search_system
    
    if _integrated_search_system is None:
        _integrated_search_system = IntegratedHighPerformanceSearch(project_root)
        await _integrated_search_system.initialize()
    
    return _integrated_search_system


if __name__ == "__main__":
    async def comprehensive_search_demo():
        """Comprehensive demonstration of integrated search system."""
        print("🚀 Integrated High-Performance Einstein Search Demo")
        print("=" * 60)
        
        # Initialize system
        search_system = await get_integrated_search_system()
        
        # Individual search test
        print("\n1️⃣ Individual Search Performance Test:")
        test_queries = [
            "class WheelStrategy",
            "def calculate_delta",
            "import numpy", 
            "TODO: implement",
            "async def process",
            "Exception handling"
        ]
        
        for i, query in enumerate(test_queries, 1):
            results, metrics = await search_system.search(query)
            search_metrics = metrics["search_metrics"]
            
            status = "✅" if search_metrics["target_met"] else "❌"
            cache_status = "CACHED" if search_metrics["cache_hit"] else "FRESH"
            
            print(f"   Query {i} [{cache_status}]: {search_metrics['total_time_ms']:.1f}ms {status}")
            print(f"     - '{query}' → {search_metrics['result_count']} results")
        
        # Batch search test
        print("\n2️⃣ Batch Search Performance Test:")
        batch_queries = [
            "performance optimization", "machine learning", "database query",
            "error handling", "unit testing", "async processing"
        ]
        
        batch_results = await search_system.batch_search(batch_queries)
        
        batch_times = []
        for i, (results, metrics) in enumerate(batch_results, 1):
            if "search_metrics" in metrics:
                response_time = metrics["search_metrics"]["total_time_ms"]
                result_count = metrics["search_metrics"]["result_count"]
                batch_times.append(response_time)
                
                status = "✅" if response_time < 50 else "❌"
                print(f"   Batch {i}: {response_time:.1f}ms → {result_count} results {status}")
        
        # Performance validation
        print("\n3️⃣ Comprehensive Performance Validation:")
        validation_report = await search_system.run_performance_validation()
        
        single_perf = validation_report["single_query_performance"]
        batch_perf = validation_report["batch_performance"]
        load_test = validation_report["load_test"]
        
        print(f"   Single Query Performance:")
        print(f"     - Average: {single_perf['avg_response_time_ms']:.1f}ms")
        print(f"     - P99: {single_perf['p99_response_time_ms']:.1f}ms")
        print(f"     - Success rate: {single_perf['target_success_rate']:.1f}%")
        
        print(f"   Batch Performance:")
        print(f"     - QPS: {batch_perf['queries_per_second']:.1f}")
        print(f"     - P99: {batch_perf['p99_response_time_ms']:.1f}ms")
        print(f"     - Success rate: {batch_perf['target_success_rate']:.1f}%")
        print(f"     - Cache hit rate: {batch_perf['cache_hit_rate']:.1f}%")
        
        print(f"   Load Test ({load_test['duration_seconds']:.1f}s):")
        print(f"     - QPS: {load_test['queries_per_second']:.1f}")
        print(f"     - P99: {load_test['p99_response_time_ms']:.1f}ms")
        print(f"     - Error rate: {load_test['error_rate_percent']:.1f}%")
        
        # System status
        print("\n4️⃣ System Status:")
        system_status = search_system.get_system_status()
        health = system_status["health_summary"]
        
        print(f"   Performance Grade: {validation_report['performance_grade']}")
        print(f"   Memory Pressure: {health['memory_pressure']}")
        print(f"   Active Alerts: {health['active_alerts']}")
        print(f"   Target Compliance: {'✅' if health['target_compliance'] else '❌'}")
        
        # Final assessment
        print("\n5️⃣ Performance Assessment:")
        grade = validation_report['performance_grade']
        
        if grade in ['A+', 'A']:
            print("   🎉 EXCELLENT: Search system exceeds performance targets!")
        elif grade == 'B':
            print("   ✅ GOOD: Search system meets performance targets")
        elif grade == 'C':
            print("   ⚠️  ACCEPTABLE: Search system meets minimum requirements")
        else:
            print("   ❌ NEEDS IMPROVEMENT: Search system below targets")
        
        # Show specific achievements
        p99_under_50 = (single_perf['p99_response_time_ms'] < 50 and 
                       batch_perf['p99_response_time_ms'] < 50 and
                       load_test['p99_response_time_ms'] < 50)
        
        success_rate_good = (single_perf['target_success_rate'] >= 95 and
                           batch_perf['target_success_rate'] >= 95)
        
        if p99_under_50:
            print("   🎯 P99 latency under 50ms target achieved!")
        
        if success_rate_good:
            print("   📈 95%+ success rate achieved!")
        
        if batch_perf['queries_per_second'] >= 20:
            print("   ⚡ High throughput (>20 QPS) achieved!")
        
        # Cleanup
        await search_system.cleanup()
        
        print("\n✅ Integrated search system demo complete!")
    
    asyncio.run(comprehensive_search_demo())
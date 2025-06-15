#!/usr/bin/env python3
"""
Einstein Optimization Suite for Claude Code CLI

Complete optimization suite that integrates all Einstein optimizations:
1. Sub-50ms search responses
2. Minimal startup time (<500ms)
3. Efficient memory usage (<2GB)
4. Comprehensive coverage (100% of codebase)

This is the main entry point for Claude Code CLI integration.
"""

import asyncio
import time
import json
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Import all optimization modules
from einstein.claude_code_optimizer import ClaudeCodeOptimizer, get_claude_code_optimizer
from einstein.rapid_startup import RapidStartupManager, get_rapid_startup_manager
from einstein.memory_optimizer import MemoryOptimizer, get_memory_optimizer
from einstein.coverage_analyzer import CoverageAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResults:
    """Complete optimization results."""
    startup_time_ms: float
    avg_search_time_ms: float
    memory_usage_mb: float
    coverage_percentage: float
    
    # Performance targets
    startup_target_met: bool
    search_target_met: bool
    memory_target_met: bool
    coverage_target_met: bool
    
    # Detailed metrics
    startup_profile: Dict[str, Any]
    search_performance: Dict[str, Any]
    memory_profile: Dict[str, Any]
    coverage_metrics: Dict[str, Any]
    
    # Overall assessment
    overall_score: float
    assessment: str
    recommendations: List[str]


class EinsteinOptimizationSuite:
    """Complete optimization suite for Claude Code CLI."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        
        # Performance targets for Claude Code CLI
        self.targets = {
            'max_startup_time_ms': 500,
            'max_search_time_ms': 50,
            'max_memory_usage_mb': 2048,
            'min_coverage_percentage': 95.0
        }
        
        # Initialize optimization components
        self.claude_optimizer = get_claude_code_optimizer()
        self.startup_manager = get_rapid_startup_manager()
        self.memory_optimizer = get_memory_optimizer()
        self.coverage_analyzer = CoverageAnalyzer(self.project_root)
        
        # Performance tracking
        self.optimization_history: List[OptimizationResults] = []
        self.last_optimization_time = 0
        
    async def run_complete_optimization(self, enable_monitoring: bool = True) -> OptimizationResults:
        """Run complete optimization suite and return results."""
        print("🚀 Einstein Optimization Suite - Optimizing for Claude Code CLI")
        print("=" * 70)
        
        optimization_start = time.time()
        
        # Phase 1: Rapid Startup Optimization
        print("\n📈 Phase 1: Rapid Startup Optimization")
        startup_profile = await self._optimize_startup()
        
        # Phase 2: Search Performance Optimization
        print("\n⚡ Phase 2: Search Performance Optimization")
        search_performance = await self._optimize_search_performance()
        
        # Phase 3: Memory Usage Optimization
        print("\n🧠 Phase 3: Memory Usage Optimization")
        memory_profile = await self._optimize_memory_usage()
        
        # Phase 4: Coverage Analysis and Gap Filling
        print("\n📊 Phase 4: Coverage Analysis and Gap Filling")
        coverage_metrics = await self._optimize_coverage()
        
        # Phase 5: System Integration and Monitoring
        if enable_monitoring:
            print("\n🔧 Phase 5: System Integration and Monitoring")
            await self._setup_monitoring()
        
        # Calculate overall results
        results = self._calculate_optimization_results(
            startup_profile, search_performance, memory_profile, coverage_metrics
        )
        
        # Store in history
        self.optimization_history.append(results)
        self.last_optimization_time = time.time()
        
        total_time = (time.time() - optimization_start) * 1000
        
        # Display final results
        self._display_optimization_summary(results, total_time)
        
        return results
    
    async def _optimize_startup(self) -> Dict[str, Any]:
        """Optimize startup performance."""
        print("   Initializing rapid startup manager...")
        
        startup_start = time.time()
        profile = await self.startup_manager.rapid_initialize()
        startup_time = (time.time() - startup_start) * 1000
        
        print(f"   ✅ Startup completed in {startup_time:.1f}ms")
        print(f"   📊 Critical path: {profile.critical_path_ms:.1f}ms")
        print(f"   🔧 Components loaded: {profile.components_loaded}")
        
        return {
            'total_time_ms': startup_time,
            'critical_path_ms': profile.critical_path_ms,
            'background_init_ms': profile.background_init_ms,
            'components_loaded': profile.components_loaded,
            'memory_usage_mb': profile.memory_usage_mb,
            'target_met': startup_time <= self.targets['max_startup_time_ms']
        }
    
    async def _optimize_search_performance(self) -> Dict[str, Any]:
        """Optimize search performance."""
        print("   Applying Claude Code CLI optimizations...")
        
        # Apply Claude Code specific optimizations
        await self.claude_optimizer.optimize_for_claude_code()
        
        # Test search performance with Claude Code patterns
        test_queries = [
            "class WheelStrategy",
            "def calculate_delta", 
            "import pandas",
            "async def process",
            "Exception",
            "TODO",
            "test_",
            "@property",
            "if __name__",
            "logging.getLogger"
        ]
        
        search_times = []
        results_counts = []
        
        print("   🔍 Testing search performance...")
        
        for query in test_queries:
            search_start = time.time()
            results = await self.claude_optimizer.rapid_search(query, max_results=10)
            search_time = (time.time() - search_start) * 1000
            
            search_times.append(search_time)
            results_counts.append(len(results))
            
            if search_time > self.targets['max_search_time_ms']:
                print(f"   ⚠️ '{query}': {search_time:.1f}ms (above target)")
            else:
                print(f"   ✅ '{query}': {search_time:.1f}ms")
        
        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)
        min_search_time = min(search_times)
        avg_results = sum(results_counts) / len(results_counts)
        
        # Get performance statistics
        perf_stats = await self.claude_optimizer.get_performance_stats()
        
        return {
            'avg_search_time_ms': avg_search_time,
            'max_search_time_ms': max_search_time,
            'min_search_time_ms': min_search_time,
            'avg_results_count': avg_results,
            'total_searches': len(search_times),
            'target_met_count': len([t for t in search_times if t <= self.targets['max_search_time_ms']]),
            'target_met_percentage': len([t for t in search_times if t <= self.targets['max_search_time_ms']]) / len(search_times) * 100,
            'performance_stats': perf_stats,
            'target_met': avg_search_time <= self.targets['max_search_time_ms']
        }
    
    async def _optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        print("   Analyzing memory usage...")
        
        initial_memory = self.memory_optimizer.get_memory_usage_mb()
        
        # Run memory optimization
        freed_mb = await self.memory_optimizer.optimize_memory_usage()
        
        # Get detailed memory profile
        memory_profile = self.memory_optimizer.get_memory_profile()
        
        # Get cache statistics
        cache_stats = self.memory_optimizer.get_cache_stats()
        
        # Get optimization recommendations
        recommendations = self.memory_optimizer.get_optimization_recommendations()
        
        print(f"   📊 Memory usage: {memory_profile.total_mb:.1f}MB")
        print(f"   🧹 Memory freed: {freed_mb:.1f}MB")
        print(f"   💾 Cache usage: {memory_profile.cache_mb:.1f}MB")
        
        return {
            'total_mb': memory_profile.total_mb,
            'heap_mb': memory_profile.heap_mb,
            'cache_mb': memory_profile.cache_mb,
            'indexes_mb': memory_profile.indexes_mb,
            'freed_mb': freed_mb,
            'gc_collections': memory_profile.gc_collections,
            'cache_stats': cache_stats,
            'recommendations': recommendations,
            'target_met': memory_profile.total_mb <= self.targets['max_memory_usage_mb']
        }
    
    async def _optimize_coverage(self) -> Dict[str, Any]:
        """Optimize codebase coverage."""
        print("   Performing comprehensive coverage scan...")
        
        # Perform coverage analysis
        coverage_metrics = await self.coverage_analyzer.perform_comprehensive_scan()
        
        print(f"   📊 Coverage: {coverage_metrics.coverage_percentage:.1f}%")
        print(f"   📁 Files: {coverage_metrics.indexed_files}/{coverage_metrics.total_files}")
        print(f"   📝 Lines: {coverage_metrics.indexed_lines:,}/{coverage_metrics.total_lines:,}")
        
        # Fill coverage gaps if below target
        gap_fill_result = None
        if coverage_metrics.coverage_percentage < self.targets['min_coverage_percentage']:
            print("   🔧 Filling coverage gaps...")
            gap_fill_result = await self.coverage_analyzer.auto_fill_gaps(max_gaps=100)
            print(f"   ✅ Filled {gap_fill_result['gaps_filled']} gaps")
            
            # Re-scan after gap filling
            coverage_metrics = await self.coverage_analyzer.perform_comprehensive_scan()
            print(f"   📊 Updated coverage: {coverage_metrics.coverage_percentage:.1f}%")
        
        # Generate coverage report
        coverage_report = self.coverage_analyzer.get_coverage_report()
        
        return {
            'coverage_percentage': coverage_metrics.coverage_percentage,
            'total_files': coverage_metrics.total_files,
            'indexed_files': coverage_metrics.indexed_files,
            'total_lines': coverage_metrics.total_lines,
            'indexed_lines': coverage_metrics.indexed_lines,
            'modality_coverage': coverage_metrics.modality_coverage,
            'gap_files_count': len(coverage_metrics.gap_files),
            'gap_fill_result': gap_fill_result,
            'detailed_report': coverage_report,
            'target_met': coverage_metrics.coverage_percentage >= self.targets['min_coverage_percentage']
        }
    
    async def _setup_monitoring(self) -> None:
        """Setup system monitoring."""
        print("   Setting up real-time monitoring...")
        
        # Start memory monitoring
        await self.memory_optimizer.start_memory_monitoring(interval_seconds=60)
        
        # Start coverage monitoring (less frequent)
        asyncio.create_task(
            self.coverage_analyzer.monitor_coverage_real_time(interval_seconds=300)
        )
        
        print("   ✅ Monitoring systems active")
    
    def _calculate_optimization_results(self, startup_profile: Dict[str, Any],
                                      search_performance: Dict[str, Any],
                                      memory_profile: Dict[str, Any],
                                      coverage_metrics: Dict[str, Any]) -> OptimizationResults:
        """Calculate overall optimization results."""
        
        # Extract key metrics
        startup_time_ms = startup_profile['total_time_ms']
        avg_search_time_ms = search_performance['avg_search_time_ms']
        memory_usage_mb = memory_profile['total_mb']
        coverage_percentage = coverage_metrics['coverage_percentage']
        
        # Check targets
        startup_target_met = startup_time_ms <= self.targets['max_startup_time_ms']
        search_target_met = avg_search_time_ms <= self.targets['max_search_time_ms']
        memory_target_met = memory_usage_mb <= self.targets['max_memory_usage_mb']
        coverage_target_met = coverage_percentage >= self.targets['min_coverage_percentage']
        
        # Calculate overall score (0-100)
        score_components = []
        
        # Startup score (25% weight)
        startup_score = min(100, (self.targets['max_startup_time_ms'] / startup_time_ms) * 100)
        score_components.append(startup_score * 0.25)
        
        # Search score (35% weight - most important for Claude Code)
        search_score = min(100, (self.targets['max_search_time_ms'] / avg_search_time_ms) * 100)
        score_components.append(search_score * 0.35)
        
        # Memory score (20% weight)
        memory_score = min(100, (self.targets['max_memory_usage_mb'] / memory_usage_mb) * 100)
        score_components.append(memory_score * 0.20)
        
        # Coverage score (20% weight)
        coverage_score = min(100, (coverage_percentage / self.targets['min_coverage_percentage']) * 100)
        score_components.append(coverage_score * 0.20)
        
        overall_score = sum(score_components)
        
        # Generate assessment
        if overall_score >= 90 and all([startup_target_met, search_target_met, memory_target_met, coverage_target_met]):
            assessment = "EXCELLENT - All targets exceeded"
        elif overall_score >= 80 and sum([startup_target_met, search_target_met, memory_target_met, coverage_target_met]) >= 3:
            assessment = "GOOD - Most targets met"
        elif overall_score >= 70:
            assessment = "ACCEPTABLE - Some targets met"
        else:
            assessment = "NEEDS IMPROVEMENT - Targets not met"
        
        # Generate recommendations
        recommendations = []
        if not startup_target_met:
            recommendations.append(f"Optimize startup time: {startup_time_ms:.1f}ms > {self.targets['max_startup_time_ms']}ms target")
        if not search_target_met:
            recommendations.append(f"Optimize search performance: {avg_search_time_ms:.1f}ms > {self.targets['max_search_time_ms']}ms target")
        if not memory_target_met:
            recommendations.append(f"Reduce memory usage: {memory_usage_mb:.1f}MB > {self.targets['max_memory_usage_mb']}MB target")
        if not coverage_target_met:
            recommendations.append(f"Improve coverage: {coverage_percentage:.1f}% < {self.targets['min_coverage_percentage']}% target")
        
        if not recommendations:
            recommendations.append("System is fully optimized for Claude Code CLI")
        
        return OptimizationResults(
            startup_time_ms=startup_time_ms,
            avg_search_time_ms=avg_search_time_ms,
            memory_usage_mb=memory_usage_mb,
            coverage_percentage=coverage_percentage,
            startup_target_met=startup_target_met,
            search_target_met=search_target_met,
            memory_target_met=memory_target_met,
            coverage_target_met=coverage_target_met,
            startup_profile=startup_profile,
            search_performance=search_performance,
            memory_profile=memory_profile,
            coverage_metrics=coverage_metrics,
            overall_score=overall_score,
            assessment=assessment,
            recommendations=recommendations
        )
    
    def _display_optimization_summary(self, results: OptimizationResults, total_time_ms: float):
        """Display comprehensive optimization summary."""
        print("\n" + "=" * 70)
        print("🏆 EINSTEIN OPTIMIZATION SUITE - FINAL RESULTS")
        print("=" * 70)
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        print(f"   Startup Time:      {results.startup_time_ms:.1f}ms {'✅' if results.startup_target_met else '❌'} (target: {self.targets['max_startup_time_ms']}ms)")
        print(f"   Avg Search Time:   {results.avg_search_time_ms:.1f}ms {'✅' if results.search_target_met else '❌'} (target: {self.targets['max_search_time_ms']}ms)")
        print(f"   Memory Usage:      {results.memory_usage_mb:.1f}MB {'✅' if results.memory_target_met else '❌'} (target: {self.targets['max_memory_usage_mb']}MB)")
        print(f"   Coverage:          {results.coverage_percentage:.1f}% {'✅' if results.coverage_target_met else '❌'} (target: {self.targets['min_coverage_percentage']}%)")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        print(f"   Score: {results.overall_score:.1f}/100")
        print(f"   Status: {results.assessment}")
        print(f"   Optimization Time: {total_time_ms:.1f}ms")
        
        # Targets met summary
        targets_met = sum([results.startup_target_met, results.search_target_met, 
                          results.memory_target_met, results.coverage_target_met])
        print(f"   Targets Met: {targets_met}/4")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for i, recommendation in enumerate(results.recommendations, 1):
            print(f"   {i}. {recommendation}")
        
        # Claude Code CLI readiness
        claude_ready = all([results.startup_target_met, results.search_target_met, 
                           results.memory_target_met, results.coverage_target_met])
        
        print(f"\n🚀 Claude Code CLI Readiness:")
        if claude_ready:
            print("   ✅ READY - System fully optimized for Claude Code CLI")
        else:
            missing_targets = []
            if not results.startup_target_met:
                missing_targets.append("startup")
            if not results.search_target_met:
                missing_targets.append("search")
            if not results.memory_target_met:
                missing_targets.append("memory")
            if not results.coverage_target_met:
                missing_targets.append("coverage")
            print(f"   ⚠️ NEEDS WORK - Missing targets: {', '.join(missing_targets)}")
        
        print("\n" + "=" * 70)
    
    async def quick_health_check(self) -> Dict[str, Any]:
        """Perform quick health check of optimized system."""
        print("🔍 Einstein System Health Check...")
        
        health_start = time.time()
        
        # Test basic functionality
        diagnostics = await self.startup_manager.get_startup_diagnostics()
        
        # Test search performance
        search_start = time.time()
        test_results = await self.startup_manager.rapid_search("test", max_results=5)
        search_time = (time.time() - search_start) * 1000
        
        # Check memory usage
        current_memory = self.memory_optimizer.get_memory_usage_mb()
        
        # Get cache hit rates
        cache_stats = self.memory_optimizer.get_cache_stats()
        
        health_time = (time.time() - health_start) * 1000
        
        health_report = {
            'health_check_time_ms': health_time,
            'system_uptime_ms': diagnostics.get('uptime_ms', 0),
            'components_loaded': diagnostics.get('components_loaded', 0),
            'search_time_ms': search_time,
            'search_results_count': len(test_results),
            'memory_usage_mb': current_memory,
            'cache_stats': cache_stats,
            'system_healthy': (
                search_time <= self.targets['max_search_time_ms'] and
                current_memory <= self.targets['max_memory_usage_mb'] and
                diagnostics.get('components_loaded', 0) > 0
            )
        }
        
        print(f"✅ Health check complete in {health_time:.1f}ms")
        print(f"📊 Search test: {search_time:.1f}ms, {len(test_results)} results")
        print(f"🧠 Memory usage: {current_memory:.1f}MB")
        print(f"🏥 System healthy: {'✅' if health_report['system_healthy'] else '❌'}")
        
        return health_report
    
    def save_optimization_results(self, results: OptimizationResults, output_path: Path = None):
        """Save optimization results to file."""
        if output_path is None:
            output_path = self.project_root / ".einstein" / "optimization_results.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        results_dict = asdict(results)
        results_dict['timestamp'] = time.time()
        results_dict['targets'] = self.targets
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"💾 Optimization results saved to {output_path}")


async def main():
    """Main entry point for Einstein optimization suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Einstein Optimization Suite for Claude Code CLI")
    parser.add_argument("command", nargs='?', choices=[
        "optimize", "health-check", "benchmark"
    ], default="optimize", help="Command to execute")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--enable-monitoring", action="store_true", 
                       help="Enable real-time monitoring")
    parser.add_argument("--save-results", action="store_true",
                       help="Save optimization results to file")
    
    args = parser.parse_args()
    
    # Initialize optimization suite
    suite = EinsteinOptimizationSuite(args.project_root)
    
    try:
        if args.command == "optimize":
            print("🚀 Running complete Einstein optimization...")
            results = await suite.run_complete_optimization(
                enable_monitoring=args.enable_monitoring
            )
            
            if args.save_results:
                suite.save_optimization_results(results)
        
        elif args.command == "health-check":
            print("🔍 Running Einstein health check...")
            health_report = await suite.quick_health_check()
            
        elif args.command == "benchmark":
            print("📊 Running Einstein benchmark...")
            results = await suite.run_complete_optimization(enable_monitoring=False)
            
            # Additional benchmark output
            print(f"\n📈 Benchmark Results:")
            print(f"   Overall Score: {results.overall_score:.1f}/100")
            print(f"   Assessment: {results.assessment}")
            
    except KeyboardInterrupt:
        print("\n👋 Einstein optimization interrupted")
    except Exception as e:
        print(f"\n❌ Einstein optimization error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())